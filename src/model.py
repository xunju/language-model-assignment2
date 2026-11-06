"""原创的教学实现：显式因果注意力、LayerNorm、Pre-LN 解码器与采样。

不调用 MultiheadAttention、Transformer、scaled_dot_product_attention 或 LayerNorm。
Linear/Embedding/Dropout/GELU 是基础算子；输出头与词嵌入共享参数。
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int = 128
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model 必须能被 n_heads 整除")
        if min(self.vocab_size, self.context_length, self.n_layers,
               self.d_model, self.n_heads) <= 0:
            raise ValueError("模型维度必须为正整数")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout 必须在 [0, 1) 内")


class ManualLayerNorm(nn.Module):
    """沿最后一维做逐词元 LayerNorm，方差使用总体方差（除以 d）。"""

    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.shift = nn.Parameter(torch.zeros(width))
        self.eps = eps

    def forward(self, x):
        # 混合精度训练中用 FP32 求均值/方差，避免数值精度下降。
        original_dtype = x.dtype
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        mean = work.mean(dim=-1, keepdim=True)
        var = (work - mean).square().mean(dim=-1, keepdim=True)
        normalized = ((work - mean) * torch.rsqrt(var + self.eps)).to(original_dtype)
        return normalized * self.scale.to(original_dtype) + self.shift.to(original_dtype)


class CausalSelfAttention(nn.Module):
    """一次线性映射产生 Q/K/V，随后手工拆头、掩码、softmax 和合头。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.output = nn.Linear(cfg.d_model, cfg.d_model)
        self.attention_dropout = nn.Dropout(cfg.dropout)
        self.output_dropout = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool))
        # mask 不是可学习参数；不持久化可避免在 checkpoint 中重复储存。
        self.register_buffer("causal_mask", mask[None, None, :, :], persistent=False)

    def forward(self, x, return_weights=False):
        batch, length, width = x.shape
        if length > self.causal_mask.shape[-1]:
            raise ValueError("序列超过注意力上下文长度")
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # softmax 前屏蔽未来位置。每行至少保留自身，不会出现整行 -inf。
        scores = scores.masked_fill(~self.causal_mask[:, :, :length, :length], -torch.inf)
        weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        heads = self.attention_dropout(weights) @ v
        merged = heads.transpose(1, 2).contiguous().reshape(batch, length, width)
        result = self.output_dropout(self.output(merged))
        return (result, weights) if return_weights else result


class DecoderBlock(nn.Module):
    """GPT 风格 Pre-LN：归一化→子层→残差，前馈维度为 4d。"""

    def __init__(self, cfg):
        super().__init__()
        self.norm_attention = ManualLayerNorm(cfg.d_model)
        self.attention = CausalSelfAttention(cfg)
        self.norm_mlp = ManualLayerNorm(cfg.d_model)
        self.expand = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.contract = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.attention(self.norm_attention(x))
        h = self.contract(F.gelu(self.expand(self.norm_mlp(x))))
        return x + self.dropout(h)


class DecoderLM(nn.Module):
    """字符/词元索引→词嵌入与可学习位置编码→解码器堆叠→词表 logits。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.context_length, cfg.d_model)
        self.embedding_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = ManualLayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # 同一 Parameter 对象只初始化、统计、优化一次。
        self.lm_head.weight = self.token_embedding.weight
        self.reset_parameters()

    def reset_parameters(self):
        for name, p in self.named_parameters():
            if p.ndim >= 2:
                # 深层残差分支缩小初始化方差，降低残差累加的幅度。
                residual = name.endswith("attention.output.weight") or name.endswith("contract.weight")
                std = 0.02 / math.sqrt(2 * self.cfg.n_layers) if residual else 0.02
                nn.init.normal_(p, mean=0.0, std=std)
            elif name.endswith("scale"):
                nn.init.ones_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, indices, targets=None):
        if indices.ndim != 2 or indices.shape[1] < 1:
            raise ValueError("输入应为非空 [batch, time] 整数张量")
        length = indices.shape[1]
        if length > self.cfg.context_length:
            raise ValueError("输入超过 context_length；生成时应裁剪左侧历史")
        positions = torch.arange(length, device=indices.device)
        x = self.embedding_dropout(self.token_embedding(indices) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        # targets 已在数据采样阶段右移，模型中不再二次移动。
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, indices, max_new_tokens=200, temperature=0.8, top_k=20, generator=None):
        if temperature <= 0 or (top_k is not None and top_k < 1):
            raise ValueError("temperature 和 top_k 必须大于零")
        previous_mode = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = indices[:, -self.cfg.context_length:]
                logits, _ = self(context)
                last = logits[:, -1, :] / temperature
                if top_k is not None:
                    cutoff = torch.topk(last, min(top_k, last.shape[-1])).values[:, -1:]
                    last = last.masked_fill(last < cutoff, -torch.inf)
                probabilities = torch.softmax(last.float(), dim=-1)
                next_token = torch.multinomial(probabilities, 1, generator=generator)
                indices = torch.cat([indices, next_token], dim=1)
        finally:
            self.train(previous_mode)
        return indices


def count_parameters(model):
    """自行逐维相乘，按对象 id 去重共享权重；buffer 不计入参数。"""
    unique = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        key = id(parameter)
        if key in unique:
            unique[key]["aliases"].append(name)
            continue
        size = 1
        for dimension in parameter.shape:
            size *= dimension
        unique[key] = {"name": name, "shape": list(parameter.shape), "count": size,
                       "trainable": parameter.requires_grad, "aliases": []}
    groups = {"token_embedding": 0, "position_embedding": 0, "blocks": 0, "final_norm": 0}
    for entry in unique.values():
        group = entry["name"].split(".")[0]
        groups[group] = groups.get(group, 0) + entry["count"]
    return {"total": sum(e["count"] for e in unique.values()),
            "trainable": sum(e["count"] for e in unique.values() if e["trainable"]),
            "groups": groups, "tensors": list(unique.values())}


def theoretical_parameters(cfg):
    """独立公式核对：共享词嵌入、位置嵌入、L 个 block 和最终 LN。"""
    d = cfg.d_model
    # 单 block：注意力 4d²+4d，MLP 8d²+5d，两组 LN 4d。
    return (cfg.vocab_size + cfg.context_length) * d + cfg.n_layers * (12 * d * d + 13 * d) + 2 * d
