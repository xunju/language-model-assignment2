"""关键语义测试：因果性、归一化数值/梯度、目标对齐、参数共享和过拟合。"""
import torch
import pytest
from src.model import (ModelConfig, ManualLayerNorm, CausalSelfAttention,
                       DecoderLM, count_parameters, theoretical_parameters)
from src.data import CharTokenizer, sample_batch

torch.set_num_threads(2)


def config(**changes):
    return ModelConfig(**({"vocab_size": 11, "context_length": 16,
                          "n_layers": 4, "d_model": 32, "n_heads": 4,
                          "dropout": 0.0} | changes))


def test_layernorm_values_and_gradients():
    torch.manual_seed(1)
    layer = ManualLayerNorm(8).double()
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    target = torch.nn.functional.layer_norm(x, (8,), layer.scale, layer.shift, layer.eps)
    actual = layer(x)
    torch.testing.assert_close(actual, target, rtol=1e-10, atol=1e-10)
    weights = torch.randn_like(x)
    grad_actual = torch.autograd.grad((actual * weights).sum(), (x, layer.scale, layer.shift), retain_graph=True)
    grad_target = torch.autograd.grad((target * weights).sum(), (x, layer.scale, layer.shift))
    for a, b in zip(grad_actual, grad_target):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
    constant = layer(torch.ones_like(x))
    assert torch.isfinite(constant).all()


def test_attention_mask_and_manual_reference():
    torch.manual_seed(2)
    att = CausalSelfAttention(config()).eval()
    x = torch.randn(2, 7, 32)
    actual, weights = att(x, return_weights=True)
    assert (weights.triu(diagonal=1) == 0).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 7))
    # 逐 head、逐时间的循环参考，与批量矩阵实现相互核对。
    q, k, v = att.qkv(x).chunk(3, dim=-1)
    q, k, v = [z.reshape(2, 7, 4, 8) for z in (q, k, v)]
    reference = torch.empty(2, 7, 4, 8)
    for b in range(2):
        for h in range(4):
            for t in range(7):
                scores = (k[b, :t + 1, h] * q[b, t, h]).sum(-1) / (8 ** 0.5)
                reference[b, t, h] = (scores.softmax(-1)[:, None] * v[b, :t + 1, h]).sum(0)
    torch.testing.assert_close(actual, att.output(reference.reshape(2, 7, 32)))


def test_full_model_future_token_invariance():
    torch.manual_seed(3)
    model = DecoderLM(config()).eval()
    ids = torch.randint(11, (2, 12))
    modified = ids.clone()
    modified[:, 6:] = (modified[:, 6:] + 1) % 11
    first = model(ids)[0]
    second = model(modified)[0]
    torch.testing.assert_close(first[:, :6], second[:, :6], rtol=0, atol=0)
    # 较短前缀应给出与完整序列一致的 logits。
    torch.testing.assert_close(first[:, :6], model(ids[:, :6])[0], atol=1e-6, rtol=1e-5)


def test_parameter_count_deduplicates_tied_head():
    model = DecoderLM(config())
    count = count_parameters(model)
    assert model.lm_head.weight is model.token_embedding.weight
    assert count["total"] == theoretical_parameters(model.cfg)
    assert count["total"] == sum(p.numel() for p in model.parameters())
    assert count["tensors"][0]["aliases"] == ["lm_head.weight"]


def test_shifted_targets_and_tokenizer():
    ids = torch.arange(100)
    x, y = sample_batch(ids, 4, 8, torch.Generator().manual_seed(0), "cpu")
    assert torch.equal(y, x + 1)
    tok = CharTokenizer(["\n", " ", "a", "b"])
    assert tok.decode(tok.encode("ab\n ba")) == "ab\n ba"
    with pytest.raises(ValueError):
        tok.encode("z")


def test_four_layer_tiny_batch_can_overfit():
    torch.manual_seed(4)
    model = DecoderLM(config())
    ids = torch.arange(9).repeat(4, 1) % 11
    x, y = ids[:, :-1], ids[:, 1:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
    initial = model(x, y)[1].item()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)[1]
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert model(x, y)[1].item() < initial * 0.1


def test_generation_crops_context_and_preserves_mode():
    model = DecoderLM(config(context_length=4)).train()
    prompt = torch.zeros(1, 7, dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=5, generator=torch.Generator().manual_seed(5))
    assert generated.shape == (1, 12)
    assert torch.equal(generated[:, :7], prompt)
    assert model.training
    with pytest.raises(ValueError):
        model(prompt)
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=10, d_model=33, n_heads=4)
