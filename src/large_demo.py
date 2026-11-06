"""约 300M 模型仅实例化、统计和推理；没有优化器，也不加载小模型权重。"""
import argparse
from dataclasses import asdict
import json
import math
import time

import torch

from .data import ROOT, prepare_data
from .model import ModelConfig, DecoderLM, count_parameters, theoretical_parameters
from .train import seed_everything, precision_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(300)
    device = torch.device(args.device)
    tokenizer, _, validation, _ = prepare_data()
    cfg = ModelConfig(vocab_size=len(tokenizer.characters), n_layers=24, d_model=1024,
                      n_heads=16, dropout=0.0)
    model = DecoderLM(cfg).to(device).eval()
    statistics = count_parameters(model)
    theoretical = theoretical_parameters(cfg)
    builtin_crosscheck = sum(p.numel() for p in model.parameters())
    assert statistics["total"] == theoretical == builtin_crosscheck
    # 真实 FP32 参数字节数，去重共享权重。
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "large_parameters.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    with torch.inference_mode():
        # 完整上下文做下词预测形状及有限值检查，不进行反向传播。
        x = validation[:cfg.context_length].unsqueeze(0).to(device)
        y = validation[1:cfg.context_length + 1].unsqueeze(0).to(device)
        with precision_context(device):
            logits, loss = model(x, y)
        assert logits.shape == (1, cfg.context_length, cfg.vocab_size)
        assert torch.isfinite(logits).all() and torch.isfinite(loss)
        probe_loss = loss.item()
        del logits, loss
        prompt = "ROMEO:\n"
        inputs = torch.tensor([tokenizer.encode(prompt)], device=device)
        rng = torch.Generator(device=device).manual_seed(301)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with precision_context(device):
            generated = model.generate(inputs, max_new_tokens=80, temperature=1.0, top_k=None, generator=rng)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
    summary = {"config": asdict(cfg), "parameter_count": statistics["total"], "formula_count": theoretical,
               "torch_crosscheck_count": builtin_crosscheck,
               "relative_to_300M_percent": 100 * (statistics["total"] / 300_000_000 - 1),
               "parameter_bytes_fp32": parameter_bytes, "initialization_seed": 300,
               "status": "RANDOM INITIALIZATION; NO TRAINING; INFERENCE PIPELINE DEMONSTRATION ONLY",
               "probe_logits_shape": [1, cfg.context_length, cfg.vocab_size],
               "probe_next_token_loss": probe_loss, "uniform_reference_loss": math.log(cfg.vocab_size),
               "generation_seed": 301, "prompt": prompt, "new_tokens": 80,
               "temperature": 1.0, "top_k": None, "text": tokenizer.decode(generated[0].tolist()),
               "generation_seconds": seconds, "generation_tokens_per_second": 80 / seconds,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    (results / "large_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (results / "large_sample.txt").write_text("Random initialization, not pretrained.\n" + summary["text"], encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
