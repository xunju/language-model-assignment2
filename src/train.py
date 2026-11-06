"""4 层模型预训练：记录原始逐步损失、固定批次评估、最终权重和采样。"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
import csv
import json
import math
import platform
import random
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

from .data import ROOT, prepare_data, sample_batch
from .model import ModelConfig, DecoderLM, count_parameters, theoretical_parameters


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def precision_context(device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@torch.no_grad()
def evaluate(model, tokens, device, batch_size, n_batches, seed):
    """每次重建同一个采样器，保证曲线各点在完全相同的评估窗口上比较。"""
    previous = model.training
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(n_batches):
        x, y = sample_batch(tokens, batch_size, model.cfg.context_length, rng, device)
        with precision_context(device):
            loss = model(x, y)[1]
        values.append(loss.item())
    model.train(previous)
    return sum(values) / len(values)


def learning_rate(step, total_steps, peak, warmup):
    if step <= warmup:
        return peak * step / warmup
    fraction = (step - warmup) / max(1, total_steps - warmup)
    return peak * (0.1 + 0.45 * (1 + math.cos(math.pi * fraction)))


@torch.no_grad()
def generate_samples(model, tokenizer, device, max_new_tokens=240):
    samples = []
    for i, prompt in enumerate(["ROMEO:\n", "First Citizen:\n", "To be, or not to be"]):
        rng = torch.Generator(device=device).manual_seed(2026 + i)
        encoded = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        with precision_context(device):
            result = model.generate(encoded, max_new_tokens, temperature=0.8, top_k=20, generator=rng)
        samples.append({"prompt": prompt, "seed": 2026 + i, "temperature": 0.8, "top_k": 20,
                        "new_tokens": max_new_tokens, "text": tokenizer.decode(result[0].tolist())})
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_interval < 1 or args.eval_batches < 1:
        parser.error("步数、batch size 和评估间隔必须为正")
    seed_everything(args.seed)
    device = torch.device(args.device)
    tokenizer, train, validation, data_info = prepare_data()
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    cfg = ModelConfig(vocab_size=len(tokenizer.characters))
    model = DecoderLM(cfg).to(device)
    param_info = count_parameters(model)
    assert param_info["total"] == theoretical_parameters(cfg)
    # 偏置和 LayerNorm 不衰减；共享嵌入通过 parameters() 自动去重。
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=5e-4, betas=(0.9, 0.95), eps=1e-8)
    sampling_rng = torch.Generator().manual_seed(args.seed + 100)
    settings = {"steps": args.steps, "batch_size": args.batch_size, "peak_lr": 5e-4,
                "min_lr": 5e-5, "warmup_steps": 100, "betas": [0.9, 0.95],
                "weight_decay_matrix": 0.1, "gradient_clip": 1.0, "seed": args.seed,
                "eval_interval": args.eval_interval, "eval_batches": args.eval_batches,
                "train_eval_seed": 142, "validation_eval_seed": 242,
                "precision": "BF16 autocast; FP32 parameters/optimizer" if device.type == "cuda" else "FP32",
                "checkpoint_selection": "final step; no validation-based checkpoint selection"}
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
                   "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "platform": platform.platform(), "deterministic_algorithms": True,
                   "cublas_workspace_config": __import__('os').environ.get('CUBLAS_WORKSPACE_CONFIG'),
                   "started_at_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
    print(json.dumps({"config": asdict(cfg), "parameters": param_info["total"], "settings": settings,
                      "environment": environment}, indent=2), flush=True)
    (results / "small_parameters.json").write_text(json.dumps(param_info, indent=2), encoding="utf-8")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    evaluations = []
    with (results / "training_log.csv").open("w", newline="") as raw_file, \
         (results / "evaluation_log.csv").open("w", newline="") as eval_file:
        writer = csv.DictWriter(raw_file, fieldnames=["step", "loss", "learning_rate", "gradient_norm_before_clip", "elapsed_seconds"])
        writer.writeheader()
        eval_writer = csv.DictWriter(eval_file, fieldnames=["step", "train_loss", "validation_loss", "validation_perplexity", "elapsed_seconds"])
        eval_writer.writeheader()

        def log_evaluation(step):
            train_loss = evaluate(model, train, device, args.batch_size, args.eval_batches, 142)
            val_loss = evaluate(model, validation, device, args.batch_size, args.eval_batches, 242)
            record = {"step": step, "train_loss": train_loss, "validation_loss": val_loss,
                      "validation_perplexity": math.exp(val_loss), "elapsed_seconds": time.perf_counter() - started}
            evaluations.append(record)
            eval_writer.writerow(record)
            eval_file.flush()
            print(f"step={step:4d} train={train_loss:.4f} val={val_loss:.4f} "
                  f"val_ppl={math.exp(val_loss):.2f} elapsed={record['elapsed_seconds']:.1f}s", flush=True)

        log_evaluation(0)
        model.train()
        for step in range(1, args.steps + 1):
            lr = learning_rate(step, args.steps, settings["peak_lr"], settings["warmup_steps"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = sample_batch(train, args.batch_size, cfg.context_length, sampling_rng, device)
            optimizer.zero_grad(set_to_none=True)
            with precision_context(device):
                loss = model(x, y)[1]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"第 {step} 步损失非有限值")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            writer.writerow({"step": step, "loss": loss.item(), "learning_rate": lr,
                             "gradient_norm_before_clip": norm.item(), "elapsed_seconds": time.perf_counter() - started})
            if step % args.eval_interval == 0 or step == args.steps:
                raw_file.flush()
                log_evaluation(step)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    checkpoint = {"config": asdict(cfg), "state_dict": model.state_dict(),
                  "characters": tokenizer.characters, "step": args.steps, "seed": args.seed}
    torch.save(checkpoint, results / "small_final.pt")
    samples = generate_samples(model, tokenizer, device)
    (results / "small_samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    sample_text = "\n\n".join(f"=== Prompt {i + 1}; seed={s['seed']}; temperature=0.8; top_k=20 ===\n{s['text']}" for i, s in enumerate(samples))
    (results / "small_samples.txt").write_text(sample_text, encoding="utf-8")
    summary = {"config": asdict(cfg), "parameters": param_info["total"], "settings": settings,
               "environment": environment, "data": data_info, "initial": evaluations[0], "final": evaluations[-1],
               "evaluations": evaluations, "training_wall_seconds_including_eval": duration,
               "peak_cuda_allocated_bytes": peak_memory,
               "processed_training_tokens": args.steps * args.batch_size * cfg.context_length,
               "loss_reduction_train_percent": 100 * (1 - evaluations[-1]["train_loss"] / evaluations[0]["train_loss"]),
               "loss_reduction_validation_percent": 100 * (1 - evaluations[-1]["validation_loss"] / evaluations[0]["validation_loss"])}
    (results / "small_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Training completed. Saved final checkpoint and three unedited samples.", flush=True)
    print(sample_text, flush=True)


if __name__ == "__main__":
    main()
