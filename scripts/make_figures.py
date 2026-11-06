"""从已完成的原始实验日志生成报告插图；不生成或修改训练测量值。"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "training_log.csv").open() as f:
        raw = list(csv.DictReader(f))
    with (RESULTS / "evaluation_log.csv").open() as f:
        evaluation = list(csv.DictReader(f))
    steps = np.array([int(r["step"]) for r in raw])
    losses = np.array([float(r["loss"]) for r in raw])
    e_steps = np.array([int(r["step"]) for r in evaluation])
    e_train = np.array([float(r["train_loss"]) for r in evaluation])
    e_val = np.array([float(r["validation_loss"]) for r in evaluation])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 200})
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.plot(steps, losses, color="#9BB7CF", linewidth=0.5, alpha=0.5, label="Training minibatch (dropout on)")
    ax.plot(e_steps, e_train, "o-", color="#1769AA", markersize=3, label="Fixed training windows (dropout off)")
    ax.plot(e_steps, e_val, "s-", color="#C65E28", markersize=3, label="Fixed validation windows (dropout off)")
    ax.axhline(np.log(65), color="#7B8188", linestyle="--", linewidth=1, label="Uniform baseline: ln(65)")
    ax.set(xlabel="Optimizer step", ylabel="Next-character cross-entropy (nats)",
           title="4-layer decoder: training and validation loss")
    ax.grid(alpha=0.15)
    ax.legend(fontsize=8, loc="upper right")
    for extension in ("png", "pdf"):
        fig.savefig(FIGURES / f"training_loss.{extension}")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), constrained_layout=True)
    axes[0].plot(steps, [float(r["learning_rate"]) for r in raw], color="#1769AA")
    axes[0].set(xlabel="Optimizer step", ylabel="Learning rate", title="Warmup and cosine decay")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].plot(e_steps, np.exp(e_val), "s-", color="#C65E28", markersize=3)
    axes[1].set(xlabel="Optimizer step", ylabel="Character perplexity", title="Fixed validation windows")
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.savefig(FIGURES / "schedule_and_perplexity.pdf")
    plt.close(fig)

    # 按原样渲染原始控制台测量行，明确这是日志文本渲染而非另一次测量。
    lines = [line for line in (RESULTS / "training_console.log").read_text().splitlines() if line.startswith("step=")]
    selected = lines[:3] + ["..."] + lines[-4:]
    fig, ax = plt.subplots(figsize=(9, 2.8), facecolor="#17212B")
    ax.set_facecolor("#17212B")
    ax.axis("off")
    ax.text(0.02, 0.96, "Training log: 4-layer decoder", transform=ax.transAxes,
            color="#89C4E1", fontsize=11, family="monospace", va="top")
    ax.text(0.02, 0.80, "\n".join(selected), transform=ax.transAxes, color="#E0E7ED",
            fontsize=9, family="monospace", va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(FIGURES / "training_log_excerpt.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    # 用真实样本创建 LaTeX 可直接读取的纯文本，不手工润色。
    samples = json.loads((RESULTS / "small_samples.json").read_text())
    for i, sample in enumerate(samples, 1):
        (RESULTS / f"sample_{i}.txt").write_text(sample["text"], encoding="utf-8")
    large = json.loads((RESULTS / "large_summary.json").read_text())
    (RESULTS / "large_sample_raw.txt").write_text(large["text"], encoding="utf-8")
    print("Figures created from existing CSV/JSON/log files.")


if __name__ == "__main__":
    main()
