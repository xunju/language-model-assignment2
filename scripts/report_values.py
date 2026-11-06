"""将真实 JSON 测量值注入 LaTeX，避免手工抄数和版本不一致。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    s = json.loads((ROOT / "results" / "small_summary.json").read_text())
    large = json.loads((ROOT / "results" / "large_summary.json").read_text())
    small_count = json.loads((ROOT / "results" / "small_parameters.json").read_text())
    large_count = json.loads((ROOT / "results" / "large_parameters.json").read_text())
    f = lambda x: f"{x:.4f}"
    values = {
        "SmallParams": f"{s['parameters']:,}", "LargeParams": f"{large['parameter_count']:,}",
        "InitialTrain": f(s['initial']['train_loss']), "InitialVal": f(s['initial']['validation_loss']),
        "FinalTrain": f(s['final']['train_loss']), "FinalVal": f(s['final']['validation_loss']),
        "InitialPPL": f"{s['initial']['validation_perplexity']:.2f}",
        "FinalPPL": f"{s['final']['validation_perplexity']:.2f}",
        "TrainReduction": f"{s['loss_reduction_train_percent']:.2f}",
        "ValReduction": f"{s['loss_reduction_validation_percent']:.2f}",
        "WallSeconds": f"{s['training_wall_seconds_including_eval']:.2f}",
        "SmallMemory": f"{s['peak_cuda_allocated_bytes'] / 2**20:.2f}",
        "LargeMemory": f"{large['peak_cuda_allocated_bytes'] / 2**20:.2f}",
        "TrainingTokens": f"{s['processed_training_tokens']:,}",
        "EffectivePasses": f"{s['processed_training_tokens'] / s['data']['train_tokens']:.2f}",
        "LargeOffset": f"{large['relative_to_300M_percent']:.3f}",
        "LargeSeconds": f"{large['generation_seconds']:.3f}",
        "LargeSpeed": f"{large['generation_tokens_per_second']:.2f}",
        "LargeProbeLoss": f(large['probe_next_token_loss']),
        "LargeParameterMiB": f"{large['parameter_bytes_fp32'] / 2**20:.2f}",
        "FinalGap": f(s['final']['validation_loss'] - s['final']['train_loss']),
        "ExperimentDate": s['environment']['started_at_beijing'][:10].replace('-', '年', 1).replace('-', '月', 1) + '日',
    }
    lines = ["% Auto-generated from actual results/*.json. Do not edit measured values."]
    lines += ["\\newcommand{\\" + key + "}{" + value + "}" for key, value in values.items()]
    (ROOT / "report" / "values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = []
    names = {"token_embedding": "词元嵌入（含共享输出头）", "position_embedding": "可学习位置嵌入",
             "blocks": "全部解码器块", "final_norm": "最终 LayerNorm"}
    for key, name in names.items():
        rows.append(f"{name} & {small_count['groups'][key]:,} & {large_count['groups'][key]:,} \\\\")
    rows += ["\\midrule", f"总计 & {s['parameters']:,} & {large['parameter_count']:,} \\\\"]
    rows.append("\\bottomrule")
    (ROOT / "report" / "parameter_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    rows = []
    for item in s['evaluations']:
        if item['step'] in (0, 100, 500, 1000, 1500, 2000):
            rows.append(f"{item['step']} & {item['train_loss']:.4f} & {item['validation_loss']:.4f} & "
                        f"{item['validation_perplexity']:.2f} \\\\")
    rows.append("\\bottomrule")
    (ROOT / "report" / "evaluation_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("LaTeX measurement tables generated from experiment results.")


if __name__ == "__main__":
    main()
