"""核对实际文件之间的一致性，独立加载权重复现首个采样，并留存验证证据。"""
import ast
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import ModelConfig, theoretical_parameters


def main():
    results = ROOT / "results"
    s = json.loads((results / "small_summary.json").read_text())
    large = json.loads((results / "large_summary.json").read_text())
    checks = {}
    check = lambda key, condition: checks.update({key: bool(condition)})
    raw_data = (ROOT / "data" / "tinyshakespeare.txt").read_bytes()
    check("data_sha256_matches", hashlib.sha256(raw_data).hexdigest() == s["data"]["sha256"])
    with (results / "training_log.csv").open() as f:
        train = list(csv.DictReader(f))
    with (results / "evaluation_log.csv").open() as f:
        evaluation = list(csv.DictReader(f))
    check("exactly_2000_consecutive_steps", [int(r["step"]) for r in train] == list(range(1, 2001)))
    check("all_training_losses_and_gradients_finite", all(
        math.isfinite(float(r["loss"])) and math.isfinite(float(r["gradient_norm_before_clip"])) for r in train))
    check("evaluation_steps_0_to_2000_every_100", [int(r["step"]) for r in evaluation] == list(range(0, 2001, 100)))
    check("summary_matches_final_csv", all(
        float(evaluation[-1][key]) == s["final"][key] for key in ("train_loss", "validation_loss", "validation_perplexity")))
    check("train_and_validation_losses_decrease", all(s["final"][k] < s["initial"][k] for k in ("train_loss", "validation_loss")))
    check("small_parameter_formula", s["parameters"] == theoretical_parameters(ModelConfig(**s["config"])))
    check("large_three_counts_agree", large["parameter_count"] == large["formula_count"] == large["torch_crosscheck_count"] == theoretical_parameters(ModelConfig(**large["config"])))
    check("large_within_one_percent_of_300M", abs(large["parameter_count"] / 300_000_000 - 1) < 0.01)
    samples = json.loads((results / "small_samples.json").read_text())
    check("three_small_samples_each_240_new_characters", len(samples) == 3 and all(
        len(x["text"]) - len(x["prompt"]) == 240 and x["text"].startswith(x["prompt"]) for x in samples))
    check("large_sample_80_new_characters", len(large["text"]) - len(large["prompt"]) == 80)
    model_ast = ast.parse((ROOT / "src" / "model.py").read_text())
    prohibited = {"MultiheadAttention", "Transformer", "TransformerEncoder", "TransformerDecoder",
                  "scaled_dot_product_attention", "LayerNorm", "layer_norm"}
    called_attributes = {node.func.attr for node in ast.walk(model_ast) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    check("model_avoids_prohibited_high_level_layers", not prohibited.intersection(called_attributes))
    demo_ast = ast.parse((ROOT / "src" / "large_demo.py").read_text())
    called_in_demo = {node.func.attr for node in ast.walk(demo_ast) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    check("large_demo_has_no_backward_or_optimizer", not {"backward", "step", "Adam", "AdamW", "SGD"}.intersection(called_in_demo))
    # 独立进程重新创建模型并加载最终权重，逐字核对训练后留存的样例。
    generated = subprocess.run([sys.executable, "-m", "src.generate"], cwd=ROOT,
                               text=True, capture_output=True, check=True)
    (results / "reproduced_sample.txt").write_text(generated.stdout, encoding="utf-8")
    check("checkpoint_reproduces_sample_one_exactly", generated.stdout == samples[0]["text"] + "\n")
    # 留存最终版本测试的控制台和机器可读结果，方便审阅而无需重新运行。
    test = subprocess.run([sys.executable, "-m", "pytest", "-q", "--junitxml=results/tests.xml"], cwd=ROOT,
                          text=True, capture_output=True)
    (results / "tests_console.log").write_text(test.stdout + test.stderr, encoding="utf-8")
    check("all_seven_semantic_tests_pass", test.returncode == 0 and "7 passed" in test.stdout)
    report = {"verified_at_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "checks": checks, "all_passed": all(checks.values()),
              "checkpoint_sha256": hashlib.sha256((results / "small_final.pt").read_bytes()).hexdigest()}
    (results / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit("实验一致性验证失败")


if __name__ == "__main__":
    main()
