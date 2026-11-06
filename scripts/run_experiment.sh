#!/usr/bin/env bash
# 注意：完整重跑会覆盖当前 results 中的实验日志和小模型权重。
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p results
python -m pytest -q
python -u -m src.train > results/training_console.log 2>&1
python -u -m src.large_demo > results/large_console.log 2>&1
python scripts/verify_results.py
python scripts/make_figures.py
python scripts/report_values.py
bash scripts/build_report.sh
echo "Experiment and report completed."
