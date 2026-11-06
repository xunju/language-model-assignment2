#!/usr/bin/env bash
# 从任何工作目录运行，连续两次编译解决目录/引用。
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root/report"
for pass in 1 2; do
    xelatex -interaction=nonstopmode -halt-on-error -file-line-error report.tex > "build_pass_${pass}.log" 2>&1
done
cp report.pdf "$task_root/实验报告_解码器语言模型.pdf"
echo "Compiled PDF: $task_root/实验报告_解码器语言模型.pdf"
