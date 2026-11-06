# 编程作业二：手写解码器语言模型

这是一个可复现的课程作业项目：使用基础 PyTorch 张量运算实现因果多头自注意力、LayerNorm 和 GPT 风格解码器语言模型，在 Tiny Shakespeare 字符语料上训练 4 层小模型，并对约 300M 参数配置执行参数统计和推理演示。

项目已经保留了一次完整实验的源码、数据校验信息、训练日志、模型 checkpoint、报告和核验结果。重新运行实验会覆盖 `results/` 中的测量文件，请先确认需要保留已有结果。

## 目录结构

```text
src/       模型、数据准备、训练、生成和约 300M 推理演示
tests/     7 项语义测试：因果性、数值、梯度、对齐、共享参数等
data/      Tiny Shakespeare 语料、来源和 SHA-256 元数据、作业要求
results/   训练日志、评估 CSV、checkpoint、采样和机器可读核验结果
report/    LaTeX 源码、自动生成的指标表、图表和编译后的 PDF
scripts/   一键实验、结果核验、画图、报告编译和提交包脚本
```

本目录包含完整的 Python 源码、真实实验结果、最终小模型权重、中文 LaTeX 报告及编译 PDF。

作业先在 4 层小模型上调试和训练，约 300M 配置只做参数统计与推理。结果如下：

| 项目 | 配置与结果 |
| --- | --- |
| 小模型 | 4 层、d=256、4 头；3,208,960 参数 |
| 预训练 | Tiny Shakespeare，字符词表 65，长度 128，batch 32，2,000 步 |
| 固定窗口训练损失 | 4.2531 → 1.4649 |
| 固定窗口验证损失 | 4.2551 → 1.6473 |
| 验证字符困惑度 | 70.46 → 5.19 |
| 大模型 | 24 层、d=1024、16 头；302,509,056 参数 |
| 大模型推理 | 随机初始化，无训练；完整前向 + 80 字符生成 |
| 语义测试 | 7 项通过；另附文件一致性与 checkpoint 采样复现核验 |

## 文件说明

- `实验报告_解码器语言模型.pdf`：修改后的报告。第一页为紧凑标题、小组信息、摘要和实验正文，不再设独立封面。
- `report/report.tex`：LaTeX 主文件，中文字体 Noto CJK，使用 XeLaTeX。
- `src/`：有注释的完整实现；注意力和 LayerNorm 均手写。
- `tests/test_model.py`：因果性、数值与梯度、共享计数、目标右移、过拟合和生成边界测试。
- `data/`：实际使用的语料和含 SHA-256 的数据记录。
- `results/`：原始逐步日志、固定窗口评估、参数明细、采样、最终 checkpoint、核验结果。
- `report/figures/training_loss.png`：训练损失图；另有适合 LaTeX 的矢量 PDF。
- `编程作业二_完整提交包.zip`：以上文件的提交归档；`MANIFEST.sha256` 为归档内容校验清单。

小组均为人智2401班：石皓文（2233316022）、张恩瑞（2244413862）、杜鹏宇（2223210437）、许程（2244313744）。
报告已删除原参考文献第一条课程作业说明，保留四条技术与数据来源。

## 实现说明

`src/model.py` 不调用 MultiheadAttention、Transformer、现成 GPT、融合 SDPA 或 LayerNorm。
Q/K/V、拆头合头、下三角 mask、softmax 由基础张量运算组织。
LayerNorm 手工计算总体方差，注意力 softmax 和低精度归一化统计在 FP32 中进行。
模型采用 Pre-LN、GELU、可学习位置编码，输出头共享词嵌入。
自定义统计函数对共享 Parameter 的对象 id 去重，再以公式和 PyTorch 统计复核。

作业允许但不强制 BPE，本次采用字符级分词。词表只从训练分区构建，数据按连续 90%/10% 划分，窗口不跨分区。
评估窗口每次用固定独立种子重建，dropout 关闭；CSV 保留 2,000 个当步训练损失和 21 个评估节点。
样例均为真实未润色输出；大模型随机权重输出不代表语言能力。

## 快速开始

建议使用 Python 3.10 或更新版本，并先安装与你的 CUDA 或 CPU 环境匹配的 PyTorch：

```bash
python -m pip install torch
python -m pip install -r requirements.txt
python -m pytest -q
```

只运行已保存 checkpoint 的生成：

```bash
python -m src.generate --device cpu --tokens 80
```

也可以分别运行单个环节：

```bash
python -m src.data                 # 准备数据并打印数据元信息
python -m src.train --device cpu   # 训练 4 层小模型
python -m src.large_demo --device cpu
python scripts/verify_results.py
python scripts/check_delivery.py
```

默认命令会自动使用 CUDA（如果可用），也可以显式指定 `--device cpu` 或 `--device cuda`。首次运行缺少语料时，`src.data` 会从公开 Tiny Shakespeare 地址下载；已经随仓库提供的 `data/tinyshakespeare.txt` 可直接离线使用。

完整复现实验、生成图表并编译报告：

```bash
bash scripts/run_experiment.sh
```

`run_experiment.sh` 会训练小模型、运行约 300M 模型演示、生成图表、更新报告中的自动测量值，并执行结果核验。编译中文 PDF 还需要安装 XeLaTeX，例如 Ubuntu/Debian：

```bash
sudo apt-get install texlive-xetex texlive-lang-chinese texlive-fonts-recommended fonts-noto-cjk
```

## 运行环境

已验证：Python 3.10.8、PyTorch 2.1.2+cu118、NumPy 1.26.4、单张 RTX 3090（24 GiB）。
先安装匹配硬件的 PyTorch，再安装 `requirements.txt` 中其余依赖。CPU 也可运行，耗时会增加。
精确依赖版本见 `results/python_environment.txt`。

实验记录的环境见 `results/python_environment.txt`。GPU 运行时可设置以下变量以获得与原实验相近的确定性行为：

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

从项目根目录运行。`src.generate` 默认会逐字复现首个小模型样例，可用 `--prompt`、`--tokens`、`--seed`、`--temperature`、`--top-k` 修改采样配置。
提示词必须包含在 65 字符英文词表中；不支持中文输入，词表外字符会明确报错。

完整重跑会覆盖当前 `results/`。跨硬件、PyTorch 版本或精度设置的采样结果可能不同。
小模型 checkpoint 约 13 MB；300M 权重不保存，通过种子 300 和配置可重建。

## 单独生成图表、编译与打包

```bash
python scripts/verify_results.py
python scripts/make_figures.py
python scripts/report_values.py
bash scripts/build_report.sh
python scripts/check_delivery.py
python scripts/package_submission.py
```

Ubuntu 编译依赖示例：`texlive-xetex texlive-lang-chinese texlive-fonts-recommended fonts-noto-cjk`。
LaTeX 表格和指标从真实 JSON 自动注入；不应手工改测量值。
提交包保留两轮编译日志及 LaTeX 运行日志。编译后的 PDF 含可点击目录与引用。

## 结果口径

训练耗时 91.86 秒含定期评估，不含数据准备、测试、最终采样及报告编译。
训练峰值 PyTorch 已分配显存约 484 MiB；300M 推理峰值约 1,766 MiB。它们不是 `nvidia-smi` 整卡占用。
困惑度是字符级 `exp(交叉熵)`，不能直接与 BPE/单词级结果比较。
只有一次训练种子运行，无独立测试集，不声称完整收敛或跨领域泛化。

数据来源：https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare 。该链接仅用于文本数据，模型源码为本项目独立实现。
