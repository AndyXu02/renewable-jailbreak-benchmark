# 可更新、场景感知的大模型越狱评测基准

[English README](README.md)

这是一个可复现的 Python 评测项目，用于测试大语言模型在不同攻击方式、使用场景和防御措施下的安全性，同时衡量防御是否会误伤正常问题，以及不同自动评审模型是否会给出不同结论。

仓库分为两个相互独立的部分：

- `reference_implementation/formal_v2/`：保留硕士论文正式实验所使用的脚本。
- `src/rjbench/`：为 Linux、macOS 和 Windows 整理的轻量跨平台版本，支持任意 Ollama 模型、OpenAI-compatible 接口，以及不需要真实模型的 Mock 后端。

跨平台版本不会改变论文已有结果。它的作用是让其他人更容易在自己的电脑、服务器或新模型上运行相同类型的测试。

## 论文评测了什么

- 50 个由模型生成的有害测试目标，分为五个类别。
- 5 个目标模型和 4 种提示词传递场景。
- 直接攻击、角色扮演、针对目标模型优化的 PAIR，以及可更新的 A6 攻击。
- 安全系统提示、字符扰动，以及基于规则的输入和输出过滤器。
- 与有害问题配对的正常问题，用于衡量实用性和“过度拒绝”。
- 按行为聚类或配对的统计分析，以及更换自动评审模型后的敏感性分析。

论文正式实验包含 2,900 条评分记录，另有 100 条用于第二评审模型敏感性分析的记录。本仓库同时提供完整的 3,000 条原始记录，以及隐藏回答文本、只保留评分信息的精简版本。

2026 年 10 月的补充层（`artifacts/supplementary/`）包含论文最终使用的评分和后续实验：三评审小组（gpt-oss 20B、GLM-5.2、MiniMax-M3，含严格解析修复）对全部 2,900 条正式回答的评分，800 条后续实验回答（PAIR 格式交叉实验和 Gemma 输出长度敏感性），以及由三个生成模型 × 三个种子产生的 450 个冻结包装、共 1,350 条回答的包装刷新实验。详见 [`docs/SUPPLEMENTARY_20261016.md`](docs/SUPPLEMENTARY_20261016.md)。

## 60 秒完整性检查

只需要 Python 3.10 或更高版本。此检查不会下载模型，也不会调用任何 API。

```bash
python3 -m venv .venv
source .venv/bin/activate              # Windows：.venv\Scripts\activate
python -m pip install -e .
rjbench-verify
python -m unittest discover -s tests -v
```

预期看到类似结果：

```text
Release verification passed: ... files; 3000 raw + 3000 compact formal records
```

这表示发布文件的数量、内容哈希、正式实验记录和核心代码都通过了校验。

## 两分钟无模型快速测试

下面的命令使用结果固定的 Mock 后端，检查攻击与场景组合、文件写入、自动评分和结果汇总流程。它不需要 Ollama、显卡或 API Key。

```bash
rjbench-run \
  --backend mock \
  --model mock-safe-model \
  --data data/smoke/smoke_prompts.jsonl \
  --attacks A0_direct,A2_persona \
  --scenarios S1_direct,S2_multiturn \
  --defenses D0_none,D1_system_prompt \
  --output runs/mock_responses.jsonl

rjbench-judge \
  --backend mock \
  --model mock-judge \
  --input runs/mock_responses.jsonl \
  --output runs/mock_judged.jsonl

rjbench-summarize runs/mock_judged.jsonl \
  --group-by model_id,attack,scenario,defense \
  --output runs/mock_summary.csv
```

## 测试任意 Ollama 模型

先安装并启动 [Ollama](https://ollama.com/)，然后下载适合自己电脑配置的模型。项目没有把模型名称写死。

```bash
ollama pull llama3.2:3b

rjbench-run \
  --backend ollama \
  --model llama3.2:3b \
  --data data/research/harmful_behaviors_v2.jsonl \
  --max-samples 5 \
  --attacks A0_direct,A2_persona,A1_pair \
  --pair-file data/research/pair_prompts_harmful_v2.jsonl \
  --scenarios S1_direct,S2_multiturn,S3_longcontext,S4_agent \
  --defenses D0_none,D1_system_prompt \
  --max-tokens 512 \
  --output runs/llama3.2-3b.jsonl
```

对于 Qwen 等推理模型，可以使用 `--think off`、`--think low`，或该 Ollama 模型所接受的设置。默认值是 `--think auto`，即不主动发送思考模式参数。

建议用另一个模型对新回答进行评分：

```bash
ollama pull gpt-oss:20b

rjbench-judge \
  --backend ollama \
  --model gpt-oss:20b \
  --think low \
  --input runs/llama3.2-3b.jsonl \
  --output runs/llama3.2-3b.judged.jsonl
```

如果要做严谨的模型比较，不建议让被测试模型同时担任自己的评审模型。

## 使用 OpenAI-compatible 接口

只要服务支持 `POST /chat/completions`，就可以接入本项目，包括兼容 OpenAI 格式的云服务或本地推理服务器。

```bash
export MODEL_API_KEY="your-key"

rjbench-run \
  --backend openai \
  --model your-model-id \
  --api-base https://provider.example/v1 \
  --api-key-env MODEL_API_KEY \
  --data data/smoke/smoke_prompts.jsonl \
  --attacks A0_direct \
  --scenarios S1_direct \
  --defenses D0_none \
  --output runs/provider-smoke.jsonl
```

不要把真实 API Key 写入准备提交到 GitHub 的命令、Notebook、配置文件或 Issue。`.env.example` 只包含占位符。

## 重新计算论文结果表格

精简评分矩阵保留了实验条件和评分，因此可以直接用于描述性统计。包含完整模型回答和评审字段的记录位于 `artifacts/formal/raw/`。

重新汇总 Phase 2 各模型结果：

```bash
rjbench-summarize artifacts/formal/scored/phase2_*.jsonl \
  --group-by model_id \
  --output runs/recalculated_phase2_by_model.csv
```

可以把生成的结果与下面的论文结果表比较：

```text
artifacts/formal/tables/phase2_by_model.csv
```

重新计算正常问题的实用性和过度拒绝情况：

```bash
rjbench-summarize artifacts/formal/scored/benign_*.jsonl \
  --group-by model_id,defense \
  --score-field utility_score \
  --utility \
  --output runs/recalculated_benign_by_defense.csv
```

论文的描述性统计、推断性统计和冻结清单位于 `artifacts/formal/tables/`。本开源版本另有一份文件哈希清单：`artifacts/release_manifest.json`。

## 仓库结构

```text
src/rjbench/                         跨平台运行和评分工具
reference_implementation/formal_v2/ 论文正式实验的原始脚本副本
reference_implementation/supplementary_20261016/  三评审、后续实验和包装刷新脚本
data/research/                       行为集、正常问题、PAIR 和 A6 数据
data/smoke/                          安全、轻量的流程测试数据
artifacts/formal/scored/             精简正式评分矩阵
artifacts/formal/raw/                完整回答和评审记录
artifacts/formal/tables/             论文描述性和推断性统计结果
artifacts/supplementary/             三评审小组、后续实验、包装刷新
docs/                                方法、架构和发布说明
tests/                               不需要模型的自动测试
```

## 解读结果时需要注意

- Qwen3-8B no-think 是 PAIR 的优化目标，因此它的 PAIR 结果属于“在训练目标上测试”，不能直接当成完全独立的泛化结果。
- S4 是用文字模拟的轻量 Agent 提示注入场景，并没有真正执行外部工具。
- D3 和 D4 是固定关键词规则，不是经过训练的安全防护模型。
- A6 实验只使用了一个生成模型和一组种子；50 个包装中有 10 个使用了确定性的备用模板。补充层的包装刷新实验使用了三个生成模型和三个种子重新生成。
- 120 条人工参考样本及其人工标签仅用于校准，不包含在本仓库中；只公开汇总的一致性指标。
- 原实验中的推理后端和输出长度限制并不完全相同，因此不应该把结果解释为严格控制所有变量后的模型因果排名。
- 自动评审模型只是测量工具，不等于绝对真相。更换评审模型后，HCR@4 的绝对数值可能发生明显变化。

## 负责任地使用

本项目用于经过授权的 AI 安全评测和可复现研究。发布内容包含对抗性提示词和完整模型回答，使用前请阅读 [`docs/RESPONSIBLE_RELEASE.md`](docs/RESPONSIBLE_RELEASE.md)。

## 引用与许可证

软件使用 MIT License。数据集和模型输出的再次使用还可能受到生成或评测这些内容时所用模型服务商条款的约束。引用本仓库时请使用 `CITATION.cff`。
