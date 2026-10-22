# 可更新场景感知越狱基准测试

[English README](README.md)

用于测试语言模型在不同攻击方法、部署场景和防御机制下安全性的可复现基准测试。

## 概览

本基准测试评估：
- **50个有害行为目标**，跨越五个风险类别
- **5个目标模型**，4种提示传递场景
- **攻击方法**：直接提示、人设包装、PAIR优化、可更新生成包装器
- **防御机制**：系统提示、字符扰动、输入/输出过滤
- **良性基线**，用于测量效用和过度拒绝
- **多评委评估**，带严格验证

仓库包含：
- **3,700条实验响应**，带验证的评委评分（实验1-3及后续验证）
- **1,350条包装器刷新响应**，来自可更新资产协议
- **便携式运行器**，支持Ollama、OpenAI兼容API和模拟后端
- **完整分析脚本**，可复现所有报告结果

## 快速开始

### 安装

需要Python 3.10+。核心验证无需强制依赖。

```bash
git clone https://github.com/AndyXu02/renewable-jailbreak-benchmark.git
cd renewable-jailbreak-benchmark
python3 -m pip install -e .
```

### 验证数据完整性

检查校验和及记录数（无需模型）：

```bash
rjbench-verify
python3 -m unittest discover -s tests -v
```

预期输出：`Release verification passed: ... files; 3000 raw + 3000 compact formal records`

### 使用模拟后端运行

无需模型测试完整流程：

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

### 使用Ollama运行

安装[Ollama](https://ollama.com/)并拉取模型：

```bash
ollama pull llama3.2:3b

rjbench-run \
  --backend ollama \
  --model llama3.2:3b \
  --data data/research/harmful_behaviors_v2.jsonl \
  --max-samples 5 \
  --attacks A0_direct,A2_persona \
  --scenarios S1_direct,S2_multiturn \
  --defenses D0_none,D1_system_prompt \
  --max-tokens 512 \
  --output runs/llama3.2-3b.jsonl
```

## 仓库结构

```
├── src/rjbench/              # 便携式CLI运行器
├── reference_implementation/ # 原始实验脚本
│   ├── formal_v2/           # 实验1-3实现
│   └── supplementary_20261016/ # 多评委小组及刷新
├── artifacts/
│   ├── formal/raw/          # 2,900条实验响应
│   └── supplementary/       # 评委小组评分、后续验证、刷新
├── data/
│   ├── research/            # 研究行为集及PAIR提示
│   └── smoke/               # 最小测试数据
├── docs/                    # 详细指南及协议
└── tests/                   # 单元及集成测试
```

## 实验记录

| 数据集 | 记录数 | 描述 |
|--------|--------|------|
| 正式实验 (实验1-3) | 2,900 | 5目标 × 4攻击 × 4场景 × 4防御 + 良性基线 |
| 后续验证 | 800 | PAIR格式交叉验证 (320) + Gemma容量敏感性 (480) |
| 包装器刷新 | 1,350 | 50行为 × 3生成器 × 3种子 × 3目标 |

所有响应包含三评委小组评分（gpt-oss 20B、GLM-5.2、MiniMax-M3），带严格验证。详见[`docs/SUPPLEMENTARY_20261016.md`](docs/SUPPLEMENTARY_20261016.md)。

## 分析

复现统计分析：

```bash
# 安装分析依赖
python3 -m pip install -e ".[analysis]"

# 运行分析脚本
cd reference_implementation/supplementary_20261016/scripts
python3 analyze_multijudge.py
python3 analyze_refresh.py
python3 refresh_generator_effects.py
```

分析输出已预计算在`artifacts/supplementary/*/analysis/`。

## 引用

```bibtex
@mastersthesis{xu2026renewable,
  author = {Xu, ChengSheng Andy},
  title = {Renewable and Scenario-Aware Jailbreak Benchmarking for Language Models},
  school = {Monash University},
  year = {2026},
  month = {October}
}
```

## 许可证

MIT许可证。详见[LICENSE](LICENSE)。

## 安全

本仓库包含对抗性提示研究材料。请参阅[SECURITY.md](SECURITY.md)了解负责任使用指南。
