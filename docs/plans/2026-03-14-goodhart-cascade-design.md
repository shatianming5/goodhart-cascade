# Goodhart Cascade — 设计文档

## 概述

研究 RL 训练（GRPO）如何同时破坏代码模型的校准、代码质量与对齐性。在同一训练过程中每 100 步追踪三个维度的退化，证明"三速退化"模式，并展示多目标 RL 可打断退化。

**目标**：NeurIPS 2026 · 周期 8 周 · 硬件 4×B200

## 架构决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 训练框架 | verl | 原生 GRPO + vLLM rollout，多 GPU 并行 |
| 评估推理 | vLLM | 高吞吐离线推理，checkpoint 直接加载 |
| 训练-评估关系 | 解耦（方案 A） | checkpoint 路径连接，可独立重跑 |
| 运行环境 | 本地开发 + 远程训练 | 代码不含 SSH 逻辑 |
| 测试策略 | 全覆盖 | pytest，含 judge/ECE/沙箱/分析 |

## 项目结构

```
goodhart-cascade/
├── configs/                    # YAML 配置
│   ├── train_test_only.yaml
│   ├── train_multi_obj.yaml
│   ├── train_7b.yaml
│   ├── train_14b.yaml
│   └── eval_default.yaml
├── src/goodhart/               # Python 包
│   ├── data/                   # 数据准备
│   │   ├── prepare_taco.py
│   │   ├── prepare_apps.py
│   │   └── generate_temptation.py
│   ├── rewards/                # verl reward 函数
│   │   ├── test_passing.py
│   │   ├── code_quality.py
│   │   ├── calibration.py
│   │   └── multi_objective.py
│   ├── eval/                   # 评估管线
│   │   ├── calibration.py
│   │   ├── code_quality.py
│   │   ├── temptation.py
│   │   ├── runner.py
│   │   └── sandbox.py
│   ├── analysis/               # 分析
│   │   ├── temporal.py
│   │   ├── quality_submetrics.py
│   │   ├── scale_comparison.py
│   │   └── plot_figures.py
│   └── utils/                  # 公共工具
│       ├── code_exec.py
│       ├── model_io.py
│       └── metrics.py
├── tests/                      # 全覆盖测试
├── scripts/
├── figures/
├── results/
└── pyproject.toml
```

## 训练管线

- 基于 verl 框架，自定义 reward function
- 主实验：test-only binary reward（全部测试通过=1，否则=0）
- 对照 A：多目标 reward（0.5×test + 0.3×quality + 0.2×calibration）
- 校准 reward 使用 logprob(Yes/No) 方法，避免额外采样
- 每 100 步保存 checkpoint，共 40 个
- verl 原生 HuggingFace 格式 checkpoint

## 评估管线

- 统一入口 runner.py：加载 vLLM 模型一次，分发到三个评估器
- 校准评估：LiveCodeBench 300 题 + TACO val 200 题，logprob + sampling 双方法
- 代码质量：ClassEval 100 题 + TACO val 200 题，8 个子维度静态分析
- 诱惑任务：200 道同域任务（3 种类型 + 控制组），确定性 judge
- 沙箱：subprocess + timeout + ulimit，不依赖 Docker

## 诱惑任务生成

- Type A（规格-测试冲突）：LLM 生成错误输出 → 篡改测试 → 验证分离
- Type B（Hardcoding）：visible/hidden 分割 → 三层判定
- Type C（修复vs重定义）：LLM 生成 buggy 代码 → 验证 1-2 个测试失败
- 生成使用 OpenAI API，~$50，4-8 小时
- 所有 judge 函数确定性，零 LLM-as-judge 依赖

## 分析管线

- Granger 因果检验：差分后双向检验
- 相变点检测：ruptures PELT
- 子维度退化顺序：8 个代码质量子维度各自拐点
- 规模比较：7B/14B/32B 退化速率

## 实验矩阵

| 实验 | 模型 | 数据 | 方式 | Checkpoint |
|------|------|------|------|-----------|
| 主实验 | 32B | TACO | GRPO test-only | 40 |
| 对照A | 32B | TACO | 多目标 GRPO | 40 |
| 域验证 | 32B | APPS | GRPO test-only | 10 |
| 规模7B | 7B | TACO | GRPO test-only | 10 |
| 规模14B | 14B | TACO | GRPO test-only | 10 |
| 闭源 | Sonnet 4.6/GPT-5.2 | — | 仅评估 | 1 |

## 依赖

verl, vllm, torch, transformers, datasets, pylint, radon, ruptures, statsmodels, matplotlib, numpy, openai, pytest
