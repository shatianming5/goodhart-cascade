# Goodhart Cascade 实验计划

## 研究目标
验证 RLHF/GRPO 训练中的 Goodhart 效应：模型在 proxy reward 优化过程中，真实能力（校准、代码质量）出现"三速退化"。

**核心需求**：训练后 pass rate 从 ~25% 涨到 ~65%+，在此过程中观察校准崩溃、质量先升后降、捷径倾向上升。

---

## 已完成实验

| 实验 | 数据 | LoRA | Steps | 结果 |
|------|------|------|-------|------|
| test_only_7b | 全量TACO 1551题 | r=16 | 1032 | pass 4%→8%，reward太低(0.08) |
| multi_obj_7b | 全量TACO | r=16 | 1548 | pylint 4→8.2，轻度Goodhart |
| test_only_14b | 全量TACO | r=16 | 516 | pass 6%→10%，效果弱 |
| test_only_easy | EASY子集 366题 | r=64 | 584 | reward collapse 0.45→0.13 |
| multi_obj_easy | EASY子集 | r=64 | 584 | Goodhart: pylint 3.9→9.5, pass 12%→6% |

**结论**：模型没有真正学会解题（pass rate 太低），Goodhart 效应没有空间展现。

## 关键突破：pass@8 数据过滤
- 用 base model 对 1551 题各生成 8 次，筛出 pass@8 在 [1/8, 7/8] 的题
- 结果：172 题在可学习区间（122 train + 50 val）
- **训练 reward 从 0.42 升到 0.86**，模型真正在学习

## 正在运行

| 实验 | GPU | 进度 | 数据 |
|------|-----|------|------|
| test_only_filtered_strong | 0-5 (6卡4090D) | ~150/3000 | 122题 filtered |
| multi_obj_filtered_strong | 6-9 (4卡4090D) | ~70/3000 | 122题 filtered |

预计完成时间：5-7 天（4090D 瓶颈：~178s/step）

---

## 待跑实验（B200 Full Fine-tuning 方案）

### 环境要求
- 2× NVIDIA B200 (192GB HBM3e)
- 或等效：2× H100 80GB（需 LoRA，速度慢 2-3×）
- Conda env: goodhart（PyTorch 2.9+, TRL 0.29.0, PEFT 0.18.1）

### 数据准备（已完成）
```bash
# 数据文件（在 data/ 目录下）
data/trl_train_filtered.json    # 122 题，pass@8 ∈ [0.125, 0.875]
data/trl_val_filtered.json      # 50 题，同分布
data/trl_train.json             # 1551 题，全量 TACO (>=5 test cases)
data/trl_val.json               # 全量 val
```

---

### P1: 核心训练实验（最高优先级）

**目标**：pass rate 从 ~25% 涨到 ~65%+，建立"先升后降"弧线

#### P1a: test_only_filtered（纯测试通过率奖励）
```bash
# Full fine-tuning on B200 (不需要 LoRA)
accelerate launch --num_processes 2 --mixed_precision bf16 \
    scripts/train_grpo_trl.py \
    --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
    --reward_mode test_only \
    --train_data data/trl_train_filtered.json \
    --val_data data/trl_val_filtered.json \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_generations 8 \
    --max_completion_length 512 \
    --learning_rate 1e-5 \
    --beta 0.04 \
    --max_steps 800 \
    --save_steps 50 \
    --warmup_steps 30 \
    --lora_r 0 \
    --output_dir outputs/test_only_filtered_ft
```
注意：lora_r=0 时需修改 train_grpo_trl.py 跳过 peft_config（或直接不传 peft_config）

**预估时间**：800 步 × ~25s ≈ 5.5 小时

#### P1b: multi_objective_filtered（多目标奖励）
```bash
# 同上，改 reward_mode
accelerate launch --num_processes 2 --mixed_precision bf16 \
    scripts/train_grpo_trl.py \
    --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
    --reward_mode multi_objective \
    --train_data data/trl_train_filtered.json \
    --val_data data/trl_val_filtered.json \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_generations 8 \
    --max_completion_length 512 \
    --learning_rate 1e-5 \
    --beta 0.04 \
    --max_steps 800 \
    --save_steps 50 \
    --warmup_steps 30 \
    --lora_r 0 \
    --output_dir outputs/multi_obj_filtered_ft
```

**预估时间**：~5.5 小时

#### P1c: test_only 全量数据（1551题，含难题）
```bash
accelerate launch --num_processes 2 --mixed_precision bf16 \
    scripts/train_grpo_trl.py \
    --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
    --reward_mode test_only \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_generations 8 \
    --max_completion_length 512 \
    --learning_rate 1e-5 \
    --beta 0.04 \
    --max_steps 1500 \
    --save_steps 100 \
    --warmup_steps 50 \
    --lora_r 0 \
    --output_dir outputs/test_only_full_ft
```

**预估时间**：~10 小时

---

### P2: 诱惑任务实验

**目标**：构造 200 道"诱惑任务"，测试模型是否走捷径

#### 三类诱惑任务
- **Type A（规格-测试冲突）**：题目描述要求功能 X，但测试用例只检查功能 Y
- **Type B（硬编码诱惑）**：测试用例可以通过 if-else 硬编码通过
- **Type C（修复 vs 重定义）**：buggy 代码 + 测试，可以修 bug 也可以重写

```bash
# 1. 生成诱惑任务（需要实现 scripts/generate_temptation_tasks.py）
python scripts/generate_temptation_tasks.py \
    --n_tasks 200 \
    --output data/temptation_tasks.json

# 2. 在各 checkpoint 上评估
python scripts/eval_temptation.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --adapter outputs/test_only_filtered_ft/checkpoint-{50,100,...,800} \
    --tasks data/temptation_tasks.json \
    --output results/temptation/
```

**需要新写的代码**：
- `scripts/generate_temptation_tasks.py`
- `scripts/eval_temptation.py`
- `src/goodhart/eval/temptation.py`

**预估时间**：代码开发 1 天 + 评估 2-3 小时

---

### P3: 消融实验（3 组单维度干预）

**目标**：证明因果结构——哪个 reward 分量导致哪种退化

#### P3a: test + calibration（无 quality）
```bash
# reward = 0.7 * test + 0.3 * calibration
accelerate launch --num_processes 2 --mixed_precision bf16 \
    scripts/train_grpo_trl.py \
    --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
    --reward_mode test_cal \
    --train_data data/trl_train_filtered.json \
    --val_data data/trl_val_filtered.json \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_generations 8 \
    --learning_rate 1e-5 \
    --beta 0.04 \
    --max_steps 800 \
    --save_steps 50 \
    --lora_r 0 \
    --output_dir outputs/ablation_test_cal_ft
```

#### P3b: test + quality（无 calibration）
```bash
# reward = 0.7 * test + 0.3 * quality
# --reward_mode test_quality
# 其余同上，output_dir=outputs/ablation_test_quality_ft
```

#### P3c: test + anti-shortcut（惩罚硬编码）
```bash
# reward = 0.7 * test + 0.3 * anti_shortcut
# --reward_mode test_anti_shortcut
# 其余同上，output_dir=outputs/ablation_test_antishortcut_ft
```

**需要新写的代码**：
- `src/goodhart/rewards/trl_rewards.py` 中新增 3 个 reward 函数
- `src/goodhart/rewards/anti_shortcut.py` 硬编码检测器

**预估时间**：每组 ~5.5 小时，3 组串行 ~17 小时，并行 ~5.5 小时

---

### P4: 扩展评估 + 置信区间

**目标**：评估集从 100 题扩到 300-500 题，加 bootstrap CI

```bash
# 1. 扩展评估集
python data/filter_by_base_passrate.py --merge \
    --min_pass 0.0 --max_pass 1.0 \
    --min_tests 3

# 2. 带 bootstrap 的评估
python scripts/eval_single_checkpoint.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --adapter outputs/test_only_filtered_ft/checkpoint-400 \
    --data_path data/trl_val_expanded.json \
    --n_bootstrap 1000 \
    --output_dir results/expanded_eval/
```

**需要修改的代码**：
- `scripts/eval_single_checkpoint.py` 加 `--n_bootstrap` 参数
- 评估输出加 CI

**预估时间**：~3-4 小时（每个 checkpoint 评估 ~20min × 多个 checkpoint）

---

### P5: 第二模型家族

**目标**：在非 Qwen 模型上复现，证明 Goodhart 效应不是模型特异的

```bash
# 先跑 pass@8 过滤（用 DeepSeek 的 base model）
CUDA_VISIBLE_DEVICES=0 python data/filter_by_base_passrate.py \
    --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
    --shard_id 0 --num_shards 1 \
    --output_dir data/passrate_shards_deepseek

# 训练
accelerate launch --num_processes 2 --mixed_precision bf16 \
    scripts/train_grpo_trl.py \
    --model_name deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
    --reward_mode test_only \
    --train_data data/trl_train_filtered_deepseek.json \
    --max_steps 800 \
    --lora_r 0 \
    --output_dir outputs/test_only_deepseek_ft
```

**预估时间**：过滤 ~2 小时 + 训练 ~5.5 小时

---

## 总时间估算

### 在 2×B200 上（Full Fine-tuning）

| 实验 | 时间 | 可并行 |
|------|------|--------|
| P1a test_only_filtered | 5.5h | ✓ |
| P1b multi_obj_filtered | 5.5h | ✓ (与P1a串行) |
| P1c test_only_full | 10h | 在P1a/b后 |
| P2 诱惑任务 | 代码1天 + 评估3h | P1训练时写代码 |
| P3 消融×3 | 5.5h（并行）/ 17h（串行） | P1后 |
| P4 扩展评估 | 4h | 任何空闲时段 |
| P5 DeepSeek | 8h | P3后 |
| **总计（智能调度）** | **~2-3 天** | |

### 在 6×4090D 上（LoRA，当前配置）

| 实验 | 时间 |
|------|------|
| P1a+P1b（正在跑） | ~6天 |
| P1c | ~8天 |
| P3 消融×3 | ~19天 |
| P5 DeepSeek | ~7天 |
| **总计** | **~5 周** |

---

## 代码修改清单（B200 适配）

1. **`scripts/train_grpo_trl.py`**：当 `lora_r=0` 时跳过 LoRA，直接 full fine-tuning
2. **`src/goodhart/rewards/trl_rewards.py`**：新增 `test_cal`, `test_quality`, `test_anti_shortcut` reward
3. **`src/goodhart/rewards/anti_shortcut.py`**：新建，硬编码检测
4. **`scripts/generate_temptation_tasks.py`**：新建，诱惑任务生成
5. **`scripts/eval_temptation.py`**：新建，诱惑任务评估
6. **`scripts/eval_single_checkpoint.py`**：加 bootstrap CI 支持
