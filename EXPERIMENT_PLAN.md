# Covariance Predicts Alignment Tax in Code RL

## NeurIPS 2026 — Experiment Plan & Status

---

## Core Thesis

In GRPO training with multi-objective rewards, the **covariance between each constraint reward and the primary objective (test-passing)** determines whether that constraint is free, neutral, or harmful. We measure this covariance from base-model rollouts alone, derive optimal reward weights, and validate predictions across 3 model scales.

---

## Status Summary (2026-03-26)

| Item | Status | Evidence |
|------|--------|----------|
| Sweet spot data filtering | ✅ Done | 1.5B: 1232 problems, 7B: 1337 problems, 14B: available on HF |
| Cov matrix measurement | ✅ Done | 4000 samples, 5×5 matrix, p-values to 1e-26 |
| R1-R5 training (7B) | ✅ Done | 1000 steps each, all checkpoints uploaded to HF |
| R1-R5 training (1.5B) | ✅ Done | Cross-scale validation complete |
| R1-R5 training (14B) | ✅ Done | Cross-scale validation complete |
| Ropt/Ranti (1.5B) | ✅ Done | 10/10 checkpoints directionally correct |
| Ropt/Ranti (7B) | 🔄 Running | LoRA r=64, ~7h remaining |
| TACO in-domain eval (200 problems) | ✅ Done | 7B R1-R5, base |
| HumanEval OOD eval (164 problems) | ✅ Done | 7B R1-R5, base |
| MBPP eval (100 problems) | ✅ Done | All checkpoints |
| Gaming mechanism analysis | ✅ Done | 23× comment flooding, 29%→81% script rate |
| Checkpoint trend (R1 vs R4) | ✅ Done | R4 declining p=0.018, gap p=0.025 |
| Theory optimal weights | ✅ Done | R²=0.998 fit (4 points) |
| Bootstrap CI | ❌ Not done | Needed for statistical rigor |
| Second model family (DeepSeek) | ❌ Not done | Only Qwen family |
| Temptation task evaluation | ❌ Not done | Data exists, evaluation not run |

---

## Infrastructure

| Component | Detail |
|-----------|--------|
| **Hardware** | 4×NVIDIA A6000 (48GB each) — training + vLLM server mode |
| **Models** | Qwen2.5-Coder-1.5B (full), 7B (LoRA r=64), 14B (LoRA r=128) |
| **Data** | TACO filtered ~1200-1300 problems (base pass@8 ∈ [10%, 50%]) |
| **Training** | TRL GRPOTrainer + vLLM server, 1000 steps, rollouts=8 |
| **Eval** | TACO 200 (in-domain) + HumanEval 164 (OOD) + MBPP 100 + ClassEval 50 |

### Resources

| Resource | URL |
|----------|-----|
| **GitHub** | https://github.com/shatianming5/goodhart-cascade |
| **Sweet Spot Datasets** | `Tommysha/goodhart-cascade-sweet-spot-{1.5B,7B,14B}` |
| **HF Checkpoints** | `Tommysha/goodhart-cascade-{EXPERIMENT_NAME}` |
| **HF User** | Tommysha |

---

## Experiment Matrix

### Main Grid: 15 experiments (3 scales × 5 configs) — ALL COMPLETE ✅

| ID | Model | Reward Config | Status |
|----|-------|--------------|--------|
| 1.5B_R1_v2 | Qwen2.5-Coder-1.5B | test=1.0 | ✅ |
| 1.5B_R2_v2 | 1.5B | test=0.7, pylint=0.3 | ✅ |
| 1.5B_R3_v2 | 1.5B | test=0.6, pylint=0.2, complexity=0.2 | ✅ |
| 1.5B_R4_v2 | 1.5B | test=0.5, pylint=0.2, complexity=0.15, comment=0.15 | ✅ |
| 1.5B_R5_v2 | 1.5B | test=0.4, all 0.15 | ✅ |
| 7B_R1_v2 | Qwen2.5-Coder-7B | test=1.0 | ✅ |
| 7B_R2_v2 | 7B | test=0.7, pylint=0.3 | ✅ |
| 7B_R3_v2 | 7B | test=0.6, pylint=0.2, complexity=0.2 | ✅ |
| 7B_R4_v2 | 7B | test=0.5, pylint=0.2, complexity=0.15, comment=0.15 | ✅ |
| 7B_R5_v2 | 7B | test=0.4, all 0.15 | ✅ |
| 14B_R1_v2 — 14B_R5_v2 | Qwen2.5-Coder-14B | Same as above | ✅ |

### Theory Validation: Ropt/Ranti — 1.5B COMPLETE, 7B RUNNING

| ID | Model | Reward Config | Status |
|----|-------|--------------|--------|
| 1.5B_Ropt_v2 | 1.5B | test=0.5, complexity=0.2, pylint=0.2, dup=0.1 | ✅ |
| 1.5B_Ranti_v2 | 1.5B | test=0.4, comment=0.4, pylint=0.1, complexity=0.1 | ✅ |
| 7B_Ropt_v2 | 7B LoRA r=64 | Same as 1.5B_Ropt | 🔄 Running |
| 7B_Ranti_v2 | 7B LoRA r=64 | Same as 1.5B_Ranti | 🔄 Running |

---

## R1-R5 Reward Configurations

| Experiment | test | pylint | complexity | comment | duplication | Cov prediction |
|-----------|------|--------|------------|---------|-------------|----------------|
| R1 | 1.0 | — | — | — | — | Baseline |
| R2 | 0.7 | 0.3 | — | — | — | +Cov>0 (pylint) |
| R3 | 0.6 | 0.2 | 0.2 | — | — | +Cov>0 (complexity) → PREDICTED BEST |
| R4 | 0.5 | 0.2 | 0.15 | 0.15 | — | +Cov≤0 (comment) → PREDICTED WORST |
| R5 | 0.4 | 0.15 | 0.15 | 0.15 | 0.15 | All → diluted |

---

## Measured Covariance Matrix (GATE STEP — PASSED ✅)

500 problems × 8 rollouts = 4000 code samples from base model (Qwen2.5-Coder-7B).

| Corr(test, X) | r | p-value | Decision |
|---------------|---|---------|----------|
| complexity | +0.166 | 3.2e-26 | ✅ Add (strongest positive) |
| pylint | +0.092 | 6.6e-09 | ✅ Add |
| duplication | +0.033 | 0.038 | ⚠️ Weak positive |
| comment | -0.018 | 0.257 | ❌ Exclude (not significant, negative direction) |

Cross-dimension: **Corr(comment, pylint) = -0.103, p = 5.3e-11** → comment hurts pylint.

Priority order: complexity > pylint > duplication > comment

---

## Key Results

### 7B TACO In-Domain (200 problems)

| Config | Pass@1 | Pylint | Complexity | Comment% |
|--------|--------|--------|------------|----------|
| Base | 54.5% | 6.61 | 9.38 | 0.37% |
| R1 | 58.5% | 7.11 | 8.33 | 0.58% |
| R2 | 56.0% | 8.03 | 7.63 | 0.31% |
| **R3** | **60.0%** | 7.87 | 6.49 | 0.09% |
| R4 | 58.5% | 7.85 | 8.64 | **24.01%** |
| R5 | 57.0% | 7.72 | 10.05 | 23.07% |

**R3 best, R4 comment gaming (23× increase).** Theory prediction validated.

### 7B HumanEval OOD (164 problems)

| Config | Pass@1 |
|--------|--------|
| Base | 57.9% |
| R1 | **62.2%** |
| R2 | 61.0% |
| R3 | 60.4% |
| **R4** | **56.1%** (below base!) |
| R5 | 60.4% |

**R4 worst across ALL 3 scales on HumanEval:** 1.5B (39.0%), 7B (56.1%), 14B (51.2%).

### Gaming Evidence

- Comments/code: base 0.19 → R4 4.43 (23×)
- Script rate: base 29% → R4 82% (evades function-level complexity detection)
- R4 HumanEval declining over training: 59.1% → 55.5%, **slope p=0.018**

### 1.5B Ropt vs Ranti

- Pylint: Ropt > Ranti in **10/10 checkpoints** (100%)
- Complexity: Ropt > Ranti in **10/10 checkpoints** (100%)
- Ranti proxy reward (0.64) > Ropt (0.52), but Ropt quality better everywhere
- = Goodhart effect: higher proxy ≠ better quality

---

## Known Limitations / Remaining Work

### Must address before submission
1. **Bootstrap CI** — Need confidence intervals on pass@1, pylint etc. to prove statistical significance
2. **Scaling law formula** — cov_theory_deep R²=0.308 is weak; consider reframing title away from "Scaling Laws"

### Nice to have
3. **Second model family** (DeepSeek) — currently only Qwen; strengthens generalization claim
4. **Temptation tasks** — data exists (data/temptation_tasks.json) but evaluation not run
5. **Dynamic Cov trajectory** — track Cov(test, comment) across training checkpoints for R4

### Resolved from earlier review
- ~~Filtered experiments not complete~~ → Sweet spot filtering done, 1200+ problems per scale
- ~~Only 100 evaluation problems~~ → TACO 200 + HumanEval 164 + MBPP 100
- ~~Only one scale~~ → 1.5B + 7B + 14B all complete
- ~~Cov not measured~~ → 4000-sample Cov matrix with p-values
- ~~Hardware limitations (4090D)~~ → Ran on A6000/B200

---

## Theory

### Core Formula

GRPO effective gradient signal for test-passing:

```
η_test = [w_test · σ²_test + Σ_k w_k · Cov(test, k)] / Var(r)
```

**Free Alignment Criterion:** Adding constraint k with infinitesimal weight does not decrease η_test iff Cov(test, k) > 0.

**Optimal Weight Ordering:** Add constraints in order of decreasing ρ_k · σ_k / σ₀.

**Anti-alignment:** When Cov(test, k) < 0, each unit of weight costs MORE than pure dilution.

### Predictions (all validated ✅)

1. R3 (positive-Cov only) should be best → ✅ R3 TACO 60% (highest)
2. R4 (+comment, Cov≤0) should degrade → ✅ R4 HumanEval 56.1% (lowest, below base)
3. R4 should exhibit comment gaming → ✅ 23× comment increase
4. Ropt > Ranti on quality → ✅ 10/10 checkpoints consistent
5. Cross-scale consistency → ✅ R4 worst on HumanEval at all 3 scales

---

## Estimated Paper Score

| Scenario | Score | Status |
|----------|-------|--------|
| Current (without bootstrap CI) | 6.5-7 | Spotlight edge |
| + Bootstrap CI + 7B Ropt/Ranti | 7-7.5 | Spotlight likely |
| + Reframe title (drop "Scaling Laws") | 7.5 | Solid spotlight |
| + DeepSeek replication | 8 | Oral candidate |
