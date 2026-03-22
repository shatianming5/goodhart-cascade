# Scaling Laws for Alignment Tax in Code Reinforcement Learning

## NeurIPS 2026 — Complete Experimental Plan

---

## Core Thesis

Code RL training (GRPO) with quality constraints exhibits a predictable **alignment tax**: each additional quality constraint reduces pass rate following a power law. We derive this law from GRPO's advantage decomposition, validate it across 3 model scales, and identify a **universal alignment ratio ρ\*** — the optimal fraction of quality dimensions to constrain.

---

## Infrastructure

| Component | Detail |
|-----------|--------|
| **Hardware** | 8×NVIDIA B200 (192GB each, 1536GB total) |
| **Models** | Qwen2.5-Coder-1.5B (full), 7B (full), 14B (LoRA r=128) |
| **Data** | TACO filtered ~1000 problems (base pass@8 ∈ [10%, 50%]) |
| **Training** | GRPO, 1000 steps, batch=16, rollouts=8, lr=5e-7, KL=0.03 |
| **Eval** | MBPP 100 (pass rate + calibration) + ClassEval 50 (6+6 dim quality) |

### Resources

| Resource | URL |
|----------|-----|
| **GitHub** | https://github.com/shatianming5/goodhart-cascade |
| **Sweet Spot Dataset** | https://huggingface.co/datasets/shatianming5/goodhart-cascade-sweet-spot |
| **HF Checkpoints** | `shatianming5/goodhart-cascade-{EXPERIMENT_NAME}` |
| **HuggingFace Token** | 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 |
| **GitHub Token** | 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 |

---

## Experiment Matrix (17 runs)

### Main: 15 experiments (3 scales × 5 configs)

| ID | Model | Config | VRAM | Est. Time |
|----|-------|--------|------|-----------|
| 1.5B_R1 | Qwen2.5-Coder-1.5B | test only | ~15GB | 3-4h |
| 1.5B_R2 | 1.5B | +pylint | ~15GB | 3-4h |
| 1.5B_R3 | 1.5B | +complexity | ~15GB | 3-4h |
| 1.5B_R4 | 1.5B | +comment | ~15GB | 3-4h |
| 1.5B_R5 | 1.5B | all 5 dims | ~15GB | 3-4h |
| 7B_R1 | Qwen2.5-Coder-7B | test only | ~90GB | 10-12h |
| 7B_R2 | 7B | +pylint | ~90GB | 10-12h |
| 7B_R3 | 7B | +complexity | ~90GB | 10-12h |
| 7B_R4 | 7B | +comment | ~90GB | 10-12h |
| 7B_R5 | 7B | all 5 dims | ~90GB | 10-12h |
| 14B_R1 | Qwen2.5-Coder-14B LoRA | test only | ~80GB | 12-14h |
| 14B_R2 | 14B LoRA | +pylint | ~80GB | 12-14h |
| 14B_R3 | 14B LoRA | +complexity | ~80GB | 12-14h |
| 14B_R4 | 14B LoRA | +comment | ~80GB | 12-14h |
| 14B_R5 | 14B LoRA | all 5 dims | ~80GB | 12-14h |

### Robustness: 2 experiments

| ID | Model | Config | Purpose |
|----|-------|--------|---------|
| 7B_R2_w1 | 7B | test:pylint = 0.5:0.5 | Weight robustness |
| 7B_R2_w2 | 7B | test:pylint = 0.8:0.2 | Weight robustness |

---

## R1-R5 Reward Configurations

| Experiment | test | pylint | complexity | comment | duplication |
|-----------|------|--------|------------|---------|-------------|
| R1 | 1.0 | — | — | — | — |
| R2 | 0.7 | 0.3 | — | — | — |
| R3 | 0.6 | 0.2 | 0.2 | — | — |
| R4 | 0.5 | 0.2 | 0.15 | 0.15 | — |
| R5 | 0.4 | 0.15 | 0.15 | 0.15 | 0.15 |

---

## Sweet Spot Construction

**Goal:** Select problems where base model pass@8 ∈ [10%, 50%] — GRPO gradient signal is strongest.

- **Too easy** (>50%): all rollouts pass → zero advantage variance → zero gradient
- **Too hard** (0%): all rollouts fail → zero advantage variance → zero gradient
- **Sweet spot** (10-50%): 1-4 of 8 pass → strong contrastive signal

**Go/No-Go:** If trainable < 500, add APPS introductory+interview problems.

---

## Evaluation Metrics (all measured on every checkpoint)

### Constrained dimensions (used in R1-R5 rewards)
1. Pass@1 rate (MBPP)
2. Pylint score (0-10)
3. Cognitive complexity (lower = better)
4. Comment ratio (%)
5. Duplication ratio (%)

### Unconstrained dimensions (never in reward — detect gaming escape)
6. Type hint coverage (%)
7. Average function length
8. Magic number count
9. Max nesting depth
10. Dead code ratio
11. Average variable name length

### Calibration
12. ECE (Expected Calibration Error)
13. Overconfidence rate

---

## Theory: Alignment Tax Scaling Law

### Derivation from GRPO advantage decomposition

GRPO advantage: `A_i = (r_i - mean(r)) / std(r)`

With multi-objective reward `r = Σ wᵢ rᵢ`, effective gradient signal:

```
η(n) = w_test² / [w_test² + (1-w_test)²/n]
```

Alignment tax:
```
tax(n) = C · (1-w_test)² / [n·w_test² + (1-w_test)²]
```

With model scale dependence:
```
tax(n, N) = C · (1-w_test)² / [n·w_test² + (1-w_test)² · N^(-δ)]
```

### Verification protocol
1. Check independence assumption (correlation matrix of components)
2. Check equal variance assumption
3. Fit C from 7B data → predict 1.5B and 14B tax
4. Compute R², prediction error
5. Find ρ* = optimal constraint ratio (knee point)

---

## Scheduling (8×B200, ~3 days)

```
DAY 0 AM:
  GPU 0: Filter TACO data (~2h)

DAY 0 PM → DAY 1 AM:
  GPU 0-4: 7B R1-R5 (parallel, ~12h)
  GPU 5-7: 1.5B R1-R3 (parallel, ~4h)

DAY 1 AM (1.5B R1-R3 done):
  GPU 5: 1.5B R4
  GPU 6: 1.5B R5
  GPU 7: 7B_R2_w1 (weight robustness)

DAY 1 PM (1.5B all done, 7B finishing):
  GPU 5: 7B_R2_w2
  GPU 6-7: Evaluate 1.5B + completed 7B checkpoints

DAY 1 EVENING (7B all done):
  *** GO/NO-GO: 7B-R1 pass rate > 50%? Tax curve convex? ***
  GPU 0-4: 14B R1-R5 (parallel, ~14h)
  GPU 5-7: Evaluate all 7B checkpoints

DAY 2 PM (14B done):
  GPU 0-7: Evaluate all 14B checkpoints

DAY 3 AM:
  Theory verification, scaling law fit, ρ*, generate all figures
```

**Total: ~3 days from data filtering to final results.**

---

## Go/No-Go Decision Points

| When | Check | If FAIL |
|------|-------|---------|
| Day 0 PM | Trainable problems ≥ 500? | Add APPS data, re-filter |
| Day 1 PM | 7B-R1 pass rate > 50%? | Adjust lr, kl_coeff, re-run |
| Day 1 PM | Tax curve convex (has knee)? | Drop ρ*, keep escape map |
| Day 3 AM | R² > 0.85 across scales? | Drop "scaling law" claim, keep empirical |
| Day 3 AM | ρ* consistent (0.3-0.4)? | Drop "universal constant" claim |

---

## Critical: Log ALL dimensions for EVERY rollout

Every rollout logs all 6 constrained + 6 unconstrained dimensions, regardless of which ones are in the reward. Without this, theory verification (correlation matrix, variance ratio) is impossible.

---

## Disk Space Management

- Each 7B checkpoint: ~14GB → 10 checkpoints = 140GB
- 17 experiments × 140GB = **2.4TB** (exceeds disk without cleanup!)
- Strategy: Upload each checkpoint to HuggingFace immediately, then delete locally
- Keep only first + last + every 500th checkpoint locally
- DiskMonitor thread runs during training, triggers cleanup at <50GB free
- HF cache cleared if >20GB

---

## Expected Figures

1. **R1 Training Dynamics**: Pass rate rise-then-fall + 5 quality dims degrading
2. **Escape Map Heatmap**: Gaming displacement R1→R5 × 3 model scales
3. **Alignment Tax Curves**: tax(n) for 1.5B/7B/14B with fitted power law
4. **Efficiency Frontier**: Quality vs pass rate, knee = ρ*
5. **Theory vs Experiment**: Predicted vs actual tax scatter (identity line)
6. **ρ* Bar Chart**: Showing ρ* ≈ 0.35 across scales

---

## Fallback Plan

| Outcome | Paper version | Score |
|---------|--------------|-------|
| R² > 0.95, ρ* stable | "Scaling Laws for Alignment Tax" | 8-9 (oral) |
| R² = 0.85-0.95 | "Alignment Tax in Code RL" (empirical) | 7-7.5 (poster) |
| R² < 0.85, escape map clear | "Proxy Gaming Escape in Code RL" | 6.5-7 (poster) |
| Training fails | Debug + resubmit ICML 2027 | — |
