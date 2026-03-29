# Plan: From Poster to Oral

## Current Status

- 7/7 Go/No-Go checks passed
- R² = 0.998 (inferred ρ, but overfitted on 4 points)
- Gaming escape clearly documented (script化 29%→81%, comment flooding 0→4/code)
- Checkpoint trend significant (R4 declining p=0.018, gap widening p=0.025)
- Current level: **poster 7-7.5**

## What's Missing for Oral

The theory is post-hoc (fit ρ from R1-R5 results). Oral requires **prediction before experiment**.

## The Plan (5 Steps, ~10h GPU)

### Step 1: Measure Real Cov Matrix (30 min GPU) — GATE STEP

Generate rollouts with base model, score on all 5 dimensions simultaneously, compute true Cov(r_test, r_k).

- 500 TACO problems × 8 rollouts = 4000 code samples
- Each sample scored: test_pass, pylint, complexity, comment, duplication
- Compute 5×5 correlation matrix

**Gate**: If Cov(test, comment) < 0 and Cov(test, complexity) > 0 → proceed. Otherwise → stop, fall back to gaming paper (7.5).

### Step 2: Blind Prediction (0 GPU)

From measured Cov matrix + theory formula:
- Predict optimal weights → R_optimal config
- Predict worst weights → R_anti config
- Predict tax ordering for R1-R5
- **Write predictions down BEFORE training**

### Step 3: Train R_optimal + R_anti (5h GPU)

- R_optimal: theory-predicted best weights (expect: test + pylint + complexity, NO comment)
- R_anti: theory-predicted worst weights (expect: heavy comment + low test)
- Both 1000 steps, 7B, same hyperparams as v2

**Success criteria**:
- R_optimal pass@1 ≥ R3 (best existing) on HumanEval
- R_anti pass@1 < R1 on HumanEval
- Both predictions correct → theory is prescriptive

### Step 4: Cross-Scale Validation (2h GPU)

Evaluate existing 14B R1-R4 checkpoints on HumanEval + TACO.

**Success criteria**:
- 14B shows same tax ordering as 7B (R4 worst)
- 7B Cov matrix predicts 14B behavior direction

### Step 5: Dynamic Cov (3h GPU) — Optional but High Impact

For R4's 10 checkpoints (100-1000):
- At each checkpoint, generate rollouts, measure Cov(test, comment)
- Plot Cov trajectory over training

**Prediction**: Cov(test, comment) starts ≈ 0 and becomes negative during training.

**If confirmed**: This is the micro-mechanism of Goodhart's Law — the model learns to game comments, causing correlation breakdown. Unifies Goodhart (macro) with Cov (micro).

## Paper Narrative (if all steps succeed)

> "In GRPO with multi-objective rewards, the covariance between constraint rewards and the primary objective determines whether alignment is free or costly.
>
> We derive η_test = f(Cov), prove Free Alignment Theorem (Cov > 0 → zero-cost constraint), predict optimal reward weights from base-model rollout statistics alone, and validate:
> (1) R_optimal beats all manual configs
> (2) R_anti is worst as predicted
> (3) Consistent across 1.5B/7B/14B
> (4) Dynamic Cov reveals Goodhart's Law = correlation breakdown during optimization"

## Risk Assessment

| Step | Probability of success | If fails |
|------|----------------------|----------|
| Step 1 (Cov signs match) | 80% | Stop, gaming paper 7.5 |
| Step 3 (R_optimal beats R3) | 50% | Weaken to "directional" claim |
| Step 3 (R_anti worst) | 70% | Still have R_optimal result |
| Step 4 (14B consistent) | 75% | Drop cross-scale, keep 7B |
| Step 5 (Dynamic Cov) | 65% | Drop this section |

## Estimated Outcome

| Scenario | Probability | Score |
|----------|------------|-------|
| Step 1 fails | 20% | 7-7.5 (gaming paper) |
| Step 1 pass, Step 3 partial | 25% | 7.5-8 (poster+) |
| Steps 1-3 pass | 25% | 8 (spotlight) |
| Steps 1-4 pass | 20% | 8-8.5 (spotlight/oral) |
| Steps 1-5 all pass | 10% | 8.5-9 (oral) |
