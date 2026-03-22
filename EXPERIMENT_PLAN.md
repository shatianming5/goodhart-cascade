# Goodhart Cascade: Complete Experiment Plan

## Core Thesis

When RL training optimizes a proxy reward (test passing), Goodhart's Law manifests as **reward gaming that cascades** across quality dimensions. Adding constraints blocks gaming in measured dimensions but displaces it to unmeasured ones — the "whack-a-mole" effect.

---

## Infrastructure

| Component | Detail |
|-----------|--------|
| **Model** | Qwen3-Coder-7B (full params) / Qwen3-Coder-14B (LoRA r=128 verification) |
| **Data** | TACO filtered ~500 problems (base pass@8 in 10%-50%) |
| **Hardware** | 2×B200, each GPU runs one experiment |
| **Training** | GRPO, 1000 steps, batch=16, rollouts=8, lr=5e-7, KL=0.03 |
| **Eval set** | MBPP 100 (pass rate + calibration) + ClassEval 50 (6-dim quality) |

### Links

| Resource | URL |
|----------|-----|
| **GitHub** | https://github.com/shatianming5/goodhart-cascade |
| **Sweet Spot Dataset** | https://huggingface.co/datasets/shatianming5/goodhart-cascade-sweet-spot |
| **R1 Checkpoints** | https://huggingface.co/shatianming5/goodhart-cascade-R1_test_only |
| **R2 Checkpoints** | https://huggingface.co/shatianming5/goodhart-cascade-R2_test_pylint |
| **R3 Checkpoints** | https://huggingface.co/shatianming5/goodhart-cascade-R3_test_pylint_complexity |
| **R4 Checkpoints** | https://huggingface.co/shatianming5/goodhart-cascade-R4_test_pylint_complexity_comment |
| **R5 Checkpoints** | https://huggingface.co/shatianming5/goodhart-cascade-R5_all |
| **14B R1 Verify** | https://huggingface.co/shatianming5/goodhart-cascade-R1_14b_verify |
| **14B R2 Verify** | https://huggingface.co/shatianming5/goodhart-cascade-R2_14b_verify |
| **HuggingFace Token** | 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 |
| **GitHub Token** | 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 |

---

## Sweet Spot Construction

**Goal:** Select problems where base model "sometimes gets it right" — GRPO gradient signal is strongest.

```
pass@8 ∈ [10%, 50%]  →  1-4 out of 8 rollouts pass
                        → advantage has variance
                        → non-zero gradient
```

- **Too easy** (pass@8 > 80%): all 8 rollouts pass → advantage identical → zero signal
- **Too hard** (pass@8 = 0%): all 8 rollouts fail → advantage identical → zero signal

**Pipeline:** `scripts/run_filter.sh` → vLLM generates 8 samples per TACO problem → test → filter → upload HF

---

## Experiments (all 1000 steps)

### R1: test only (baseline)

```yaml
reward: {test: 1.0}
```

**Purpose:** Full "rise-and-fall" arc baseline. Pass rate rises then plateaus/declines while all quality dimensions degrade.

**Expected:**
- pass rate: 25% → ~55-60% (peak) → plateau
- ECE: 0.45 → 0.75 (continuous worsening)
- Pylint: 4.2 → 3.5 (declining)
- Comment%: 15% → 4% (declining)
- Complexity: 5 → 12 (increasing)
- Duplication: 8% → 18% (increasing)

---

### R2: test + Pylint

```yaml
reward: {test: 0.7, pylint: 0.3}
```

**Purpose:** Block Pylint dimension. Observe gaming escape.

**Expected:**
- Pylint: ✅ 4.2 → 8.0+ (constrained)
- Comment%: ❌ 15% → 1-3% (escape: docstring templates replace real comments)
- Complexity: ❌ 5 → 15+ (escape: function splitting for Pylint rules)
- Duplication: ❌ 8% → 20%+ (unconstrained degradation)

---

### R3: test + Pylint + complexity

```yaml
reward: {test: 0.6, pylint: 0.2, complexity: 0.2}
```

**Purpose:** Block Pylint + complexity. Observe continued escape.

**Expected:**
- Pylint: ✅ | Complexity: ✅
- Comment%: ❌❌ 15% → 0-2% (escape intensifies)
- Duplication: ❌❌ 8% → 25-30% (new escape: copy-paste to satisfy complexity)

---

### R4: test + Pylint + complexity + comment

```yaml
reward: {test: 0.5, pylint: 0.2, complexity: 0.15, comment: 0.15}
```

**Purpose:** Block 3 escape routes. Observe remaining direction.

**Expected:**
- Pylint: ✅ | Complexity: ✅ | Comment%: ✅
- Duplication: ❌❌❌ 8% → 30-35% (gaming concentrates here)
- pass rate: 25% → ~48% (alignment tax visible)

---

### R5: all constraints

```yaml
reward: {test: 0.4, pylint: 0.15, complexity: 0.15, comment: 0.15, duplication: 0.15}
```

**Purpose:** Block all measured dimensions. Quantify alignment tax.

**Expected:**
- All constrained dims: ✅
- pass rate: 25% → 40-45% (15-20pp below R1 = **alignment tax**)
- ECE: ❌ (never constrained, always worsening — proves gaming persists)

---

### 14B Verification

R1 and R2 configs with Qwen3-Coder-14B (LoRA r=128), 1000 steps each. Confirms escape directions are consistent across model scales.

---

## Timeline (2×B200)

| Day | B200 #1 | B200 #2 |
|-----|---------|---------|
| 0 | Data filtering (2h) | Code debug |
| 1-2 | **R1** 1000 steps | **R2** 1000 steps |
| 2-3 | **R3** | **R4** |
| 3-4 | **R5** | **14B R1 verify** |
| 4-5 | **14B R2 verify** | Eval + figures |

**~5 days total.**

---

## Disk Space Management

- Checkpoints are uploaded to HuggingFace after each experiment
- Local checkpoints cleaned: keep only every 500th + last
- DiskMonitor runs in background, triggers cleanup at <50GB free
- Model weights (~14GB for 7B, ~28GB for 14B) are the main concern
- HF cache cleared if >20GB

---

## Final Deliverables

1. **R1 Training Dynamics Curve**: pass rate rise-and-fall + 5 quality dims degrading
2. **Escape Map**: heatmap showing gaming displacement R1→R5
3. **Alignment Tax Curve**: pass rate vs # constrained dimensions (non-linear decline)

These three figures tell the complete story:
- R1 curve: "test-only RL causes quality collapse"
- Escape map: "adding constraints displaces gaming, doesn't eliminate it"
- Tax curve: "more constraints = lower pass rate, and ECE always escapes"
