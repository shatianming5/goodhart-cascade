# Goodhart Cascade

**Quantifying reward gaming displacement in code RL.**

When RL training optimizes test-passing as a proxy reward, Goodhart's Law manifests as gaming that **cascades** across quality dimensions. Adding quality constraints blocks gaming in constrained dimensions but displaces it to unconstrained ones.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Filter TACO dataset for sweet spot (pass@8 ∈ [10%, 50%])
bash scripts/run_filter.sh

# 3. Run all experiments (2×B200 GPUs)
bash scripts/run_all.sh

# Or run individual experiment:
bash scripts/run_experiment.sh configs/R1_test_only.yaml 0  # GPU 0
```

## Experiments

| Experiment | Reward | Purpose |
|------------|--------|---------|
| R1 | test only | Baseline: observe full quality degradation arc |
| R2 | test + pylint | Block Pylint → gaming escapes to comments, complexity |
| R3 | + complexity | Block complexity → gaming escapes to duplication |
| R4 | + comment | Block comments → gaming concentrates in duplication |
| R5 | all 5 dims | All blocked → quantify alignment tax |

## Resources

- **Dataset:** [goodhart-cascade-sweet-spot](https://huggingface.co/datasets/shatianming5/goodhart-cascade-sweet-spot)
- **Checkpoints:** `shatianming5/goodhart-cascade-{experiment_name}` on HuggingFace
- **Full Plan:** [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)

## Key Results (Expected)

1. **R1 Dynamics**: Pass rate rises ~25%→60% then plateaus; all quality metrics degrade
2. **Escape Map**: Gaming displaced to unconstrained dimensions at each step
3. **Alignment Tax**: Pass rate drops ~15-20pp when all quality dimensions are constrained
