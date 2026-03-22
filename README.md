# Scaling Laws for Alignment Tax in Code RL

**Quantifying how quality constraints cost pass rate — and proving it follows a power law.**

When RL training optimizes test-passing as a proxy reward, Goodhart's Law manifests as gaming that **cascades** across quality dimensions. We derive a scaling law for the alignment tax: `tax(n) = C·(1-w)²/[n·w² + (1-w)²]`, validate across 3 model scales, and find a universal optimal constraint ratio ρ* ≈ 0.35.

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Filter TACO dataset for sweet spot (pass@8 ∈ [10%, 50%])
bash scripts/run_filter.sh

# 3. Run all 17 experiments on 8×B200 (~3 days)
bash scripts/run_all_8gpu.sh

# Or run a single experiment:
CUDA_VISIBLE_DEVICES=0 bash scripts/run_experiment.sh configs/7B_R1.yaml
```

## Experiments

| Scale | R1 (test) | R2 (+pylint) | R3 (+complexity) | R4 (+comment) | R5 (all) |
|-------|-----------|-------------|-----------------|--------------|---------|
| 1.5B  | 1.5B_R1   | 1.5B_R2     | 1.5B_R3         | 1.5B_R4      | 1.5B_R5 |
| 7B    | 7B_R1     | 7B_R2       | 7B_R3           | 7B_R4        | 7B_R5   |
| 14B   | 14B_R1    | 14B_R2      | 14B_R3          | 14B_R4       | 14B_R5  |

Plus 2 weight robustness experiments: 7B_R2_w1 (0.5:0.5), 7B_R2_w2 (0.8:0.2)

## Resources

- **Dataset:** [goodhart-cascade-sweet-spot](https://huggingface.co/datasets/shatianming5/goodhart-cascade-sweet-spot)
- **Checkpoints:** `shatianming5/goodhart-cascade-{experiment}` on HuggingFace
- **Full Plan:** [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)

## Key Expected Results

1. **Escape Map**: Gaming displaced to unconstrained dimensions at each step
2. **Alignment Tax Scaling Law**: `tax(n) = C·(1-w)²/[n·w²+(1-w)²]` across 3 scales
3. **Universal ρ***: Optimal to constrain ~35% of quality dimensions
