#!/usr/bin/env bash
set -Eeuo pipefail
eval "$(/home/zechuan/anaconda3/bin/conda shell.bash hook)"
conda activate goodhart
cd ~/goodhart-cascade

echo "[Tue Mar 17 12:44:04 CST 2026] Waiting for test_only 7B to finish..."
while true; do
    if ! pgrep -f 'train_grpo_trl.*test_only' > /dev/null 2>&1; then
        echo "[Tue Mar 17 12:44:04 CST 2026] test_only 7B finished. Waiting 30s before launching 14B..."
        sleep 30
        if ! pgrep -f 'train_grpo_trl.*test_only' > /dev/null 2>&1; then
            echo "[Tue Mar 17 12:44:04 CST 2026] Confirmed. Launching 14B test_only..."
            CUDA_VISIBLE_DEVICES=0,1,6,7,8,9 bash scripts/launch_train.sh 14b test_only
            break
        fi
    fi
    sleep 60
done
echo "[Tue Mar 17 12:44:04 CST 2026] 14B training launched."
rm -- "$0"
