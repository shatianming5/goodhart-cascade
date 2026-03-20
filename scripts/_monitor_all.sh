#\!/usr/bin/env bash
set -euo pipefail
eval "$(/home/zechuan/anaconda3/bin/conda shell.bash hook)"
conda activate goodhart
cd ~/goodhart-cascade
LOG=outputs/monitor.log

while true; do
    echo "========== $(date "+%Y-%m-%d %H:%M:%S") ==========" >> "$LOG"

    # multi_objective 7B
    MO_PROG=$(grep -oP "\d+/1548.*?s/it" outputs/train_7b_multi.log 2>/dev/null | tail -1 || echo "N/A")
    MO_REWARD=$(grep -oP "'reward': [0-9.]+" outputs/train_7b_multi.log 2>/dev/null | tail -1 | grep -oP "[0-9.]+" || echo "N/A")
    MO_ALIVE=$(pgrep -f "reward_mode multi_objective" > /dev/null 2>&1 && echo "running" || echo "stopped")

    # test_only 14B
    T14_PROG=$(grep -oP "\d+/516.*?s/it" outputs/train_14b.log 2>/dev/null | tail -1 || echo "N/A")
    T14_REWARD=$(grep -oP "'reward': [0-9.]+" outputs/train_14b.log 2>/dev/null | tail -1 | grep -oP "[0-9.]+" || echo "N/A")
    T14_ALIVE=$(pgrep -f "reward_mode test_only.*14b\|Coder-14B" > /dev/null 2>&1 && echo "running" || echo "stopped")

    printf "| %-20s | %-35s | %-35s |\n" "指标" "multi_objective 7B (GPU2345)" "test_only 14B (GPU016789)" >> "$LOG"
    printf "| %-20s | %-35s | %-35s |\n" "进度" "$MO_PROG" "$T14_PROG" >> "$LOG"
    printf "| %-20s | %-35s | %-35s |\n" "reward" "$MO_REWARD" "$T14_REWARD" >> "$LOG"
    printf "| %-20s | %-35s | %-35s |\n" "状态" "$MO_ALIVE" "$T14_ALIVE" >> "$LOG"

    # Auto-restart if crashed
    if [[ "$MO_ALIVE" == "stopped" && "$MO_PROG" \!= *"1548/"* ]]; then
        LAST_MO_STEP=$(echo "$MO_PROG" | grep -oP "^\d+" || echo "0")
        if [[ "$LAST_MO_STEP" -lt 1548 && "$LAST_MO_STEP" -gt 0 ]]; then
            echo "[$(date)] multi_objective 7B crashed at step $LAST_MO_STEP, restarting..." >> "$LOG"
            CUDA_VISIBLE_DEVICES=2,3,4,5 nohup bash scripts/launch_train.sh 7b multi_objective >> outputs/train_7b_multi.log 2>&1 &
        fi
    fi

    if [[ "$T14_ALIVE" == "stopped" && "$T14_PROG" \!= *"516/"* ]]; then
        LAST_14_STEP=$(echo "$T14_PROG" | grep -oP "^\d+" || echo "0")
        if [[ "$LAST_14_STEP" -lt 516 && "$LAST_14_STEP" -gt 0 ]]; then
            echo "[$(date)] test_only 14B crashed at step $LAST_14_STEP, restarting..." >> "$LOG"
            CUDA_VISIBLE_DEVICES=0,1,6,7,8,9 nohup bash scripts/launch_train.sh 14b test_only >> outputs/train_14b.log 2>&1 &
        fi
    fi

    echo "" >> "$LOG"

    # Exit when all done
    if [[ "$MO_ALIVE" == "stopped" && "$T14_ALIVE" == "stopped" ]]; then
        MO_DONE=$(grep -c "Training complete" outputs/train_7b_multi.log 2>/dev/null || echo "0")
        T14_DONE=$(grep -c "Training complete" outputs/train_14b.log 2>/dev/null || echo "0")
        if [[ "$MO_DONE" -gt 0 && "$T14_DONE" -gt 0 ]]; then
            echo "[$(date)] 所有实验已完成！" >> "$LOG"
            rm -- "$0"
            break
        fi
    fi

    sleep 1800
done
