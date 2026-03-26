#!/bin/bash
set -e

configs=("CRW" "EE" "POI" "LPOI")
max_jobs=1

mkdir -p table
manifest="table/runs_manifest_$(date +%Y%m%d_%H%M%S).csv"

echo "config,run_name,eval_name" > "$manifest"

for cfg in "${configs[@]}"
do
(
    echo "================================="
    echo "Training config: $cfg"
    echo "================================="

    run_name=$(python -m scripts.train_agent \
        --config "$cfg" \
        --agent ppo \
        --seed 42 \
        --wandb | tee /dev/tty | grep "RUN_DIR::" | cut -d':' -f3)

    echo "Run created: $run_name"
    echo "Starting evaluation..."

    eval_name=$(python -m scripts.eval_models \
        --run "$run_name" \
        --baseline centroid \
        --num-episodes 30 \
        --plot-rewards \
        --plot-heatmaps \
        --start-seed 42 | tee /dev/tty | grep "EVAL_DIR::" | cut -d':' -f3)

    echo "Finished config: $cfg"
    echo ""

    {
        flock 200
        echo "$cfg,$run_name,$eval_name" >> "$manifest"
    } 200>"$manifest.lock"

) &

while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
    sleep 2
done

done

wait

echo "All experiments finished."
echo "Manifest saved to $manifest"