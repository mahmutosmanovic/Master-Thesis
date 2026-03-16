#!/bin/bash
set -euo pipefail

configs=(
  "LPOI_SPEEDRATIO_10"
  "LPOI_SPEEDRATIO_08"
  "LPOI_SPEEDRATIO_06"
  "LPOI_SPEEDRATIO_04"
  "LPOI_SPEEDRATIO_02"
)

max_jobs=3

mkdir -p speed_sensitivity
manifest="speed_sensitivity/runs_manifest_$(date +%Y%m%d_%H%M%S).csv"

echo "config,run_name,eval_name" > "$manifest"

for cfg in "${configs[@]}"
do
(
    echo "================================="
    echo "Training config: $cfg"
    echo "================================="

    run_name=$(python -m scripts.train_agent \
        --config "$cfg" \
        --seed 42 \
        --wandb | tee /dev/tty | grep "RUN_DIR::" | sed 's/^RUN_DIR:://')

    echo "Run created: $run_name"
    echo "Starting evaluation..."

    eval_name=$(python -m scripts.eval_models \
        --run "$run_name" \
        --baseline centroid \
        --num-episodes 100 \
        --start-seed 42 \
        --weights last \
        --plot-rewards \
        --plot-heatmaps \
        | tee /dev/tty | grep "EVAL_DIR::" | sed 's/^EVAL_DIR:://')

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