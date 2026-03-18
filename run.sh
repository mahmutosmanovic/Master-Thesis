#!/bin/bash
set -e

trap 'echo "Stopping..."; kill 0; exit 1' SIGINT

configs=("CRWMOD" "CRWPPO")
max_jobs=2

for cfg in "${configs[@]}"
do
(
    echo "================================="
    echo "Training config: $cfg"
    echo "================================="

    run_name=$(python -m scripts.train_agent \
        --config "$cfg" \
        --seed 42 \
        --wandb | grep "RUN_DIR::" | cut -d':' -f3)

    echo "Run created: $run_name"
    echo "Starting evaluation..."

    python -m scripts.eval_models \
        --run "$run_name" \
        --baseline centroid \
        --num-episodes 100 \
        --start-seed 42 \
        --weights last \
        --plot-rewards \
        --plot-heatmaps

    echo "Finished config: $cfg"
    echo ""
) &

while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
    sleep 2
done

done

wait || true

echo "All experiments finished."