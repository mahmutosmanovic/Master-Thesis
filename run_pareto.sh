#!/bin/bash
set -e

configs=("PDPS0" "PDPS025" "PDPS05" "PDPS1" "PDPS2" "PDPS4" "PDPS8")
max_jobs=3

manifest="pareto/runs_manifest_$(date +%Y%m%d_%H%M%S).csv"

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
        --wandb | tee /dev/tty | grep "RUN_DIR::" | cut -d':' -f3)

    echo "Run created: $run_name"
    echo "Starting evaluation..."

    eval_name=$(python -m scripts.eval_models \
        --run "$run_name" \
        --num-episodes 100 \
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