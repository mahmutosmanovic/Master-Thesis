#!/bin/bash
set -e

configs=("CRW_MAPPO__3A_3D_M2x3")
max_jobs=1

for cfg in "${configs[@]}"
do
(
    echo "================================="
    echo "Training config: $cfg"
    echo "================================="

    run_name=$(python -m scripts.train_agent \
        --config "$cfg" \
        --agent mappo \
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

    echo "Evaluation created: $eval_name"
    echo "Starting play..."

    python -m scripts.play \
        --run "$run_name" \
        --seed 42

    echo "Finished config: $cfg"
    echo ""

) &

while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
    sleep 2
done

done

wait

echo ">>>> Experiment Finished."