#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# configs=("JACKALS" "JACKALS_GPS" "PIGEONS" "PIGEONS_GPS" "SPUR" "SPUR_GPS")
# agents=("sac")
# configs=("CRW" "EE" "POI" "LPOI")
# configs=("JACKALS" "PIGEONS" "SPUR")
# agents=("dqn" "sac" "ppo")

configs=("LPOI")
agents=("sac")

max_jobs=1

mkdir -p table
manifest="table/runs_manifest_$(date +%Y%m%d_%H%M%S).csv"

echo "config,agent,full_config,run_name,eval_name" > "$manifest"

for cfg in "${configs[@]}"
do
for agent in "${agents[@]}"
do
(
    full_cfg="${cfg}_${agent}"

    echo "================================="
    echo "Training config: $full_cfg"
    echo "================================="

    run_name=$(python -m scripts.train_agent \
        --config "$full_cfg" \
        --agent "$agent" \
        --seed 42 \
        --wandb | tee /dev/tty | grep "RUN_DIR::" | cut -d':' -f3)

    echo "Run created: $run_name"
    echo "Evaluating run $run_name"

    eval_name=$(python -m scripts.eval_models \
        --run "$run_name" \
        --baseline centroid \
        --num-episodes 100 \
        --plot-rewards \
        --plot-heatmaps \
        --start-seed 42 | tee /dev/tty | grep "EVAL_DIR::" | cut -d':' -f3)

    echo "Finished config: $full_cfg"
    echo ""

    {
        flock 200
        echo "$cfg,$agent,$full_cfg,$run_name,$eval_name" >> "$manifest"
    } 200>"$manifest.lock"

) &

while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
    sleep 2
done

done
done

wait

echo "All experiments finished."
echo "Manifest saved to $manifest"