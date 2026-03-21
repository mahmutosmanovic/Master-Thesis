#!/bin/bash
set -euo pipefail

runs=(
    "PDPS035_seed42_2026-03-16_17-24-11"
    "PDPS0_seed42_2026-03-16_17-24-11"
    "PDPS025_seed42_2026-03-16_17-24-11"
    "PDPS07_seed42_2026-03-16_17-33-28"
    "PDPS05_seed42_2026-03-16_17-33-22"
    "PDPS1_seed42_2026-03-16_17-33-40"
    "PDPS2_seed42_2026-03-16_17-42-43"
    "PDPS28_seed42_2026-03-16_17-42-56"
    "PDPS14_seed42_2026-03-16_17-42-42"
)

max_jobs=3

mkdir -p pareto
manifest="pareto/eval_manifest_$(date +%Y%m%d_%H%M%S).csv"

echo "run_name,eval_name" > "$manifest"

for run_name in "${runs[@]}"
do
(
    echo "================================="
    echo "Evaluating run: $run_name"
    echo "================================="

    if [ ! -d "runs/$run_name" ]; then
        echo "Run directory missing: runs/$run_name"
        exit 1
    fi

    eval_name=$(python -m scripts.eval_models \
        --run "$run_name" \
        --baseline centroid \
        --num-episodes 100 \
        --start-seed 42 \
        --weights last \
        --plot-rewards \
        --plot-heatmaps \
        | tee /dev/tty | grep "EVAL_DIR::" | sed 's/^EVAL_DIR:://')

    echo "Finished run: $run_name"
    echo "Eval created: $eval_name"
    echo ""

    {
        flock 200
        echo "$run_name,$eval_name" >> "$manifest"
    } 200>"$manifest.lock"

) &

while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
    sleep 2
done

done

wait

echo "All evaluations finished."
echo "Manifest saved to $manifest"