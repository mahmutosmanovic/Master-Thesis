#!/bin/bash
# Heterogeneous-morphology experiment with GSD-based observation quality.
#
# Each *__GSD config is the original CRW_MAPPO__1A_3D_* config with:
#   - view_range set to the distance at which each camera reaches the M2 GSD threshold
#     (M1 100deg: 280 m, M2 140deg: 200 m (unchanged), M3 170deg: 164.7 m)
#   - a `camera` block, so the quality term is GSD / gsd_max (12.73 cm/px)
# Speed and disturbance_mult per type are unchanged (M1 15/1.0, M2 20/1.1, M3 25/1.2).
#
# M2x3 is included as the homogeneous default-drone control trained with the same code.
# After each eval, scripts.gsd_analysis writes gsd_summary.csv into the eval dir.
#
# Usage:
#   bash bash/run_morf_gsd.sh                 # seed 42, 2 parallel jobs
#   SEEDS="42 84 126" MAX_JOBS=3 bash bash/run_morf_gsd.sh
#   CONFIGS="CRW_MAPPO__1A_3D_M1x3__GSD CRW_MAPPO__1A_3D_M3x3__GSD" bash bash/run_morf_gsd.sh
set -e

# Never log to Weights & Biases from this experiment
export WANDB_MODE=disabled

cd "$(dirname "$0")/.."

all_configs=(
    "CRW_MAPPO__1A_3D_M1x3__GSD"
    "CRW_MAPPO__1A_3D_M2x3__GSD"
    "CRW_MAPPO__1A_3D_M3x3__GSD"
    "CRW_MAPPO__1A_3D_M1_M2_M3__GSD"
    "CRW_MAPPO__1A_3D_M1x2_M2x1__GSD"
    "CRW_MAPPO__1A_3D_M1x2_M3x1__GSD"
    "CRW_MAPPO__1A_3D_M2x2_M1x1__GSD"
    "CRW_MAPPO__1A_3D_M2x2_M3x1__GSD"
    "CRW_MAPPO__1A_3D_M3x2_M1x1__GSD"
    "CRW_MAPPO__1A_3D_M3x2_M2x1__GSD"
)

if [ -n "${CONFIGS:-}" ]; then
    configs=(${CONFIGS})
else
    configs=("${all_configs[@]}")
fi

seeds=(${SEEDS:-42})
max_jobs=${MAX_JOBS:-2}
summary_dir="table/morf_gsd_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$summary_dir"

for seed in "${seeds[@]}"
do
for cfg in "${configs[@]}"
do
(
    echo "================================="
    echo "Training config: $cfg (seed $seed)"
    echo "================================="

    run_name=$(python -m scripts.train_agent \
        --config "$cfg" \
        --agent mappo \
        --seed "$seed" | tee /dev/stderr | grep "RUN_DIR::" | cut -d':' -f3)

    echo "Run created: $run_name"
    echo "Starting evaluation..."

    eval_name=$(python -m scripts.eval_models \
        --run "$run_name" \
        --baseline centroid \
        --num-episodes 30 \
        --plot-rewards \
        --plot-heatmaps \
        --start-seed 42 | tee /dev/stderr | grep "EVAL_DIR::" | cut -d':' -f3)

    echo "Evaluation created: $eval_name"
    echo "Starting GSD analysis..."

    python -m scripts.gsd_analysis "evals/$(basename "$eval_name")" \
        --out "$summary_dir/${cfg}_seed${seed}.csv"

    echo "Finished config: $cfg (seed $seed)"
    echo ""
) &

while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
    sleep 2
done

done
done

wait

python - "$summary_dir" <<'EOF'
import sys, glob, pandas as pd
d = sys.argv[1]
files = sorted(glob.glob(f"{d}/*.csv"))
if files:
    pd.concat([pd.read_csv(f) for f in files]).to_csv(f"{d}/all.csv", index=False)
    print(f"Combined summary: {d}/all.csv")
EOF

echo ">>>> Experiment Finished."
