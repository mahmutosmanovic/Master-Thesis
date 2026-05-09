#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

MANIFEST="table/runs_manifest_20260421_141025_2.csv"

mkdir -p imgtemp

# Skip header and read CSV
tail -n +2 "$MANIFEST" | while IFS=',' read -r config agent full_config run_name eval_name
do
    echo "Running for eval_name=${eval_name}, agent=${agent}"

    python -m scripts.plots.policy_heatmap \
        --bins 100 \
        --csv "evals/${eval_name}/${agent}.csv"

    # --- lowercase conversion ---
    config_lc=$(echo "$config" | tr '[:upper:]' '[:lower:]')
    agent_lc=$(echo "$agent" | tr '[:upper:]' '[:lower:]')

    # --- source files ---
    src_rz="evals/${eval_name}/${agent_lc}_policy_heatmap_disturbance_bg.png"
    src_xy="evals/${eval_name}/${agent_lc}_policy_heatmap_xy.png"

    # --- destination files ---
    dst_rz="imgtemp/${config_lc}_${agent_lc}_policy_rz.png"
    dst_xy="imgtemp/${config_lc}_${agent_lc}_policy_xy.png"

    # --- copy (with overwrite) ---
    cp "$src_rz" "$dst_rz"
    cp "$src_xy" "$dst_xy"

done