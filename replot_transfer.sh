#!/usr/bin/env bash

set -euo pipefail

BASE="evals/eval_animals/behaviors_t5"
OUT="imgtemp_transfer"

mkdir -p "$OUT"

animals=(
  "jackals_km_sm"
  "pigeons_km_sm"
  "spur_winged_lapwings_km_sm"
)

agents=("dqn" "ppo" "sac")

get_config() {
    local agent="$1"
    local animal="$2"

    case "${agent}:${animal}" in
        dqn:jackals_km_sm) echo "CRW" ;;
        dqn:pigeons_km_sm) echo "CRW" ;;
        dqn:spur_winged_lapwings_km_sm) echo "CRW" ;;

        ppo:jackals_km_sm) echo "EE" ;;
        ppo:pigeons_km_sm) echo "POI" ;;
        ppo:spur_winged_lapwings_km_sm) echo "LPOI" ;;

        sac:jackals_km_sm) echo "EE" ;;
        sac:pigeons_km_sm) echo "LPOI" ;;
        sac:spur_winged_lapwings_km_sm) echo "LPOI" ;;

        *) echo "UNKNOWN"; return 1 ;;
    esac
}

for animal in "${animals[@]}"; do
    for agent in "${agents[@]}"; do
        config=$(get_config "$agent" "$animal")

        echo "Processing animal=${animal}, agent=${agent}, config=${config}"

        run_dir=$(find "$BASE/$animal" -maxdepth 1 -type d -name "${config}_${agent}*" | head -n 1)

        if [[ -z "$run_dir" ]]; then
            echo "Missing run folder for ${animal}, ${agent}, ${config}" >&2
            exit 1
        fi

        agent_lc=$(echo "$agent" | tr '[:upper:]' '[:lower:]')

        # remove suffix
        animal_clean=${animal%_km_sm}

        # custom shortening
        case "$animal_clean" in
            spur_winged_lapwings) animal_clean="swl" ;;
        esac

        animal_lc=$(echo "$animal_clean" | tr '[:upper:]' '[:lower:]')

        # --- RUN PYTHON SCRIPT ---
        python -m scripts.plots.policy_heatmap \
            --bins 100 \
            --csv "${run_dir}/${agent_lc}.csv"

        # --- FILE PATHS ---
        src_rz="${run_dir}/${agent_lc}_policy_heatmap_disturbance_bg.png"
        src_xy="${run_dir}/${agent_lc}_policy_heatmap_xy.png"

        dst_rz="${OUT}/${animal_lc}_${agent_lc}_policy_rz.png"
        dst_xy="${OUT}/${animal_lc}_${agent_lc}_policy_xy.png"

        cp "$src_rz" "$dst_rz"
        cp "$src_xy" "$dst_xy"
    done
done