#!/bin/bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <runs_manifest.csv>"
    exit 1
fi

selected_manifest="$1"
max_jobs="${MAX_JOBS:-2}"
runs_dir="${RUNS_DIR:-runs}"
evals_dir="${EVALS_DIR:-evals}"

animals=("behaviors_t4/jackals_km_sm" "behaviors_t4/pigeons_km_sm" "behaviors_t4/spur_winged_lapwings_km_sm")

mkdir -p table
out_manifest="table/eval_animals_manifest_$(date +%Y%m%d_%H%M%S).csv"
echo "config,agent,full_config,run_name,animal,behavior_cfg,eval_name,dest_path" > "$out_manifest"

patch_config() {
    local config_path="$1"
    local behavior_cfg_path="$2"

    python - "$config_path" "$behavior_cfg_path" <<'PY'
import sys
import yaml

config_path = sys.argv[1]
behavior_cfg_path = sys.argv[2]

with open(config_path, "r") as f:
    data = yaml.safe_load(f) or {}

# behavior_cfg must be a STRING path, not a dict
data["behavior_cfg"] = behavior_cfg_path

animal = data.get("animal")
if not isinstance(animal, dict):
    animal = {}
data["animal"] = animal

init = animal.get("init")
if not isinstance(init, dict):
    init = {}
animal["init"] = init
init["behavior"] = "REPLAY_CFG"

with open(config_path, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)
PY
}

run_eval_animals() {
    local cfg="$1"
    local agent="$2"
    local full_cfg="$3"
    local run_name="$4"

    local run_dir="$runs_dir/$run_name"
    local config_path="$run_dir/config.yaml"
    local backup_path="$run_dir/config.yaml.eval_animals.bak"

    local animal
    local behavior_cfg_path
    local eval_name
    local dest_dir
    local dest_path

    if [ ! -f "$config_path" ]; then
        echo "Missing config: $config_path" >&2
        exit 1
    fi

    cp "$config_path" "$backup_path"
    trap 'mv -f "$backup_path" "$config_path"' EXIT

    for animal in "${animals[@]}"; do
        behavior_cfg_path="$animal"

        echo "================================="
        echo "Run: $run_name | animal: $animal"
        echo "================================="

        cp "$backup_path" "$config_path"
        patch_config "$config_path" "$behavior_cfg_path"

        eval_name=$(python -m scripts.eval_models \
            --run "$run_name" \
            --baseline centroid \
            --num-episodes 100 \
            --plot-rewards \
            --plot-heatmaps \
            --start-seed 42 | tee /dev/tty | grep "EVAL_DIR::" | cut -d':' -f3)

        dest_dir="$evals_dir/eval_animals/$animal"
        dest_path="$dest_dir/$eval_name"

        mkdir -p "$dest_dir"
        mv "$evals_dir/$eval_name" "$dest_path"

        {
            flock 200
            echo "$cfg,$agent,$full_cfg,$run_name,$animal,$behavior_cfg_path,$eval_name,$dest_path" >> "$out_manifest"
        } 200>"$out_manifest.lock"

        echo "Moved eval to $dest_path"
        echo ""
    done
}

while IFS=, read -r cfg agent full_cfg run_name prev_eval_name; do
    [ "$cfg" = "config" ] && continue
    [ -z "${run_name// }" ] && continue

    run_name="${run_name%$'\r'}"
    cfg="${cfg%$'\r'}"
    agent="${agent%$'\r'}"
    full_cfg="${full_cfg%$'\r'}"

    (
        run_eval_animals "$cfg" "$agent" "$full_cfg" "$run_name"
    ) &

    while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
        sleep 2
    done
done < "$selected_manifest"

wait

echo "All animal evals finished."
echo "Manifest saved to $out_manifest"