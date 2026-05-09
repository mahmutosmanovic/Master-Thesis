#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ $# -ne 1 ]; then
    echo "Usage: $0 <runs_manifest.csv>"
    exit 1
fi

selected_manifest="$1"

max_jobs="${MAX_JOBS:-2}"
runs_dir="${RUNS_DIR:-runs}"
evals_dir="${EVALS_DIR:-evals}"

mkdir -p table
out_manifest="table/replay_eval_manifest_$(date +%Y%m%d_%H%M%S).csv"
echo "config,agent,full_config,run_name,prev_eval_name,eval_name,dest_path" > "$out_manifest"

patch_config() {
    local config_path="$1"

    python - "$config_path" <<'PY'
import sys
import yaml

config_path = sys.argv[1]

with open(config_path, "r") as f:
    data = yaml.safe_load(f) or {}

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

run_replay_eval() {
    local cfg="$1"
    local agent="$2"
    local full_cfg="$3"
    local run_name="$4"
    local prev_eval_name="$5"

    local run_dir="$runs_dir/$run_name"
    local config_path="$run_dir/config.yaml"
    local backup_path="$run_dir/config.yaml.replay_eval.bak"

    local eval_name
    local dest_dir
    local dest_path

    if [ ! -f "$config_path" ]; then
        echo "Missing config: $config_path" >&2
        exit 1
    fi

    cp "$config_path" "$backup_path"
    trap 'mv -f "$backup_path" "$config_path"' EXIT

    echo "================================="
    echo "Run: $run_name"
    echo "Config: $full_cfg"
    echo "Previous eval: $prev_eval_name"
    echo "================================="

    patch_config "$config_path"

    eval_name=$(python -m scripts.eval_models \
        --run "$run_name" \
        --baseline centroid \
        --num-episodes 100 \
        --plot-rewards \
        --plot-heatmaps \
        --start-seed 42 | tee /dev/tty | grep "EVAL_DIR::" | cut -d':' -f3)

    dest_dir="$evals_dir/replay"
    dest_path="$dest_dir/$eval_name"

    mkdir -p "$dest_dir"
    mv "$evals_dir/$eval_name" "$dest_path"

    {
        flock 200
        echo "$cfg,$agent,$full_cfg,$run_name,$prev_eval_name,$eval_name,$dest_path" >> "$out_manifest"
    } 200>"$out_manifest.lock"

    echo "Moved eval to $dest_path"
    echo ""

    mv -f "$backup_path" "$config_path"
    trap - EXIT
}

while IFS=, read -r cfg agent full_cfg run_name prev_eval_name; do
    [ "$cfg" = "config" ] && continue
    [ -z "${run_name// }" ] && continue

    cfg="${cfg%$'\r'}"
    agent="${agent%$'\r'}"
    full_cfg="${full_cfg%$'\r'}"
    run_name="${run_name%$'\r'}"
    prev_eval_name="${prev_eval_name%$'\r'}"

    (
        run_replay_eval "$cfg" "$agent" "$full_cfg" "$run_name" "$prev_eval_name"
    ) &

    while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
        sleep 2
    done
done < "$selected_manifest"

wait

echo "All replay evals finished."
echo "Manifest saved to $out_manifest"