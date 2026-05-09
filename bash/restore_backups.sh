#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

runs_dir="${1:-runs}"

find "$runs_dir" -type f -name "config.yaml.eval_animals.bak" -print0 |
while IFS= read -r -d '' backup_path; do
    config_path="$(dirname "$backup_path")/config.yaml"
    mv -f "$backup_path" "$config_path"
    echo "Restored: $config_path"
done

echo "Done."