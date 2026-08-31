#!/bin/bash
# Recursively submits an evaluate.sh sbatch job for every YAML config found
# under the given folder (including nested subfolders).

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: configs/batch_folder.sh <folder-path>" >&2
    exit 1
fi

folder_path="$1"

if [[ ! -d "$folder_path" ]]; then
    echo "Error: '$folder_path' is not a directory" >&2
    exit 1
fi

found_any=false

while IFS= read -r -d '' config_path; do
    found_any=true
    echo "Submitting $config_path"
    sbatch scripts/jobs/evaluate.sh "$config_path"
done < <(find "$folder_path" -type f -name '*.yaml' -print0)

if [[ "$found_any" == false ]]; then
    echo "No YAML configs found under '$folder_path'"
fi
