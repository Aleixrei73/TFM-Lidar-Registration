#!/bin/bash
# Recursively submits a single SLURM job array (see evaluate_array.sh) that
# runs every YAML config found under the given folder (including nested
# subfolders), capped at a configurable number of concurrently running tasks.

set -euo pipefail

if [[ $# -ne 1 && $# -ne 2 ]]; then
    echo "Usage: scripts/jobs/batch_folder.sh <folder-path> [max-concurrent-tasks]" >&2
    exit 1
fi

folder_path="$1"
max_concurrent="${2:-20}"

if [[ ! -d "$folder_path" ]]; then
    echo "Error: '$folder_path' is not a directory" >&2
    exit 1
fi

if ! [[ "$max_concurrent" =~ ^[0-9]+$ ]] || [[ "$max_concurrent" -eq 0 ]]; then
    echo "Error: max-concurrent-tasks must be a positive integer, got '$max_concurrent'" >&2
    exit 1
fi

manifest_path="experiments/logs/batch_folder_configs_$(date +%Y%m%d_%H%M%S).txt"

find "$folder_path" -type f -name '*.yaml' | sort > "$manifest_path"

num_configs=$(wc -l < "$manifest_path")
if [[ "$num_configs" -eq 0 ]]; then
    echo "No YAML configs found under '$folder_path'"
    rm "$manifest_path"
    exit 0
fi

echo "Submitting $num_configs config(s) (max $max_concurrent concurrent), listed in $manifest_path"
sbatch --array="0-$((num_configs - 1))%${max_concurrent}" scripts/jobs/evaluate_array.sh "$manifest_path"
