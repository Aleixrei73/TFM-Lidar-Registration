#!/bin/bash
#SBATCH --job-name=evaluate_array
#SBATCH --partition=gpi.compute
#SBATCH --nodelist=gpic15,gpic16
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --output=experiments/logs/%x_%A_%a.out
#SBATCH --error=experiments/logs/%x_%A_%a.err

# No --gres requested, so the job is scheduled CPU-only despite the
# partition's GPU nodes. --array (and its %-throttle) is set at submission
# time via `sbatch --array=...`, since it depends on the config list length.
# --nodes=1 is required: without it, sbatch defaults the node count to the
# number of entries in --nodelist, so each task would reserve every listed
# node instead of sharing the pool with the other array tasks.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch --array=0-N%20 scripts/jobs/evaluate_array.sh <config-list-file>" >&2
    exit 1
fi

: "${SLURM_ARRAY_TASK_ID:?must be run as a SLURM array job (missing SLURM_ARRAY_TASK_ID)}"

config_list_file="$1"
config_path="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$config_list_file")"

if [[ -z "$config_path" ]]; then
    echo "No config at line $((SLURM_ARRAY_TASK_ID + 1)) of $config_list_file" >&2
    exit 1
fi

python src/main.py evaluate --config-path "$config_path"
