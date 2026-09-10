#!/bin/bash
#SBATCH --job-name=create_rosbag
#SBATCH --partition=gpi.compute
#SBATCH --nodelist=gpic16
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err

# No --gres requested, so the job is scheduled CPU-only despite the
# partition's GPU nodes.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: sbatch scripts/jobs/create_rosbag.sh <config-path> <output-path> [index] [fps]" >&2
    exit 1
fi

config_path="$1"
output_path="$2"
index="${3:-}"
fps="${4:-}"

args=(--config-path "$config_path" --output-path "$output_path")
if [[ -n "$index" ]]; then
    args+=(--index "$index")
fi
if [[ -n "$fps" ]]; then
    args+=(--fps "$fps")
fi

python src/main.py create-rosbag "${args[@]}"
