#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --partition=gpi.compute
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=gpic16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err

# No --gres requested, so the job is scheduled CPU-only despite the
# partition's GPU nodes.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: sbatch scripts/jobs/inference.sh <config-path> [index] [--save-transform]" >&2
    exit 1
fi

config_path="$1"
shift

index=""
save_transform=false
for arg in "$@"; do
    if [[ "$arg" == "--save-transform" ]]; then
        save_transform=true
    else
        index="$arg"
    fi
done

args=(--config-path "$config_path")
if [[ -n "$index" ]]; then
    args+=(--index "$index")
fi
if [[ "$save_transform" == true ]]; then
    args+=(--save-transform)
fi

python src/main.py inference "${args[@]}"
