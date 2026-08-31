#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --partition=gpi.compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err

# No --gres requested, so the job is scheduled CPU-only despite the
# partition's GPU nodes.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch scripts/jobs/evaluate.sh <config-path>" >&2
    exit 1
fi

config_path="$1"

python src/main.py evaluate --config-path "$config_path"
