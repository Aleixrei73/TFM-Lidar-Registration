#!/bin/bash
# Submits every missing high-noise VGICP grid-search combo (see
# report_missing_sweep.py) as a single SLURM job array, throttled per
# evaluate_array.sh's --array setting and restricted to its node list.

set -euo pipefail

manifest_path="experiments/logs/missing_sweep_configs_$(date +%Y%m%d_%H%M%S).txt"

# Config paths live under a grid-search-{low,high}-noise folder per noise
# type, so filtering on that path segment selects high noise only.
python scripts/report_missing_sweep.py --paths-only | grep "/grid-search-high-noise/" > "$manifest_path" || true

num_configs=$(wc -l < "$manifest_path")
if [[ "$num_configs" -eq 0 ]]; then
    echo "Nothing missing for high noise, no jobs submitted."
    rm "$manifest_path"
    exit 0
fi

echo "Submitting $num_configs missing config(s), listed in $manifest_path"
sbatch --array="0-$((num_configs - 1))%30" scripts/jobs/evaluate_array.sh "$manifest_path"
