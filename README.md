# TFM-Lidar-Registration

Evaluator of multiple point cloud registration algorithms for lidar
datasets in Autonomous Driving.

Given a dataset of infrastructure/vehicle (or vehicle/vehicle) lidar
point cloud pairs and a ground-truth relative transform, the pipeline
runs a chosen registration method on every pair, scores the result with
a set of metrics, and persists per-sample and aggregate results. Runs
are driven entirely by YAML configuration files, so new methods,
datasets, and metrics can be compared without touching the CLI.

You can find the thesis document [here](TFM.pdf).

## Overview

- **Registration methods** (`src/registration/`): a common
  `RegistrationMethod` interface (`preprocess` → `fit` → `register`)
  implemented by an `identity` baseline, a TEASER++-style robust solver
  (`KISSMatcher`), and a family of Voxelized GICP (VGICP, via
  `small_gicp`) variants that add coarse-to-fine voxel sizing, ground
  plane removal or flattening before matching, and a brute-force
  coarse-to-fine grid search over the initial pose.
- **Datasets** (`src/datasets/`): PyTorch `Dataset` loaders for
  V2X-Real (infrastructure/vehicle pairs, or same-vehicle
  frame-to-frame pairs, with optional synthetic pose noise) and for the
  Beamagine crowdsourced dataset (`beamagine` and `beamagine_v2`), each yielding point clouds plus an
  IMU/GPS-derived initial transform and the ground-truth transform.
- **Metrics** (`src/metrics/`): a `Metric` interface, currently
  implemented by point-based Relative Rotation Error (RRE) and
  Relative Translation Error (RTE).
- **Pipeline** (`src/pipeline.py`): `ExperimentPipeline` ties a method,
  dataset, and metrics together, with optional overlap-based sample
  skipping and sequence tracking of the initial transform guess
  (previous-transform chaining, constant-velocity extrapolation, or
  frame-to-frame composition), see
  [Sequence-tracking options](#sequence-tracking-options).
- **CLI** (`src/main.py`): `evaluate`, `inference`,
  `visualize-single`, and `create-rosbag` commands, see [Usage](#usage).
- **Scripts** (`scripts/`): grid-search config generation, sweep-status
  reporting, and Slurm job scripts under `scripts/jobs/`.
- **Configs** (`configs/`): one YAML file per experiment; `configs/`
  holds a large set of sweeps used for the underlying thesis work.
- **Experiments** (`experiments/`): output of `evaluate` runs, one
  UUID-named subfolder per run containing `config.yaml` and
  `metrics.json`.

## Installation

Requires Python 3.10+.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras can be added in brackets, e.g. `pip install -e ".[plots]"`:

- `plots`: `matplotlib` and `seaborn`.
- `registration`: `small_gicp` and `kiss-matcher`, if they can be
  resolved from your package index (see below).
- `dev`: `pytest` and `ruff`, for running tests and linting.

Two registration dependencies are not on PyPI and must be installed
separately otherwise:

- **small_gicp** (used by every VGICP variant): follow the
  [small_gicp installation guide](https://github.com/koide3/small_gicp).
- **kiss_matcher** (used by the `kiss_matcher` method, a TEASER++-based
  robust solver): follow the
  [KISS-Matcher installation guide](https://github.com/MIT-SPARK/KISS-Matcher).

## Usage

Experiments are driven by a YAML configuration file describing the
registration method, dataset, and metrics to use. See
[configs/template.yaml](configs/template.yaml) for the expected format:

```yaml
output_path: experiments/my_run
transforms_path: transforms/my_run.npy
overlap_threshold: 1.0
checkpoint_every: 100    # optional, default 10
use_previous: false      # optional, default false
overlap: 0.0             # optional, default 0.0, only used with use_previous
order: 0                 # optional, default 0; one of -2, -1, 0, 1

registration_method:
  name: identity          # identity | kiss_matcher | vgicp | vgicp-param |
                           # vgicp-cofi | vgicp-plane | vgicp-plane-init |
                           # vgicp-plane-remove | vgicp-plane-remove-init |
                           # vgicp-plane-remove-match | vgicp-grid
  params: {}               # forwarded as **kwargs to load_method

dataset:
  name: v2x_real            # v2x_real | v2x_real_same_vehicle | beamagine |
                            # beamagine_v2
  params:                   # forwarded as **kwargs to load_dataset
    root_dir: /path/to/V2X-Real-Lidar-128/train

metrics:
  - name: point_based
    params:
      rte_threshold: 0.2
      rre_threshold: 0.2
```

`output_path`, `transforms_path`, `overlap_threshold`,
`registration_method`, `dataset`, and `metrics` are all required; each
`registration_method`/`dataset`/metric entry must have a `name` and a
`params` mapping (use `{}` for no parameters). `checkpoint_every`,
`use_previous`, `overlap`, and `order` are optional. See
[configs/](configs/) for real examples of each method/dataset
combination.

#### Sequence-tracking options

By default every sample is registered independently, starting from its
own IMU-derived initial transform. For sequential datasets, three
optional keys reuse the results of earlier samples instead:

- `use_previous`: if `true`, the initial guess of a sample is the
  previous sample's registered transform. Samples flagged as first
  frames of a scene (and the first sample of each chain) always use
  their own IMU-derived transform.
- `overlap`: a float in `[0, 1]`, only used together with
  `use_previous`. Samples whose overlap ratio does not exceed it are
  skipped (only the ratio is reported), and the first sample above it
  after such a gap starts a new chain with its own IMU-derived
  transform, marked `"first_frame": true` in its metrics.
- `order`: how the initial guess is built along a chain.

| `order` | Requires `use_previous` | Behaviour |
|---------|-------------------------|-----------|
| `0` (default) | no | The previous registered transform is the initial guess. |
| `1` | no | Constant-velocity extrapolation from the previous two registered transforms (once two are available). The history resets at each new chain. |
| `-1` | yes | Only first frames are registered infrastructure-to-vehicle. Every other sample registers the previous vehicle cloud onto the current one, and the result is composed with the previous registered transform. |
| `-2` | yes | Like `-1`, but the composed transform is only the initial guess for a full infrastructure-to-vehicle registration, whose output is the result. |

An `order` outside `{-2, -1, 0, 1}`, or a negative `order` with
`use_previous: false`, raises a `ValueError`. Example configs live in
[configs/beamagine/scenes_v2/](configs/beamagine/scenes_v2/) (orders 1
and -1) and
[configs/beamagine/vgicp-parametrized-best-v2.yaml](configs/beamagine/vgicp-parametrized-best-v2.yaml)
(order 0).

### `evaluate`

Runs the registration method over the full dataset, computes all
configured metrics, and saves per-sample and aggregate results plus the
resolved run configuration to a fresh UUID-named subfolder under
`output_path`:

```bash
python src/main.py evaluate --config-path configs/template.yaml
```

### `inference`

Runs registration on one or more dataset samples without computing
metrics. `--index` takes a comma-separated list of indices and/or
inclusive ranges (e.g. `0,5-8,12`), and defaults to `all`:

```bash
# Run on the whole dataset
python src/main.py inference --config-path configs/template.yaml

# Run on specific samples and persist the resulting transforms
python src/main.py inference \
  --config-path configs/template.yaml \
  --index 0,1,2 \
  --save-transform
```

### `create-rosbag`

Writes a ROS1 `.bag` file (via the pure-Python `rosbags` library, no
ROS installation required) containing the infrastructure and vehicle
point clouds plus the transform between them, for the requested
samples. Requires transforms previously saved with
`inference --save-transform` at `config["transforms_path"]`:

```bash
python src/main.py create-rosbag \
  --config-path configs/template.yaml \
  --index 0-10 \
  --output-path experiments/my_run.bag \
  --fps 15
```

## Tests

```bash
pytest
```

Tests live under `test/`, mirroring the `src/` structure.
