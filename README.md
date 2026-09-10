# TFM-Lidar-Registration

Evaluator of multiple point cloud registration algorithms for lidar
datasets in Autonomous Driving.

Given a dataset of infrastructure/vehicle (or vehicle/vehicle) lidar
point cloud pairs and a ground-truth relative transform, the pipeline
runs a chosen registration method on every pair, scores the result with
a set of metrics, and persists per-sample and aggregate results. Runs
are driven entirely by YAML configuration files, so new methods,
datasets, and metrics can be compared without touching the CLI.

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
  Beamagine crowdsourced dataset, each yielding point clouds plus an
  IMU/GPS-derived initial transform and the ground-truth transform.
- **Metrics** (`src/metrics/`): a `Metric` interface, currently
  implemented by point-based Relative Rotation Error (RRE) and
  Relative Translation Error (RTE).
- **Pipeline** (`src/pipeline.py`): `ExperimentPipeline` ties a method,
  dataset, and metrics together, with optional overlap-based sample
  skipping and constant-velocity (order-1) extrapolation of the initial
  transform guess across a sequence.
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

For running tests and linting, install the development extras instead:

```bash
pip install -e ".[dev]"
```

Two registration dependencies are not on PyPI and must be installed
separately:

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

registration_method:
  name: identity          # identity | kiss_matcher | vgicp | vgicp-param |
                           # vgicp-cofi | vgicp-plane | vgicp-plane-init |
                           # vgicp-plane-remove | vgicp-plane-remove-init |
                           # vgicp-plane-remove-match | vgicp-grid
  params: {}               # forwarded as **kwargs to load_method

dataset:
  name: v2x_real            # v2x_real | v2x_real_same_vehicle | beamagine
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
`params` mapping (use `{}` for no parameters). See
[configs/](configs/) for real examples of each method/dataset
combination, including the `use_previous`, `overlap`, and `order`
sequence-tracking options documented on `ExperimentPipeline`.

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

### `visualize-single`

Runs registration on a single sample and opens an Open3D window showing
the point clouds before and after alignment. The resulting transform is
not saved to disk:

```bash
python src/main.py visualize-single --config-path configs/template.yaml --index 0
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
