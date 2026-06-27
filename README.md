# TFM-Lidar-Registration

Evaluator of multiple point cloud registration algorithms for lidar
datasets in Autonomous Driving.

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

## Usage

Experiments are driven by a YAML configuration file describing the
registration method, dataset, and metrics to use. See
[configs/template.yaml](configs/template.yaml) for the expected format:

```yaml
output_path: [Path to experiments folder]
transforms_path: [Path to save transformations]
overlap_threshold: 1.0

registration_method:
  name: icp
  params: {}

dataset:
  name: kitti
  params:
    root_path: [Path to dataset]

metrics:
  - name: rmse
    params: {}
  - name: chamfer_distance
    params: {}
```

`output_path`, `transforms_path`, `registration_method`, `dataset`, and
`metrics` are all required; each `registration_method`/`dataset`/metric
entry must have a `name` and a `params` mapping (use `{}` for no
parameters).

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
metrics. `--index` takes a comma-separated list of indices and defaults
to `all`:

```bash
# Run on the whole dataset
python src/main.py inference --config-path configs/template.yaml

# Run on specific samples and persist the resulting transforms
python src/main.py inference \
  --config-path configs/template.yaml \
  --index 0,1,2 \
  --save-transform
```
