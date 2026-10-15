# Contact-UAN

Train a humanoid UAN residual world model from data collected on your own
Unitree G1, then train running, force-control, or jumping policies with the UAN
frozen. This repository builds on [MJLab](https://github.com/mujocolab/mjlab)
and retains its `mjlab` Python package name.

Start with the [hardware pipeline guide](docs/SIM2REAL.md): train WTW, collect
robot data, train the paper's six separate state estimators, relabel the data,
train a UAN, and train downstream policies. The
[deployment reference](docs/DEPLOYMENT.md) covers controller setup and export.
[Dataset format](DATASETS.md) describes the
files passed between stages.

## Install

Training requires Linux and an NVIDIA GPU. Install with
[uv](https://docs.astral.sh/uv/) from the repository root:

```bash
uv sync --locked --extra cu128
uv run --locked --extra cu128 python -c "import mjlab, mjlab.tasks; print('mjlab import ok')"
```

For CPU-only offline work, including macOS, use `uv sync --locked --extra cpu`
instead. Controller setup also needs the optional `deploy` extra; follow the
[deployment prerequisites](docs/DEPLOYMENT.md#0-prepare-dependencies-offline).

Pretrained policies and hardware datasets are not included. Keep collected
data, checkpoints, W&B logs, and generated figures under ignored `data/` or
`logs/` directories, or external storage.
