# Preparing a source release

This repository provides the UAN code and a guide for running the pipeline with
data collected on the user's own robot. The release entry points are
[README](README.md) and [Hardware pipeline](docs/SIM2REAL.md).

Before releasing:

1. Check that a clean checkout installs with the locked dependencies:
   `uv sync --locked --extra cpu --extra deploy --group dev`.
2. Run `make check` and `make test`. On a Linux NVIDIA machine, also verify the
   training setup with `uv sync --locked --extra cu128 --group dev` and a short
   simulation run for each documented training stage.
3. Check the guides against the actual command-line interfaces, task names,
   input schemas, and checkpoint locations. In particular, jumping continues
   from `JumpingTrack` pretraining to `JumpingHeight` fine-tuning and exports
   the 161-dimensional `jumping_track` policy format.
4. Verify a trained checkpoint's export and local simulation playback with its
   matching `params/env.yaml` and reference motion. Keep all training inputs
   and outputs explicit so the instructions work outside the original workspace.
5. Update the upcoming changelog and check that links point to files included
   in the release. Keep generated logs, datasets, and training checkpoints out
   of the release, including pretrained deployment policy packages.

Publish the reviewed source revision through this repository's GitHub release
process. The `mjlab` package name and upstream attribution remain because this
code builds on MJLab; this repository does not publish the upstream PyPI package
or its container images.
