# Hardware Pipeline

This workflow trains WTW, collects data on your own G1 with that WTW policy,
trains six separate proprioceptive estimators, relabels hardware data, trains a UAN
and downstream frozen-UAN policies, then exports a deployable controller.

Complete the [installation](../README.md#install) first. Run the commands in
Bash from the repository root, replacing quoted `<...>` placeholders with your
paths. Keep the path variables available between stages. Hardware setup and
operator-run collection are covered in the [deployment reference](DEPLOYMENT.md).

## 1. Train WTW

Train a WTW source policy:

```bash
uv run --locked --extra cu128 train Mjlab-WTW-Unitree-G1 \
  --video True \
  --agent.logger wandb \
  --agent.max-iterations 100000 \
  --agent.run-name "<wtw_run_name>"
```

Set the checkpoint path:

```bash
WTW_PT="logs/rsl_rl/g1_wtw/<wtw_run>/model_<iter>.pt"
```

## 2. Export And Verify WTW For Deployment

Export the WTW checkpoint to a MuJoCo-order TorchScript policy:

```bash
uv run --locked --extra cu128 python deploy/export_mjlab_checkpoint_to_jit.py \
  --policy-type wtw \
  --checkpoint "$WTW_PT"

WTW_JIT="${WTW_PT%.pt}.jit"
```

Verify deploy config resolution:

```bash
uv run --locked --extra cu128 python deploy/verify_deploy_config.py \
  --policy "$WTW_JIT" \
  --policy-joint-order mujoco
```

All deploy scripts expect MuJoCo-order policy observations and actions.

## 3. Collect Hardware Data With WTW

Follow the [WTW hardware collection instructions](DEPLOYMENT.md#5-collect-hardware-data)
using `WTW_JIT` from the previous step. After collection, copy each run's
directory to the training machine. Give each run a unique directory name,
such as `wtw_run_001`, and set its local path:

```bash
HARDWARE_RUN_DIR="<local_collection_parent>/wtw_run_001"
```

The collector records MuJoCo-order state and final joint targets with
`state[i], action[i] -> state[i + 1]` timing; see [dataset format](../DATASETS.md).

Prepare the dataset offline. `sync` copies the run into `raw/wtw_run_001/`
under a separate dataset root; `inspect` produces plots and trim suggestions:

```bash
HARDWARE_DATA_ROOT="data/hardware/<dataset_name>"

uv run --locked --extra cu128 python scripts/uan_pipeline/hardware_pipeline.py sync \
  --dataset-root "$HARDWARE_DATA_ROOT" \
  --task wtw \
  --policy "<wtw_run_name>" \
  --collection-datetime "<collection_timestamp>" \
  --source "$HARDWARE_RUN_DIR"

uv run --locked --extra cu128 python scripts/uan_pipeline/hardware_pipeline.py inspect \
  --dataset-root "$HARDWARE_DATA_ROOT"
```

Review `inspection/wtw_run_001/summary.png` and `summary.json` under that root.
Edit `artifacts/trim_manifest.json` so each run's `approved_intervals` retains
only valid collection intervals, excluding initialization and failed motion.
Inspection prepopulates this field with suggestions, so review it before
continuing. `start_row` and `end_row` are zero-based and inclusive; use an empty
list to exclude a run. Then materialize the selected intervals:

```bash
uv run --locked --extra cu128 python scripts/uan_pipeline/hardware_pipeline.py trim \
  --dataset-root "$HARDWARE_DATA_ROOT"
```

This creates `trimmed/<run>/<segment>/` CSV shards for the relabeling step.
For additional runs, repeat `sync` with their uniquely named local directories
before inspecting and trimming the dataset.

## 4. Collect WTW-Only Estimator Data

Estimator data is collected in simulation from the same WTW checkpoint on a
plane terrain. The absolute root-height and stance-foot world-height labels are
not observable from proprioception on randomized terrain, so the standard
sim-to-real estimator dataset uses 20 seconds with 4096 plane environments.

```bash
EST_DATA_ROOT="logs/estimator_data/<estimator_data_run>"

uv run --locked --extra cu128 python scripts/uan_pipeline/collect_estimator_data.py \
  --checkpoint_dir "$(dirname "$WTW_PT")" \
  --iterations "$(basename "$WTW_PT" .pt | sed 's/model_//')" \
  --num_envs 4096 \
  --seconds 20 \
  --terrain plane \
  --output "$EST_DATA_ROOT/estimator_training_data.pt"
```

Collection always records three proprioceptive frames and the labels needed for
all six estimators, including foot contact and stance-foot orientation.

## 5. Train The Six Separate Estimators

Use the separate estimators from the paper. This is the supported configuration;
combined estimation was less reliable in our experiments. The default command
trains the full set with ELU activations and a three-frame, 279-dimensional input:

| Estimator | Networks | Hidden layers | Outputs per network |
| --- | --- | --- | --- |
| Base linear velocity | 1 | 256, 256, 256 | 3 |
| Foot contact | 1 | 128, 128 | 2 contact logits |
| Stance-foot height | 2, one per foot | 256, 256, 256 | 1 |
| Stance-foot orientation | 2, one per foot | 256, 256, 256 | 3 projected-gravity components |

Foot-height and orientation networks train only on samples where their foot is
in contact. Failed episodes are excluded from training by default.

```bash
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
EST_ROOT="logs/estimators/wtw_${RUN_STAMP}"

uv run --locked --extra cu128 python scripts/uan_pipeline/train_estimators.py \
  --data "$EST_DATA_ROOT/estimator_training_data.pt" \
  --epochs 200 \
  --lr 3e-4 \
  --batch_size 4096 \
  --val_fraction 0.1 \
  --device cuda:0 \
  --wandb_project mjlab-state-estimators \
  --output_dir "$EST_ROOT" \
  --run_name "wtw_${RUN_STAMP}"
```

The trainer checks the dataset before training and writes
`paper_estimators_meta.json` after all six networks finish. Each component has
its own `state_estimator.jit` and `estimator_meta.json`.

To retrain one component, select its name and its existing output directory. For
example, to update only the left stance-foot height estimator:

```bash
uv run --locked --extra cu128 python scripts/uan_pipeline/train_estimators.py \
  --data "$EST_DATA_ROOT/estimator_training_data.pt" \
  --target_group left_stance_foot_height \
  --output_dir "$EST_ROOT/stance_foot_height/left_stance_foot_height"
```

## 6. Relabel Hardware Data

Relabel the prepared dataset's `trimmed/` intervals with estimator predictions
and stance-foot geometry:

```bash
LABELED_DATA_ROOT="<labeled_hardware_root>"

uv run --locked --extra cu128 python \
  scripts/uan_pipeline/relabel_hardware_stance_geometry_mjlab.py \
  --source-root "$HARDWARE_DATA_ROOT" \
  --output-root "$LABELED_DATA_ROOT" \
  --estimator-root "$EST_ROOT"
```

Relabeling validates all six models, their input histories and target layouts
before creating output files. Existing paper-configuration exports can be used
without retraining; combined estimators and incompatible histories are rejected.

Use the relabeled `processed` directory directly as UAN replay data.

## 7. Train UAN And Downstream Policies

Supply the labeled hardware replay directory explicitly with
`--env.commands.replay.data-dir`:

```bash
UAN_DATA_DIR="$LABELED_DATA_ROOT/processed"

uv run --locked --extra cu128 train Mjlab-UAN-Rollout-Unitree-G1 \
  --env.commands.replay.data-dir "$UAN_DATA_DIR" \
  --env.commands.replay.rollout-horizon 24 \
  --env.commands.replay.require-contact-at-reset True \
  --env.actions.residual-target.residual-clip 1.0 \
  --agent.num-steps-per-env 24 \
  --agent.max-iterations 2000 \
  --video True \
  --agent.logger wandb \
  --agent.run-name "<uan_run_name>"

UAN_PT="logs/rsl_rl/uan_rollout/<uan_run>/model_1999.pt"
```

### Running

Train a velocity-tracking baseline and a corresponding policy with the UAN
frozen. These tasks use the `g1_velocity_custom` experiment directory.

```bash
uv run --locked --extra cu128 train Mjlab-Velocity-Custom-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 30000 \
  --video True \
  --agent.logger wandb \
  --agent.run-name running_baseline

UAN_CHECKPOINT_PATH="$UAN_PT" \
uv run --locked --extra cu128 train Mjlab-Velocity-Custom-UAN-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 30000 \
  --video True \
  --agent.logger wandb \
  --agent.run-name running_uan
```

### Force Control

Train the force-control baseline and frozen-UAN policy. Both use the
`g1_force_control` experiment directory.

```bash
uv run --locked --extra cu128 train Mjlab-ForceControl-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 30000 \
  --video True \
  --agent.logger wandb \
  --agent.run-name force_control_baseline

UAN_CHECKPOINT_PATH="$UAN_PT" \
uv run --locked --extra cu128 train Mjlab-ForceControl-UAN-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 30000 \
  --video True \
  --agent.logger wandb \
  --agent.run-name force_control_uan
```

### Jumping: Tracking Pretraining And Height Fine-Tuning

Jumping uses two stages with the same 161-dimensional actor observations and
the same reference motion. `JumpingTrack` learns reference tracking;
`JumpingHeight` inherits that task and adds the height objective and fall
terminations for fine-tuning. In `src/mjlab/tasks/`, `jumping_track` contains
the G1 baseline/UAN configurations and shared PPO configuration, while
`jumping_height` contains the fine-tuning configuration and its reward and
metric helpers in `mdp/`.

First train reference tracking with your frozen UAN. The example run labels
identify the stage and variant within the shared experiment directory:

```bash
JUMP_MOTION="motions/02_04_jump_balance_pad50.npz"

UAN_CHECKPOINT_PATH="$UAN_PT" \
uv run --locked --extra cu128 train Mjlab-JumpingTrack-UAN-Unitree-G1 \
  --env.commands.motion.motion-file "$JUMP_MOTION" \
  --env.scene.num-envs 16384 \
  --agent.max-iterations 30000 \
  --agent.logger wandb \
  --agent.run-name stage1_track_pretrain_uan
```

Then resume that policy in `JumpingHeight`. Set `JUMP_TRACK_RUN` to the exact
directory name printed by pretraining under `logs/rsl_rl/g1_jumping_track/`,
including its timestamp. Select an existing checkpoint explicitly; the example
uses the final checkpoint from the 30,000-iteration run above. Both stages share
the `g1_jumping_track` experiment directory.

```bash
JUMP_TRACK_RUN="<timestamp>_stage1_track_pretrain_uan"

UAN_CHECKPOINT_PATH="$UAN_PT" \
uv run --locked --extra cu128 train Mjlab-JumpingHeight-UAN-Unitree-G1 \
  --env.commands.motion.motion-file "$JUMP_MOTION" \
  --env.scene.num-envs 16384 \
  --env.rewards.jump-com-ballistic-rise.weight 5.0 \
  --agent.resume True \
  --agent.load-run "^${JUMP_TRACK_RUN}$" \
  --agent.load-checkpoint 'model_29999[.]pt$' \
  --agent.max-iterations 30000 \
  --agent.logger wandb \
  --agent.run-name stage2_height_finetune_uan
```

Keep `UAN_PT` and `JUMP_MOTION` unchanged between stages. `JumpingHeight`
retains the six motion-tracking rewards and action-rate, joint-limit, and
self-collision penalties, and adds a one-shot ballistic center-of-mass rise
bonus. Its default weight is `5.0`; change the explicit weight argument to
tune it. Fine-tuning also replaces tracking-error terminations with root-height
and tilt limits, allowing the policy to jump beyond the reference motion.

For a no-UAN comparison, use `Mjlab-JumpingTrack-Baseline-Unitree-G1` followed
by `Mjlab-JumpingHeight-Baseline-Unitree-G1`, omit the `UAN_CHECKPOINT_PATH`
assignment, and resume the matching baseline pretraining run. Use
`stage1_track_pretrain_baseline` and `stage2_height_finetune_baseline` as the
corresponding run labels.

Inspect the fine-tuned policy in the training simulator before export:

```bash
JUMP_PT="logs/rsl_rl/g1_jumping_track/<fine_tuning_run>/model_<iter>.pt"

UAN_CHECKPOINT_PATH="$UAN_PT" \
uv run --locked --extra cu128 play Mjlab-JumpingHeight-UAN-Unitree-G1 \
  --checkpoint-file "$JUMP_PT" \
  --motion-file "$JUMP_MOTION" \
  --num-envs 1
```

Use the matching baseline task when inspecting a baseline checkpoint, or the
matching `JumpingTrack` task when inspecting a pretraining checkpoint.

## 8. Export And Deploy The Downstream Policy

Export and verify the final downstream checkpoint:

```bash
FORCE_PT="logs/rsl_rl/g1_force_control/<run>/model_<iter>.pt"

uv run --locked --extra cu128 python deploy/export_mjlab_checkpoint_to_jit.py \
  --policy-type force_control \
  --checkpoint "$FORCE_PT"

FORCE_JIT="${FORCE_PT%.pt}.jit"

uv run --locked --extra cu128 python deploy/verify_deploy_config.py \
  --policy "$FORCE_JIT" \
  --policy-joint-order mujoco
```

Both jumping stages use the `jumping_track` export format. Export the selected
checkpoint and verify its runtime configuration:

```bash
uv run --locked --extra cu128 python deploy/export_mjlab_checkpoint_to_jit.py \
  --policy-type jumping_track \
  --checkpoint "$JUMP_PT"

JUMP_JIT="${JUMP_PT%.pt}.jit"

uv run --locked --extra cu128 python deploy/verify_deploy_config.py \
  --policy "$JUMP_JIT" \
  --policy-joint-order mujoco
```

Keep the exported policy with its run's `params/env.yaml`. Check the exported
controller locally with the same reference clip:

```bash
uv run --locked --extra cu128 python deploy/deploy_jumping_track.py \
  --policy "$JUMP_JIT" \
  --motion-file "$JUMP_MOTION" \
  --interface mujoco \
  --render
```

This last command uses the local MuJoCo interface. The exported controller runs
the task policy alone; the UAN is used during policy training, not by this
controller.

See [Deployment](DEPLOYMENT.md) for the robot launch sequence and deploy script
details.
