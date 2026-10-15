# Deployment

Deployment scripts live under `deploy/`.

All deploy scripts expect MuJoCo joint order for policy observations and
actions:

- `deploy/deploy_wtw.py`
- `deploy/deploy_velocity.py`
- `deploy/deploy_force_control.py`
- `deploy/deploy_jumping_track.py`

Each script requires `--policy "<your_policy.jit>"`; checkpoints are not included.

## 0. Prepare Dependencies Offline

Run the setup, export, and verification commands from the repository root on
a machine isolated from robot control. Use the same repository revision on
the training and controller machines.

| Machine | Required setup |
| --- | --- |
| Training/export | The [project Python environment](../README.md) and training dependencies. Export and config verification do not require LCM or a robot connection. |
| Python controller | The project environment plus the `deploy` extra below. Policies run on CPU; CUDA is not required for controller inference. |
| C++ bridge | Linux `x86_64` or `aarch64`, a C++17 compiler, CMake, and LCM development files. The vendored SDK supplies Unitree and CycloneDDS libraries for those two architectures. |

The Python controller and bridge can share a machine. If they run on separate
machines, their LCM transport and network route must be configured for that
deployment; the bridge's DDS interface argument does not configure LCM.

### Python controller environment

For a controller-only machine, install CPU PyTorch and the pinned Python LCM
binding:

```bash
uv sync --locked --extra cpu --extra deploy
```

The generated message bindings are already vendored. Make their parent
directory importable in each controller terminal, then perform an import-only
check; this does not construct an LCM connection or robot interface:

```bash
export PYTHONPATH="$PWD/third-party/openhomie/HomieDeploy/unitree_sdk2${PYTHONPATH:+:$PYTHONPATH}"
uv run --locked --extra cpu --extra deploy python -c \
  "import lcm; from deploy.interfaces.lcm_interface import G1LCMInterface; print('LCM imports OK')"
```

Use `--extra deploy` for subsequent LCM controller commands so `uv` retains the
optional dependency.

### Build the C++ bridge

Build on an isolated Linux machine matching the bridge host's architecture.
The vendored [SDK README](../third-party/openhomie/HomieDeploy/unitree_sdk2/README.md)
records Ubuntu 20.04 and GCC 9.4 as its prebuilt-library environment. On Ubuntu,
install the compiler and [LCM development package](https://packages.ubuntu.com/noble/liblcm-dev):

```bash
sudo apt-get update
sudo apt-get install build-essential cmake liblcm-dev

cmake -S third-party/openhomie/HomieDeploy/unitree_sdk2 \
  -B third-party/openhomie/HomieDeploy/unitree_sdk2/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build third-party/openhomie/HomieDeploy/unitree_sdk2/build \
  --target g1_control --parallel
```

This builds only `build/bin/g1_control`; it does not start it. The Python LCM
wheel and system LCM development package serve different builds: install the
former on the Python controller machine and the latter on the bridge build
machine. Keep the vendored SDK/DDS libraries with the bridge checkout.

## 1. Export A Checkpoint

After installing the environment in Section 0, export MJLab `.pt` actor
checkpoints to deploy-ready TorchScript. `--no-sync` retains the CPU or CUDA
dependencies already selected for that machine:

```bash
uv run --no-sync python deploy/export_mjlab_checkpoint_to_jit.py \
  --policy-type wtw \
  --checkpoint "logs/rsl_rl/g1_wtw/<run>/model_<iter>.pt"

uv run --no-sync python deploy/export_mjlab_checkpoint_to_jit.py \
  --policy-type velocity \
  --checkpoint "logs/rsl_rl/g1_velocity_custom/<run>/model_<iter>.pt"

uv run --no-sync python deploy/export_mjlab_checkpoint_to_jit.py \
  --policy-type force_control \
  --checkpoint "logs/rsl_rl/g1_force_control/<run>/model_<iter>.pt"

uv run --no-sync python deploy/export_mjlab_checkpoint_to_jit.py \
  --policy-type jumping_track \
  --checkpoint "logs/rsl_rl/g1_jumping_track/<run>/model_<iter>.pt"
```

The default export target is MuJoCo order for every policy type used by the
deployment scripts.

When transferring a policy to the controller machine, retain this layout:

```text
<run>/
  model_<iter>.jit
  model_<iter>.pt
  params/env.yaml
```

The sibling `.pt` supplies checkpoint metadata needed to resolve command
curricula. Keep `env.yaml` from the same training run.

The `jumping_track` format covers both reference-tracking and height-fine-tuned
policies with the 161-dimensional actor observation. Keep the policy with its
run's `params/env.yaml` and use the same reference motion used for training.
The [sim-to-real guide](SIM2REAL.md#jumping-tracking-pretraining-and-height-fine-tuning)
documents both training stages.

## 2. Verify Runtime Config

Verify that the exported policy and its sibling `params/env.yaml` resolve to a
deploy-compatible runtime config:

```bash
uv run --no-sync python deploy/verify_deploy_config.py \
  --policy "<policy.jit>" \
  --policy-joint-order mujoco
```

If you only have `env.yaml`, pass it directly:

```bash
uv run --no-sync python deploy/verify_deploy_config.py \
  --env-yaml "logs/rsl_rl/<experiment>/<run>/params/env.yaml" \
  --policy-common-step-counter "<checkpoint_common_step_counter>" \
  --policy-joint-order mujoco
```

For a local jumping controller check, provide an exported policy explicitly:

```bash
uv run --no-sync python deploy/deploy_jumping_track.py \
  --policy "<jumping_policy.jit>" \
  --motion-file "motions/02_04_jump_balance_pad50.npz" \
  --interface mujoco \
  --render
```

Use your training clip if it differs from the padded reference shown here.

### Torque-speed clipping

Running/jumping training and the shared deployment controller use asymmetric
torque-speed bounds. With `r = joint_velocity / no_load_speed`,
`tau_min = -stall_torque * (1 + clamp(r, -1, 0))` and
`tau_max = stall_torque * (1 - clamp(r, 0, 1))`. Torque in the direction of motion
decreases to zero at the no-load speed, while full braking torque remains
available. Beyond that speed, zero torque remains allowed; braking is not forced.
Both sides of the position-target interval are clipped so the nominal PD torque
lies in `[tau_min, tau_max]` at each policy step. Deployment reads motor-limit
overrides and PD gains from the policy's `env.yaml`. During UAN-augmented training, the residual
is added after clipping the nominal target; the UAN is not exported to hardware.

## 3. Start The Robot Control Bridge

**USER-RUN ONLY — MAY ACTUATE THE PHYSICAL ROBOT**

Before using Sections 3–5, establish the robot's physical setup, operating mode,
and approved physical stop procedure. Identify the bridge and controller hosts,
robot DDS interface, and LCM route, and replace the command placeholders with
the values for your setup.

Starting the vendored bridge enables joint commands and moves all 29 joints
toward zero over approximately three seconds after robot state arrives,
before a Python policy is running. It then uses compiled-in `Kp`/`Kd` values
from `g1_control.cpp`. The Python LCM interface checks the policy's resolved
gains against those source constants before connecting and rejects unsupported
gain overrides. Rebuild the bridge from the same checkout so its compiled gains
match the constants checked by Python.

Use the robot's approved physical stop procedure. R2, Y, and Ctrl-C are not
substitutes: the Python method named
`damping_mode()` sends zero position targets, while this bridge retains its
nonzero gains; Y exits the bridge. The bridge has no command timeout.

For hardware deployment, start the OpenHomie Unitree bridge on the
bridge host before launching the Python controller, using its robot-facing
DDS network interface:

```bash
./third-party/openhomie/HomieDeploy/unitree_sdk2/build/bin/g1_control "<robot-dds-interface>"
```

## 4. Launch The Python Controller

**USER-RUN ONLY — MAY ACTUATE THE PHYSICAL ROBOT**

After the Section 3 prerequisites are satisfied, open a second terminal on
the Python controller machine at the repository root and set `PYTHONPATH` as
shown in Section 0. Initializing the LCM interface itself sends zero joint
targets; later pose initialization and policy execution command robot motion.
Use an explicit policy path and the same approved physical stop procedure.

WTW:

```bash
uv run --locked --extra cpu --extra deploy python deploy/deploy_wtw.py \
  --policy "<wtw_policy.jit>" \
  --interface lcm
```

Velocity:

```bash
uv run --locked --extra cpu --extra deploy python deploy/deploy_velocity.py \
  --policy "<velocity_policy.jit>" \
  --interface lcm
```

Force control:

```bash
uv run --locked --extra cpu --extra deploy python deploy/deploy_force_control.py \
  --policy "<force_control_policy.jit>" \
  --interface lcm
```

Use `--interface mujoco --render` for local simulation smoke tests instead of
hardware.

## 5. Collect Hardware Data

**USER-RUN ONLY — MAY ACTUATE THE PHYSICAL ROBOT**

Use this in place of the WTW launch in Section 4, after completing its setup
and the Section 3 operator prerequisites. It runs the WTW policy and records
pipeline-format CSV shards; `--collect` does not make execution passive.
Supply an explicit output directory:

```bash
uv run --locked --extra cpu --extra deploy python deploy/deploy_wtw.py \
  --policy "<wtw_policy.jit>" \
  --interface lcm \
  --collect \
  --output_dir "<hardware_collection_root>"
```

`DataCollector` writes rows as:

```text
state_96 | action_29 | extension_15 | contact_2 | termination_1
```

The state and action joint blocks are MuJoCo order. The action block is the
final clipped absolute joint-position target sent to the low-level controller.
The row timing is always:

```text
state[i], action[i] -> state[i + 1]
```
