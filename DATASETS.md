# Dataset Format

The [hardware guide](docs/SIM2REAL.md) produces the datasets and checkpoints
below from your own G1 data. Store generated files under ignored `data/` or
`logs/` directories, or external storage, and pass their paths explicitly.

## Replay Row Schema

Replay rows use fixed pre-action timing:

```text
state_96 | action_29 | extension_15 | contact_2 | termination_1
```

where row `i` stores the proprioceptive state at time `i` and the absolute
MuJoCo-order joint-position target issued from that state. The replay target is
the next state, row `i + 1`.

The deploy `DataCollector` records the final clipped absolute joint targets sent
to the controller in MuJoCo order. Recorded actions can be consumed directly by
the pipeline.

## Files Passed Between Stages

| File or directory | Produced by | Used for |
| --- | --- | --- |
| WTW checkpoint and exported policy | WTW training and export | Hardware collection and simulated estimator data |
| Hardware collection directory | WTW deployment collector | Relabeling |
| `estimator_training_data.pt` and its `.npy` arrays | Simulated WTW collection | Training the six separate estimators with three-frame histories |
| Estimator root containing six component directories and `paper_estimators_meta.json` | Estimator training | Validated base velocity, foot contact, and stance-foot geometry labeling |
| Relabeled `processed/` directory | Hardware relabeling | UAN replay training |
| UAN checkpoint | UAN training | Frozen actuator model for downstream training |
| Downstream checkpoint and `params/env.yaml` | Running, force-control, or jumping training | Export and deployment |

Jumping also uses the included `motions/02_04_jump_balance_pad50.npz` reference.
Keep the same clip, including padding, for pretraining, fine-tuning, and
evaluation. Keep exported policies with their run's `params/env.yaml`.
