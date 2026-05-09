# Stealth-Fleet

Stealth-Fleet is a reinforcement learning project for training drone policies that monitor animals while minimizing disturbance. The repository combines a custom multi-agent simulation environment, PPO/MAPPO training code, a handcrafted centroid baseline, evaluation and plotting utilities, and preprocessing for real GPS track replay.

The core idea is to learn surveillance behavior that keeps animals visible and well-framed without pushing them into avoidance or escape behavior. Reward shaping in the environment balances monitoring quality against disturbance and control effort.

## What the project includes

- A custom drone-animal simulation environment in `environment/`
- Two RL algorithms in `model/`:
  - PPO with per-drone observations
  - MAPPO with centralized critic / decentralized actors
- A handcrafted centroid standoff baseline in `scripts/centroid.py`
- Evaluation, reward plots, and policy heatmaps in `scripts/eval_models.py` and `scripts/plots/`
- Config-driven experiment setup through YAML files in `config/`
- GPS preprocessing utilities for replayable animal tracks in `scripts/prepare_tracks.py`

## Environment summary

The environment simulates one or more drones observing one or more animals in 2D/3D space. Each drone controls:

- velocity direction
- speed
- camera rotation

Observations are built per drone and include:

- drone view direction
- normalized altitude
- per-animal visibility
- normalized distance
- vertical and horizontal viewing angles
- animal motion in the drone camera frame

Animals respond to drone-induced disturbance. Depending on disturbance magnitude, they can remain calm, switch to avoidance, or flee. Episodes terminate when:

- the maximum episode length is reached, or
- too few animals remain visible, or
- a replay track segment completes

## Reward design

The reward encourages useful monitoring while penalizing disruptive behavior.

Positive terms include:

- keeping animals visible
- maintaining good viewing distance
- aligning the camera with the target

Penalties include:

- disturbing the animals
- excessive drone speed
- excessive camera rotation
- losing visibility entirely

The environment also tracks per-episode behavior statistics such as calm, avoidance, and flee fractions.

## Algorithms

Training is implemented in `scripts/train_agent.py`.

- `ppo`: independent per-drone policy/value updates using local observations
- `mappo`: shared actor with centralized critic over the joint observation

Both implementations use:

- squashed Gaussian continuous actions
- GAE
- clipped PPO objective
- entropy regularization
- checkpoint saving for `best` and `last`

## Config system

Experiment settings live in `config/` and are loaded by `config/loader.py`.

Configs define:

- episode length and timestep
- drone counts and sensing limits
- animal counts and motion behavior
- reward scales
- disturbance model parameters
- PPO/MAPPO optimization hyperparameters
- action space type (`rel` or `abs`)

Representative configs in the repo include:

- `CRW` / `CRWLSTD`
- `POI`, `LPOI`, `EE`
- `PDPS*`
- `PARETO_*`
- `REPLAY`

## Repository layout

```text
config/           Experiment YAMLs and config loader
environment/      Simulation environment, entities, disturbance, viewer, vector math
model/            PPO and MAPPO implementations
scripts/          Training, evaluation, playback, baseline, preprocessing, plotting
data/             Raw movement/GPS data
track_segments/   Preprocessed replay segments
runs/             Training outputs and checkpoints
evals/            Evaluation outputs, CSV logs, plots
figures/          Generated figures
recordings/       Rendered outputs / saved media
notebooks/        Analysis notebooks
tests/            Small test and render checks
```

## Setup

You mentioned you use an Ubuntu terminal, so the examples below assume that workflow.

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want Weights & Biases logging, you will also need `wandb` available in the environment:

```bash
pip install wandb
```

The training script expects the following environment variables when `--wandb` is enabled:

- `API_TOKEN`
- `WANDB_ENTITY`
- `WANDB_PROJECT`

These can be placed in `.env`.

## Training

Train a PPO agent:

```bash
python3 -m scripts.train_agent --config CRW --agent ppo --seed 42
```

Train a MAPPO agent:

```bash
python3 -m scripts.train_agent --config CRW --agent mappo --seed 42
```

Train with Weights & Biases logging:

```bash
python3 -m scripts.train_agent --config CRW --agent ppo --seed 42 --wandb
```

Training creates a timestamped run directory under `runs/` and stores:

- the resolved `config.yaml`
- saved checkpoints
- metadata needed for later playback and evaluation

## Playback

Replay a trained model with rendering:

```bash
python3 -m scripts.play --run latest --seed 99 --weights best
```

Or target a specific run:

```bash
python3 -m scripts.play --run CRW_seed42_YYYY-MM-DD_HH-MM-SS --weights last
```

## Baseline controller

The project includes a classical non-learning baseline that tracks the visible animal centroid while trying to maintain a useful standoff distance.

Run a single baseline episode:

```bash
python3 -m scripts.centroid --mode run --config CRW --seed 42
```

Run a parameter grid search:

```bash
python3 -m scripts.centroid --mode grid --config CRW --eval-seeds 10 --seed 42 --render-best
```

## Evaluation

Evaluate a trained run and compare it against the centroid baseline:

```bash
python3 -m scripts.eval_models \
  --run latest \
  --baseline centroid \
  --num-episodes 100 \
  --start-seed 42 \
  --weights last \
  --plot-rewards \
  --plot-heatmaps
```

Evaluation writes a new directory under `evals/` containing:

- step-level CSV logs
- episode-level CSV summaries
- reward distribution plots
- radial / XY policy heatmaps
- a copied config snapshot

## Real track preprocessing and replay

`scripts/prepare_tracks.py` preprocesses raw GPS movement data into replayable track segments. The current script contains pipelines for:

- pigeons
- jackals
- spur-winged lapwings

The preprocessing pipeline:

1. loads raw GPS files
2. converts latitude/longitude to UTM coordinates
3. computes time deltas
4. splits trajectories into segments using time and motion gaps
5. filters short or low-extent segments
6. saves compressed `.npz` segment files plus a manifest

Run it with:

```bash
python3 -m scripts.prepare_tracks
```

To use replay behavior, switch to the `REPLAY` config and point it at prepared track segments.

## Convenience scripts

The repository also includes batch helpers for running repeated experiments:

- `run.sh` for selected training + evaluation runs
- `run_eval.sh` for evaluating a list of stored runs
- `run_pareto.sh`
- `run_dist_ablation.sh`
- `run_sensitivity_speed.sh`

These are useful once the project environment is already set up.

## Typical workflow

1. Pick a YAML config in `config/`
2. Train with `scripts.train_agent`
3. Inspect the produced run in `runs/`
4. Replay the policy with `scripts.play`
5. Benchmark against the centroid baseline with `scripts.eval_models`
6. Inspect figures and CSV outputs in `evals/` and `figures/`

## Current notes

- The codebase is centered on research experiments rather than packaged library usage
- Some folders such as `pareto/`, `reward_ablation/`, and `speed_sensitivity/` store experiment artifacts and analysis outputs
- The `tests/` folder currently looks lightweight, so most validation appears to happen through scripted training, playback, and evaluation runs

## License

This repository includes a `LICENSE` file at the project root.
