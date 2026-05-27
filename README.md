# MSc. Low-Disturbance Wildlife Monitoring with Drones

Low-Disturbance Wildlife Monitoring with Drones is a reinforcement learning project for training drone policies that monitor animals while minimizing disturbance, the project was conducted at Jönköping School of Engineering as part of a MSc. thesis. The repository combines a custom multi-agent simulation environment, PPO/MAPPO training code, a handcrafted centroid baseline, evaluation and plotting utilities, and preprocessing for real GPS track replay.

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

The reward encourages monitoring animals from useful viewpoints while penalizing disruptive or inefficient drone behaviour. The main monitoring reward combines animal disturbance and target-viewing quality through a trade-off controlled by $\alpha$.

Positive terms include:

- keeping animals visible in the camera view
- keeping the assigned target visible
- maintaining a useful viewing distance to the target
- aligning the camera with the target
- completing the episode without losing track

Penalties include:

- animal disturbance, included through the monitoring trade-off
- excessive drone speed
- excessive camera rotation
- abrupt changes in flight direction
- losing sight of the assigned target for repeated steps

Multi-agent terms:

- bonus when all drones keep their assigned targets visible
- reward for separated viewpoints around the same target
- penalty when animals enter avoid or flee states
- penalty for drones flying too close to each other
- penalty for drones flying too close to animals
- hard safety penalty and termination for severe proximity violations

The environment also tracks per-episode behavior statistics such as calm, avoidance, and flee fractions.

## Algorithms

Training is implemented in `scripts/train_agent.py`.

- `ppo`: independent per-drone policy/value updates using local observations
- `mappo`: shared actor with centralized critic over the joint observation
- `sac`: centralized continuous-control actor with twin soft Q-critics over the joint drone observation/action space
- `dqn`: single-drone branching dueling Q-network with discrete direction, speed, and theta action branches

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
bash/             Bash batch run and conveniece scripts
config/           Experiment YAMLs and config loader
environment/      Simulation environment, entities, disturbance, viewer, vector math
model/            PPO, DQN, SAC and MAPPO implementations
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

Train a SAC agent:

```bash
python3 -m scripts.train_agent --config CRW --agent sac --seed 42
```

Train a DQN agent:

```bash
python3 -m scripts.train_agent --config CRW --agent dqn --seed 42
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

### `replay_test.sh`
Runs replay-based evaluations for runs listed in a manifest.

```bash
bash bash/replay_test.sh table/runs_manifest.csv
```

Uses `MAX_JOBS`, `RUNS_DIR`, and `EVALS_DIR` as optional environment variables.

### `replot.sh`
Regenerates policy heatmaps from a fixed runs manifest and copies them to `imgtemp/`.

```bash
bash bash/replot.sh
```

### `replot_transfer.sh`
Regenerates transfer-evaluation policy heatmaps and copies them to `imgtemp_transfer/`.

```bash
bash bash/replot_transfer.sh
```

### `restore_backups.sh`
Restores backed-up `config.yaml` files after animal-evaluation edits.

```bash
bash bash/restore_backups.sh
```

Optionally specify a runs directory:

```bash
bash bash/restore_backups.sh <run dir>
```

### `run_animal_table.sh`
Runs animal-transfer evaluations for runs listed in a manifest.

```bash
bash bash/run_animal_table.sh <bash run manifest>
```

Uses `MAX_JOBS`, `RUNS_DIR`, and `EVALS_DIR` as optional environment variables.

### `run_fit.sh`
Fits movement-behaviour models for the animal trajectory datasets.

```bash
bash bash/run_fit.sh
```

### `run_morf.sh`
Trains and evaluates the MORF/MAPPO configuration set.

```bash
bash bash/run_morf.sh
```

### `run_one.sh`
Trains, evaluates, and plays a single MAPPO configuration.

```bash
bash bash/run_one.sh
```

### `run_table.sh`
Runs training and evaluation jobs and writes a manifest to `table/`.

```bash
bash bash/run_table.sh
```

### `run_wind.sh`
Trains and evaluates the wind-condition experiment configurations.

```bash
bash bash/run_wind.sh
```

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

## Data Availability
For purposes of reproducability, users can access part of the data which has already been made public by [Orr Spiegel](https://scholar.google.com/citations?user=1lEq3csAAAAJ&hl=en): https://datarepository.movebank.org/entities/datapackage/a2e2f6f1-e0b4-476f-b108-2208f4b29baa
The data which has been made public is for the Spur-Winged Lapwings. Folder: _ATLAS tracking of spur-winged lapwings-1of6.csv.zip_ was utilized, containing about 21.5M datapoints. After filteration, where we consider distance traveled, steps traveled, etc, we end up with approximately 400k points. A similar processing procedure was conducted with the pigeon and jackal data.

With regards to ethical collection of the data, all trapping and tagging procedures were authorized by permits 2020/42481, 2021/42733, and 2022/42989 from the Israel Nature and Parks Authority, and the researchers state that no birds were harmed during the tagging process and that all individuals remained active in their respective home ranges after release. For the full public statement of the authors, please see the following article, https://royalsocietypublishing.org/rspb/article/292/2038/20242471/104772/Spur-winged-lapwings-show-spatial-behavioural.

## License

This repository includes a `LICENSE` file at the project root.
