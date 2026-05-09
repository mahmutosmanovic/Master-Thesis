## Training Instructions
Example:

`config`: specifies config yaml file name in ./config folder

`seed`: specifies seed to run training run with

`wandb` activates wandb logging

`ppo` choices to train with "ppo" or "mappo"

    python -m scripts.train_agent --config CRW --agent ppo --seed 42 --wandb

## Inference Instructions
Example:

`run`: expects a folder name within ./runs folder, which in turn contains *actor* and *critic* weights in addition to model configurations

`latest`: runs the latest locally trained model in ./runs folder

`weights`: "best" or "last" to select trained model of run with highest obtained reward or last obtained reward

    python -m scripts.play --run train_seed42_2026-02-24_09-19-44

OR

    python -m scripts.play --run latest

## Baseline Instructions

> The baseline implements a handcrafted Centroid Standoff controller, which tracks the centroid of visible animals while maintaining a desired distance and altitude. It serves as a non-learning reference policy for comparison against trained RL agents.

**Run a Single Baseline Episode**
- Runs one episode using fixed controller parameters and renders the environment.

`config`: specifies config yaml file name in ./config folder

`seed`: environment seed for reproducibility

    python -m scripts.centroid --mode run --config train --seed 42

This will:

1. load the environment configuration

2. initialize the centroid controller

3. run a single episode

4. render drone behaviour

5. print normalized reward and behavior statistics

**Grid Search (Parameter Tuning)**
- Performs a grid search over predefined controller parameters to find a strong classical baseline.

`config`: config yaml file used for evaluation

`seed`: base seed (used to generate evaluation seeds)

`eval-seeds`: number of seeds averaged per parameter set

`render-best`: optionally render the best-performing controller after search

    python -m scripts.centroid --mode grid --config CRW --eval-seeds 10 --seed 42 --render-best

This will:

1. evaluate all parameter combinations

2. run multiple episodes per configuration

3. compute mean and standard deviation of rewards

4. report the best-performing parameter set

5. After optimization finishes, the environment will open in render mode and replay the best controller.

> Variables:

| Parameter             | Explanation                  |
| --------------------- | ------------------------------------------ |
| target_range_ratio    | desired equilibrium distance from target   |
| target_altitude_ratio | preferred operating height setpoint        |
| xy_gain               | horizontal position correction strength    |
| z_gain                | vertical position correction strength      |
| theta_gain            | heading alignment sensitivity              |
| search_theta          | scan rate during target loss               |
| min_speed_norm        | baseline motion bias (anti-stall velocity) |

# Evaluate Models Instructions

`run`: expects a folder name within ./runs folder, which in turn contains *actor* and *critic* weights in addition to model configurations

`baseline`: specifies the baseline to benchmark against, default is None

`num-episodes`: specicies over how many episode to run baseline and RL model

`start_seed`: specifies which the starting seed is, all ran episodes are seeded

`plot-rewards`: saves a plot with mean and std of the RL agent and baseline if both are present

`plot-heatmaps`: Plots radial occupancy of drones for included methods

`weights`: "best" or "last" to select trained model of run with highest obtained reward or last obtained reward

    python -m scripts.eval_models --run latest --baseline centroid --num-episodes 100 --start-seed 99 --plot-rewards --plot-heatmaps --weights last

    python -m scripts.eval_models \
    --run "CRW_seed42_2026-03-23_20-39-50" \
    --checkpoint-prefix ppo_ \
    --checkpoint-episodes \
    --num-episodes 100 \
    --start-seed 42 \
    --append-summary-csv table/all_model_episode_rewards.csv

    python -m scripts.eval_models \
    --run "CRW_seed42_2026-03-23_20-39-50" \
    --checkpoint-prefix ppo_ \
    --checkpoint-episodes \
    --num-episodes 1 \
    --start-seed 42 \
    --append-summary-csv table/all_model_episode_rewards.csv
    
## Simulate Real GPS Behavior
> GPS DATA PRE-PROCESSING

Do not forget to set REPLAY in config

    python -m scripts.prepare_tracks

runs/LPOI_seed42_2026-03-23_13-34-52