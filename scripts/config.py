from environment import Behavior, MovementDim
from environment.config import CRW_CFG, EE_CFG, POI_CFG

example_crw = CRW_CFG(
    persistence = 0.9,
    turn_sigma = 0.15,
    target_speed = 10,
    speed_sigma = 0.03,
    speed_smooth = 0.2,
    bias_gain = 0.0
)

example_ee = EE_CFG(
    explore_cfg = CRW_CFG(
        persistence = 0.9,
        turn_sigma = 0.15,
        target_speed = 10,
        speed_sigma = 0.03,
        speed_smooth = 0.2,
        bias_gain = 0.0,
    ),
    exploit_cfg = CRW_CFG(
        persistence = 0.9,
        turn_sigma = 0.3,
        target_speed = 3,
        speed_sigma = 0.03,
        speed_smooth = 0.2,
        bias_gain = 0.0,
    ),
    time_to_leave = 10,
)

example_poi = POI_CFG(
    explore_cfg = CRW_CFG(
        persistence = 0.9,
        turn_sigma = 0.15,
        target_speed = 10,
        speed_sigma = 0.03,
        speed_smooth = 0.4,
        bias_gain = 0.3,
    ),
    exploit_cfg = CRW_CFG(
        persistence = 0.9,
        turn_sigma = 0.3,
        target_speed = 3,
        speed_sigma = 0.03,
        speed_smooth = 0.2,
        bias_gain = 0.0,
    ),
    time_to_leave = 10,
    arrive_dist = 10
)

cfg_train = {
    "dt": 0.1, # seconds
    "max_episode_steps": 512, # max steps per epsiode

    "model": {
        "path": "checkpoints",
        "mode": "train",
        "space": {
            "n_actions": 5,
            "features": 8,
        },
        "optimization": {
            "gamma": 0.997,
            "actor_lr": 0.0003,
            "critic_lr": 0.0005,
            "gae_lambda": 0.95,
            "policy_clip": 0.2,
            "val_loss_coef": 0.5,
            "entropy_coef": 0.01
        },
        "sampling": {
            "total_timesteps": 200*1024,
            "rollout_steps": 1024,
            "mini_batch_size": 128,
            "n_epochs": 8,
        }
    },
    "drone": {
        "small": {
            "count": 1,
            "view_range": 200,
            "disturbance_mult": 1,
            "min_speed": 0,
            "max_speed": 16,
            "ver_angle": 90,
            "hor_angle": 140,
            "max_cam_rot": 90,
            "spawn_dist": [30, 80],
            "view_dir": [1, 0, -0.7],
            "max_altitude": 150
        },
        "large": {
            "count": 1,
            "view_range": 400,
            "disturbance_mult": 1.25,
            "min_speed": 0,
            "max_speed": 20,
            "ver_angle": 90,
            "hor_angle": 140,
            "max_cam_rot": 90,
            "spawn_dist": [40, 100],
            "view_dir": [1, 0, -0.7],
            "max_altitude": 300
        }
    },
    "animal": {
        "env": {
            "count": 2, # animal count
        },
        "init": {
            "min_speed": 0, # min animal speed
            "max_speed": 14, # max animal speed
            "behavior": example_poi, # type of behavior
            "movement_dim": MovementDim.TWO_D, # 2d or 3d
            "max_spawn_radius": 500 # meters
        },
    },
    "resource": {
        "p_wavelenght": 200, # (meters) wavelenght of major resource noise
        "p_reduction": 0.2,  # Reduction on raw encounter probability
        "p_scale": 0.4,      # Scaling of reduced probability
        "sample_res": 5,     # (meters per sample) Sample resolution for poi generation
        "kernel_size": 250,  # (meters) kernel size for poi generation (local maxima)
        "min_poi_p": 1e-2,   # minimum value for a local maxima to be considered a poi
    },
}