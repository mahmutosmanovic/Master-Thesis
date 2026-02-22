from environment import Behavior, MovementDim

cfg_train = {
    "dt": 0.01, # seconds
    "max_episode_steps": 1024, # max steps per epsiode

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
            "total_timesteps": 100*2048,
            "rollout_steps": 2048,
            "mini_batch_size": 256,
            "n_epochs": 10,
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
            "spawn_dist": [40, 80],
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
            "spawn_dist": [80, 160],
            "view_dir": [1, 0, -0.7],
            "max_altitude": 300
        }
    },
    "animal": {
        "env": {
            "count": 1, # animal count
        },
        "init": {
            "min_speed": 0, # min animal speed
            "max_speed": 14, # max animal speed
            "epsilon": 0.1, # how often dir change
            "ver_dir_angle": 0, # max animal vertical abs(deg) change
            "hor_dir_angle": 60, # max animal horizontal abs(deg) change
            "behavior": Behavior.RANDOM, # type of behavior
            "movement_dim": MovementDim.TWO_D, # 2d or 3d
            "max_spawn_radius": 200 # meters
        },
    },
}