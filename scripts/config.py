from environment import Behavior, MovementDim

cfg_train = {
    "dt": 0.5, # seconds,

    "model": {
        "path": "checkpoints",
        "mode": "train",
        "optimization": {
            "gamma": 0.99,
            "value_lr": 1e-4,
            "policy_lr": 3e-4,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        },
        "sampling": {
            "total_timesteps": 1,
            "rollout_steps": 200,
            "mini_batch_size": 100,
            "epochs_per_rollout": 5,
        }
    },

    "drone": {
        "env": {
            "count": 1, # drone count
        },
        "init": {
            "min_speed": 0, # m/s2
            "max_speed": 8, # m/s2
            "ver_angle": 90, # frustum, vertical angle
            "hor_angle": 140, # frustum, horizontal angle
            "max_cam_rot": 90, # abs(deg)
            "view_range": 400, # meters
            "spawn_dist": [40,100], # euclidean spawn distance from animal
            "view_dir": [1,0,-0.7], # camera direction
        },
    },
    "animal": {
        "env": {
            "count": 5, # animal count
        },
        "init": {
            "min_speed": 1, # min animal speed
            "max_speed": 8, # max animal speed
            "epsilon": 0.0, # how often dir change
            "ver_dir_angle": 0, # max animal vertical abs(deg) change
            "hor_dir_angle": 60, # max animal vertical abs(deg) change
            "behavior": Behavior.RANDOM, # type of behavior
            "movement_dim": MovementDim.TWO_D, # 2d or 3d
            "max_spawn_radius": 200 # meters
        },
    },
}