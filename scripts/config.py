from environment import Behavior, MovementDim

cfg_train = {
    "dt": 0.1, # seconds,

    "model": {
        "path": "checkpoints",
        "mode": "train",
        "space": {
            "n_actions": 5,
            "features": 4,
        },
        "optimization": {
            "gamma": 0.99,
            "lr": 0.0003,
            "gae_lambda": 0.95,
            "policy_clip": 0.2,
            "val_loss_coef": 0.5,
            "entropy_coef": 0.01
        },
        "sampling": {
            "total_timesteps": 50*2048,
            "rollout_steps": 2048,
            "mini_batch_size": 64,
            "n_epochs": 10,
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
            "count": 1, # animal count
        },
        "init": {
            "min_speed": 4, # min animal speed
            "max_speed": 6, # max animal speed
            "epsilon": 0.10, # how often dir change
            "ver_dir_angle": 0, # max animal vertical abs(deg) change
            "hor_dir_angle": 60, # max animal horizontal abs(deg) change
            "behavior": Behavior.RANDOM, # type of behavior
            "movement_dim": MovementDim.TWO_D, # 2d or 3d
            "max_spawn_radius": 200 # meters
        },
    },
}