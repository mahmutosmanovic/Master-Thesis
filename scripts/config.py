from environment import Behavior, MovementDim

cfg_train = {
    "dt": 0.05, # seconds
    "max_episode_steps": 256, # max steps per epsiode

    "model": {
        "path": "checkpoints",
        "mode": "train",
        "space": {
            "n_actions": 5,
            "features": 8,
        },
        "optimization": {
            "gamma": 0.995,
            "actor_lr": 0.0003,
            "critic_lr": 0.001,
            "gae_lambda": 0.95,
            "policy_clip": 0.2,
            "val_loss_coef": 0.5,
            "entropy_coef": 0.10
        },
        "sampling": {
            "total_timesteps": 50*1024,
            "rollout_steps": 1024,
            "mini_batch_size": 128,
            "n_epochs": 4,
        }
    },
    "drone": {
        "env": {
            "count": 1, # drone count
        },
        "init": {
            "min_speed": 0, # m/s
            "max_speed": 16, # m/s
            "ver_angle": 90, # frustum, vertical angle
            "hor_angle": 140, # frustum, horizontal angle
            "max_cam_rot": 90, # abs(deg)
            "view_range": 150, # meters
            "spawn_dist": [40,100], # euclidean spawn distance from animal
            "view_dir": [1,0,-0.7], # camera direction
            "max_altitude": 150
        },
    },
    "animal": {
        "env": {
            "count": 1, # animal count
        },
        "init": {
            "min_speed": 8, # min animal speed
            "max_speed": 12, # max animal speed
            "epsilon": 0.0, # how often dir change
            "ver_dir_angle": 0, # max animal vertical abs(deg) change
            "hor_dir_angle": 60, # max animal horizontal abs(deg) change
            "behavior": Behavior.RANDOM, # type of behavior
            "movement_dim": MovementDim.TWO_D, # 2d or 3d
            "max_spawn_radius": 200 # meters
        },
    },
}