import numpy as np
from .agent import Agent
from dataclasses import dataclass
from utils.vec_utils import *
from environment.agents.sensor import Sensor

class Drone(Agent):
    def __init__(self, agent_id, pos, direction, params, sensor, seed, yaw=0.0, pos_scale=np.array([1.0, 1.0, 1.0])):
        super().__init__(agent_id, pos, direction, seed)
        self.params = params
        self.pos = pos
        self.pos_scale = pos_scale
        self.sensor = sensor
        rad_camera_pitch = np.deg2rad(self.params.camera_pitch)
        self.view_dir = np.array([ # Used by camera, drone facing direction
            np.cos(rad_camera_pitch) * np.cos(yaw),
            np.cos(rad_camera_pitch) * np.sin(yaw),
            np.sin(rad_camera_pitch),
        ])

    @property
    def obs_dim(self) -> int:
        return 6 + self.sensor.obs_dim()

    def observe(self, animals) -> np.ndarray:
        self_obs = np.concatenate([
            # self.pos[[2]] / self.pos_scale[[2]], #Just altitude
            # self.direction * self.norm_speed,
            self.view_dir
        ]).astype(np.float32)

        return self_obs, self.sensor.observe(self, animals)
    
    def update(self, action, dt):
        direction, norm_speed, view_yaw_rate = action
        self.apply_control(direction, norm_speed, dt)
        self.apply_view_yaw(view_yaw_rate, dt)
        self.move(dt)
    
    def apply_view_yaw(self, view_yaw_rate, dt):
        # Clamp yaw rate
        view_yaw_rate *= self.params.max_view_yaw
        view_yaw_rate = np.clip(
            view_yaw_rate,
            -self.params.max_view_yaw,
            self.params.max_view_yaw,
        )

        yaw = view_yaw_rate * dt
        if abs(yaw) < 1e-8:
            return

        c = np.cos(yaw)
        s = np.sin(yaw)

        # Rotation about Z axis
        Rz = np.array([
            [ c, -s, 0.0],
            [ s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ])

        self.view_dir = unit(Rz @ self.view_dir)
    
    def to_dict(self):
        view_x, view_y, view_z = self.view_dir
        return{
            **super().to_dict(),
            "view_x": view_x,
            "view_y": view_y,
            "view_z": view_z,
        }