import numpy as np
from ..agent import Agent
from dataclasses import dataclass
from utils.vec_utils import *
from environment.agents.drones.sensor import Sensor

class Drone(Agent):
    def __init__(self, agent_id, pos, params, seed, mode=None, yaw=0.0, pos_scale=np.array([1.0, 1.0, 1.0])):
        super().__init__(agent_id, pos, seed, mode)
        self.params = params
        self.pos = pos
        self.pos_scale = pos_scale
        self.sensors = []
        self.view_dir = np.array([ # Used by camera, drone facing direction
            np.cos(self.params.camera_pitch) * np.cos(yaw),
            np.cos(self.params.camera_pitch) * np.sin(yaw),
            np.sin(self.params.camera_pitch),
        ])

    @property
    def obs_dim(self) -> int:
        return 1 + sum(s.obs_dim for s in self.sensors)

    def observe(self, animals) -> np.ndarray:
        # full_state = np.concatenate([
        #     self.pos / self.pos_scale,
        #     (self.direction * self.speed) / self.params.max_speed,
        #     self.view_dir
        # ]).astype(np.float32)

        parts = [self.view_dir]
        for sensor in self.sensors:  # list order is deterministic
            parts.append(sensor.observe(self, animals).astype(np.float32))
        obs = np.concatenate(parts, axis=0).astype(np.float32)
        return obs
    
    def reward(self, animals):
        reward = 0
        for sensor in self.sensors:
            reward += sensor.reward(self, animals)
        return reward
    
    def add_sensor(self, sensor):
        self.sensors.append(sensor)
    
    def update(self, action, dt):
        direction, speed, view_yaw_rate = action
        self.apply_control(direction, speed, dt)
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

@dataclass
class DroneParams:
   # metadata
   name: str

   # geometry
   is_planar: bool

   # movement
   max_speed: float
   max_turn: float
   max_view_yaw: float
   max_accel: float

   camera_pitch: float

def drone_params():
    return DroneParams(
        name="drone",
        is_planar=False,
        max_speed=12.0,
        max_turn=4.0,
        max_view_yaw=2.0,
        max_accel=4.0,
        camera_pitch=np.deg2rad(-30)
    )