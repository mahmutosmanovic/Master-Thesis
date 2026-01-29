import numpy as np
from ..agent import Agent
from dataclasses import dataclass
from utils.vec_utils import *
from environment.agents.drones.sensor import Sensor
from environment.settings import *

class Drone(Agent):
    def __init__(self, agent_id, pos, params, seed, mode=None):
        super().__init__(agent_id, pos, seed, mode)
        self.params = params
        self.pos = pos
        self.sensors = []
        self.view_dir = np.array([ # Used by camera, drone facing direction
            np.cos(self.params.camera_pitch),
            0.0,
            np.sin(self.params.camera_pitch),
        ])

    @property
    def state_obs_dim(self):
        return 9

    def get_state_obs(self) -> np.ndarray:
        return np.concatenate([
            self.pos / POS_NORM,
            (self.direction * self.speed) / self.params.max_speed,
            self.view_dir
        ]).astype(np.float32)

    @property
    def obs_dim(self) -> int:
        return self.state_obs_dim + sum(s.obs_dim for s in self.sensors)

    def get_obs(self, world) -> np.ndarray:
        parts = [] #[self.get_state_obs()]
        for s in self.sensors:  # list order is deterministic
            parts.append(s.get_obs(self, world).astype(np.float32))
        obs = np.concatenate(parts, axis=0).astype(np.float32)

        # optional sanity check in debug mode
        # assert obs.shape[0] == self.obs_dim

        return obs
    
    def add_sensor(self, sensor):
        self.sensors.append(sensor)

    def sense(self, world):
        observations = {}
        for sensor in self.sensors:
            observations[sensor.__class__.__name__] = sensor.sense(self, world)
        return observations
    
    def update(self, action, dt):
        direction, speed, view_yaw_rate = action
        self.apply_control(direction, speed, dt)
        self.apply_view_yaw(view_yaw_rate, dt)
        self.move(dt)
    
    def apply_view_yaw(self, view_yaw_rate, dt):
        # Clamp yaw rate
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
        max_view_yaw=4.0,
        max_accel=4.0,
        camera_pitch=np.deg2rad(-45)
    )