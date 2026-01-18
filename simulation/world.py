import os
import csv
import numpy as np
from random import uniform

from simulation.settings import *
from simulation.agents.animals import Eagle, Jackal, Pigeon
from simulation.paths import CirclePath


class World:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = []
        self.log = []
        self.t = 0.0

    # Spawning
    def spawn(self):
        path = self._create_path_if_needed()

        self._spawn_species(Jackal, JACKAL_COUNT, JACKAL_MODE, path)
        self._spawn_species(Eagle,  EAGLE_COUNT,  EAGLE_MODE,  path)
        self._spawn_species(Pigeon, PIGEON_COUNT, PIGEON_MODE, path)

    def _create_path_if_needed(self):
        if not self._any_path_following():
            return None

        # default path (circle)
        return CirclePath(
            center=[MAP_WIDTH / 2, MAP_HEIGHT / 2, 0.0],
            radius=min(MAP_WIDTH, MAP_HEIGHT) * 0.8
        )

    def _any_path_following(self):
        return any(mode == "path_follow" for mode in (
            JACKAL_MODE,
            EAGLE_MODE,
            PIGEON_MODE,
        ))

    def _spawn_species(self, cls, count, mode, path):
        for _ in range(count):
            agent = cls(self.random_position(), mode=mode)

            if mode == "path_follow":
                agent.path = path

            self.agents.append(agent)

    # Simulation
    def random_position(self):
        return np.array([
            uniform(0, MAP_WIDTH),
            uniform(0, MAP_HEIGHT),
            0.0
        ])

    def get_observation(self, agent):
        return {
            "pos": agent.pos.copy(),
            "speed": agent.speed,
            "direction": agent.direction,
            "rng": np.random.normal()
        }

    def step(self, dt):
        for agent in self.agents:
            obs = self.get_observation(agent)
            angular_velocity, accel = agent.update(dt, obs)
            self.log_agent_state(agent, angular_velocity, accel)

        self.t += dt

    def reset(self):
        self.agents.clear()
        self.log.clear()
        self.t = 0.0

    # Logging
    def log_agent_state(self, agent, angular_velocity, accel):
        vx, vy, vz = agent.direction * agent.speed

        self.log.append({
            "t": self.t,
            "agent_id": agent.agent_id,
            "species": type(agent).__name__,
            "mode": agent.mode,

            "x": agent.pos[0],
            "y": agent.pos[1],
            "z": agent.pos[2],

            "vx": vx,
            "vy": vy,
            "vz": vz,

            "speed": agent.speed,
            "angular_velocity": angular_velocity,
            "accel": accel,
        })

    def save_log_csv(self, path):
        if not self.log:
            print("Warning: log is empty, nothing to save.")
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.log[0].keys())
            writer.writeheader()
            writer.writerows(self.log)

        print(f"Saved log to {path}")
