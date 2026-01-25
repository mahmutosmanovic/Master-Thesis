import os
import csv
import numpy as np
from random import uniform

from environment.settings import *
from environment.agents.animals.animal import Animal, jackal_params, pigeon_params, eagle_params
from environment.agents.behaviour import RandomWalk, PathFollow, POI
from environment.paths import CirclePath
from environment.agents.drones.drone import Drone, drone_params
from environment.disturbance import DisturbanceField


class Environment:
    def __init__(self, seed=None):
        self.seed_seq = np.random.SeedSequence(seed)
        self.rng = np.random.default_rng(self.seed_seq.spawn(1)[0])

        self.agents = []
        self.log = []
        self.t = 0.0

        self.pois = self._init_pois()

        self.disturbance = DisturbanceField()

        self.drone_ids = None
        self.animal_ids = None

    # Spawning
    def spawn(self):
        path = self._create_path_if_needed()

        self._spawn_animal(jackal_params, JACKAL_COUNT, JACKAL_MODE, path)
        self._spawn_animal(eagle_params,  EAGLE_COUNT,  EAGLE_MODE,  path)
        self._spawn_animal(pigeon_params, PIGEON_COUNT, PIGEON_MODE, path)

        self._spawn_drone(drone_params, DRONE_COUNT)

        observation = []

        self.drone_ids = [agent.agent_id for agent in self.agents if type(agent) == Drone]
        self.animal_ids = [agent.agent_id for agent in self.agents if type(agent) == Animal]

        info = {"drone_ids": self.drone_ids,
                "animal_ids": self.animal_ids}
        
        return observation, info
    
    def _create_path_if_needed(self):
        if not self._any_path_following():
            return None

        # default path (circle)
        return CirclePath(
            center=[0, 0, 20.0],
            radius=min(MAP_WIDTH, MAP_HEIGHT) * 0.4
        )

    def _any_path_following(self):
        return any(mode == "path_follow" for mode in (
            JACKAL_MODE,
            EAGLE_MODE,
            PIGEON_MODE,
        ))

    def _spawn_animal(self, animal_params_fn, count, mode, path):
        for _ in range(count):
            match mode:
                case "random":
                    behaviour = RandomWalk(self.seed_seq.spawn(1)[0])
                case "path_follow":
                    behaviour = PathFollow(path, self.seed_seq.spawn(1)[0])
                case "poi":
                    behaviour = POI(self.pois, self.seed_seq.spawn(1)[0])

            agent = Animal(pos=self.random_position(),
                           params=animal_params_fn(),
                           behaviour=behaviour,
                           seed=self.seed_seq.spawn(1)[0],
                           mode=mode)
            
            self.agents.append(agent)

    def _spawn_drone(self, drone_params_fn, count):
        for _ in range(count):
            agent = Drone(pos=self.random_position(), 
                          params=drone_params_fn(),
                          seed=self.seed_seq.spawn(1)[0],
                          mode="external")
            self.agents.append(agent)

    # Simulation
    def random_position(self):
        return np.array([
            self.rng.uniform(0, MAP_WIDTH),
            self.rng.uniform(0, MAP_HEIGHT),
            0.0
        ])
    
    def _init_pois(self):
        if POI_POINTS is not None:
            pts = POI_POINTS
        else:
            pts = [(self.rng.uniform(0, MAP_WIDTH), self.rng.uniform(0, MAP_HEIGHT), 0.0) for _ in range(POI_COUNT)]

        return [np.array(p, dtype=float) for p in pts]

    def get_observation(self, agent):
        return {
            "pos": agent.pos.copy(),
            "speed": agent.speed,
            "direction": agent.direction,
        }

    def step(self, external_actions):

        observations = []
        for agent in self.agents:
            obs = self.get_observation(agent)
            observations.append(obs)
            if type(agent) == Drone:
                action = external_actions[agent.agent_id]
            elif type(agent) == Animal:
                action = agent.policy(obs, DT)
            else:
                raise NotImplementedError
            
            yaw_rate, pitch_rate, accel = agent.update(action, DT)
            
            self.log_agent_state(agent, yaw_rate, pitch_rate, accel)

        self.t += DT

        animal_disturbance = {}
        for agent in self.agents:
            if type(agent) == Animal:
                disturbance = {drone_id: self.disturbance.disturbance_at(agent, self.agents[drone_id], external_actions[drone_id][2]) for drone_id in self.drone_ids}
                animal_disturbance[agent.agent_id] = disturbance
        
        drone_disturbance = {}
        for drone_id in self.drone_ids:
            total = 0
            for animal in animal_disturbance.values():
                total += animal[drone_id]
            drone_disturbance[drone_id] = total

        reward = {drone_id: -drone_disturbance[drone_id] for drone_id in self.drone_ids}

        done = False
        info = {}
        return observations, reward, done, info

    def reset(self):
        self.agents.clear()
        self.log.clear()
        self.t = 0.0

    # Logging
    def log_agent_state(self, agent, yaw_rate, pitch_rate, accel):
        vx, vy, vz = agent.direction * agent.speed

        self.log.append({
            "t": self.t,
            "agent_id": agent.agent_id,
            "species": agent.params.name,
            "mode": agent.mode,

            "x": agent.pos[0],
            "y": agent.pos[1],
            "z": agent.pos[2],

            "vx": vx,
            "vy": vy,
            "vz": vz,

            "speed": agent.speed,
            "yaw_rate": yaw_rate,
            "pitch_rate": pitch_rate,
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
