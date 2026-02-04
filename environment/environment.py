import os
import csv
import numpy as np
import pandas as pd
from collections import defaultdict

from environment.paths import CirclePath
from environment.agents.drones.drone import Drone
from environment.agents.behaviour import RandomWalk, PathFollow, POI
from environment.agents.animals.animal import Animal
from environment.agents.drones.sensor import Camera, GPSSensor
from environment.disturbance import DisturbanceField
from utils.vec_utils import *


class Environment:
    def __init__(self, config):
        self.cfg = config
        self.seed_seq = None
        self.rng = None

        self.pos_scale = np.array([self.cfg.map_width, self.cfg.map_height, self.cfg.map_altitude])

        self.agents = []
        self.log = []
        self.t = 0.0
        self.max_t = config.max_t

        self.pois = self._init_pois()

        self.disturbance = DisturbanceField()
        self.animal_disturbance = None

        self.drone_ids = []
        self.animal_ids = []

    def reset(self, seed=None):
        # Optional reseed
        if seed is not None:
            self.seed_seq = np.random.SeedSequence(seed)
            self.rng = np.random.default_rng(self.seed_seq.spawn(1)[0])

        self.agents.clear()
        self.log.clear()
        self.drone_ids.clear()
        self.animal_ids.clear()
        self.t = 0.0

        self.pois = self._init_pois()

        path = self._create_path_if_needed()

        for group in self.cfg.animals:
            self._spawn_animal(
                group["params"],
                group["count"],
                group["mode"],
                path
            )

        for group in self.cfg.drones:
            self._spawn_drone(
                group["params"],
                group["count"],
                group["sensor"],
            )

        info = {"drone_ids": self.drone_ids, "animal_ids": self.animal_ids}

        self.calc_animal_disturbance()

        animals = [self.agents[animal_id] for animal_id in self.animal_ids]
        drone_observations = {drone_id: self.agents[drone_id].observe(animals) for drone_id in self.drone_ids}

        return drone_observations, info
    
    def _create_path_if_needed(self):
        if not self._any_path_following():
            return None

        # default path (circle)
        return CirclePath(
            center=[0, 0, 20.0],
            radius=min(self.cfg.map_width, self.cfg.map_height) * 0.4
        )

    def _any_path_following(self):
        return any(group.get("mode") == "path_follow" for group in self.cfg.animals)

    def _spawn_animal(self, animal_params, count, mode, path):
        for _ in range(count):
            # If list, select a random mode from the list
            if isinstance(mode, list):
                selected_idx = self.rng.choice(np.arange(len(mode)))
                selected_mode = mode[selected_idx]
            else:
                selected_mode = mode

            match selected_mode:
                case "random":
                    behaviour = RandomWalk(self.seed_seq.spawn(1)[0])
                case "path_follow":
                    behaviour = PathFollow(path, self.seed_seq.spawn(1)[0])
                case "poi":
                    behaviour = POI(self.pois, self.seed_seq.spawn(1)[0])

            agent = Animal(agent_id=len(self.agents),
                           pos=self.random_position(),
                           params=animal_params,
                           behaviour=behaviour,
                           seed=self.seed_seq.spawn(1)[0],
                           mode=selected_mode)
            
            self.animal_ids.append(agent.agent_id)
            self.agents.append(agent)
    
    def _spawn_drone(self, drone_params, count, sensor):
        match sensor:
            case "camera":
                sensor = Camera(np.deg2rad(90), np.deg2rad(56), far=200, reward_scale=self.cfg.reward_scale, seed=self.seed_seq.spawn(1)[0])
            case "gps":
                sensor = GPSSensor(1, reward_scale=self.cfg.reward_scale, pos_scale=self.pos_scale, seed=self.seed_seq.spawn(1)[0])

        for _ in range(count):
            target_id = self.rng.choice(self.animal_ids)
            target_pos = self.agents[target_id].pos.copy()
            pos, yaw = self.position_on_circle(target_pos)

            agent = Drone(agent_id=len(self.agents),
                          pos=pos, 
                          params=drone_params,
                          seed=self.seed_seq.spawn(1)[0],
                          mode="external",
                          yaw=yaw,
                          pos_scale=self.pos_scale)
            
            self.drone_ids.append(agent.agent_id)
            agent.add_sensor(sensor)
            self.agents.append(agent)

    # Simulation
    def random_position(self):
        return np.array([
            self.rng.uniform(0, self.cfg.map_width),
            self.rng.uniform(0, self.cfg.map_height),
            0.0
        ])
    
    def position_on_circle(self, target_pos, distance=120, altitude=60):
        angle = self.rng.uniform(0, 2*np.pi)
        offset = np.array([distance * np.cos(angle), distance * np.sin(angle), altitude], dtype=float)
        pos = target_pos + offset
        yaw = np.arctan2(target_pos[1] - pos[1], target_pos[0] - pos[0])

        return pos, yaw
    
    def _init_pois(self):
        if self.cfg.poi_points is not None:
            pts = self.cfg.poi_points
        else:
            pts = [(self.rng.uniform(0, self.cfg.map_width), self.rng.uniform(0, self.cfg.map_height), 0.0) for _ in range(self.cfg.poi_count)]

        return [np.array(p, dtype=float) for p in pts]

    def get_animal_observation(self, agent):
        return {
            "pos": agent.pos.copy(),
            "norm_speed": agent.norm_speed,
            "direction": agent.direction,
            "disturbance_info": self.animal_disturbance[agent.agent_id],
        }

    def calc_animal_disturbance(self):
        animal_disturbance = {}
        for animal_id in self.animal_ids:
            disturbance = {drone_id: self.disturbance.get_disturbance(self.agents[animal_id], self.agents[drone_id]) for drone_id in self.drone_ids}
            animal_disturbance[self.agents[animal_id].agent_id] = disturbance
        
        self.animal_disturbance = animal_disturbance
    
    def step(self, external_actions):
        # Update agent states
        for agent in self.agents:
            if type(agent) == Drone:
                action = external_actions[agent.agent_id]
            elif type(agent) == Animal:
                obs = self.get_animal_observation(agent)
                action = agent.policy(obs, self.cfg.dt)
            else:
                raise NotImplementedError
            
            agent.update(action, self.cfg.dt)
            
        self.t += self.cfg.dt

        self.calc_animal_disturbance()
        animals = [self.agents[animal_id] for animal_id in self.animal_ids]

        reward = {}
        drone_observations = {}
        for drone_id in self.drone_ids:
            # per drone disturbance
            disturbances = np.array([animal[drone_id]["val"] for animal in self.animal_disturbance.values()], dtype=np.float32)

            # calculate total reward
            reward[drone_id] = self.agents[drone_id].reward(animals)              # Observation reward
            reward[drone_id] -= self.cfg.penalty_scale * float(np.mean(disturbances)) # Disturbance penalty

            # assign observation
            drone_observations[drone_id] = self.agents[drone_id].observe(animals)

        for agent in self.agents: self.log_agent_state(agent)

        info = {}
        if self.t > self.max_t:
            done = True
        else:
            done = False

        return drone_observations, reward, done, info

    def episode_statistics(self):
        log = self.log
        scalars = {f"behaviour/{b}_percent": 0.0 for b in Animal.STATES} # Init with behaviour fallback values

        behaviours = np.array(
            [r.get("behaviour_state") for r in log if r.get("behaviour_state") is not None],
            dtype=object,
        )
        if behaviours.size:
            uniq, cnt = np.unique(behaviours, return_counts=True)
            pct = 100.0 * cnt / behaviours.size
            for b, p in zip(uniq, pct):
                scalars[f"behaviour/{b}_percent"] = float(p)

        by_species = defaultdict(list)
        for r in log:
            sp = r.get("species")
            d = r.get("disturbance")
            if sp is None or d is None:
                continue
            by_species[sp].append(d)

        for sp, vals in by_species.items():
            arr = np.asarray(vals, dtype=np.float32)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue

            scalars[f"disturbance/{sp}_min"] = float(arr.min())
            scalars[f"disturbance/{sp}_max"] = float(arr.max())
            scalars[f"disturbance/{sp}_mean"] = float(arr.mean())

        return scalars

    # Logging
    def log_agent_state(self, agent):
        disturbance = self.animal_disturbance.get(agent.agent_id, None)
        if disturbance:
            disturbance = np.sum([d["val"] for d in disturbance.values()])

        self.log.append({
            "t": self.t,
            **agent.to_dict(),
            "disturbance": disturbance,
        })

    def _get_log_fieldnames(self):
        keys = set()
        for row in self.log:
            keys.update(row.keys())
        return sorted(keys)

    def save_log_csv(self, path):
        if not self.log:
            print("Warning: log is empty, nothing to save.")
            return

        fieldnames = self._get_log_fieldnames()

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.log)

        print(f"Saved log to {path}")
