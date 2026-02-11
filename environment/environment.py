import os
import csv
import numpy as np
import pandas as pd
from collections import defaultdict

from environment.paths import CirclePath
from environment.agents.drone import Drone
from environment.agents.behaviour import make_behaviour
from environment.agents.animal import Animal
from environment.agents.sensor import make_sensor
from environment.agents.disturbance import DisturbanceField
from environment.reward import tracking_reward
from utils.vec_utils import position_on_cylinder, random_position, random_direction


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

        self.drone_ids = []
        self.animal_ids = []

    # Setup

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

        for group in self.cfg.animals:
            self._spawn_animal(
                group["params"],
                group["count"],
                group["behaviour_cfg"],
            )

        for group in self.cfg.drones:
            self._spawn_drone(
                group["params"],
                group["count"],
                group["sensor_cfg"],
            )

        info = {"drone_ids": self.drone_ids, "animal_ids": self.animal_ids}

        self.disturb_animals()
        dummy_actions = {drone_id: ((0, 0, 0), 0, 0) for drone_id in self.drone_ids}
        drone_obs, _ = self.gather_drone_obs_rew(dummy_actions)

        return drone_obs, info

    def _spawn_animal(self, animal_params, count, behaviour_cfg):
        for _ in range(count):
            agent = Animal(agent_id=len(self.agents),
                           pos=random_position(self.rng, self.cfg.map_width, self.cfg.map_height, 0),
                           direction=random_direction(self.rng),
                           params=animal_params,
                           behaviour=make_behaviour(behaviour_cfg, self.seed_seq.spawn(1)[0]),
                           disturbance_field=DisturbanceField(),
                           seed=self.seed_seq.spawn(1)[0])
            
            self.animal_ids.append(agent.agent_id)
            self.agents.append(agent)
    
    def _spawn_drone(self, drone_params, count, sensor_cfg):
        for _ in range(count):
            target_id = self.rng.choice(self.animal_ids)
            target_pos = self.agents[target_id].pos.copy()
            pos, yaw = position_on_cylinder(target_pos, self.rng)

            agent = Drone(agent_id=len(self.agents),
                          pos=pos,
                          direction=random_direction(self.rng),
                          params=drone_params,
                          sensor=make_sensor(sensor_cfg, self.seed_seq.spawn(1)[0]),
                          seed=self.seed_seq.spawn(1)[0],
                          yaw=yaw,
                          pos_scale=self.pos_scale)
            
            self.drone_ids.append(agent.agent_id)
            self.agents.append(agent)
    
    # Simulation

    def gather_actions(self, external_actions):
        animal_obs = {animal_id: self.agents[animal_id].observe() for animal_id in self.animal_ids}
        animal_actions = {animal_id: self.agents[animal_id].policy(animal_obs[animal_id], self.cfg.dt) for animal_id in self.animal_ids}
        return {**external_actions, **animal_actions}
    
    def update_state(self, actions):
        for agent in self.agents:
            action = actions[agent.agent_id]
            agent.update(action, self.cfg.dt)
        self.t += self.cfg.dt

    def disturb_animals(self):
        drones = [self.agents[drone_id] for drone_id in self.drone_ids]
        for animal_id in self.animal_ids:
            self.agents[animal_id].disturb(drones)

    def gather_drone_obs_rew(self, actions):
        animals = [self.agents[animal_id] for animal_id in self.animal_ids]
        rewards = {}
        observations = {}
        for drone_id in self.drone_ids:
            # keep separate and handle concat in env, easier to modify later with heterogeneous agents
            drone_obs, sensor_obs = self.agents[drone_id].observe(animals)
            sensor_metrics = self.agents[drone_id].sensor.metrics(sensor_obs)

            # per drone disturbance
            disturbances = np.array([animal.disturbance_info[drone_id]["val"] for animal in animals], dtype=np.float32)
            disturbance = np.mean(disturbances)

            # calculate reward from metrics, explicit reward stated in reward.py
            rewards[drone_id] = tracking_reward(sensor_metrics, disturbance, actions[drone_id], self.cfg)

            # Assemble full observation
            observations[drone_id] = np.concatenate([drone_obs, sensor_obs], axis=0)
        
        return observations, rewards

    def is_done(self):
        if self.t > self.max_t:
            return True
        else:
            return False
        
    def step(self, external_actions):
        actions = self.gather_actions(external_actions)
        self.update_state(actions)  
        self.disturb_animals()
        drone_obs, rewards = self.gather_drone_obs_rew(actions)
        for agent in self.agents: self.log_agent_state(agent)
        done = self.is_done()
        info = {}
        return drone_obs, rewards, done, info

    # Logging

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
            sp = r.get("type")
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

    def log_agent_state(self, agent):
        self.log.append({
            "t": self.t,
            **agent.to_dict(),
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
