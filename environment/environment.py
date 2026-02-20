import os
import csv
import numpy as np
import pandas as pd
from collections import defaultdict

from environment.agents.drone import Drone
from environment.agents.behaviour import make_behaviour
from environment.agents.animal import Animal
from environment.agents.disturbance import DisturbanceField
from environment.reward import tracking_reward
from environment.resource_map import ResourceMap
from environment.utils.vec_utils import position_on_cylinder, position_in_view, random_position, random_direction
from environment.viewer import animate_from_log
from environment.immutables import BehaviourState


class Environment:
    def __init__(self, config):
        self.cfg = config
        self.seed_seq = None
        self.rng = None
        self.next_episode_seed = None

        self.pos_scale = np.array([self.cfg.map_size, self.cfg.map_size, self.cfg.map_altitude])

        self.agents = []
        self.log = []
        self.t = 0.0
        self.max_t = config.max_t
        self.resource_map = None
        self.resource_seed = None

        self.drone_ids = []
        self.animal_ids = []

    # Setup
    def set_seed(self, seed):
        # Optional reseed
        if seed == None:
            self.episode_seed = self.next_episode_seed
        else:
            self.episode_seed = int(seed)

        # reseed episode RNG tree
        self.seed_seq = np.random.SeedSequence(self.episode_seed)
        self.rng = np.random.default_rng(self.seed_seq.spawn(1)[0])
        self.next_episode_seed = int(self.rng.integers(0, 2**31 - 1))
        self.resource_seed = int(self.rng.integers(0, 2**31 - 1))

    def reset(self, seed=None):
        self.set_seed(seed)
        self.agents.clear()
        self.log.clear()
        self.drone_ids.clear()
        self.animal_ids.clear()
        self.t = 0.0

        self.resource_map = ResourceMap(
            p_wavelenght=self.cfg.p_wavelenght,
            p_reduction=self.cfg.p_reduction,
            p_scale=self.cfg.p_scale,
            sample_res=self.cfg.sample_res,
            min_poi_p=self.cfg.min_poi_p,
            world_size=self.cfg.map_size,
            kernel_size=self.cfg.kernel_size,
            seed=self.resource_seed
            )

        for group in self.cfg.animals:
            self._spawn_animal(
                group["config"],
                group["count"],
                group["behaviour_cfg"],
            )

        for group in self.cfg.drones:
            self._spawn_drone(
                group["config"],
                group["count"],
                group["spawn_range"],
            )

        info = {"seed": self.episode_seed,
                "resource_seed": self.resource_seed}

        self.disturb_animals()
        dummy_actions = {drone_id: ((0, 0, 0), 0, 0) for drone_id in self.drone_ids}
        drone_obs, _ = self.gather_drone_obs_rew(dummy_actions)

        return drone_obs, info

    def _spawn_animal(self, animal_config, count, behaviour_cfg):
        for _ in range(count):
            agent = Animal(agent_id=len(self.agents),
                           pos=random_position(self.rng, self.cfg.map_size, self.cfg.map_size, 0),
                           direction=random_direction(self.rng),
                           cfg=animal_config,
                           behaviour=make_behaviour(behaviour_cfg),
                           disturbance_field=DisturbanceField(),
                           resource_map=self.resource_map,
                           force_bounds=self.cfg.force_bounds,
                           xy_bound=self.cfg.map_size,
                           seed=self.seed_seq.spawn(1)[0])
            
            self.animal_ids.append(agent.agent_id)
            self.agents.append(agent)
    
    def select_drone_target_id(self):
        match self.cfg.drone_target_order:
            case "round_robin":
                target_idx = len(self.drone_ids) % len(self.animal_ids)
                return self.animal_ids[target_idx]
            case "random":
                return self.rng.choice(self.animal_ids)
            case _:
                raise NotImplementedError("valid options for target order [round_robin, random]")

    def _spawn_drone(self, drone_config, count, spawn_range):
        for _ in range(count):
            target_agent_id = self.select_drone_target_id()
            target_pos = self.agents[target_agent_id].pos.copy()
            distance = self.rng.uniform(*spawn_range)
            yaw_rad = self.rng.uniform(0, np.pi*2)
            pos = position_in_view(target_pos, np.deg2rad(drone_config.camera_pitch), yaw_rad, distance)

            agent = Drone(agent_id=len(self.agents),
                          pos=pos,
                          direction=random_direction(self.rng),
                          cfg=drone_config,
                          seed=self.seed_seq.spawn(1)[0],
                          yaw_rad=yaw_rad,
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
    
    def forage_animals(self):
        for animal_id in self.animal_ids:
            self.agents[animal_id].forage()

    def gather_drone_obs_rew(self, actions):
        animals = [self.agents[animal_id] for animal_id in self.animal_ids]
        rewards = {}
        observations = {}
        for drone_id in self.drone_ids:
            # keep separate and handle concat in env, easier to modify later with heterogeneous agents
            drone_obs, camera_obs = self.agents[drone_id].observe(animals)
            reward_metrics = self.agents[drone_id].reward_metrics(camera_obs)

            # per drone disturbance
            disturbances = np.array([animal.disturbance_info[drone_id]["val"] for animal in animals], dtype=np.float32)
            disturbance = np.mean(disturbances)

            # calculate reward from metrics, explicit reward stated in reward.py
            rewards[drone_id] = tracking_reward(reward_metrics, disturbance, actions[drone_id], self.cfg)

            # Assemble full observation
            observations[drone_id] = np.concatenate([drone_obs, camera_obs], axis=0)
        
        return observations, rewards
    
    def gather_global_state(self):
        global_state = []
        for animal_id in self.animal_ids:
            animal = self.agents[animal_id]
            global_state.append(np.concatenate([
                animal.pos / self.pos_scale,
                animal.direction * animal.norm_speed,
            ]))

        for drone_id in self.drone_ids:
            drone = self.agents[drone_id]
            global_state.append(np.concatenate([
                drone.pos / self.pos_scale,
                drone.direction * drone.norm_speed,
                drone.view_dir
            ]))
        
        return np.concatenate(global_state).astype(np.float32)
            
    def is_done(self):
        if self.t > self.max_t:
            return True
        else:
            return False
        
    def step(self, external_actions):
        actions = self.gather_actions(external_actions)
        self.update_state(actions)  
        self.disturb_animals()
        self.forage_animals()
        drone_obs, rewards = self.gather_drone_obs_rew(actions)
        for agent in self.agents: self.log_agent_state(agent)
        done = self.is_done()
        info = {}
        return drone_obs, rewards, done, info

    @property
    def global_state_dim(self):
        return len(self.animal_ids) * 6 + len(self.drone_ids) * 9
    
    @property
    def obs_dim(self):
        return {drone_id: self.agents[drone_id].obs_dim for drone_id in self.drone_ids}
    
    @property
    def action_dim(self):
        return {drone_id: self.agents[drone_id].action_dim for drone_id in self.drone_ids}

    # Logging

    def render_episode(self, save_path="recordings/default.mp4"):
        animate_from_log(self.log,
                         save_path=save_path,
                         interval_ms=1000*self.cfg.dt)
        

    def episode_statistics(self):
        log = self.log
        scalars = {f"behaviour/{b.name.lower()}_percent": 0.0 for b in BehaviourState} # Init with behaviour fallback values

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
