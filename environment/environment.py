import os
import csv
import numpy as np
from random import uniform

from environment.settings import *
from environment.agents.animals.animal import Animal, jackal_params, pigeon_params, eagle_params
from environment.agents.behaviour import RandomWalk, PathFollow, POI
from environment.paths import CirclePath
from environment.agents.drones.drone import Drone, drone_params
from environment.agents.drones.sensor import Camera, GPSSensor
from environment.disturbance import DisturbanceField


class Environment:
    def __init__(self, seed=None):
        self.seed_seq = np.random.SeedSequence(seed)
        self.rng = np.random.default_rng(self.seed_seq.spawn(1)[0])

        self.agents = []
        self.log = []
        self.t = 0.0
        self.max_t = MAX_T

        self.pois = self._init_pois()

        self.disturbance = DisturbanceField()
        self.animal_disturbance = None

        self.drone_ids = None
        self.animal_ids = None

    def reset(self, seed=None):
        # Optional reseed
        if seed is not None:
            self.seed_seq = np.random.SeedSequence(seed)
            self.rng = np.random.default_rng(self.seed_seq.spawn(1)[0])

        self.agents.clear()
        self.log.clear()
        self.t = 0.0

        self.pois = self._init_pois()

        path = self._create_path_if_needed()

        self._spawn_animal(jackal_params, JACKAL_COUNT, JACKAL_MODE, path)
        self._spawn_animal(eagle_params,  EAGLE_COUNT,  EAGLE_MODE,  path)
        self._spawn_animal(pigeon_params, PIGEON_COUNT, PIGEON_MODE, path)

        self._spawn_drone(drone_params, DRONE_COUNT)

        self.drone_ids = np.array([agent.agent_id for agent in self.agents if type(agent) == Drone])
        self.animal_ids = np.array([agent.agent_id for agent in self.agents if type(agent) == Animal])

        info = {"drone_ids": self.drone_ids, "animal_ids": self.animal_ids}

        self.calc_animal_disturbance()

        animals = [self.agents[animal_id] for animal_id in self.animal_ids]
        drone_observations = {did: self.agents[did].get_obs(animals) for did in self.drone_ids}

        return drone_observations, info
    
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
        
        self.calc_animal_disturbance()

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

            agent = Animal(agent_id=len(self.agents),
                           pos=self.random_position(),
                           params=animal_params_fn(),
                           behaviour=behaviour,
                           seed=self.seed_seq.spawn(1)[0],
                           mode=mode)
            
            self.agents.append(agent)

    def _spawn_drone(self, drone_params_fn, count):
        # sensor = Camera(np.deg2rad(90), np.deg2rad(56), far=100)
        sensor = GPSSensor(1, reward_scale=5, pos_scale=POS_SCALE)

        for _ in range(count):
            pos = self.random_position()
            pos[2] = 60
            agent = Drone(agent_id=len(self.agents),
                          pos=pos, 
                          params=drone_params_fn(),
                          seed=self.seed_seq.spawn(1)[0],
                          mode="external")
            agent.add_sensor(sensor)
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

    def get_animal_observation(self, agent):
        return {
            "pos": agent.pos.copy(),
            "speed": agent.speed,
            "direction": agent.direction,
            "disturbance_info": self.animal_disturbance[agent.agent_id],
        }

    def calc_animal_disturbance(self):
        animal_disturbance = {}
        for animal_id in self.animal_ids:
            disturbance = {drone_id: self.disturbance.disturbance_at(self.agents[animal_id], self.agents[drone_id]) for drone_id in self.drone_ids}
            animal_disturbance[self.agents[animal_id].agent_id] = disturbance
        
        self.animal_disturbance = animal_disturbance
    
    def step(self, external_actions):
        # Update agent states
        for agent in self.agents:
            if type(agent) == Drone:
                action = external_actions[agent.agent_id]
            elif type(agent) == Animal:
                obs = self.get_animal_observation(agent)
                action = agent.policy(obs, DT)
            else:
                raise NotImplementedError
            
            agent.update(action, DT)
            
            self.log_agent_state(agent)

        self.t += DT

        self.calc_animal_disturbance()
        animals = [self.agents[animal_id] for animal_id in self.animal_ids]

        reward = {}
        drone_observations = {}
        for drone_id in self.drone_ids:
            disturbances = np.array([animal[drone_id]["val"] for animal in self.animal_disturbance.values()], dtype=np.float32)

            # calculate total reward
            reward[drone_id] = self.agents[drone_id].sensors[0].reward(self.agents[drone_id], animals) # hard coded sensor for now
            reward[drone_id] -= DIST_PENALTY_SCALE * float(np.mean(disturbances))

            # assign observation
            drone_observations[drone_id] = self.agents[drone_id].get_obs(animals)

        if self.t > self.max_t:
            done = True
        else:
            done = False

        info = {}
        return drone_observations, reward, done, info

    # Logging
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
