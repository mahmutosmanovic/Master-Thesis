import numpy as np
from .agent import Agent
from dataclasses import dataclass
from utils.vec_utils import *

class Animal(Agent):
    STATE_BASE  = "base"
    STATE_AVOID = "avoid"
    STATE_FLEE  = "flee"
    STATES = [STATE_BASE, STATE_AVOID, STATE_FLEE]

    def __init__(self, agent_id, pos, direction, params, behaviour, disturbance_field, seed, mode=None):
        super().__init__(agent_id, pos, direction, seed, mode)
        self.params = params
        self.behaviour = behaviour
        self.state = Animal.STATE_BASE

        self.disturbance_field = disturbance_field
        self.disturbance_info = None
        self.total_disturbance = None

    # Behavior 
    def disturb(self, drones):
        self.disturbance_info = {drone.agent_id: self.disturbance_field.get_disturbance(self, drone) for drone in drones}
        self.total_disturbance = np.sum([d["val"] for d in self.disturbance_info.values()])

    def observe(self):
        return {
            "pos": self.pos.copy(),
            "norm_speed": self.norm_speed,
            "direction": self.direction,
            "disturbance_info": self.disturbance_info,
        }
    
    def policy(self, obs, dt):
        direction, norm_speed = self.behaviour.act(obs, dt) # always act, to maintain determinism
        disturbance = np.sum([drone["val"] for drone in obs["disturbance_info"].values()])

        if disturbance > self.params.flight_threshold: # Flee !!!
            mean_disturbance_dir = self.calc_weighted_disturbance_dir(obs)
            self.state = Animal.STATE_FLEE
            return mean_disturbance_dir, 1
        elif disturbance > self.params.avoidance_threshold: # Avoid !
            mean_disturbance_dir = self.calc_weighted_disturbance_dir(obs)

            w = (disturbance - self.params.avoidance_threshold) / (self.params.flight_threshold - self.params.avoidance_threshold)
            w = np.clip(w, 0.0, 1.0)
            blended_dir = (1 - w) * direction + w * mean_disturbance_dir
            self.state = Animal.STATE_AVOID
            return unit(blended_dir), norm_speed
        else: # Base
            self.state = self.behaviour.get_state()
            return direction, norm_speed

    def calc_weighted_disturbance_dir(self, obs):
        drone_dist_vals = np.array([drone["val"] for drone in obs["disturbance_info"].values()])
        drone_directions = np.array([drone["dir"] for drone in obs["disturbance_info"].values()])
        weighted_dir = np.sum(drone_directions * drone_dist_vals[:, None], axis=0)
        return -unit(weighted_dir)
    
    def to_dict(self):
        return{
            **super().to_dict(),
            "behaviour_state": self.state,
            "disturbance": self.total_disturbance,
            "behaviour": type(self.behaviour).__name__,
            # Add aditional things to log
        }

    def __repr__(self):
        x, y, z = self.pos
        return f"{self.params.name}([{round(x,1)}, {round(y,1)}, {round(z,1)}], behaviour={type(self.behaviour).__name__}, id={self.agent_id})"