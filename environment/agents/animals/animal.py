import numpy as np
from ..agent import Agent
from dataclasses import dataclass
from utils.vec_utils import *

class Animal(Agent):
    STATE_BASE  = "base"
    STATE_AVOID = "avoid"
    STATE_FLEE  = "flee"
    STATES = [STATE_BASE, STATE_AVOID, STATE_FLEE]

    def __init__(self, agent_id, pos, params, behaviour, seed, mode=None):
        super().__init__(agent_id, pos, seed, mode)
        self.params = params
        self.behaviour = behaviour
        self.state = Animal.STATE_BASE

    # Behavior
    def policy(self, obs, dt):
        direction, norm_speed = self.behaviour.act(obs, self.params, dt) # always act, to maintain determinism
        disturbance = np.sum([drone["val"] for drone in obs["disturbance_info"].values()])

        if disturbance > 0.95: # Flee !!! hardcoded atm
            mean_disturbance_dir = self.calc_mean_disturbance_dir(obs)
            self.state = Animal.STATE_FLEE
            return mean_disturbance_dir, 1
        elif disturbance > 0.6: # Avoid ! hardcoded atm
            mean_disturbance_dir = self.calc_mean_disturbance_dir(obs)

            w = (disturbance - 0.4) / 0.4
            blended_dir = (1 - w) * direction + w * mean_disturbance_dir
            self.state = Animal.STATE_AVOID
            return unit(blended_dir), norm_speed
        else: # Base
            self.state = Animal.STATE_BASE
            return direction, norm_speed
    
    def calc_mean_disturbance_dir(self, obs):
        drone_directions = [drone["dir"] for drone in obs["disturbance_info"].values()]
        mean_disturbance_dir = np.mean(drone_directions, axis=0)

        return -unit(mean_disturbance_dir)
    
    def to_dict(self):
        return{
            **super().to_dict(),
            "behaviour_state": self.state,
            # Add aditional things to log
        }

    def __repr__(self):
        x, y, z = self.pos
        return f"{self.params.name}([{round(x,1)}, {round(y,1)}, {round(z,1)}], behaviour={type(self.behaviour).__name__}, id={self.agent_id})"