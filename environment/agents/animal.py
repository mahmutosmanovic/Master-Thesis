import numpy as np
from .agent import Agent
from environment.utils.vec_utils import unit
from environment.immutables import BehaviourState

class Animal(Agent):
    def __init__(self, agent_id, pos, direction, cfg, behaviour, disturbance_field, resource_map, force_bounds, xy_bound, seed):
        super().__init__(agent_id, pos, direction, seed)
        self.cfg = cfg
        self.behaviour = behaviour
        self.state = BehaviourState.EXPLORE

        self.force_bounds = force_bounds
        self.xy_bound = xy_bound

        self.disturbance_field = disturbance_field
        self.disturbance_info = None
        self.total_disturbance = None

        self.resource_map = resource_map

        self.encounter = False
        self.p_resource = None

    # Behavior 
    def disturb(self, drones):
        self.disturbance_info = {drone.agent_id: self.disturbance_field.get_disturbance(self, drone) for drone in drones}
        self.total_disturbance = np.sum([d["val"] for d in self.disturbance_info.values()])

    def forage(self):
        self.encounter, self.p_resource = self.resource_map.is_encounter(self.pos[0:2], self.rng)

    def observe(self):
        return {
            "pos": self.pos.copy(),
            "norm_speed": self.norm_speed,
            "direction": self.direction,
            "disturbance_info": self.disturbance_info,
            "encounter": bool(self.encounter),
            "pois": self.resource_map.get_pois(),
        }
    
    def policy(self, obs, dt):
        direction, norm_speed = self.behaviour.act(obs, self.rng, dt) # always act, to maintain determinism
        disturbance = np.sum([drone["val"] for drone in obs["disturbance_info"].values()])

        if disturbance > self.cfg.flight_threshold: # Flee !!!
            mean_disturbance_dir = self.calc_weighted_disturbance_dir(obs)
            self.state = BehaviourState.FLIGHT
            return mean_disturbance_dir, 1
        elif disturbance > self.cfg.avoidance_threshold: # Avoid !
            mean_disturbance_dir = self.calc_weighted_disturbance_dir(obs)

            w = (disturbance - self.cfg.avoidance_threshold) / (self.cfg.flight_threshold - self.cfg.avoidance_threshold)
            w = np.clip(w, 0.0, 1.0)
            blended_dir = (1 - w) * direction + w * mean_disturbance_dir
            self.state = BehaviourState.AVOID
            return unit(blended_dir), norm_speed
        else: # Base
            self.state = self.behaviour.get_state()
            return direction, norm_speed

    def calc_weighted_disturbance_dir(self, obs):
        drone_dist_vals = np.array([drone["val"] for drone in obs["disturbance_info"].values()])
        drone_directions = np.array([drone["dir"] for drone in obs["disturbance_info"].values()])
        weighted_dir = np.sum(drone_directions * drone_dist_vals[:, None], axis=0)
        return -unit(weighted_dir)
    
    def update(self, action, dt):
        direction, norm_speed = action
        self.apply_control(direction, norm_speed, dt)
        self.move(dt)
        if self.force_bounds:
            self.reflect_bounds()
    
    def to_dict(self):
        return{
            **super().to_dict(),
            "behaviour_state": self.state.name.lower(),
            "disturbance": self.total_disturbance,
            "behaviour": type(self.behaviour).__name__,
            "encounter": self.encounter,
            "p_resource": self.p_resource
            # Add aditional things to log
        }

    def __repr__(self):
        x, y, z = self.pos
        return f"{self.cfg.name}([{round(x,1)}, {round(y,1)}, {round(z,1)}], behaviour={type(self.behaviour).__name__}, id={self.agent_id})"