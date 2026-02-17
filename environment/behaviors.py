import random
from .immutables import Behavior

def random_walk(animal, rng):
    if random.random() < animal.epsilon:
        if ~animal.use_random_unit_3d:
            deg = rng.triangular(-animal.hor_dir_angle, 0, animal.hor_dir_angle)
            animal.vel_dir.rotate_z(deg)
            animal.vel_speed = rng.uniform(animal.min_speed, animal.max_speed)
        elif animal.use_random_unit_3d:
            ...
        else:
            raise NotImplementedError 

def poi_patrol(animal, rng):
    ...

def parametric_path(animal, rng):
    ...

BEHAVIOR_FNs = {
    Behavior.RANDOM: random_walk,
    Behavior.POI: poi_patrol,
    Behavior.PATH: parametric_path,
}
        