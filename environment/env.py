from .agent import Drone
from .agent import Animal
from scripts import Behavior, MovementDim

class Env:
    def __init__(self, config):
        
        self.animals = [Animal(config=config, 
                               behaviors=Behavior, 
                               movement_dims=MovementDim)
                        for _ in range(config["animal"]["env"]["count"])]
        
        self.drones = [Drone(config=config) 
                       for _ in range(config["drone"]["env"]["count"])]

    
        print(self.animals[0].pos)