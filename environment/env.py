from .agent import Drone, Animal
from scripts import Behavior, MovementDim

class Env:
    def __init__(self, config):
        
        self.animal_count = config["animal"]["env"]["count"]
        self.animals = [Animal(config=config, 
                               behaviors=Behavior, 
                               movement_dims=MovementDim)
                        for _ in range(self.animal_count)]
        
        self.drone_count = config["drone"]["env"]["count"]
        self.drones = [Drone(config=config) 
                       for _ in range(self.drone_count)]
        self._init_drone_pos()

    def _init_drone_pos(self):
        for drone, animal in zip(self.drones, self.animals):
            animal_pos = animal.pos.getter()    
            view_dir = drone.view_dir.getter()
            inv_scale_view_dir = view_dir.scale(-drone.spawn_dist)
            new_drone_pos = animal_pos.add(inv_scale_view_dir)
            drone.pos.setter(new_drone_pos)


            
        