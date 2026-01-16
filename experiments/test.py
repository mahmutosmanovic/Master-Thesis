from .settings import *
from simulation import World

world = World.random_world(seed=1)

dt = TIME_STEP

for step in range(50):
    world.step(dt)

    for i, a in enumerate(world.agents):
        print(f"Step {step+1}:", a)
    
world.save_log_csv("logs/simulations/test_jackal_random_single.csv")

    
