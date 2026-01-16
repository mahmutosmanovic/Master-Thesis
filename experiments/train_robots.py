from settings import *
from simulation.world import World
from learning.ugv_policy import UGVPolicy

def train_robots():
    world = World.load_pretrained_animals()

    ugv = world.get_robot()
    policy = UGVPolicy()

    for episode in range(EPISODES_TRAIN_ROBOTS):
        world.reset()

        while not world.done:
            obs = world.observe(ugv)
            action = policy.act(obs)

            world.step({ugv: action})

        policy.learn(world.rewards_for(ugv))
