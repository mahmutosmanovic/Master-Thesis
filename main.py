from view import *
from agent import *
from drone import *
from train import *
from test import *
from utils import *
from logger import *
from animal import *
from settings import *

def main(train=False, test=False, model_path="ppo_drone.pt", steps=STEPS):
    animal_config = AnimalConfig()
    drone_config = DroneConfig()

    animal = Animal(animal_config, BEHAVIOR, start_pos=(0, 0, 0))
    drone = Drone(drone_config, (animal.x, animal.y, animal.z))

    agent = PPOAgent(obs_dim, act_dim, lr=learning_rate)

    logger = Logger()

    if train:
        train_script(EPISODES, ROLLOUT_EPS, agent, model_path, animal, drone, logger, steps)
    elif test:
        test_script(agent, model_path, animal, drone, logger, steps, CSV_PATH, draw_trail_3D)
    else:
        raise ValueError("Choose one: --train or --run")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--test", action="store_true")

    parser.add_argument("--model", type=str, default="ppo_drone.pt")
    parser.add_argument("--steps", type=int, default=STEPS)

    args = parser.parse_args()

    main(train=args.train, test=args.test, model_path=args.model, steps=args.steps)
