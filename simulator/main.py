import numpy as np

from config import default_config
from world import World
from agent import Agent, AgentParams
from controllers import RandomWalk


def main():
    cfg = default_config()
    world = World(cfg, seed=42)

    world_lo = np.array(cfg.bounds_min, dtype=float)
    world_hi = np.array(cfg.bounds_max, dtype=float)

    animal_lo = world_lo.copy()
    animal_hi = world_hi.copy()
    animal_lo[2] = 0.0
    animal_hi[2] = 0.0

    animal_behaviours = [
        RandomWalk(speed=0.9, change_prob=0.2, seed=1),
    ]

    for i in range(6):
        beh = animal_behaviours[i % len(animal_behaviours)]
        a = Agent(
            name=f"animal_{i}",
            params=AgentParams(max_speed=1.2, bounds_min=animal_lo, bounds_max=animal_hi, kind="animal"),
            controller=beh,
        )
        world.add_agent(a)

    drone_lo = world_lo.copy()
    drone_hi = world_hi.copy()
    drone_lo[2] = 5.0
    drone_hi[2] = 30.0

    scripted_drone = Agent(
        name="drone_scripted",
        params=AgentParams(max_speed=3.0, bounds_min=drone_lo, bounds_max=drone_hi, kind="drone"),
        controller=RandomWalk(speed=2.0, change_prob=0.1, seed=10),
    )
    world.add_agent(scripted_drone)

    rl_drone = Agent(
        name="drone_rl",
        params=AgentParams(max_speed=3.0, bounds_min=drone_lo, bounds_max=drone_hi, kind="drone"),
        controller=None,  # external control
    )
    world.add_agent(rl_drone)

    world.reset(seed=0)

    for t in range(cfg.max_steps):
        external_actions = {
            "drone_rl": np.random.uniform(-1, 1, size=3) * 3.0
        }
        world.step(external_actions=external_actions)

        if t % 50 == 0:
            print(f"t={t:03d} | drone_rl pos={rl_drone.state.pos.round(2)} | animal_0 pos={world.agents[0].state.pos.round(2)}")

    print("Done.")


if __name__ == "__main__":
    main()