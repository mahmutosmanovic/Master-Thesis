from environment import Environment
from model.model import Model
from collections import Counter

env = Environment(seed=42)
observation, info = env.spawn()
agents = info["drone_ids"]
print(info)
ppo = Model() 

rewards = {drone_id: 0.0 for drone_id in agents}

for step in range(5):
    action = {drone_id: ppo.policy(observation) for drone_id in agents}
    observation, reward, done, info = env.step(action)

    for k, v in reward.items():
        rewards[k] += v
    print(rewards)

    # for i, a in enumerate(env.agents):
    #     print(f"Step {step+1}:", a)
    
env.save_log_csv("logs/simulations/test_jackal_random_single.csv")

    
