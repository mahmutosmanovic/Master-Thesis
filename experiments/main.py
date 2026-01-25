from environment import Environment
from model.model import Model

env = Environment(seed=42)
observation = env.spawn()

ppo = Model() 

rewards = 0
for step in range(5):
    action = ppo.policy(observation)
    observation, reward, done = env.step(action)

    # rewards += reward 

    # for i, a in enumerate(env.agents):
    #     print(f"Step {step+1}:", a)
    
env.save_log_csv("logs/simulations/test_jackal_random_single.csv")

    
