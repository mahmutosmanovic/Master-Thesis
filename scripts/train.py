# script folder
from .config import cfg_train

# environment folder
from environment import Env

# model folder
from model import Agent

# standard modules
from collections import deque
from tqdm import trange, tqdm

def main(config):
    env = Env(config, seed=42)
    agent = Agent()
    agent.load(config["model_path"])
    agent.train()

    global_step = 0
    best_avg = float('-inf')
    best_ep = float('-inf')
    reward_buffer = deque(maxlen=100)
    loss_buffer = deque(maxlen=100)
    try:
        for ep in trange(config["episodes"]):

            obs, info = env.reset() # obs: (animals, drones)

            episodic_reward = 0
            episodic_loss = 0
            step_count = 0

            for _ in range(config["steps"]):
                
                global_step += 1
                step_count += 1
                
                action = agent.policy(obs)

                next_obs, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated

                loss = agent.learn(obs, action, reward, next_obs, done)
                
                obs = next_obs

                episodic_reward += reward
                episodic_loss += loss
                
                if done:
                    break

            if step_count > 0:
                episodic_loss /= step_count

            reward_buffer.append(episodic_reward)
            loss_buffer.append(episodic_loss)

            reward_avg = sum(reward_buffer) / len(reward_buffer)
            loss_avg = sum(loss_buffer) / len(loss_buffer)

            tqdm.write(f"Ep {ep:4d} | Steps: {global_step:8d} | Reward_Avg100: {reward_avg:8.2f} | Loss_Avg100: {loss_avg:3.5f}")

            new_best_avg = reward_avg > best_avg
            new_best_ep  = episodic_reward > best_ep

            if new_best_avg:
                best_avg = reward_avg
                agent.save("best_avg.pt", meta={"ep": ep, "reward": best_avg})
                tqdm.write(f"New best_avg model: {best_avg:.2f}")

            if new_best_ep :
                best_ep = episodic_reward
                agent.save("best_ep.pt", meta={"ep": ep, "reward": best_ep})
                tqdm.write(f"New best_ep model: {best_ep:.2f}")
            
            if ((ep + 1) % config["save_every"] == 0) and not(new_best_avg or new_best_ep):
                agent.save()


    finally:
        env.close()

if __name__ == "__main__":
    # RUN WITH:
    # python -m scripts.train
    main(cfg_train)
