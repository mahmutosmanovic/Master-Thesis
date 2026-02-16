import os
import torch as T
import numpy as np
from torch import nn
from torch import optim
from torch.distributions import Normal

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def get_length(self):
        return len(self.states)

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches
        )
    
    def store_memory(self, state, action, probs, vals, rewards, done):
        self.states.append(state.reshape(-1))
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(rewards)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir="tmp/ppo"):
        super(ActorNetwork, self).__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, "actor_torch_ppo.pt")
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LeakyReLU(),
        )
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.log_std = nn.Parameter(T.zeros(n_actions))
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = self.actor(state)
        mu = self.mu(x)
        std = self.log_std.clamp(-20, 2).exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
                 chkpt_dir="tmp/ppo"):
        super().__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, "critic_torch_ppo.pt")
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LeakyReLU(),
            nn.Linear(fc2_dims, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, state):
        value = self.critic(state)
        return value
        
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))      

class Agent:
    def __init__(self, config):
        # config
        self.optim_hpt = config.model.optimization
        self.space_hpt = config.model.space
        self.samp_hpt = config.model.sampling
        self.path = config.model.path

        self.act_dim = config.drone.env.count * self.space_hpt.n_actions
        self.input_dims = config.drone.env.count * config.animal.env.count * self.space_hpt.features
        self.actor = ActorNetwork(self.act_dim,
                                  self.input_dims,
                                  self.optim_hpt.lr)
        self.critic = CriticNetwork(self.input_dims,
                                    self.optim_hpt.lr)
        self.memory = PPOMemory(self.samp_hpt.mini_batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)
        return self.memory.get_length()

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
    
    def load_models(self):
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation, deterministic=False):
        state = T.tensor(observation.reshape(-1), dtype=T.float).unsqueeze(0).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)

        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
            
        probs = dist.log_prob(action).sum(dim=-1).item()
        action = action.squeeze(0).cpu().numpy()
        value = T.squeeze(value).item()

        return (
            action,
            probs,
            value
        )
    
    def get_last_value(self, observation, done):
        if done:
            return 0.0

        with T.no_grad():
            state = T.tensor(
                observation.reshape(-1),
                dtype=T.float,
                device=self.actor.device
            ).unsqueeze(0)

            value = self.critic(state).item()

        return value

    def learn(self, last_value):

        if self.memory.get_length() == 0:
            return

        (
            state_arr,
            action_arr,
            old_probs_arr,
            vals_arr,
            reward_arr,
            done_arr,
            batches
        ) = self.memory.generate_batches()

        device = self.actor.device

        states_tensor = T.tensor(state_arr, dtype=T.float, device=device)
        actions_tensor = T.tensor(action_arr, dtype=T.float, device=device)
        old_probs_tensor = T.tensor(old_probs_arr, dtype=T.float, device=device)
        values = T.tensor(vals_arr, dtype=T.float, device=device)
        rewards = T.tensor(reward_arr, dtype=T.float, device=device)
        dones = T.tensor(done_arr, dtype=T.float, device=device)

        # =========================
        # GAE with proper bootstrap
        # =========================
        advantages = T.zeros_like(rewards, device=device)
        gae = T.zeros(1, device=device)

        for t in reversed(range(len(rewards))):

            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + \
                    self.optim_hpt.gamma * next_value * (1 - dones[t]) - \
                    values[t]

            gae = delta + \
                self.optim_hpt.gamma * self.optim_hpt.gae_lambda * \
                (1 - dones[t]) * gae

            advantages[t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # =========================
        # PPO Update
        # =========================
        for _ in range(self.samp_hpt.n_epochs):

            indices = np.arange(len(states_tensor))
            np.random.shuffle(indices)

            for start in range(0, len(indices), self.memory.batch_size):
                batch_idx = indices[start:start + self.memory.batch_size]

                batch_states = states_tensor[batch_idx]
                batch_actions = actions_tensor[batch_idx]
                batch_old_probs = old_probs_tensor[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                dist = self.actor(batch_states)
                critic_value = self.critic(batch_states).squeeze(-1)

                new_probs = dist.log_prob(batch_actions).sum(dim=1)
                prob_ratio = (new_probs - batch_old_probs).exp()

                unclipped = prob_ratio * batch_adv
                clipped = T.clamp(
                    prob_ratio,
                    1 - self.optim_hpt.policy_clip,
                    1 + self.optim_hpt.policy_clip
                ) * batch_adv

                actor_loss = -T.min(unclipped, clipped).mean()
                critic_loss = (batch_returns - critic_value).pow(2).mean()
                entropy = dist.entropy().sum(dim=1).mean()

                total_loss = (
                    actor_loss
                    + self.optim_hpt.val_loss_coef * critic_loss
                    - self.optim_hpt.entropy_coef * entropy
                )

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
