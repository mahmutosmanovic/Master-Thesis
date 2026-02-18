import os
import numpy as np
import torch as T
from torch import nn, optim
from torch.distributions import Normal

EPS = 1e-6

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

    def store_memory(self, state, action, logp, val, reward, done):
        self.states.append(state.reshape(-1).astype(np.float32))
        self.actions.append(action.astype(np.float32))
        self.probs.append(np.float32(logp))
        self.vals.append(np.float32(val))
        self.rewards.append(np.float32(reward))
        self.dones.append(np.float32(done))

    def generate_batches(self):
        n = self.get_length()
        idx = np.arange(n, dtype=np.int64)
        np.random.shuffle(idx)

        batches = []
        for start in range(0, n, self.batch_size):
            batches.append(idx[start:start + self.batch_size])

        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.float32),
            np.array(self.probs, dtype=np.float32),
            np.array(self.vals, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
            batches
        )

    def clear_memory(self):
        self.states, self.probs, self.vals = [], [], []
        self.actions, self.rewards, self.dones = [], [], []


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir="tmp/ppo"):
        super().__init__()
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, "actor_torch_ppo.pt")

        self.net = nn.Sequential(
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
        x = self.net(state)
        mu = self.mu(x)
        std = self.log_std.exp().expand_as(mu)
        return mu, std

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

        self.net = nn.Sequential(
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
        return self.net(state)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


def _atanh(x: T.Tensor) -> T.Tensor:
    # numerically safe inverse tanh
    x = T.clamp(x, -1 + EPS, 1 - EPS)
    return 0.5 * (T.log1p(x) - T.log1p(-x))


def _squashed_log_prob(normal_dist: Normal, raw_action: T.Tensor, squashed_action: T.Tensor) -> T.Tensor:
    """
    log pi(a) where a = tanh(raw_action).
    Correction: log|det(d tanh / d raw)| = sum log(1 - tanh(raw)^2)
    """
    # log p(raw)
    logp_raw = normal_dist.log_prob(raw_action).sum(dim=-1)
    # correction term
    correction = T.log(1 - squashed_action.pow(2) + EPS).sum(dim=-1)
    return logp_raw - correction


class Agent:
    def __init__(self, config):
        self.optim_hpt = config.model.optimization
        self.space_hpt = config.model.space
        self.samp_hpt = config.model.sampling

        self.act_dim = config.drone.env.count * self.space_hpt.n_actions
        self.input_dims = config.drone.env.count * config.animal.env.count * self.space_hpt.features

        self.actor = ActorNetwork(self.act_dim, self.input_dims, self.optim_hpt.actor_lr)
        self.critic = CriticNetwork(self.input_dims, self.optim_hpt.critic_lr)
        self.memory = PPOMemory(self.samp_hpt.mini_batch_size)

    def remember(self, state, action, logp, val, reward, done):
        self.memory.store_memory(state, action, logp, val, reward, done)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
    
    def load_models(self):
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation, deterministic=False):
        state = T.tensor(observation.reshape(-1), dtype=T.float32, device=self.actor.device).unsqueeze(0)

        mu, std = self.actor(state)
        dist = Normal(mu, std)
        value = self.critic(state).squeeze(-1)

        if deterministic:
            raw_action = mu
        else:
            raw_action = dist.rsample()

        action = T.tanh(raw_action)
        logp = _squashed_log_prob(dist, raw_action, action)

        return (
            action.squeeze(0).detach().cpu().numpy(),
            float(logp.item()),
            float(value.item())
        )

    def get_last_value(self, observation, done):
        if done:
            return 0.0
        with T.no_grad():
            state = T.tensor(observation.reshape(-1), dtype=T.float32, device=self.actor.device).unsqueeze(0)
            return float(self.critic(state).item())

    def learn(self, last_value):
        if self.memory.get_length() == 0:
            return

        state_arr, action_arr, old_logp_arr, vals_arr, reward_arr, done_arr, batches = self.memory.generate_batches()
        device = self.actor.device

        states = T.tensor(state_arr, dtype=T.float32, device=device)
        actions = T.tensor(action_arr, dtype=T.float32, device=device)
        old_logp = T.tensor(old_logp_arr, dtype=T.float32, device=device)
        values = T.tensor(vals_arr, dtype=T.float32, device=device)
        rewards = T.tensor(reward_arr, dtype=T.float32, device=device)
        dones = T.tensor(done_arr, dtype=T.float32, device=device)

        # GAE 
        advantages = T.zeros_like(rewards, device=device)
        gae = T.tensor(0.0, device=device)

        for t in reversed(range(len(rewards))):
            next_value = T.tensor(last_value, device=device) if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.optim_hpt.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.optim_hpt.gamma * self.optim_hpt.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO 
        for _ in range(self.samp_hpt.n_epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(indices), self.memory.batch_size):
                batch_idx = indices[start:start + self.memory.batch_size]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_logp = old_logp[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                mu, std = self.actor(batch_states)
                dist = Normal(mu, std)
                critic_value = self.critic(batch_states).squeeze(-1)

                # IMPORTANT: invert tanh to get raw_action for correct log-prob
                raw_action = _atanh(batch_actions)
                new_logp = _squashed_log_prob(dist, raw_action, batch_actions)

                ratio = (new_logp - batch_old_logp).exp()
                unclipped = ratio * batch_adv
                clipped = T.clamp(ratio,
                                  1 - self.optim_hpt.policy_clip,
                                  1 + self.optim_hpt.policy_clip) * batch_adv

                actor_loss = -T.min(unclipped, clipped).mean()
                critic_loss = (batch_returns - critic_value).pow(2).mean()

                entropy = 0.5 * dist.entropy().sum(dim=-1).mean()

                total_loss = (
                    actor_loss
                    + self.optim_hpt.val_loss_coef * critic_loss
                    - self.optim_hpt.entropy_coef * entropy
                )

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
