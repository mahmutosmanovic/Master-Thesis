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
        self.states.append(state.astype(np.float32))
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
        self.chkpt_dir = chkpt_dir

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

    def save_checkpoint(self, name="last"):
        path = os.path.join(self.chkpt_dir, f"mappo_actor_{name}.pt")
        T.save(self.state_dict(), path)

    def load_checkpoint(self, name="last"):
        path = os.path.join(self.chkpt_dir, f"mappo_actor_{name}.pt")
        self.load_state_dict(T.load(path, map_location=self.device))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
                 chkpt_dir="tmp/ppo"):
        super().__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.chkpt_dir = chkpt_dir

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

    def save_checkpoint(self, name="last"):
        path = os.path.join(self.chkpt_dir, f"mappo_critic_{name}.pt")
        T.save(self.state_dict(), path)

    def load_checkpoint(self, name="last"):
        path = os.path.join(self.chkpt_dir, f"mappo_critic_{name}.pt")
        self.load_state_dict(T.load(path, map_location=self.device))


def _atanh(x: T.Tensor) -> T.Tensor:
    x = T.clamp(x, -1 + EPS, 1 - EPS)
    return 0.5 * (T.log1p(x) - T.log1p(-x))


def _squashed_log_prob(normal_dist: Normal, raw_action: T.Tensor, squashed_action: T.Tensor) -> T.Tensor:
    logp_raw = normal_dist.log_prob(raw_action).sum(dim=-1)
    correction = T.log(1 - squashed_action.pow(2) + EPS).sum(dim=-1)
    return logp_raw - correction


class MAPPOAgent:
    def __init__(self, config):
        self.optim_hpt = config.model.optimization
        self.space_hpt = config.model.space
        self.samp_hpt = config.model.sampling

        self.act_dim = self.space_hpt.n_actions

        self.entropy_start = self.optim_hpt.entropy_start_coef
        self.entropy_end = self.optim_hpt.entropy_end_coef
        self.total_steps = self.samp_hpt.total_timesteps
        self.train_step = 0

        self.n_animals = config.animal.env.count
        self.total_drone_count = sum(config.drone[d_type].count for d_type in config.drone)

        drone_features = config.model.space.drone_features
        animal_features = config.model.space.animal_features

        self.actor_input_dims = drone_features + self.n_animals * animal_features
        self.critic_input_dims = self.total_drone_count * self.actor_input_dims

        self.actor = ActorNetwork(
            self.act_dim,
            self.actor_input_dims,
            self.optim_hpt.actor_lr,
            chkpt_dir=config.run_dir
        )

        self.critic = CriticNetwork(
            self.critic_input_dims,
            self.optim_hpt.critic_lr,
            chkpt_dir=config.run_dir
        )

        self.actor_lr_start = self.optim_hpt.actor_lr
        self.critic_lr_start = self.optim_hpt.critic_lr
        self.lr_end_frac = 0.1


        self.memory = PPOMemory(self.samp_hpt.mini_batch_size)

    def update_learning_rates(self):
        frac = 1.0 - (self.train_step / self.total_steps)
        new_actor_lr = self.actor_lr_start * max(self.lr_end_frac, frac)
        new_critic_lr = self.critic_lr_start * max(self.lr_end_frac, frac)

        for param_group in self.actor.optimizer.param_groups:
            param_group["lr"] = new_actor_lr

        for param_group in self.critic.optimizer.param_groups:
            param_group["lr"] = new_critic_lr

        return new_actor_lr, new_critic_lr

    def get_entropy_coef(self):
        frac = min(self.train_step / max(1, self.total_steps), 1.0)
        return self.entropy_start + frac * (self.entropy_end - self.entropy_start)

    def remember(self, state, action, logp, val, reward, done):
        self.memory.store_memory(state, action, logp, val, reward, done)

    def save_models(self, name="last"):
        path = os.path.join(self.actor.chkpt_dir, f"mappo_{name}.pt")

        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor.optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic.optimizer.state_dict(),
            "train_step": self.train_step,
        }

        T.save(checkpoint, path)

    def load_models(self, name="last"):
        path = os.path.join(self.actor.chkpt_dir, f"mappo_{name}.pt")

        checkpoint = T.load(path, map_location=self.actor.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])

        self.actor.optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic.optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        self.train_step = checkpoint.get("train_step", 0)

    def choose_action(self, observation, deterministic=False):
        device = self.actor.device
        obs_np = np.asarray(observation, dtype=np.float32)
        n_drones = obs_np.shape[0]

        global_obs = T.as_tensor(
            obs_np.reshape(-1),
            dtype=T.float32,
            device=device
        ).unsqueeze(0)

        with T.no_grad():
            value = self.critic(global_obs).squeeze(-1)

        local_batch = T.as_tensor(
            obs_np.reshape(n_drones, -1),
            dtype=T.float32,
            device=device
        )

        with T.no_grad():
            mu, std = self.actor(local_batch)
            dist = Normal(mu, std)

            raw_action = mu if deterministic else dist.rsample()
            actions = T.tanh(raw_action)

            logps_per_agent = _squashed_log_prob(dist, raw_action, actions)
            joint_logp = logps_per_agent.sum()

        return (
            actions.cpu().numpy(),
            float(joint_logp.item()),
            float(value.item())
        )

    def get_last_value(self, observation, done):
        if done:
            return 0.0

        with T.no_grad():
            state = T.as_tensor(
                np.asarray(observation, dtype=np.float32).reshape(-1),
                dtype=T.float32,
                device=self.actor.device
            ).unsqueeze(0)

            return float(self.critic(state).item())

    def learn(self, last_value):
        new_actor_lr, new_critic_lr = self.update_learning_rates()

        if self.memory.get_length() == 0:
            return {
                "entropy_coef": self.get_entropy_coef(),
                "policy_entropy": None,
                "actor_loss": None,
                "critic_loss": None,
                "actor_lr": None,
                "critic_lr": None,
            }

        state_arr, action_arr, old_logp_arr, vals_arr, reward_arr, done_arr, _ = self.memory.generate_batches()
        device = self.actor.device

        states = T.as_tensor(state_arr, dtype=T.float32, device=device)      # (T, D, obs_dim)
        actions = T.as_tensor(action_arr, dtype=T.float32, device=device)    # (T, D, act_dim)
        old_logp = T.as_tensor(old_logp_arr, dtype=T.float32, device=device) # (T,)
        values = T.as_tensor(vals_arr, dtype=T.float32, device=device)       # (T,)
        rewards = T.as_tensor(reward_arr, dtype=T.float32, device=device)    # (T,)
        dones = T.as_tensor(done_arr, dtype=T.float32, device=device)        # (T,)

        T_steps, n_drones, obs_dim = states.shape
        act_dim = self.act_dim

        advantages = T.zeros_like(rewards, device=device)
        gae = T.tensor(0.0, dtype=T.float32, device=device)
        last_v = T.as_tensor(last_value, dtype=T.float32, device=device)

        for t in reversed(range(T_steps)):
            next_value = last_v if t == T_steps - 1 else values[t + 1]
            delta = rewards[t] + self.optim_hpt.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.optim_hpt.gamma * self.optim_hpt.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        global_states = states.reshape(T_steps, -1)

        last_actor_loss = None
        last_critic_loss = None
        last_policy_entropy = None
        last_entropy_coef = None

        for _ in range(self.samp_hpt.n_epochs):
            indices = np.arange(T_steps)
            np.random.shuffle(indices)

            for start in range(0, T_steps, self.memory.batch_size):
                batch_idx = indices[start:start + self.memory.batch_size]

                batch_states = states[batch_idx]         # (B, D, obs_dim)
                batch_global = global_states[batch_idx]  # (B, D*obs_dim)
                batch_actions = actions[batch_idx]       # (B, D, act_dim)
                batch_old_logp = old_logp[batch_idx]     # (B,)
                batch_adv = advantages[batch_idx]        # (B,)
                batch_returns = returns[batch_idx]       # (B,)

                critic_value = self.critic(batch_global).squeeze(-1)

                local_obs = batch_states.reshape(-1, obs_dim)
                mu, std = self.actor(local_obs)
                dist = Normal(mu, std)

                agent_actions = batch_actions.reshape(-1, act_dim)
                raw_action = _atanh(agent_actions)

                new_logp_per_agent = _squashed_log_prob(dist, raw_action, agent_actions)
                new_logp = new_logp_per_agent.view(-1, n_drones).sum(dim=1)

                ratio = (new_logp - batch_old_logp).exp()

                unclipped = ratio * batch_adv
                clipped = T.clamp(
                    ratio,
                    1 - self.optim_hpt.policy_clip,
                    1 + self.optim_hpt.policy_clip
                ) * batch_adv

                actor_loss = -T.min(unclipped, clipped).mean()
                critic_loss = (batch_returns - critic_value).pow(2).mean()

                entropy = dist.entropy().sum(dim=-1)
                entropy = entropy.view(-1, n_drones).sum(dim=1).mean()

                entropy_coef = self.get_entropy_coef()

                total_loss = (
                    actor_loss
                    + self.optim_hpt.val_loss_coef * critic_loss
                    - entropy_coef * entropy
                )

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                total_loss.backward()

                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                self.actor.optimizer.step()
                self.critic.optimizer.step()

                last_actor_loss = float(actor_loss.item())
                last_critic_loss = float(critic_loss.item())
                last_policy_entropy = float(entropy.item())
                last_entropy_coef = float(entropy_coef)

        self.train_step += self.samp_hpt.rollout_steps
        self.train_step = min(self.train_step, self.total_steps)

        self.memory.clear_memory()

        return {
            "entropy_coef": last_entropy_coef if last_entropy_coef is not None else self.get_entropy_coef(),
            "policy_entropy": last_policy_entropy,
            "actor_loss": last_actor_loss,
            "critic_loss": last_critic_loss,
            "actor_lr": new_actor_lr,
            "critic_lr": new_critic_lr,
        }