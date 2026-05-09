import os
import numpy as np
import torch as T
from torch import nn, optim
from torch.distributions import Normal

EPS = 1e-6

def _atanh(x: T.Tensor) -> T.Tensor:
    x = T.clamp(x, -1 + EPS, 1 - EPS)
    return 0.5 * (T.log1p(x) - T.log1p(-x))


def _squashed_log_prob(dist: Normal, raw_action: T.Tensor, squashed_action: T.Tensor):
    logp_raw = dist.log_prob(raw_action).sum(dim=-1)
    correction = T.log(1 - squashed_action.pow(2) + EPS).sum(dim=-1)
    return logp_raw - correction


class ActorNetwork(nn.Module):
    def __init__(
        self,
        n_actions,
        input_dims,
        alpha,
        fc1_dims=256,
        fc2_dims=256,
        chkpt_dir="tmp/mappo",
        device="cpu",
    ):
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
        self.log_std = nn.Parameter(T.full((n_actions,), -1.0))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device(device)
        self.to(self.device)

    def forward(self, state):
        x = self.net(state)
        mu = self.mu(x)
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def save_checkpoint(self, name="last"):
        path = os.path.join(self.chkpt_dir, f"actor_{name}.pt")
        T.save(self.state_dict(), path)

    def load_checkpoint(self, name="last"):
        path = os.path.join(self.chkpt_dir, f"actor_{name}.pt")
        self.load_state_dict(T.load(path, map_location=self.device))


class CriticNetwork(nn.Module):
    def __init__(
        self,
        input_dims,
        alpha,
        fc1_dims=256,
        fc2_dims=256,
        chkpt_dir="tmp/mappo",
        device="cpu",
    ):
        super().__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.chkpt_dir = chkpt_dir

        self.net = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LeakyReLU(),
            nn.Linear(fc2_dims, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device(device)
        self.to(self.device)

    def forward(self, state):
        return self.net(state)

    def save_checkpoint(self, name="last"):
        path = os.path.join(self.chkpt_dir, f"critic_{name}.pt")
        T.save(self.state_dict(), path)

    def load_checkpoint(self, name="last"):
        path = os.path.join(self.chkpt_dir, f"critic_{name}.pt")
        self.load_state_dict(T.load(path, map_location=self.device))


class MAPPOMemory:
    def __init__(self, batch_size):
        self.local_obs = []
        self.global_obs = []
        self.actions = []
        self.logp = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def get_length(self):
        return len(self.rewards)

    def store_memory(self, local_obs, global_obs, actions, logp, value, reward, done):
        self.local_obs.append(np.asarray(local_obs, dtype=np.float32))    # [N, obs_dim]
        self.global_obs.append(np.asarray(global_obs, dtype=np.float32))  # [critic_dim]
        self.actions.append(np.asarray(actions, dtype=np.float32))        # [N, act_dim]
        self.logp.append(np.asarray(logp, dtype=np.float32))              # [N]
        self.values.append(np.float32(value))                             # scalar
        self.rewards.append(np.float32(reward))                           # scalar
        self.dones.append(np.float32(done))                               # scalar

    def generate_batches(self):
        n_steps = self.get_length()
        indices = np.arange(n_steps, dtype=np.int64)
        np.random.shuffle(indices)

        batches = []
        for start in range(0, n_steps, self.batch_size):
            batches.append(indices[start:start + self.batch_size])

        return (
            np.array(self.local_obs, dtype=np.float32),    # [T, N, obs_dim]
            np.array(self.global_obs, dtype=np.float32),   # [T, critic_dim]
            np.array(self.actions, dtype=np.float32),      # [T, N, act_dim]
            np.array(self.logp, dtype=np.float32),         # [T, N]
            np.array(self.values, dtype=np.float32),       # [T]
            np.array(self.rewards, dtype=np.float32),      # [T]
            np.array(self.dones, dtype=np.float32),        # [T]
            batches,
        )

    def clear_memory(self):
        self.local_obs = []
        self.global_obs = []
        self.actions = []
        self.logp = []
        self.values = []
        self.rewards = []
        self.dones = []


class MAPPOAgent:
    def __init__(self, config, device="cpu"):
        self.optim_hpt = config.model.optimization
        self.space_hpt = config.model.space
        self.samp_hpt = config.model.sampling

        # total number of controlled drones
        self.n_agents = int(sum(d.count for d in config.drone.values()))
        self.act_dim = self.space_hpt.n_actions

        # observation dimensions
        drone_features = config.model.space.drone_features
        animal_features = config.model.space.animal_features * config.animal.env.count
        other_drone_features = config.model.space.other_drone_features * (self.n_agents - 1)

        self.local_obs_dim = drone_features + animal_features + other_drone_features
        self.global_obs_dim = self.n_agents * self.local_obs_dim

        self.entropy_start = self.optim_hpt.entropy_start_coef
        self.entropy_end = self.optim_hpt.entropy_end_coef
        self.total_steps = self.samp_hpt.total_timesteps
        self.train_step = 0

        self.actor = ActorNetwork(
            n_actions=self.act_dim,
            input_dims=self.local_obs_dim,
            alpha=self.optim_hpt.actor_lr,
            chkpt_dir=config.run_dir,
            device=device,
        )

        self.critic = CriticNetwork(
            input_dims=self.global_obs_dim,
            alpha=self.optim_hpt.critic_lr,
            chkpt_dir=config.run_dir,
            device=device,
        )

        self.actor_lr_start = self.optim_hpt.actor_lr
        self.critic_lr_start = self.optim_hpt.critic_lr
        self.lr_end_frac = 0.1

        self.memory = MAPPOMemory(self.samp_hpt.mini_batch_size)

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

    def choose_actions(self, observations, deterministic=False):
        device = self.actor.device

        obs = np.asarray(observations, dtype=np.float32)
        assert obs.ndim == 2, f"Expected [n_agents, obs_dim], got {obs.shape}"
        assert obs.shape[0] == self.n_agents, f"Expected {self.n_agents} agents, got {obs.shape[0]}"
        assert obs.shape[1] == self.local_obs_dim, f"Expected obs_dim={self.local_obs_dim}, got {obs.shape[1]}"

        local_obs = T.as_tensor(obs, dtype=T.float32, device=device)                  # [N, obs_dim]
        global_obs = T.as_tensor(obs.reshape(1, -1), dtype=T.float32, device=device)  # [1, N*obs_dim]

        with T.no_grad():
            mu, std = self.actor(local_obs)    # [N, act_dim]
            dist = Normal(mu, std)

            raw_action = mu if deterministic else dist.rsample()
            action = T.tanh(raw_action)        # [N, act_dim]

            logp = _squashed_log_prob(dist, raw_action, action)   # [N]
            value = self.critic(global_obs).squeeze(-1)           # [1]

        return (
            action.cpu().numpy(),
            logp.cpu().numpy(),
            float(value.item()),
        )

    def get_last_value(self, observations, done):
        if done:
            return 0.0

        device = self.actor.device
        obs = np.asarray(observations, dtype=np.float32)
        global_obs = T.as_tensor(obs.reshape(1, -1), dtype=T.float32, device=device)

        with T.no_grad():
            value = self.critic(global_obs).squeeze(-1)

        return float(value.item())

    def remember(self, observations, actions, logp, value, reward, done):
        obs = np.asarray(observations, dtype=np.float32)
        global_obs = obs.reshape(-1).astype(np.float32)

        self.memory.store_memory(
            local_obs=obs,
            global_obs=global_obs,
            actions=np.asarray(actions, dtype=np.float32),
            logp=np.asarray(logp, dtype=np.float32),
            value=value,
            reward=reward,
            done=done,
        )

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

        (
            local_obs,
            global_obs,
            actions,
            old_logp,
            values,
            rewards,
            dones,
            batches,
        ) = self.memory.generate_batches()

        device = self.actor.device

        local_obs = T.tensor(local_obs, dtype=T.float32, device=device)     # [T, N, obs_dim]
        global_obs = T.tensor(global_obs, dtype=T.float32, device=device)   # [T, critic_dim]
        actions = T.tensor(actions, dtype=T.float32, device=device)         # [T, N, act_dim]
        old_logp = T.tensor(old_logp, dtype=T.float32, device=device)       # [T, N]
        values = T.tensor(values, dtype=T.float32, device=device)           # [T]
        rewards = T.tensor(rewards, dtype=T.float32, device=device)         # [T]
        dones = T.tensor(dones, dtype=T.float32, device=device)             # [T]

        T_steps = rewards.shape[0]
        advantages = T.zeros_like(rewards, device=device)

        gae = 0.0
        last_v = T.tensor(last_value, dtype=T.float32, device=device)

        for t in reversed(range(T_steps)):
            next_value = last_v if t == T_steps - 1 else values[t + 1]
            delta = rewards[t] + self.optim_hpt.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.optim_hpt.gamma * self.optim_hpt.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        last_actor_loss = None
        last_critic_loss = None
        last_policy_entropy = None
        last_entropy_coef = None

        for _ in range(self.samp_hpt.n_epochs):
            for batch in batches:
                batch_local_obs = local_obs[batch]       # [B, N, obs_dim]
                batch_global_obs = global_obs[batch]     # [B, critic_dim]
                batch_actions = actions[batch]           # [B, N, act_dim]
                batch_old_logp = old_logp[batch]         # [B, N]
                batch_adv = advantages[batch]            # [B]
                batch_returns = returns[batch]           # [B]

                B = batch_local_obs.shape[0]
                N = batch_local_obs.shape[1]

                flat_local_obs = batch_local_obs.reshape(B * N, self.local_obs_dim)
                flat_actions = batch_actions.reshape(B * N, self.act_dim)
                flat_old_logp = batch_old_logp.reshape(B * N)

                # same team advantage for all agents at a timestep
                flat_adv = batch_adv.unsqueeze(1).expand(B, N).reshape(B * N)

                mu, std = self.actor(flat_local_obs)
                dist = Normal(mu, std)

                raw_action = _atanh(flat_actions)
                new_logp = _squashed_log_prob(dist, raw_action, flat_actions)

                ratio = (new_logp - flat_old_logp).exp()

                unclipped = ratio * flat_adv
                clipped = T.clamp(
                    ratio,
                    1 - self.optim_hpt.policy_clip,
                    1 + self.optim_hpt.policy_clip,
                ) * flat_adv

                actor_loss = -T.min(unclipped, clipped).mean()

                critic_value = self.critic(batch_global_obs).squeeze(-1)   # [B]
                critic_loss = (batch_returns - critic_value).pow(2).mean()

                entropy = dist.entropy().sum(dim=-1).mean()
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