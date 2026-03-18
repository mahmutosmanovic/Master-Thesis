import os
import numpy as np
import torch as T
from torch import nn, optim
from torch.distributions import Normal

EPS = 1e-6


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

        # optional auxiliary supervision
        self.dist_targets = []
        self.track_targets = []

        self.batch_size = batch_size

    def get_length(self):
        return len(self.states)

    def store_memory(self, state, action, logp, val, reward, done, disturbance_target=None, track_target=None):
        self.states.append(state.astype(np.float32))
        self.actions.append(action.astype(np.float32))
        self.probs.append(np.float32(logp))
        self.vals.append(np.float32(val))
        self.rewards.append(np.float32(reward))
        self.dones.append(np.float32(done))

        self.dist_targets.append(
            np.float32(np.nan if disturbance_target is None else disturbance_target)
        )
        self.track_targets.append(
            np.float32(np.nan if track_target is None else track_target)
        )

    def compute_future_track_loss(self, horizon=5):
        vis = np.array(self.track_targets, dtype=np.float32)  # currently visibility
        N = len(vis)

        future_loss = np.zeros_like(vis)

        for t in range(N):
            end = min(N, t + horizon)

            # if ANY future step is not visible → mark as risky
            if np.any(vis[t:end] == 0.0):
                future_loss[t] = 1.0
            else:
                future_loss[t] = 0.0

        self.track_targets = future_loss.tolist()

    def generate_batches(self):
        n = self.get_length()
        indices = np.arange(n, dtype=np.int64)
        np.random.shuffle(indices)

        batches = []
        for start in range(0, n, self.batch_size):
            batches.append(indices[start:start + self.batch_size])

        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.float32),
            np.array(self.probs, dtype=np.float32),
            np.array(self.vals, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
            np.array(self.dist_targets, dtype=np.float32),
            np.array(self.track_targets, dtype=np.float32),
            batches,
        )

    def clear_memory(self):
        self.states, self.actions = [], []
        self.probs, self.vals = [], []
        self.rewards, self.dones = [], []
        self.dist_targets, self.track_targets = [], []

class ActorNetwork(nn.Module):
    is_modular = False
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
        self.log_std = nn.Parameter(T.full((n_actions,), -1.5))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
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
        self.load_state_dict(T.load(path))

class ModularActorNetwork(nn.Module):
    """
    Disturbance-gated modular policy:
      - aggressive head
      - conservative head
      - safety residual head
      - disturbance risk predictor
      - tracking-loss risk predictor

    final_mu = blend(aggressive, conservative) + safety_residual
    """
    is_modular = True

    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir="tmp/ppo"):
        super().__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.chkpt_dir = chkpt_dir
        hidden_dim = fc2_dims

        self.encoder = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(fc1_dims, hidden_dim),
            nn.LeakyReLU(),
        )

        # specialized control heads
        self.mu_aggressive = nn.Linear(hidden_dim, n_actions)
        self.mu_conservative = nn.Linear(hidden_dim, n_actions)
        self.mu_safe_residual = nn.Linear(hidden_dim, n_actions)

        # risk heads
        self.dist_risk_head = nn.Linear(hidden_dim, 1)
        self.track_risk_head = nn.Linear(hidden_dim, 1)

        self.log_std = nn.Parameter(T.full((n_actions,), -1.5))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        h = self.encoder(state)

        mu_aggr = self.mu_aggressive(h)
        mu_cons = self.mu_conservative(h)
        mu_safe = self.mu_safe_residual(h)

        dist_risk = T.sigmoid(self.dist_risk_head(h))   # [B, 1]
        track_risk = T.sigmoid(self.track_risk_head(h)) # [B, 1]

        d = dist_risk
        t = track_risk

        g_safe = t
        g_cons = d * (1.0 - t)
        g_aggr = (1.0 - d) * (1.0 - t)

        # normalize
        g_sum = g_safe + g_cons + g_aggr + 1e-8
        g_safe = g_safe / g_sum
        g_cons = g_cons / g_sum
        g_aggr = g_aggr / g_sum

        g_safe = g_safe.detach()
        g_cons = g_cons.detach()
        g_aggr = g_aggr.detach()

        mu = (
            g_aggr * mu_aggr +
            g_cons * mu_cons +
            g_safe * mu_safe
        )

        std = self.log_std.exp().expand_as(mu)

        return mu, std, dist_risk.squeeze(-1), track_risk.squeeze(-1)

    def save_checkpoint(self, name="last"):
        path = os.path.join(self.chkpt_dir, f"actor_{name}.pt")
        T.save(self.state_dict(), path)

    def load_checkpoint(self, name="last"):
        path = os.path.join(self.chkpt_dir, f"actor_{name}.pt")
        self.load_state_dict(T.load(path, map_location=self.device))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir="tmp/ppo"):
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
        path = os.path.join(self.chkpt_dir, f"critic_{name}.pt")
        T.save(self.state_dict(), path)

    def load_checkpoint(self, name="last"):
        path = os.path.join(self.chkpt_dir, f"critic_{name}.pt")
        self.load_state_dict(T.load(path))


def _atanh(x: T.Tensor) -> T.Tensor:
    x = T.clamp(x, -1 + EPS, 1 - EPS)
    return 0.5 * (T.log1p(x) - T.log1p(-x))


def _squashed_log_prob(dist: Normal, raw_action: T.Tensor, squashed_action: T.Tensor):
    logp_raw = dist.log_prob(raw_action).sum(dim=-1)
    correction = T.log(1 - squashed_action.pow(2) + EPS).sum(dim=-1)
    return logp_raw - correction

class PPOAgent:
    def __init__(self, config):

        self.optim_hpt = config.model.optimization
        self.space_hpt = config.model.space
        self.samp_hpt = config.model.sampling

        self.act_dim = self.space_hpt.n_actions

        # linear entropy schedule over total env timesteps
        self.entropy_start = self.optim_hpt.entropy_start_coef
        self.entropy_end = self.optim_hpt.entropy_end_coef
        self.total_steps = self.samp_hpt.total_timesteps
        self.train_step = 0

        drone_features = config.model.space.drone_features
        animal_features = config.model.space.animal_features * config.animal.env.count

        self.obs_dim = drone_features + animal_features

        self.policy_type = getattr(config.model, "policy_type", "standard")
        self.aux_dist_coef = getattr(config.model, "aux_dist_coef", 0.0)
        self.aux_track_coef = getattr(config.model, "aux_track_coef", 0.0)

        if self.policy_type == "modular":
            self.actor = ModularActorNetwork(
                self.act_dim,
                self.obs_dim,
                self.optim_hpt.actor_lr,
                chkpt_dir=config.run_dir
            )
        else:
            self.actor = ActorNetwork(
                self.act_dim,
                self.obs_dim,
                self.optim_hpt.actor_lr,
                chkpt_dir=config.run_dir
            )

        self.critic = CriticNetwork(
            self.obs_dim,
            self.optim_hpt.critic_lr,
            chkpt_dir=config.run_dir
        )

        self.actor_lr_start = self.optim_hpt.actor_lr
        self.critic_lr_start = self.optim_hpt.critic_lr
        self.lr_end_frac = 0.1

        self.memory = PPOMemory(self.samp_hpt.mini_batch_size)

    def update_learning_rates(self):
        """Linearly decreases the learning rate of both actor and critic."""
        frac = 1.0 - (self.train_step / self.total_steps)
        new_actor_lr = self.actor_lr_start * max(self.lr_end_frac, frac)
        new_critic_lr = self.critic_lr_start * max(self.lr_end_frac, frac)

        for param_group in self.actor.optimizer.param_groups:
            param_group['lr'] = new_actor_lr
        
        for param_group in self.critic.optimizer.param_groups:
            param_group['lr'] = new_critic_lr

        return new_actor_lr, new_critic_lr
        

    def get_entropy_coef(self):
        frac = min(self.train_step / max(1, self.total_steps), 1.0)
        return self.entropy_start + frac * (self.entropy_end - self.entropy_start)

    def remember(self, state, action, logp, val, reward, done, disturbance_target=None, track_target=None):
        self.memory.store_memory(state, action, logp, val, reward, done, disturbance_target=disturbance_target, track_target=track_target)

    def save_models(self, name="last"):
        path = os.path.join(self.actor.chkpt_dir, f"ppo_{name}.pt")

        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor.optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic.optimizer.state_dict(),
            "train_step": self.train_step,
            "policy_type": self.policy_type,
        }

        T.save(checkpoint, path)

    def load_models(self, name="last"):
        path = os.path.join(self.actor.chkpt_dir, f"ppo_{name}.pt")

        checkpoint = T.load(path, map_location=self.actor.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])

        self.actor.optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic.optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        self.train_step = checkpoint.get("train_step", 0)
    
    def _actor_forward(self, state):
        """
        Returns:
            mu, std, pred_dist, pred_track
        pred_* are None for standard PPO.
        """
        out = self.actor(state)
        if self.actor.is_modular:
            mu, std, pred_dist, pred_track = out
            return mu, std, pred_dist, pred_track
        else:
            mu, std = out
            return mu, std, None, None

    def choose_action(self, observation, deterministic=False):
        device = self.actor.device
        state = T.as_tensor(observation, dtype=T.float32, device=device).unsqueeze(0)

        with T.no_grad():
            mu, std, _, _ = self._actor_forward(state)
            dist = Normal(mu, std)

            raw_action = mu if deterministic else dist.rsample()
            action = T.tanh(raw_action)

            logp = _squashed_log_prob(dist, raw_action, action)
            value = self.critic(state)

        return (
            action.squeeze(0).cpu().numpy(),
            float(logp.item()),
            float(value.item())
        )

    def get_last_value(self, observation, done):
        if done:
            return 0.0

        device = self.actor.device

        with T.no_grad():
            obs = T.as_tensor(observation, dtype=T.float32, device=device)

            # obs shape: (num_drones, obs_dim)
            values = self.critic(obs).squeeze(-1)

            # shared reward assumption
            value = values.mean()

        return float(value.item())

    def learn(self, last_value):
        new_actor_lr, new_critic_lr = self.update_learning_rates()

        if self.aux_track_coef > 0.0:
            self.memory.compute_future_track_loss(horizon=5)

        if self.memory.get_length() == 0:
            return {
                "train_entropy_coef": self.get_entropy_coef(),
                "train_policy_entropy": None,
                "actor_loss": None,
                "critic_loss": None,
                "aux_dist_loss": None,
                "aux_track_loss": None,
                "actor_lr": None,
                "critic_lr": None,
            }

        (
            states,
            actions,
            old_logp,
            values,
            rewards,
            dones,
            dist_targets,
            track_targets,
            batches,
        ) = self.memory.generate_batches()

        device = self.actor.device

        states = T.tensor(states, dtype=T.float32, device=device)
        actions = T.tensor(actions, dtype=T.float32, device=device)
        old_logp = T.tensor(old_logp, dtype=T.float32, device=device)
        values = T.tensor(values, dtype=T.float32, device=device)
        rewards = T.tensor(rewards, dtype=T.float32, device=device)
        dones = T.tensor(dones, dtype=T.float32, device=device)
        dist_targets = T.tensor(dist_targets, dtype=T.float32, device=device)
        track_targets = T.tensor(track_targets, dtype=T.float32, device=device)

        T_steps = len(rewards)

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
        last_aux_dist_loss = None
        last_aux_track_loss = None

        for _ in range(self.samp_hpt.n_epochs):
            for batch in batches:
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_logp = old_logp[batch]
                batch_adv = advantages[batch]
                batch_returns = returns[batch]
                batch_dist_targets = dist_targets[batch]
                batch_track_targets = track_targets[batch]

                mu, std, pred_dist, pred_track = self._actor_forward(batch_states)
                dist = Normal(mu, std)

                raw_action = _atanh(batch_actions)
                new_logp = _squashed_log_prob(dist, raw_action, batch_actions)

                ratio = (new_logp - batch_old_logp).exp()

                unclipped = ratio * batch_adv
                clipped = T.clamp(
                    ratio,
                    1 - self.optim_hpt.policy_clip,
                    1 + self.optim_hpt.policy_clip
                ) * batch_adv

                actor_loss = -T.min(unclipped, clipped).mean()

                critic_value = self.critic(batch_states).squeeze(-1)
                critic_loss = (batch_returns - critic_value).pow(2).mean()

                entropy = dist.entropy().sum(dim=-1).mean()
                entropy_coef = self.get_entropy_coef()

                aux_dist_loss = T.tensor(0.0, device=device)
                aux_track_loss = T.tensor(0.0, device=device)

                if pred_dist is not None and self.aux_dist_coef > 0.0:
                    valid = T.isfinite(batch_dist_targets)
                    if valid.any():
                        aux_dist_loss = ((pred_dist[valid] - batch_dist_targets[valid]) ** 2).mean()

                if pred_track is not None and self.aux_track_coef > 0.0:
                    valid = T.isfinite(batch_track_targets)
                    if valid.any():
                        aux_track_loss = T.nn.functional.binary_cross_entropy(
                            pred_track[valid],
                            batch_track_targets[valid]
                        )

                total_loss = (
                    actor_loss
                    + self.optim_hpt.val_loss_coef * critic_loss
                    - entropy_coef * entropy
                    + self.aux_dist_coef * aux_dist_loss
                    + self.aux_track_coef * aux_track_loss
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
                last_aux_dist_loss = float(aux_dist_loss.item())
                last_aux_track_loss = float(aux_track_loss.item())

        self.train_step += self.samp_hpt.rollout_steps
        self.train_step = min(self.train_step, self.total_steps)

        self.memory.clear_memory()

        return {
            "train_entropy_coef": last_entropy_coef if last_entropy_coef is not None else self.get_entropy_coef(),
            "train_policy_entropy": last_policy_entropy,
            "actor_loss": last_actor_loss,
            "critic_loss": last_critic_loss,
            "aux_dist_loss": last_aux_dist_loss,
            "aux_track_loss": last_aux_track_loss,
            "actor_lr": new_actor_lr,
            "critic_lr": new_critic_lr,
        }