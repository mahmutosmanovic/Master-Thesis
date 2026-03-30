import os
import copy
import numpy as np
import torch as T
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Normal

EPS = 1e-6
LOG_STD_MIN = -20
LOG_STD_MAX = 2

class ReplayBuffer:
    def __init__(self, max_size, input_dims, n_actions, batch_size):
        self.max_size = int(max_size)
        self.batch_size = int(batch_size)
        self.mem_cntr = 0

        self.states = np.zeros((self.max_size, input_dims), dtype=np.float32)
        self.next_states = np.zeros((self.max_size, input_dims), dtype=np.float32)
        self.actions = np.zeros((self.max_size, n_actions), dtype=np.float32)
        self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.max_size, 1), dtype=np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        idx = self.mem_cntr % self.max_size

        self.states[idx] = np.asarray(state, dtype=np.float32)
        self.actions[idx] = np.asarray(action, dtype=np.float32)
        self.rewards[idx] = np.float32(reward)
        self.next_states[idx] = np.asarray(next_state, dtype=np.float32)
        self.dones[idx] = np.float32(done)

        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.max_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        return (
            self.states[batch],
            self.actions[batch],
            self.rewards[batch],
            self.next_states[batch],
            self.dones[batch],
        )

    def ready(self):
        return self.mem_cntr >= self.batch_size

    def __len__(self):
        return min(self.mem_cntr, self.max_size)

class ActorNetwork(nn.Module):
    def __init__(
        self,
        n_actions,
        input_dims,
        alpha,
        fc1_dims=256,
        fc2_dims=256,
        chkpt_dir="tmp/sac",
        device="cpu",
    ):
        super().__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.chkpt_dir = chkpt_dir
        self.device = T.device(device)

        self.net = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LeakyReLU(),
        )

        self.mu = nn.Linear(fc2_dims, n_actions)
        self.log_std = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)

    def forward(self, state):
        x = self.net(state)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = T.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        return mu, std

    def sample_normal(self, state, reparameterize=True):
        mu, std = self.forward(state)
        dist = Normal(mu, std)

        if reparameterize:
            z = dist.rsample()
        else:
            z = dist.sample()

        action = T.tanh(z)

        log_prob = dist.log_prob(z)
        log_prob -= T.log(1 - action.pow(2) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        deterministic_action = T.tanh(mu)

        return action, log_prob, deterministic_action

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
        n_actions,
        alpha,
        fc1_dims=256,
        fc2_dims=256,
        chkpt_dir="tmp/sac",
        name="critic",
        device="cpu",
    ):
        super().__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.chkpt_dir = chkpt_dir
        self.name = name
        self.device = T.device(device)

        self.net = nn.Sequential(
            nn.Linear(input_dims + n_actions, fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LeakyReLU(),
            nn.Linear(fc2_dims, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)

    def forward(self, state, action):
        x = T.cat([state, action], dim=-1)
        return self.net(x)

    def save_checkpoint(self, name="last"):
        path = os.path.join(self.chkpt_dir, f"{self.name}_{name}.pt")
        T.save(self.state_dict(), path)

    def load_checkpoint(self, name="last"):
        path = os.path.join(self.chkpt_dir, f"{self.name}_{name}.pt")
        self.load_state_dict(T.load(path, map_location=self.device))


class SACAgent:
    def __init__(self, config, device="cpu"):
        self.optim_hpt = config.model.optimization
        self.space_hpt = config.model.space

        replay_hpt = getattr(config.model, "replay", None)
        sampling_hpt = getattr(config.model, "sampling", None)

        self.device = T.device(device)

        drone_features = config.model.space.drone_features
        animal_features = config.model.space.animal_features * config.animal.env.count

        per_drone_obs_dim = drone_features + animal_features
        per_drone_act_dim = self.space_hpt.n_actions
        n_drones = config.drone.large.count

        self.obs_dim = n_drones * per_drone_obs_dim
        self.act_dim = n_drones * per_drone_act_dim

        self.gamma = self.optim_hpt.gamma
        self.tau = getattr(self.optim_hpt, "tau", 0.005)
        self.reward_scale = getattr(self.optim_hpt, "reward_scale", 1.0)

        self.actor_lr = self.optim_hpt.actor_lr
        self.critic_lr = self.optim_hpt.critic_lr
        self.alpha_lr = getattr(self.optim_hpt, "alpha_lr", self.actor_lr)

        self.buffer_size = getattr(
            replay_hpt, "buffer_size",
            getattr(self.optim_hpt, "buffer_size", 1_000_000)
        )
        self.batch_size = getattr(
            replay_hpt, "batch_size",
            getattr(sampling_hpt, "mini_batch_size", 256)
        )
        self.learn_after = getattr(replay_hpt, "learn_after", self.batch_size)
        self.learn_every = getattr(replay_hpt, "learn_every", 8)
        self.gradient_steps = getattr(replay_hpt, "gradient_steps", 1)

        self.learn_alpha = getattr(self.optim_hpt, "learn_alpha", True)
        self.target_entropy = getattr(
            self.optim_hpt, "target_entropy", -float(self.act_dim)
        )
        init_temperature = getattr(self.optim_hpt, "init_temperature", 0.2)

        self.actor = ActorNetwork(
            self.act_dim,
            self.obs_dim,
            self.actor_lr,
            chkpt_dir=config.run_dir,
            device=device,
        )

        self.critic_1 = CriticNetwork(
            self.obs_dim,
            self.act_dim,
            self.critic_lr,
            chkpt_dir=config.run_dir,
            name="critic_1",
            device=device,
        )

        self.critic_2 = CriticNetwork(
            self.obs_dim,
            self.act_dim,
            self.critic_lr,
            chkpt_dir=config.run_dir,
            name="critic_2",
            device=device,
        )

        self.target_critic_1 = CriticNetwork(
            self.obs_dim,
            self.act_dim,
            self.critic_lr,
            chkpt_dir=config.run_dir,
            name="target_critic_1",
            device=device,
        )

        self.target_critic_2 = CriticNetwork(
            self.obs_dim,
            self.act_dim,
            self.critic_lr,
            chkpt_dir=config.run_dir,
            name="target_critic_2",
            device=device,
        )

        self.update_network_parameters(tau=1.0)

        if self.learn_alpha:
            self.log_alpha = T.tensor(
                np.log(init_temperature),
                dtype=T.float32,
                device=self.device,
                requires_grad=True,
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self.alpha = T.tensor(init_temperature, dtype=T.float32, device=self.device)

        self.memory = ReplayBuffer(
            self.buffer_size,
            self.obs_dim,
            self.act_dim,
            self.batch_size,
        )

        self.env_step = 0
        self.learn_step = 0

    def update_network_parameters(self, tau=None):
        tau = self.tau if tau is None else tau

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)
        self.env_step += 1

    def choose_action(self, observation, deterministic=False):
        state = T.as_tensor(observation, dtype=T.float32, device=self.device).unsqueeze(0)

        with T.no_grad():
            sampled_action, log_prob, deterministic_action = self.actor.sample_normal(
                state, reparameterize=False
            )

            if deterministic:
                action = deterministic_action
                logp_out = 0.0
            else:
                action = sampled_action
                logp_out = float(log_prob.item())

            q1 = self.critic_1(state, action)
            q2 = self.critic_2(state, action)
            q = T.min(q1, q2)

        return (
            action.squeeze(0).cpu().numpy(),
            logp_out,
            float(q.item()),
        )

    def save_models(self, name="last"):
        path = os.path.join(self.actor.chkpt_dir, f"sac_{name}.pt")

        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_1_state_dict": self.critic_1.state_dict(),
            "critic_2_state_dict": self.critic_2.state_dict(),
            "target_critic_1_state_dict": self.target_critic_1.state_dict(),
            "target_critic_2_state_dict": self.target_critic_2.state_dict(),
            "actor_optimizer_state_dict": self.actor.optimizer.state_dict(),
            "critic_1_optimizer_state_dict": self.critic_1.optimizer.state_dict(),
            "critic_2_optimizer_state_dict": self.critic_2.optimizer.state_dict(),
            "env_step": self.env_step,
            "learn_step": self.learn_step,
            "alpha": float(self.alpha.item()),
            "learn_alpha": self.learn_alpha,
        }

        if self.learn_alpha:
            checkpoint["log_alpha"] = self.log_alpha.detach().cpu()
            checkpoint["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()

        T.save(checkpoint, path)

    def load_models(self, name="last"):
        path = os.path.join(self.actor.chkpt_dir, f"sac_{name}.pt")
        checkpoint = T.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic_1.load_state_dict(checkpoint["critic_1_state_dict"])
        self.critic_2.load_state_dict(checkpoint["critic_2_state_dict"])
        self.target_critic_1.load_state_dict(checkpoint["target_critic_1_state_dict"])
        self.target_critic_2.load_state_dict(checkpoint["target_critic_2_state_dict"])

        self.actor.optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_1.optimizer.load_state_dict(checkpoint["critic_1_optimizer_state_dict"])
        self.critic_2.optimizer.load_state_dict(checkpoint["critic_2_optimizer_state_dict"])

        self.env_step = checkpoint.get("env_step", 0)
        self.learn_step = checkpoint.get("learn_step", 0)

        if self.learn_alpha and "log_alpha" in checkpoint:
            self.log_alpha = checkpoint["log_alpha"].to(self.device).requires_grad_(True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.alpha = T.tensor(
                checkpoint.get("alpha", 0.2),
                dtype=T.float32,
                device=self.device,
            )

    def learn(self):
        if len(self.memory) < self.learn_after:
            return {
                "actor_loss": None,
                "critic_loss": None,
                "alpha_loss": None,
                "alpha": float(self.alpha.item()),
                "mean_q": None,
            }

        if self.env_step % self.learn_every != 0:
            return {
                "actor_loss": None,
                "critic_loss": None,
                "alpha_loss": None,
                "alpha": float(self.alpha.item()),
                "mean_q": None,
            }

        last_actor_loss = None
        last_critic_loss = None
        last_alpha_loss = None
        last_mean_q = None

        for _ in range(self.gradient_steps):
            states, actions, rewards, next_states, dones = self.memory.sample_buffer()

            states = T.tensor(states, dtype=T.float32, device=self.device)
            actions = T.tensor(actions, dtype=T.float32, device=self.device)
            rewards = T.tensor(rewards, dtype=T.float32, device=self.device)
            next_states = T.tensor(next_states, dtype=T.float32, device=self.device)
            dones = T.tensor(dones, dtype=T.float32, device=self.device)

            with T.no_grad():
                next_actions, next_log_probs, _ = self.actor.sample_normal(
                    next_states, reparameterize=False
                )

                target_q1 = self.target_critic_1(next_states, next_actions)
                target_q2 = self.target_critic_2(next_states, next_actions)
                target_q = T.min(target_q1, target_q2) - self.alpha * next_log_probs

                q_target = self.reward_scale * rewards + self.gamma * (1.0 - dones) * target_q

            q1 = self.critic_1(states, actions)
            q2 = self.critic_2(states, actions)

            critic_1_loss = F.mse_loss(q1, q_target)
            critic_2_loss = F.mse_loss(q2, q_target)
            critic_loss = critic_1_loss + critic_2_loss

            self.critic_1.optimizer.zero_grad()
            self.critic_2.optimizer.zero_grad()
            critic_loss.backward()
            T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0)
            T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0)
            self.critic_1.optimizer.step()
            self.critic_2.optimizer.step()

            new_actions, log_probs, _ = self.actor.sample_normal(
                states, reparameterize=True
            )

            q1_pi = self.critic_1(states, new_actions)
            q2_pi = self.critic_2(states, new_actions)
            q_pi = T.min(q1_pi, q2_pi)

            actor_loss = (self.alpha.detach() * log_probs - q_pi).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            T.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor.optimizer.step()

            if self.learn_alpha:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.alpha = self.log_alpha.exp().detach()
                last_alpha_loss = float(alpha_loss.item())
            else:
                last_alpha_loss = None

            self.update_network_parameters()
            self.learn_step += 1

            last_actor_loss = float(actor_loss.item())
            last_critic_loss = float(critic_loss.item())
            last_mean_q = float(q_pi.mean().item())

        return {
            "actor_loss": last_actor_loss,
            "critic_loss": last_critic_loss,
            "alpha_loss": last_alpha_loss,
            "alpha": float(self.alpha.item()),
            "mean_q": last_mean_q,
        }