import os
import itertools
import numpy as np
import torch as T
import torch.nn.functional as F
from torch import nn, optim


class ReplayBuffer:
    def __init__(self, max_size, input_dims, batch_size):
        self.max_size = int(max_size)
        self.batch_size = int(batch_size)
        self.mem_cntr = 0

        self.states = np.zeros((self.max_size, input_dims), dtype=np.float32)
        self.next_states = np.zeros((self.max_size, input_dims), dtype=np.float32)
        self.actions = np.zeros((self.max_size,), dtype=np.int64)   # discrete action index
        self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.max_size, 1), dtype=np.float32)

    def store_transition(self, state, action_idx, reward, next_state, done):
        idx = self.mem_cntr % self.max_size

        self.states[idx] = np.asarray(state, dtype=np.float32)
        self.actions[idx] = int(action_idx)
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

    def __len__(self):
        return min(self.mem_cntr, self.max_size)


class QNetwork(nn.Module):
    def __init__(
        self,
        input_dims,
        n_actions,
        lr,
        fc1_dims=256,
        fc2_dims=256,
        chkpt_dir="tmp/dqn",
        name="q_net",
        device="cpu",
    ):
        super().__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.chkpt_dir = chkpt_dir
        self.name = name
        self.device = T.device(device)

        self.net = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LeakyReLU(),
            nn.Linear(fc2_dims, n_actions),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, state):
        return self.net(state)

    def save_checkpoint(self, name="last"):
        path = os.path.join(self.chkpt_dir, f"{self.name}_{name}.pt")
        T.save(self.state_dict(), path)

    def load_checkpoint(self, name="last"):
        path = os.path.join(self.chkpt_dir, f"{self.name}_{name}.pt")
        self.load_state_dict(T.load(path, map_location=self.device))


class DQNAgent:
    """
    Discrete-action DQN for your current continuous env by using an action catalog.

    Important:
    - this implementation assumes ONE drone
    - action catalog entries are 5D vectors matching env action format:
      [v_forward, v_right, v_up, norm_speed, norm_theta]
    """

    def __init__(self, config, device="cpu"):
        self.optim_hpt = config.model.optimization
        self.space_hpt = config.model.space

        replay_hpt = getattr(config.model, "replay", None)
        sampling_hpt = getattr(config.model, "sampling", None)
        discrete_hpt = getattr(config.model, "discrete", None)

        self.device = T.device(device)

        drone_features = config.model.space.drone_features
        animal_features = config.model.space.animal_features * config.animal.env.count
        per_drone_obs_dim = drone_features + animal_features

        n_drones = config.drone.large.count
        if n_drones != 1:
            raise ValueError(
                "This DQNAgent currently supports exactly 1 drone. "
                "For multiple drones, use independent DQNs or a factorized action design."
            )

        self.obs_dim = per_drone_obs_dim

        self.gamma = getattr(self.optim_hpt, "gamma", 0.99)
        self.lr = getattr(self.optim_hpt, "actor_lr", 3e-4)

        self.tau = getattr(self.optim_hpt, "tau", None)  # if None, hard target updates are used
        self.target_update_interval = getattr(self.optim_hpt, "target_update_interval", 1000)

        self.buffer_size = getattr(
            replay_hpt, "buffer_size",
            getattr(self.optim_hpt, "buffer_size", 200_000)
        )
        self.batch_size = getattr(
            replay_hpt, "batch_size",
            getattr(sampling_hpt, "mini_batch_size", 256)
        )
        self.learn_after = getattr(replay_hpt, "learn_after", self.batch_size)
        self.learn_every = getattr(replay_hpt, "learn_every", 4)
        self.gradient_steps = getattr(replay_hpt, "gradient_steps", 1)

        # epsilon-greedy exploration
        self.eps_start = getattr(self.optim_hpt, "eps_start", 1.0)
        self.eps_end = getattr(self.optim_hpt, "eps_end", 0.05)
        self.eps_decay_steps = getattr(self.optim_hpt, "eps_decay_steps", 300_000)
        self.epsilon = self.eps_start

        # build discrete action catalog
        self.action_map = self._build_action_map(discrete_hpt)
        self.n_discrete_actions = len(self.action_map)

        self.q_net = QNetwork(
            input_dims=self.obs_dim,
            n_actions=self.n_discrete_actions,
            lr=self.lr,
            chkpt_dir=config.run_dir,
            name="q_net",
            device=device,
        )

        self.target_q_net = QNetwork(
            input_dims=self.obs_dim,
            n_actions=self.n_discrete_actions,
            lr=self.lr,
            chkpt_dir=config.run_dir,
            name="target_q_net",
            device=device,
        )
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.memory = ReplayBuffer(
            max_size=self.buffer_size,
            input_dims=self.obs_dim,
            batch_size=self.batch_size,
        )

        self.env_step = 0
        self.learn_step = 0

    def _build_action_map(self, discrete_hpt):
        """
        Creates a catalog of continuous 5D actions from per-dimension bins.

        Default:
            forward/right/up/speed/theta each in {-1, 0, 1}
            => 3^5 = 243 discrete actions
        """
        if discrete_hpt is None:
            bins = {
                "forward": [-1.0, 0.0, 1.0],
                "right":   [-1.0, 0.0, 1.0],
                "up":      [-1.0, 0.0, 1.0],
                "speed":   [-1.0, 0.0, 1.0],
                "theta":   [-1.0, 0.0, 1.0],
            }
        else:
            bins = {
                "forward": list(getattr(discrete_hpt, "forward_bins", [-1.0, 0.0, 1.0])),
                "right":   list(getattr(discrete_hpt, "right_bins",   [-1.0, 0.0, 1.0])),
                "up":      list(getattr(discrete_hpt, "up_bins",      [-1.0, 0.0, 1.0])),
                "speed":   list(getattr(discrete_hpt, "speed_bins",   [-1.0, 0.0, 1.0])),
                "theta":   list(getattr(discrete_hpt, "theta_bins",   [-1.0, 0.0, 1.0])),
            }

        catalog = list(itertools.product(
            bins["forward"],
            bins["right"],
            bins["up"],
            bins["speed"],
            bins["theta"],
        ))

        return np.asarray(catalog, dtype=np.float32)

    def _update_epsilon(self):
        frac = min(1.0, self.env_step / max(1, self.eps_decay_steps))
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

    def update_target_network(self, hard=False):
        if self.tau is not None and not hard:
            for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        else:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def remember(self, state, action_idx, reward, next_state, done):
        self.memory.store_transition(state, action_idx, reward, next_state, done)
        self.env_step += 1
        self._update_epsilon()

    def choose_action(self, observation, deterministic=False):
        state = T.as_tensor(observation, dtype=T.float32, device=self.device).unsqueeze(0)

        with T.no_grad():
            q_values = self.q_net(state).squeeze(0)

        if deterministic:
            action_idx = int(T.argmax(q_values).item())
        else:
            if np.random.random() < self.epsilon:
                action_idx = np.random.randint(self.n_discrete_actions)
            else:
                action_idx = int(T.argmax(q_values).item())

        action_vec = self.action_map[action_idx]
        q_val = float(q_values[action_idx].item())

        return action_vec.copy(), action_idx, q_val

    def save_models(self, name="last"):
        path = os.path.join(self.q_net.chkpt_dir, f"dqn_{name}.pt")

        checkpoint = {
            "q_net_state_dict": self.q_net.state_dict(),
            "target_q_net_state_dict": self.target_q_net.state_dict(),
            "q_optimizer_state_dict": self.q_net.optimizer.state_dict(),
            "env_step": self.env_step,
            "learn_step": self.learn_step,
            "epsilon": self.epsilon,
            "action_map": self.action_map,
        }

        T.save(checkpoint, path)

    def load_models(self, name="last"):
        path = os.path.join(self.q_net.chkpt_dir, f"dqn_{name}.pt")
        checkpoint = T.load(path, map_location=self.device, weights_only=False)
        
        self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.target_q_net.load_state_dict(checkpoint["target_q_net_state_dict"])
        self.q_net.optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])

        self.env_step = checkpoint.get("env_step", 0)
        self.learn_step = checkpoint.get("learn_step", 0)
        self.epsilon = checkpoint.get("epsilon", self.eps_start)

        if "action_map" in checkpoint:
            self.action_map = checkpoint["action_map"]
            self.n_discrete_actions = len(self.action_map)

    def learn(self):
        if len(self.memory) < self.learn_after:
            return {
                "loss": None,
                "epsilon": float(self.epsilon),
                "mean_q": None,
            }

        if self.env_step % self.learn_every != 0:
            return {
                "loss": None,
                "epsilon": float(self.epsilon),
                "mean_q": None,
            }

        last_loss = None
        last_mean_q = None

        for _ in range(self.gradient_steps):
            states, actions, rewards, next_states, dones = self.memory.sample_buffer()

            states = T.tensor(states, dtype=T.float32, device=self.device)
            actions = T.tensor(actions, dtype=T.long, device=self.device).unsqueeze(1)
            rewards = T.tensor(rewards, dtype=T.float32, device=self.device)
            next_states = T.tensor(next_states, dtype=T.float32, device=self.device)
            dones = T.tensor(dones, dtype=T.float32, device=self.device)

            q_pred_all = self.q_net(states)
            q_pred = q_pred_all.gather(1, actions)

            with T.no_grad():
                # Double DQN target
                next_q_online = self.q_net(next_states)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)

                next_q_target = self.target_q_net(next_states)
                next_q = next_q_target.gather(1, next_actions)

                q_target = rewards + self.gamma * (1.0 - dones) * next_q

            loss = F.mse_loss(q_pred, q_target)

            self.q_net.optimizer.zero_grad()
            loss.backward()
            T.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.q_net.optimizer.step()

            if self.tau is not None:
                self.update_target_network(hard=False)
            elif self.learn_step % self.target_update_interval == 0:
                self.update_target_network(hard=True)

            self.learn_step += 1
            last_loss = float(loss.item())
            last_mean_q = float(q_pred.mean().item())

        return {
            "loss": last_loss,
            "epsilon": float(self.epsilon),
            "mean_q": last_mean_q,
        }