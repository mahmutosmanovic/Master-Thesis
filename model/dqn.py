import os
import numpy as np
import torch as T
import torch.nn.functional as F
from torch import nn, optim


class ReplayBuffer:
    def __init__(self, max_size, input_dims, batch_size, n_branches=3):
        self.max_size = int(max_size)
        self.batch_size = int(batch_size)
        self.n_branches = int(n_branches)
        self.mem_cntr = 0

        self.states = np.zeros((self.max_size, input_dims), dtype=np.float32)
        self.next_states = np.zeros((self.max_size, input_dims), dtype=np.float32)
        self.actions = np.zeros((self.max_size, self.n_branches), dtype=np.int64)
        self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.max_size, 1), dtype=np.float32)

    def store_transition(self, state, action_idx, reward, next_state, done):
        idx = self.mem_cntr % self.max_size

        self.states[idx] = np.asarray(state, dtype=np.float32)
        self.actions[idx] = np.asarray(action_idx, dtype=np.int64)
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


class BranchingQNetwork(nn.Module):
    """
    Shared trunk + dueling-style branching heads.

    Branches:
      0: direction index
      1: speed index
      2: theta index
    """

    def __init__(
        self,
        input_dims,
        branch_sizes,
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
        self.branch_sizes = list(branch_sizes)

        self.trunk = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LeakyReLU(),
        )

        self.value_head = nn.Linear(fc2_dims, 1)
        self.adv_heads = nn.ModuleList(
            [nn.Linear(fc2_dims, branch_size) for branch_size in self.branch_sizes]
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, state):
        """
        Returns list of branch-wise Q tensors:
          [Q_dir, Q_speed, Q_theta]
        each of shape [B, branch_size]
        """
        feat = self.trunk(state)
        value = self.value_head(feat)  # [B, 1]

        q_branches = []
        for head in self.adv_heads:
            adv = head(feat)  # [B, A]
            q = value + (adv - adv.mean(dim=1, keepdim=True))
            q_branches.append(q)

        return q_branches

    def save_checkpoint(self, name="last"):
        path = os.path.join(self.chkpt_dir, f"{self.name}_{name}.pt")
        T.save(self.state_dict(), path)

    def load_checkpoint(self, name="last"):
        path = os.path.join(self.chkpt_dir, f"{self.name}_{name}.pt")
        self.load_state_dict(T.load(path, map_location=self.device))


class DQNAgent:
    """
    Branching DQN:
      - direction branch
      - speed branch
      - theta branch

    Env action format remains:
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

        self.tau = getattr(self.optim_hpt, "tau", None)
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

        self.eps_start = getattr(self.optim_hpt, "eps_start", 1.0)
        self.eps_end = getattr(self.optim_hpt, "eps_end", 0.05)
        self.eps_decay_steps = getattr(self.optim_hpt, "eps_decay_steps", 300_000)
        self.epsilon = self.eps_start

        # action branches
        self.direction_table = self._build_direction_table(discrete_hpt)
        self.speed_bins, self.theta_bins = self._build_scalar_bins(discrete_hpt)

        self.n_dir = len(self.direction_table)
        self.n_speed = len(self.speed_bins)
        self.n_theta = len(self.theta_bins)

        self.branch_sizes = [self.n_dir, self.n_speed, self.n_theta]

        self.q_net = BranchingQNetwork(
            input_dims=self.obs_dim,
            branch_sizes=self.branch_sizes,
            lr=self.lr,
            chkpt_dir=config.run_dir,
            name="q_net",
            device=device,
        )

        self.target_q_net = BranchingQNetwork(
            input_dims=self.obs_dim,
            branch_sizes=self.branch_sizes,
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
            n_branches=3,
        )

        self.env_step = 0
        self.learn_step = 0

        print(
            f"[BDQN] directions={self.n_dir}, "
            f"speed_bins={self.n_speed}, "
            f"theta_bins={self.n_theta}, "
            f"joint_action_count={self.n_dir * self.n_speed * self.n_theta}"
        )

    def _build_direction_table(self, discrete_hpt):
        if discrete_hpt is not None and hasattr(discrete_hpt, "direction_vectors"):
            raw_dirs = np.asarray(getattr(discrete_hpt, "direction_vectors"), dtype=np.float32)
        else:
            raw_dirs = np.asarray(
                [
                    [ 1,  0,  0], [-1,  0,  0], [ 0,  1,  0], [ 0, -1,  0], [ 0,  0,  1], [ 0,  0, -1],
                    [ 1,  1,  0], [ 1, -1,  0], [-1,  1,  0], [-1, -1,  0],
                    [ 1,  0,  1], [ 1,  0, -1], [-1,  0,  1], [-1,  0, -1],
                    [ 0,  1,  1], [ 0,  1, -1], [ 0, -1,  1], [ 0, -1, -1],
                    [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
                    [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1],
                ],
                dtype=np.float32,
            )

        if raw_dirs.ndim != 2 or raw_dirs.shape[1] != 3:
            raise ValueError("direction_vectors must have shape [N, 3].")
        if not np.all(np.isfinite(raw_dirs)):
            raise ValueError("direction_vectors contains NaN or inf.")

        norms = np.linalg.norm(raw_dirs, axis=1, keepdims=True)
        if np.any(norms.squeeze(-1) == 0.0):
            raise ValueError("direction_vectors cannot contain [0, 0, 0].")

        unit_dirs = raw_dirs / norms

        unique_dirs = []
        seen = set()
        for d in unit_dirs:
            key = tuple(np.round(d, 6).tolist())
            if key not in seen:
                seen.add(key)
                unique_dirs.append(d)

        return np.asarray(unique_dirs, dtype=np.float32)

    def _build_scalar_bins(self, discrete_hpt):
        if discrete_hpt is None:
            speed_bins = [-1.0, -0.80, -0.76, -0.72, -0.56, -0.52, -0.48, 1.0]
            theta_bins = [-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0]
        else:
            speed_bins = list(getattr(discrete_hpt, "speed_bins", [-1.0, -0.80, -0.76, -0.72, -0.56, -0.52, -0.48, 1.0]))
            theta_bins = list(getattr(discrete_hpt, "theta_bins", [-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0]))

        def _dedupe_preserve_order(vals):
            out = []
            seen = set()
            for v in vals:
                fv = float(v)
                key = round(fv, 8)
                if key not in seen:
                    seen.add(key)
                    out.append(fv)
            return np.asarray(out, dtype=np.float32)

        speed_bins = _dedupe_preserve_order(speed_bins)
        theta_bins = _dedupe_preserve_order(theta_bins)

        if speed_bins.ndim != 1 or len(speed_bins) == 0:
            raise ValueError("speed_bins must be a non-empty 1D list.")
        if theta_bins.ndim != 1 or len(theta_bins) == 0:
            raise ValueError("theta_bins must be a non-empty 1D list.")
        if np.any(speed_bins < -1.0) or np.any(speed_bins > 1.0):
            raise ValueError("speed_bins must lie in [-1, 1].")
        if np.any(theta_bins < -1.0) or np.any(theta_bins > 1.0):
            raise ValueError("theta_bins must lie in [-1, 1].")

        return speed_bins, theta_bins

    def _update_epsilon(self):
        frac = min(1.0, self.env_step / max(1, self.eps_decay_steps))
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

    def update_target_network(self, hard=False):
        if self.tau is not None and not hard:
            for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        else:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def _assemble_action(self, dir_idx, speed_idx, theta_idx):
        direction = self.direction_table[int(dir_idx)]
        speed = float(self.speed_bins[int(speed_idx)])
        theta = float(self.theta_bins[int(theta_idx)])

        action = np.array(
            [direction[0], direction[1], direction[2], speed, theta],
            dtype=np.float32,
        )
        return action

    def remember(self, state, action_idx, reward, next_state, done):
        self.memory.store_transition(state, action_idx, reward, next_state, done)
        self.env_step += 1
        self._update_epsilon()

    def choose_action(self, observation, deterministic=False):
        state = T.as_tensor(observation, dtype=T.float32, device=self.device).unsqueeze(0)

        with T.no_grad():
            q_dir, q_speed, q_theta = self.q_net(state)

        if deterministic:
            dir_idx = int(q_dir.argmax(dim=1).item())
            speed_idx = int(q_speed.argmax(dim=1).item())
            theta_idx = int(q_theta.argmax(dim=1).item())
        else:
            if np.random.random() < self.epsilon:
                dir_idx = np.random.randint(self.n_dir)
                speed_idx = np.random.randint(self.n_speed)
                theta_idx = np.random.randint(self.n_theta)
            else:
                dir_idx = int(q_dir.argmax(dim=1).item())
                speed_idx = int(q_speed.argmax(dim=1).item())
                theta_idx = int(q_theta.argmax(dim=1).item())

        action_idx = np.array([dir_idx, speed_idx, theta_idx], dtype=np.int64)
        action_vec = self._assemble_action(dir_idx, speed_idx, theta_idx)

        q_val = float(T.stack([
            q_dir[0, dir_idx],
            q_speed[0, speed_idx],
            q_theta[0, theta_idx],
        ]).mean().item())

        return action_vec, action_idx, q_val

    def save_models(self, name="last"):
        path = os.path.join(self.q_net.chkpt_dir, f"dqn_{name}.pt")

        checkpoint = {
            "q_net_state_dict": self.q_net.state_dict(),
            "target_q_net_state_dict": self.target_q_net.state_dict(),
            "q_optimizer_state_dict": self.q_net.optimizer.state_dict(),
            "env_step": self.env_step,
            "learn_step": self.learn_step,
            "epsilon": self.epsilon,
            "direction_table": self.direction_table,
            "speed_bins": self.speed_bins,
            "theta_bins": self.theta_bins,
            "branch_sizes": self.branch_sizes,
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

        if "direction_table" in checkpoint:
            self.direction_table = checkpoint["direction_table"]
        if "speed_bins" in checkpoint:
            self.speed_bins = checkpoint["speed_bins"]
        if "theta_bins" in checkpoint:
            self.theta_bins = checkpoint["theta_bins"]

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
            actions = T.tensor(actions, dtype=T.long, device=self.device)        # [B, 3]
            rewards = T.tensor(rewards, dtype=T.float32, device=self.device)     # [B, 1]
            next_states = T.tensor(next_states, dtype=T.float32, device=self.device)
            dones = T.tensor(dones, dtype=T.float32, device=self.device)

            q_dir, q_speed, q_theta = self.q_net(states)

            q_taken_dir = q_dir.gather(1, actions[:, 0:1])
            q_taken_speed = q_speed.gather(1, actions[:, 1:2])
            q_taken_theta = q_theta.gather(1, actions[:, 2:3])

            q_pred = T.stack(
                [q_taken_dir, q_taken_speed, q_taken_theta],
                dim=0,
            ).mean(dim=0)  # [B, 1]

            with T.no_grad():
                next_q_dir_online, next_q_speed_online, next_q_theta_online = self.q_net(next_states)

                next_dir_idx = next_q_dir_online.argmax(dim=1, keepdim=True)
                next_speed_idx = next_q_speed_online.argmax(dim=1, keepdim=True)
                next_theta_idx = next_q_theta_online.argmax(dim=1, keepdim=True)

                next_q_dir_target, next_q_speed_target, next_q_theta_target = self.target_q_net(next_states)

                next_q_dir = next_q_dir_target.gather(1, next_dir_idx)
                next_q_speed = next_q_speed_target.gather(1, next_speed_idx)
                next_q_theta = next_q_theta_target.gather(1, next_theta_idx)

                next_q = T.stack(
                    [next_q_dir, next_q_speed, next_q_theta],
                    dim=0,
                ).mean(dim=0)  # [B, 1]

                q_target = rewards + self.gamma * (1.0 - dones) * next_q

            loss = F.smooth_l1_loss(q_pred, q_target)

            self.q_net.optimizer.zero_grad()
            loss.backward()
            T.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.q_net.optimizer.step()

            self.learn_step += 1
            if self.tau is not None:
                self.update_target_network(hard=False)
            elif self.learn_step % self.target_update_interval == 0:
                self.update_target_network(hard=True)

            last_loss = float(loss.item())
            last_mean_q = float(q_pred.mean().item())

        return {
            "loss": last_loss,
            "epsilon": float(self.epsilon),
            "mean_q": last_mean_q,
        }