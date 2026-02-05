from reward import *
from settings import *

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
EPS = 1e-6


def mlp(in_dim, out_dim, hidden=128):
    net = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )
    return net


class PPOAgent:
    """
    PPO with squashed Gaussian policy (tanh) and correct time-limit bootstrapping.

    Important expectations:
      - obs is a 1D np array of length obs_dim (recommended: 5D from Drone.observe)
      - env does NOT further distort actions beyond the same bounds (Drone.step matches action_scale)
      - rollout 'done' should be True only for *true terminals*, not time limits
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        lr=3e-4,
        gamma=0.99,
        clip=0.2,
        gae_lambda=0.95,
        value_coef=0.5,
        entropy_coef=0.01,  # small entropy helps exploration
        max_grad_norm=0.5,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.gamma = gamma
        self.clip = clip
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Action scale applied AFTER tanh squashing
        self.action_scale = torch.tensor(
            [MAX_DX, MAX_DY, MAX_DZ, MAX_DYAW],
            dtype=torch.float32,
            device=self.device,
        )
        assert self.action_scale.numel() == act_dim, "action_scale must match act_dim"

        self.actor = mlp(obs_dim, act_dim).to(self.device)
        self.critic = mlp(obs_dim, 1).to(self.device)

        # Start with slightly stochastic policy
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5, device=self.device))

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std],
            lr=lr,
        )

        self.reset_buffer()

    # -------------------------

    def save(self, path="ppo_agent.pt"):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "log_std": self.log_std.detach().cpu(),
            },
            path,
        )

    def load(self, path="ppo_agent.pt"):
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.log_std.data = data["log_std"].to(self.device)

    # -------------------------

    def reset_buffer(self):
        self.obs_buf = []
        self.act_buf = []
        self.logp_buf = []
        self.rew_buf = []
        self.val_buf = []
        self.done_buf = []
        self.bootstrap_value = 0.0  # V(s_T) for time-limit truncation

    # -------------------------
    # Squashed Gaussian helpers
    # -------------------------

    def _get_std(self):
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        return torch.exp(log_std)

    def _policy(self, obs_t):
        mean = self.actor(obs_t)
        std = self._get_std()
        dist = torch.distributions.Normal(mean, std)
        return dist

    def _squash_action_and_logp(self, pre_tanh, dist):
        """
        pre_tanh: R^act_dim
        returns:
          action: tanh(pre_tanh) * action_scale
          logp: log prob under squashed distribution
        """
        tanh_a = torch.tanh(pre_tanh)
        action = tanh_a * self.action_scale

        # log p(u)
        logp_u = dist.log_prob(pre_tanh).sum(dim=-1)

        # change-of-variables correction: sum log(1 - tanh(u)^2)
        log_det = torch.log(1.0 - tanh_a.pow(2) + EPS).sum(dim=-1)

        logp = logp_u - log_det
        return action, logp

    def _logp_from_action(self, obs_batch, act_batch):
        """
        Compute log pi(a|s) for stored (squashed+scaled) actions.
        """
        dist = self._policy(obs_batch)

        a_scaled = torch.clamp(act_batch / self.action_scale, -1.0 + EPS, 1.0 - EPS)
        pre_tanh = 0.5 * torch.log((1 + a_scaled) / (1 - a_scaled))  # atanh

        _, logp = self._squash_action_and_logp(pre_tanh, dist)
        entropy = dist.entropy().sum(dim=-1)  # pre-squash entropy (good enough for PPO)
        return logp, entropy

    # -------------------------

    @torch.no_grad()
    def act(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"Bad obs shape: {obs.shape}, expected ({self.obs_dim},)")

        obs_t = torch.from_numpy(obs).to(self.device)

        dist = self._policy(obs_t)

        # Sampling for exploration (no need for rsample during data collection)
        pre_tanh = dist.sample()
        action_t, logp_t = self._squash_action_and_logp(pre_tanh, dist)

        value_t = self.critic(obs_t).squeeze(-1)

        return (
            action_t.cpu().numpy(),
            logp_t.cpu(),
            value_t.cpu(),
        )

    @torch.no_grad()
    def act_deterministic(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        obs_t = torch.from_numpy(obs).to(self.device)

        mean = self.actor(obs_t)
        action = torch.tanh(mean) * self.action_scale
        return action.cpu().numpy()

    # -------------------------

    def store(self, obs, act, logp, val, rew, done):
        self.obs_buf.append(np.asarray(obs, dtype=np.float32))
        self.act_buf.append(np.asarray(act, dtype=np.float32))
        self.logp_buf.append(float(logp.item()))
        self.val_buf.append(float(val.item()))
        self.rew_buf.append(float(rew))
        self.done_buf.append(bool(done))

    def finalize_rollout(self, last_obs, last_done):
        """
        Call after finishing a rollout to set bootstrap_value correctly.
        For time-limit truncation: last_done should be False, and we bootstrap with V(last_obs).
        For true terminal: last_done should be True, bootstrap_value = 0.
        """
        if last_done:
            self.bootstrap_value = 0.0
            return

        with torch.no_grad():
            o = torch.from_numpy(np.asarray(last_obs, dtype=np.float32)).to(self.device)
            self.bootstrap_value = float(self.critic(o).squeeze(-1).cpu().item())

    # -------------------------

    def compute_returns_and_advs(self):
        """
        GAE(lambda) with correct bootstrap at end of rollout.
        Resets at true terminals (done=True).
        """
        T = len(self.rew_buf)
        rews = np.asarray(self.rew_buf, dtype=np.float32)
        vals = np.asarray(self.val_buf, dtype=np.float32)
        dones = np.asarray(self.done_buf, dtype=np.bool_)

        advs = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)

        last_gae = 0.0
        next_val = float(self.bootstrap_value)

        for t in reversed(range(T)):
            mask = 0.0 if dones[t] else 1.0
            delta = rews[t] + self.gamma * next_val * mask - vals[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae * mask

            advs[t] = last_gae
            returns[t] = advs[t] + vals[t]

            # For the next step backward, V_{t+1} is either V_t (if not terminal) or 0 (if terminal)
            next_val = 0.0 if dones[t] else vals[t]

        return returns, advs

    # -------------------------

    def rollout_episode(self, animal, drone, logger, ep=1, steps=STEPS, train=False):
        animal.reset()
        drone.reset((animal.x, animal.y, animal.z))

        obs = drone.observe(animal)

        ep_reward = 0.0
        ep_monitor = 0.0
        ep_disturb = 0.0

        for t in range(1, steps + 1):
            if train:
                action, logp, val = self.act(obs)
            else:
                action = self.act_deterministic(obs)
                logp, val = None, None

            drone.step(action)
            drone.z = max(0.0, drone.z)

            animal.step()
            next_obs = drone.observe(animal)

            animal_pos = (animal.x, animal.y, animal.z)
            drone_pos  = (drone.x, drone.y, drone.z)

            reward, monitoring_r, disturbance_pen = compute_total_reward(
                next_obs, animal_pos, drone_pos, obs
            )

            # Time-limit truncation is NOT a true terminal:
            done = False

            if train:
                self.store(obs, action, logp, val, reward, done)

            step_id = (ep - 1) * steps + (t - 1)

            logger.write(
                CSV_PATH,
                ep,
                step_id,
                "PIGEON",
                (animal.x, animal.y, animal.z, 0),
                0,
                0,
                0,
            )

            logger.write(
                CSV_PATH,
                ep,
                step_id,
                "DRONE",
                (drone.x, drone.y, drone.z, drone.yaw),
                reward,
                monitoring_r,
                disturbance_pen,
            )

            ep_reward += reward
            ep_monitor += monitoring_r
            ep_disturb += disturbance_pen

            obs = next_obs

        if train:
            # Bootstrap at time limit
            self.finalize_rollout(last_obs=obs, last_done=False)

        return ep_reward, ep_monitor, ep_disturb

    # -------------------------

    def update(self, epochs=5, batch_size=64):
        if len(self.obs_buf) == 0:
            return

        obs = torch.from_numpy(np.asarray(self.obs_buf, dtype=np.float32)).to(self.device)
        act = torch.from_numpy(np.asarray(self.act_buf, dtype=np.float32)).to(self.device)
        old_logp = torch.from_numpy(np.asarray(self.logp_buf, dtype=np.float32)).to(self.device)

        returns, advs = self.compute_returns_and_advs()
        returns = torch.from_numpy(returns).to(self.device)
        advs = torch.from_numpy(advs).to(self.device)

        # Normalize advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        N = obs.shape[0]
        batch_size = min(int(batch_size), int(N))

        for _ in range(int(epochs)):
            idx = np.random.permutation(N)

            for start in range(0, N, batch_size):
                batch = idx[start:start + batch_size]

                o = obs[batch]
                a = act[batch]
                ret = returns[batch]
                adv = advs[batch]
                old_lp = old_logp[batch]

                logp, entropy = self._logp_from_action(o, a)
                ratio = torch.exp(logp - old_lp)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value = self.critic(o).squeeze(-1)
                value_loss = 0.5 * (ret - value).pow(2).mean()

                ent_bonus = entropy.mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * ent_bonus

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std],
                    self.max_grad_norm,
                )
                self.optimizer.step()

        self.reset_buffer()
