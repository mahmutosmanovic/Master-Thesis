from settings import *


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
EPS = 1e-6


def mlp(in_dim, out_dim, hidden=128):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class PPOAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        lr=3e-4,
        gamma=0.99,
        clip=0.2,
        gae_lambda=0.95,
        value_coef=0.5,
        entropy_coef=0.0,
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

        # Device (must come BEFORE tensors that use self.device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Action scale (applied AFTER tanh squashing)
        self.action_scale = torch.tensor(
            [MAX_DX, MAX_DY, MAX_DZ, MAX_DYAW],
            dtype=torch.float32,
            device=self.device,
        )
        assert self.action_scale.numel() == act_dim, "action_scale must match act_dim"

        # Actor outputs mean in R^act_dim (we tanh-squash samples later)
        self.actor = mlp(obs_dim, act_dim).to(self.device)

        # Critic outputs scalar V(s)
        self.critic = mlp(obs_dim, 1).to(self.device)

        # Trainable log std (state-independent)
        self.log_std = nn.Parameter(torch.zeros(act_dim, device=self.device))

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) +
            list(self.critic.parameters()) +
            [self.log_std],
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

    # -------------------------
    # Squashed Gaussian helpers
    # -------------------------

    def _get_std(self):
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        return torch.exp(log_std)

    def _squash_action_and_logp(self, pre_tanh, normal_dist):
        """
        pre_tanh: sampled action in R^act_dim
        action: tanh(pre_tanh) * scale
        logp: log prob of squashed action (change of variables)
        """
        tanh_a = torch.tanh(pre_tanh)
        action = tanh_a * self.action_scale

        # logp(pre_tanh) from Normal, sum over dims
        logp = normal_dist.log_prob(pre_tanh).sum(dim=-1)

        # Change-of-variables correction for tanh:
        # log |det(d tanh(u) / du)| = sum log(1 - tanh(u)^2)
        # We subtract it because: logp(a) = logp(u) - log|det(J)|
        log_det = torch.log(1.0 - tanh_a.pow(2) + EPS).sum(dim=-1)
        logp = logp - log_det

        return action, logp

    def _policy(self, obs_t):
        """
        Returns:
          dist (Normal), mean, std
        """
        mean = self.actor(obs_t)
        std = self._get_std()
        dist = torch.distributions.Normal(mean, std)
        return dist, mean, std

    # -------------------------

    def act(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"Bad obs shape: {obs.shape}, expected ({self.obs_dim},)")

        obs_t = torch.from_numpy(obs).to(self.device)

        dist, _, _ = self._policy(obs_t)

        # Reparameterized sample is usually nicer
        pre_tanh = dist.rsample()
        action_t, logp_t = self._squash_action_and_logp(pre_tanh, dist)

        value_t = self.critic(obs_t).squeeze(-1)

        return (
            action_t.detach().cpu().numpy(),
            logp_t.detach().cpu(),
            value_t.detach().cpu(),
        )

    @torch.no_grad()
    def act_deterministic(self, obs):
        obs = np.array(obs, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"Bad obs shape: {obs.shape}, obs={obs}")

        obs_t = torch.from_numpy(obs).to(self.device)

        # If your actor ends with Tanh, mean is already bounded [-1,1]
        mean = self.actor(obs_t) * self.action_scale
        return mean.detach().cpu().numpy()
    
    # -------------------------

    def store(self, obs, act, logp, val, rew, done):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.logp_buf.append(logp.detach().cpu())
        self.val_buf.append(val.detach().cpu())
        self.rew_buf.append(float(rew))
        self.done_buf.append(bool(done))

    # -------------------------

    def compute_returns_and_advs(self):
        """
        GAE(lambda) with bootstrap = 0 (or episode terminal)
        """
        T = len(self.rew_buf)

        returns = np.zeros(T, dtype=np.float32)
        advs = np.zeros(T, dtype=np.float32)

        last_gae = 0.0
        last_val = 0.0

        for t in reversed(range(T)):
            r = self.rew_buf[t]
            v = float(self.val_buf[t].item())

            done = self.done_buf[t]
            mask = 0.0 if done else 1.0

            # delta_t = r_t + gamma * V_{t+1} - V_t
            delta = r + self.gamma * last_val * mask - v

            # GAE
            last_gae = delta + self.gamma * self.gae_lambda * last_gae * mask

            advs[t] = last_gae
            returns[t] = advs[t] + v

            last_val = v

            if done:
                last_gae = 0.0
                last_val = 0.0

        return returns, advs

    # -------------------------

    def update(self, epochs=5, batch_size=64):
        if len(self.obs_buf) == 0:
            return

        obs = torch.from_numpy(np.asarray(self.obs_buf, dtype=np.float32)).to(self.device)
        act = torch.from_numpy(np.asarray(self.act_buf, dtype=np.float32)).to(self.device)
        old_logp = torch.stack(self.logp_buf).to(self.device)

        returns, advs = self.compute_returns_and_advs()
        returns = torch.from_numpy(returns).to(self.device)
        advs = torch.from_numpy(advs).to(self.device)

        # Normalize advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        N = obs.shape[0]
        batch_size = min(batch_size, N)

        for _ in range(epochs):
            idx = np.random.permutation(N)

            for start in range(0, N, batch_size):
                batch = idx[start:start + batch_size]

                o = obs[batch]
                a = act[batch]
                ret = returns[batch]
                adv = advs[batch]
                old_lp = old_logp[batch]

                # Policy distribution
                dist, _, _ = self._policy(o)

                # We need logp of *the stored action a*.
                # Since a is already squashed+scaled, we invert:
                #   tanh(u) = a/scale  -> u = atanh(a/scale)
                a_scaled = torch.clamp(a / self.action_scale, -1.0 + EPS, 1.0 - EPS)
                pre_tanh = 0.5 * torch.log((1 + a_scaled) / (1 - a_scaled))  # atanh

                _, logp = self._squash_action_and_logp(pre_tanh, dist)

                ratio = torch.exp(logp - old_lp)

                clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
                policy_loss = -(torch.min(ratio * adv, clipped * adv)).mean()

                value = self.critic(o).squeeze(-1)
                value_loss = (ret - value).pow(2).mean()

                # Entropy bonus (optional)
                entropy = dist.entropy().sum(dim=-1).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std],
                    self.max_grad_norm,
                )

                self.optimizer.step()

        self.reset_buffer()
