import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PPOAgent:

    def __init__(self,
                 obs_dim,
                 act_dim,
                 lr=3e-4,
                 gamma=0.99,
                 clip=0.2):

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.gamma = gamma
        self.clip = clip

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) +
            list(self.critic.parameters()),
            lr=lr
        )

        # Log std (for continuous actions)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Rollout buffer
        self.reset_buffer()

    # ------------------------------

    def reset_buffer(self):

        self.obs_buf = []
        self.act_buf = []
        self.logp_buf = []
        self.rew_buf = []
        self.val_buf = []
        self.done_buf = []

    # ------------------------------

    def act(self, obs):

        obs = torch.tensor(obs, dtype=torch.float32)

        mean = self.actor(obs)
        std = torch.exp(self.log_std)

        dist = torch.distributions.Normal(mean, std)

        action = dist.sample()
        logp = dist.log_prob(action).sum()

        value = self.critic(obs)

        return (
            action.detach().numpy(),
            logp.detach(),
            value.detach()
        )

    # ------------------------------

    def store(self, obs, act, logp, val, rew, done):

        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.logp_buf.append(logp)
        self.val_buf.append(val)
        self.rew_buf.append(rew)
        self.done_buf.append(done)

    # ------------------------------

    def compute_returns(self, last_val=0):

        returns = []
        advs = []

        ret = last_val
        adv = 0

        for i in reversed(range(len(self.rew_buf))):

            ret = self.rew_buf[i] + self.gamma * ret
            delta = self.rew_buf[i] + self.gamma * last_val - self.val_buf[i]
            adv = delta + self.gamma * adv

            returns.insert(0, ret)
            advs.insert(0, adv)

            last_val = self.val_buf[i]

        return returns, advs

    # ------------------------------

    def update(self, epochs=5, batch_size=64):

        obs = torch.tensor(self.obs_buf, dtype=torch.float32)
        act = torch.tensor(self.act_buf, dtype=torch.float32)
        old_logp = torch.stack(self.logp_buf).detach()
        val = torch.stack(self.val_buf).detach()

        returns, advs = self.compute_returns()
        returns = torch.tensor(returns, dtype=torch.float32)
        advs = torch.tensor(advs, dtype=torch.float32)

        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for _ in range(epochs):

            idx = np.random.permutation(len(obs))

            for i in range(0, len(obs), batch_size):

                batch = idx[i:i+batch_size]

                o = obs[batch]
                a = act[batch]
                r = returns[batch]
                adv = advs[batch]
                old_lp = old_logp[batch]

                mean = self.actor(o)
                std = torch.exp(self.log_std)

                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(a).sum(axis=1)

                ratio = torch.exp(logp - old_lp)

                clip_adv = torch.clamp(
                    ratio,
                    1-self.clip,
                    1+self.clip
                ) * adv

                policy_loss = -torch.min(
                    ratio * adv,
                    clip_adv
                ).mean()

                value = self.critic(o).squeeze()
                value_loss = (r - value).pow(2).mean()

                loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.reset_buffer()
