import os
from typing import Dict, Iterator, Optional, Tuple

import torch
from torch import nn
from torchsummary import summary
from torch.distributions import Categorical

class Agent:
    def __init__(self, config):
        # config
        self.optimization_hpt = config.model.optimization
        self.update_hpt = config.model.sampling
        self.path = config.model.path
        self.setup(config.model.mode)

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # rollout buffer
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def mlp(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def initialize_networks(self, observation):
        obs = torch.tensor(observation, dtype=torch.float32)

        n_agents, n_animals, n_features = obs.shape

        self.n_agents = n_agents
        self.n_animals = n_animals
        self.n_features = n_features

        self.action_dim = 5  # [dx, dy, dz, speed, theta]

        self.actor_obs_dim = n_animals * n_features
        self.critic_obs_dim = n_agents * n_animals * n_features

        hidden = 128

        # Actor backbone
        self.actor_body = nn.Sequential(
            nn.Linear(self.actor_obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        ).to(self.device)

        self.actor_mu = nn.Linear(hidden, self.action_dim).to(self.device)

        # Learnable log_std (shared across agents)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, device=self.device))

        # Critic 
        self.critic = self.mlp(self.critic_obs_dim, 1).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.actor_body.parameters()) +
            list(self.actor_mu.parameters()) +
            [self.log_std] +
            list(self.critic.parameters()),
            lr=self.optimization_hpt.lr
        )

    def act(self, observation):
        obs = torch.tensor(observation, dtype=torch.float32, device=self.device)
        n_agents = obs.shape[0]

        # ACTOR 
        actor_obs = obs.reshape(n_agents, -1)
        features = self.actor_body(actor_obs)

        mu = self.actor_mu(features)
        std = torch.exp(self.log_std).expand_as(mu)
        dist = torch.distributions.Normal(mu, std)

        raw_action = dist.rsample()
        log_probs = dist.log_prob(raw_action).sum(-1)

        # CRITIC
        critic_obs = obs.reshape(1, -1)
        value = self.critic(critic_obs).squeeze(-1)
        values = value.repeat(n_agents)

        return (
            raw_action.detach(),
            log_probs.detach(),
            values.detach()
        )

    def learn(self, obs, action, reward, next_obs, done):
        ...        
    
    def clear_buffer(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add_to_buffer(self, obs, action, reward, done, log_prob, value):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        reward = torch.full((self.n_agents,), reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        self.obs.append(obs)
        self.rewards.append(reward)
        self.dones.append(done)

        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def package_buffer(self):
        gamma = self.optimization_hpt.gamma
        lam = self.optimization_hpt.gae_lambda

        obs = torch.stack(self.obs)               # (T, n_agents, ...)
        actions = torch.stack(self.actions)       # (T, n_agents, act_dim)
        rewards = torch.stack(self.rewards)       # (T, n_agents)
        dones = torch.stack(self.dones)           # (T,)
        log_probs = torch.stack(self.log_probs)   # (T, n_agents)
        values = torch.stack(self.values)         # (T, n_agents)

        T = rewards.shape[0]
        n_agents = rewards.shape[1]

        advantages = torch.zeros_like(rewards)
        last_adv = torch.zeros(n_agents, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = torch.zeros(n_agents, device=self.device)
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
            advantages[t] = last_adv

        returns = advantages + values

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return obs, actions, log_probs, returns, advantages

    def setup(self, mode="train"):
        match mode:
            case "train":
                pass
            case "eval":
                self.load(self.path)
            case _:
                raise ValueError(f"Unexpected value: {mode!r}")
    
    def load(self, path):
        """
        load model at path, use it for inference
        """
        ...

    def save(self):
        ...