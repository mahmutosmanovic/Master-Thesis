import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0.0)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.v  = nn.Linear(hidden, 1)

        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2)))
        orthogonal_init(self.mu, gain=0.01)
        orthogonal_init(self.v, gain=1.0)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        v = self.v(h).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std, v

def tanh_gaussian_sample(mu, std):
    eps = torch.randn_like(mu)
    pre_tanh = mu + std * eps
    a = torch.tanh(pre_tanh)
    return a, pre_tanh

def tanh_gaussian_logprob(mu, std, pre_tanh, a):
    var = std.pow(2)
    logp = -0.5 * (((pre_tanh - mu).pow(2) / (var + 1e-8)) + 2*torch.log(std + 1e-8) + np.log(2*np.pi))
    logp = logp.sum(-1)

    correction = torch.log(1 - a.pow(2) + 1e-6).sum(-1)
    return logp - correction

class RolloutBuffer:
    def __init__(self, T, obs_dim, act_dim, device):
        self.T = T
        self.device = device
        self.reset()

    def reset(self):
        self.ptr = 0

        self.obs = torch.zeros((self.T, 0), device=self.device) # Placeholder init
        self.act = torch.zeros((self.T, 0), device=self.device)
    
    def init_storage(self, obs_dim, act_dim):
        if self.obs.shape[1] != obs_dim:
            self.obs = torch.zeros((self.T, obs_dim), device=self.device)
            self.act = torch.zeros((self.T, act_dim), device=self.device)
            self.raw_act = torch.zeros((self.T, act_dim), device=self.device) # <--- ADDED
            self.logp = torch.zeros((self.T,), device=self.device)
            self.rew = torch.zeros((self.T,), device=self.device)
            self.done = torch.zeros((self.T,), device=self.device)
            self.val = torch.zeros((self.T,), device=self.device)

    def add(self, obs, act, raw_act, logp, rew, done, val):
        if self.ptr == 0:
            self.init_storage(obs.shape[0], act.shape[0])

        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.raw_act[i] = raw_act # <--- Store raw pre-tanh
        self.logp[i] = logp
        self.rew[i] = rew
        self.done[i] = done
        self.val[i] = val
        self.ptr += 1

@torch.no_grad()
def compute_gae(rew, done, val, last_val, gamma=0.99, lam=0.95):
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = last_val if t == T-1 else val[t+1]
        nonterminal = 1.0 - done[t]
        delta = rew[t] + gamma * next_val * nonterminal - val[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    ret = adv + val
    return adv, ret

class PPO:
    def __init__(
        self,
        obs_dim,
        act_dim,
        lr=3e-4,
        gamma=0.95,
        lam=0.95,
        clip=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.ac = ActorCritic(obs_dim, act_dim).to(self.device)
        self.opt = optim.Adam(self.ac.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    @torch.no_grad()
    def act(self, obs_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        mu, std, v = self.ac(obs)
        a, pre_tanh = tanh_gaussian_sample(mu, std)
        logp = tanh_gaussian_logprob(mu, std, pre_tanh, a)

        return (
            a.squeeze(0).cpu().numpy(), 
            pre_tanh.squeeze(0).cpu().numpy(), 
            logp.squeeze(0).cpu().item(), 
            v.squeeze(0).cpu().item()
        )

    def update(self, buf, last_val, epochs=10, batch_size=64):
        adv, ret = compute_gae(buf.rew, buf.done, buf.val, last_val, self.gamma, self.lam)
        
        # Normalize advantages (Critical for stability)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = buf.obs
        act = buf.act
        raw_act = buf.raw_act
        old_logp = buf.logp
        old_val = buf.val

        N = obs.shape[0]
        idx = torch.arange(N, device=self.device)

        for _ in range(epochs):
            perm = idx[torch.randperm(N)]
            for start in range(0, N, batch_size):
                mb = perm[start:start+batch_size]

                mu, std, v = self.ac(obs[mb])
                
                curr_logp = tanh_gaussian_logprob(mu, std, raw_act[mb], act[mb])

                ratio = torch.exp(curr_logp - old_logp[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv[mb]
                pi_loss = -(torch.min(surr1, surr2)).mean()

                vf_loss = 0.5 * (v - ret[mb]).pow(2).mean() 

                ent = (0.5 + 0.5*np.log(2*np.pi) + torch.log(std + 1e-8)).sum(-1).mean()
                
                loss = pi_loss + self.vf_coef * vf_loss - self.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.opt.step()