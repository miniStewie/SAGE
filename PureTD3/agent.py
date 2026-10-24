from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim: Tuple[int, ...], out_act=None):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dim:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_act = out_act

    def forward(self, x):
        y = self.net(x)
        if self.out_act is not None:
            y = self.out_act(y)
        return y


class Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_dim: Tuple[int, ...] = (256, 256, 256),
        out_act=nn.Tanh(),
    ):
        super().__init__()
        self.mlp = MLP(state_dim, act_dim, hidden_dim, out_act)

    def forward(self, state):
        return self.mlp(state)


class Critic(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim: Tuple[int, ...] = (256, 256, 256)):
        super().__init__()
        self.q1 = MLP(state_dim + act_dim, 1, hidden_dim)
        self.q2 = MLP(state_dim + act_dim, 1, hidden_dim)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.len = 0
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.len = min(self.len + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.len, size=batch_size)
        return (
            torch.tensor(self.states[idx], dtype=torch.float32),
            torch.tensor(self.actions[idx], dtype=torch.float32),
            torch.tensor(self.rewards[idx], dtype=torch.float32),
            torch.tensor(self.next_states[idx], dtype=torch.float32),
            torch.tensor(self.dones[idx], dtype=torch.float32),
        )


@dataclass
class TD3config:
    state_dim: int
    action_dim: int
    actor_lr: float
    critic_lr: float
    gamma: float
    tau: float
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    batch_size: int = 128
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3Agent:
    def __init__(self, cfg: TD3config):
        self.cfg = cfg
        self.actor = Actor(cfg.state_dim, cfg.action_dim).to(cfg.device)
        self.actor_tgt = Actor(cfg.state_dim, cfg.action_dim).to(cfg.device)
        self.actor_tgt.load_state_dict(self.actor.state_dict())

        self.critic = Critic(cfg.state_dim, cfg.action_dim).to(cfg.device)
        self.critic_tgt = Critic(cfg.state_dim, cfg.action_dim).to(cfg.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.total_updates = 0

    @torch.no_grad()
    def act(self, state: np.ndarray, noise_std: float = 0.1):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
        action = self.actor(state_t).cpu().numpy().squeeze(0)
        if noise_std > 0:
            action += np.random.normal(0, noise_std, size=action.shape)
        return np.clip(action, -1, 1)

    def update(self, replay: ReplayBuffer):
        self.total_updates += 1
        cfg = self.cfg

        s, a, r, s2, d = replay.sample(cfg.batch_size)
        s, a, r, s2, d = (t.to(cfg.device) for t in (s, a, r, s2, d))

        with torch.no_grad():
            noise = (torch.randn_like(a) * cfg.policy_noise).clamp(
                -cfg.noise_clip, cfg.noise_clip
            )
            a2 = (self.actor_tgt(s2) + noise).clamp(-1.0, 1.0)
            q1_tgt, q2_tgt = self.critic_tgt(s2, a2)
            y = r + cfg.gamma * (1.0 - d) * torch.min(q1_tgt, q2_tgt)

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        if self.total_updates % cfg.policy_delay == 0:
            a_pi = self.actor(s)
            q1_pi, _ = self.critic(s, a_pi)
            actor_loss = -torch.mean(q1_pi)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            with torch.no_grad():
                for p, p_t in zip(self.actor.parameters(), self.actor_tgt.parameters()):
                    p_t.data.mul_(1.0 - cfg.tau).add_(p, alpha=cfg.tau)
                for p, p_t in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                    p_t.data.mul_(1.0 - cfg.tau).add_(p, alpha=cfg.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "q1_mean": float(q1.mean().item()),
            "q2_mean": float(q2.mean().item()),
            "y_tgt": float(y.mean().item()),
        }


def build_agent_cfg(cfg, state_dim: int) -> TD3config:
    return TD3config(
        state_dim=state_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        tau=cfg.tau,
        policy_noise=cfg.policy_noise,
        noise_clip=cfg.noise_clip,
        policy_delay=cfg.policy_delay,
        batch_size=cfg.batch_size,
        device=cfg.device,
    )
