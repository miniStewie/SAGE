from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#全连接神经网络
class MLP(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_dim:Tuple[int, ...],
                 out_act = None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
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
    def __init__(self,state_dim, act_dim=3,hidden_dim:Tuple[int, ...] = (256, 256, 256), out_act = nn.Tanh()):
        super().__init__()
        self.mlp = MLP(in_dim=state_dim, out_dim=act_dim, hidden_dim=hidden_dim, out_act=out_act)

    def forward(self, state):
        return self.mlp(state)

class Critic(nn.Module):
    def __init__(self, state_dim, act_dim=3, hidden_dim: Tuple[int, ...] = (256, 256, 256)):
        super().__init__()
        self.q1 = MLP(in_dim=state_dim+act_dim, out_dim=1, hidden_dim=hidden_dim)
        self.q2 = MLP(in_dim=state_dim+act_dim, out_dim=1, hidden_dim=hidden_dim)

    def forward(self, state, action):
        x = torch.cat(tensors=(state, action), dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2

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
        state_dim = self.cfg.state_dim
        action_dim = self.cfg.action_dim

        self.actor = Actor(state_dim, action_dim).to(cfg.device)
        self.actor_tgt = Actor(state_dim, action_dim).to(cfg.device)
        self.actor_tgt.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(cfg.device)
        self.critic_tgt = Critic(state_dim, action_dim).to(cfg.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.total_updates = 0

    @torch.no_grad()
    def act(self, state: np.ndarray, noise_std: float = 0.1):
        state = torch.tensor(state, dtype=torch.float32, device= self.cfg.device).unsqueeze(0)
        action = self.actor(state).cpu().numpy().squeeze(0)
        #添加噪声
        if noise_std > 0:
            action += np.random.normal(0, noise_std, size=action.shape)
        return np.clip(action, -1, 1)

    def save(self, checkpoint_path, metadata=None):
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": 1,
            "state_dim": int(self.cfg.state_dim),
            "action_dim": int(self.cfg.action_dim),
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_tgt.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_tgt.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_updates": int(self.total_updates),
            "metadata": metadata or {},
        }, path)
        return str(path.resolve())

    def load(self, checkpoint_path, load_optimizers=False):
        path = Path(checkpoint_path)
        try:
            checkpoint = torch.load(path, map_location=self.cfg.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.cfg.device)

        state_dim = int(checkpoint.get("state_dim", self.cfg.state_dim))
        action_dim = int(checkpoint.get("action_dim", self.cfg.action_dim))
        if state_dim != self.cfg.state_dim or action_dim != self.cfg.action_dim:
            raise ValueError(
                f"checkpoint dimensions ({state_dim}, {action_dim}) do not match "
                f"agent dimensions ({self.cfg.state_dim}, {self.cfg.action_dim})"
            )

        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_tgt.load_state_dict(checkpoint.get("actor_target", checkpoint["actor"]))
        if "critic" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic"])
        if "critic_target" in checkpoint:
            self.critic_tgt.load_state_dict(checkpoint["critic_target"])
        elif "critic" in checkpoint:
            self.critic_tgt.load_state_dict(checkpoint["critic"])
        if load_optimizers:
            if "actor_optimizer" in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            if "critic_optimizer" in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_updates = int(checkpoint.get("total_updates", 0))
        self.actor.eval()
        self.actor_tgt.eval()
        return checkpoint.get("metadata", {})

    def update(self, replay: ReplayBuffer):
        self.total_updates += 1
        cfg = self.cfg

        s, a, r, s2, d = replay.sample(cfg.batch_size)
        s, a, r, s2, d = (t.to(cfg.device) for t in (s, a, r, s2, d))

        with torch.no_grad():
            noise = (torch.randn_like(a) * cfg.policy_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
            a2 = (self.actor_tgt(s2) + noise).clamp(-1.0, 1.0)
            q1_tgt, q2_tgt = self.critic_tgt(s2, a2)
            qtg = torch.min(q1_tgt, q2_tgt)
            y = r + cfg.gamma * (1.0 - d) * qtg

        q1, q2 = self.critic(s, a)
        # critic_loss = F.smooth_l1_loss(q1, y) + F.smooth_l1_loss(q2, y)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_loss = None
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

        actor_loss_val = np.nan
        if (self.total_updates % cfg.policy_delay) == 0:
            actor_loss_val = float(actor_loss.item())

        return {
            "critic_loss": float(critic_loss.item()),
            #"actor_loss": actor_loss_val,  # 多数 step 是 NaN（因为 policy_delay）
            # 可选：用于诊断稳定性
            "q1_mean": float(q1.mean().item()),
            "q2_mean": float(q2.mean().item()),
            "y_tgt": float(y.mean().item()),
        }
