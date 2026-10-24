from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim: Tuple[int, ...]):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dim:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        max_action_combos,
        packet_action_dim,
        sat_action_dim,
        slot_action_dim,
        hidden_dim: Tuple[int, ...] = (256, 256, 256),
    ):
        super().__init__()
        self.max_action_combos = max_action_combos
        self.packet_action_dim = packet_action_dim
        self.sat_action_dim = sat_action_dim
        self.slot_action_dim = slot_action_dim
        self.logits_dim = max_action_combos * (
            packet_action_dim + sat_action_dim + slot_action_dim
        )

        self.policy = MLP(state_dim, self.logits_dim, hidden_dim)
        self.value = MLP(state_dim, 1, hidden_dim)

    def forward(self, state):
        logits = self.policy(state)
        value = self.value(state).squeeze(-1)
        return logits, value

    def split_logits(self, logits):
        bsz = logits.shape[0]
        per_combo = self.packet_action_dim + self.sat_action_dim + self.slot_action_dim
        logits = logits.view(bsz, self.max_action_combos, per_combo)

        p_end = self.packet_action_dim
        s_end = p_end + self.sat_action_dim
        packet_logits = logits[:, :, :p_end]
        sat_logits = logits[:, :, p_end:s_end]
        slot_logits = logits[:, :, s_end:]
        return packet_logits, sat_logits, slot_logits

    def get_action_and_value(self, state, action=None):
        logits, value = self.forward(state)
        packet_logits, sat_logits, slot_logits = self.split_logits(logits)

        packet_dist = Categorical(logits=packet_logits)
        sat_dist = Categorical(logits=sat_logits)
        slot_dist = Categorical(logits=slot_logits)

        if action is None:
            packet_action = packet_dist.sample()
            sat_action = sat_dist.sample()
            slot_action = slot_dist.sample()
            action = torch.stack((packet_action, sat_action, slot_action), dim=-1)
        else:
            packet_action = action[:, :, 0]
            sat_action = action[:, :, 1]
            slot_action = action[:, :, 2]

        logprob = (
            packet_dist.log_prob(packet_action)
            + sat_dist.log_prob(sat_action)
            + slot_dist.log_prob(slot_action)
        ).sum(dim=-1)
        entropy = (
            packet_dist.entropy() + sat_dist.entropy() + slot_dist.entropy()
        ).sum(dim=-1)

        return action, logprob, entropy, value


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, state, action, logprob, reward, done, value):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.int64))
        self.logprobs.append(float(logprob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def __len__(self):
        return len(self.rewards)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()


@dataclass
class PPOAgentConfig:
    state_dim: int
    max_action_combos: int
    packet_action_dim: int
    sat_action_dim: int
    slot_action_dim: int
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ppo_epochs: int = 10
    minibatch_size: int = 128
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOAgent:
    def __init__(self, cfg: PPOAgentConfig):
        self.cfg = cfg
        self.model = ActorCritic(
            state_dim=cfg.state_dim,
            max_action_combos=cfg.max_action_combos,
            packet_action_dim=cfg.packet_action_dim,
            sat_action_dim=cfg.sat_action_dim,
            slot_action_dim=cfg.slot_action_dim,
        ).to(cfg.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def act(self, state: np.ndarray):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
        action, logprob, entropy, value = self.model.get_action_and_value(state_t)
        return (
            action.squeeze(0).cpu().numpy().astype(np.int64),
            float(logprob.item()),
            float(value.item()),
            float(entropy.item()),
        )

    def update(self, rollout: RolloutBuffer):
        if len(rollout) == 0:
            return {
                "policy_loss": np.nan,
                "value_loss": np.nan,
                "entropy": np.nan,
                "approx_kl": np.nan,
                "clipfrac": np.nan,
            }

        cfg = self.cfg
        states_np = np.asarray(rollout.states, dtype=np.float32)
        actions_np = np.asarray(rollout.actions, dtype=np.int64)
        old_logprobs_np = np.asarray(rollout.logprobs, dtype=np.float32)
        rewards_np = np.asarray(rollout.rewards, dtype=np.float32)
        dones_np = np.asarray(rollout.dones, dtype=np.float32)
        values_np = np.asarray(rollout.values, dtype=np.float32)

        advantages_np = np.zeros_like(rewards_np, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards_np))):
            if t == len(rewards_np) - 1:
                next_nonterminal = 1.0 - dones_np[t]
                next_value = 0.0
            else:
                next_nonterminal = 1.0 - dones_np[t]
                next_value = values_np[t + 1]
            delta = rewards_np[t] + cfg.gamma * next_value * next_nonterminal - values_np[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
            advantages_np[t] = last_gae
        returns_np = advantages_np + values_np

        states = torch.tensor(states_np, dtype=torch.float32, device=cfg.device)
        actions = torch.tensor(actions_np, dtype=torch.long, device=cfg.device)
        old_logprobs = torch.tensor(old_logprobs_np, dtype=torch.float32, device=cfg.device)
        advantages = torch.tensor(advantages_np, dtype=torch.float32, device=cfg.device)
        returns = torch.tensor(returns_np, dtype=torch.float32, device=cfg.device)

        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = states.shape[0]
        minibatch_size = min(cfg.minibatch_size, batch_size)
        losses = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clipfrac": [],
        }

        indices = np.arange(batch_size)
        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                mb_idx_t = torch.tensor(mb_idx, dtype=torch.long, device=cfg.device)

                _, new_logprob, entropy, new_value = self.model.get_action_and_value(
                    states[mb_idx_t],
                    actions[mb_idx_t],
                )
                logratio = new_logprob - old_logprobs[mb_idx_t]
                ratio = logratio.exp()

                mb_adv = advantages[mb_idx_t]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio,
                    1.0 - cfg.clip_coef,
                    1.0 + cfg.clip_coef,
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = F.mse_loss(new_value, returns[mb_idx_t])
                entropy_loss = entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()

                losses["policy_loss"].append(float(policy_loss.item()))
                losses["value_loss"].append(float(value_loss.item()))
                losses["entropy"].append(float(entropy_loss.item()))
                losses["approx_kl"].append(float(approx_kl.item()))
                losses["clipfrac"].append(float(clipfrac.item()))

        return {key: float(np.mean(val)) for key, val in losses.items()}


def build_agent_cfg(cfg, state_dim: int) -> PPOAgentConfig:
    return PPOAgentConfig(
        state_dim=state_dim,
        max_action_combos=cfg.max_action_combos,
        packet_action_dim=cfg.packet_action_dim,
        sat_action_dim=cfg.sat_action_dim,
        slot_action_dim=cfg.slot_action_dim,
        lr=cfg.lr,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_coef=cfg.clip_coef,
        ppo_epochs=cfg.ppo_epochs,
        minibatch_size=cfg.minibatch_size,
        entropy_coef=cfg.entropy_coef,
        value_coef=cfg.value_coef,
        max_grad_norm=cfg.max_grad_norm,
        device=cfg.device,
    )
