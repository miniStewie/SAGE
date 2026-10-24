from dataclasses import dataclass
from ppo_compat import TrainConfig


@dataclass
class PurePPOConfig(TrainConfig):
    """SAGE defaults plus direct-policy settings; None dimensions resolve at build time."""

    direct_backend: str = "numba"  # numba: accelerated; python: original reference
    C_base: float = TrainConfig.C_base
    info_dir: str = "./pure_ppo_info"
    max_packets: int | None = None
    max_visible_sats: int | None = None
    max_direct_slots: int | None = None
    max_action_combos: int | None = None
    invalid_action_strategy: str = "discard"

    # PPO parameters
    lr: float = 3e-4
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ppo_epochs: int = 10
    minibatch_size: int = 128
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    @property
    def packet_action_dim(self) -> int:
        return self.max_packets + 1

    @property
    def sat_action_dim(self) -> int:
        return self.max_visible_sats

    @property
    def slot_action_dim(self) -> int:
        return self.max_direct_slots

    @property
    def action_shape(self):
        return self.max_action_combos, 3

    @property
    def logits_dim(self) -> int:
        per_combo = self.packet_action_dim + self.sat_action_dim + self.slot_action_dim
        return self.max_action_combos * per_combo
