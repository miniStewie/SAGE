from dataclasses import dataclass
from td3_compat import TrainConfig


@dataclass
class PureTD3Config(TrainConfig):
    """SAGE defaults plus direct-policy settings; None dimensions resolve at build time."""

    direct_backend: str = "numba"  # numba: accelerated; python: original reference
    C_base: float = TrainConfig.C_base
    info_dir: str = "./pure_td3_info"
    max_packets: int | None = None
    max_visible_sats: int | None = None
    max_direct_slots: int | None = None
    max_action_combos: int | None = None
