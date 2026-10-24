from pathlib import Path
import importlib.util
import sys


TD3_DIR = Path(__file__).resolve().parent.parent / "TD3"


def _load_td3_module(module_name: str, filename: str):
    module_path = TD3_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_train_config = _load_td3_module("_puretd3_train_config", "train_config.py")
_reward = _load_td3_module("_puretd3_reward", "reward.py")
_compute_sats_info = _load_td3_module("_puretd3_compute_sats_info", "utils/compute_sats_info.py")
_save_info = _load_td3_module("_puretd3_save_info", "utils/save_info.py")

set_random_seed = _train_config.set_random_seed

reward_fn = _reward._reward

satellite_info_dict = _compute_sats_info.satellite_info_dict

print_avg_info_to_excel = _save_info.print_avg_info_to_excel
print_info_to_excel = _save_info.print_info_to_excel
save_to_excel = _save_info.save_to_excel

_physics = _load_td3_module("_puretd3_physics", "../baseline_environment.py")
EnvironmentPhysics = _physics.EnvironmentPhysics
Satellite = _physics.Satellite
align_direct_dimensions = _physics.align_direct_dimensions
TrainConfig = _train_config.TrainConfig

comparison_grid = _physics.comparison_grid

_compiled = sys.modules.get("baseline_compiled") or _load_td3_module("baseline_compiled", "utils/baseline_compiled.py")
select_actions = _compiled.select_actions
NUMBA_AVAILABLE = _compiled.NUMBA_AVAILABLE

make_state_fast = _compiled.make_state
