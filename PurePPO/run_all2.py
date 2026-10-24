import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
INFO_DIR = "C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\PurePPO\pure_ppo_info_beijing\\59"

from ppo_compat import comparison_grid

reward_method_list = ["u/c-lambda1",]
beta_list = [0.5,]
seed_list = [5,6,7,8,9]
C_base_list = [1.0, 3.0,  5.0,  7.0]

for beta in beta_list:
    for seed in seed_list:
        for reward_method in reward_method_list:
            for C_base in C_base_list:
                subprocess.run(
                    [
                        sys.executable,
                        str(THIS_DIR / "train.py"),
                        "--beta",
                        str(beta),
                        "--reward_method",
                        reward_method,
                        "--seed",
                        str(seed),
                        "--C_base",
                        str(C_base),
                        "--info_dir",
                        str(INFO_DIR),
                    ],
                    check=True,
                )
