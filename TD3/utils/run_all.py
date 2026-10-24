import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = THIS_DIR / "checkpoints" / "td3_final.pt"

# ====== info.xlsx 保存目录 ======
INFO_DIR = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\TD3\100info"

# ====== 参数网格 ======

"""
"u/c-lambda1-lambda2+n",
"u/c-lambda1",
"u/c-lambda1-lambda2",目前来看这个比较合理
"u/c-lambda1*1-lambda2*2",
"u-nc-lambda1*1-lambda2*2",
"tu/tc-lambda1",
"tu/tc-lambda1-lambda2",
"tu/tc-lambda1-lambda2+n",
"tu/tc-lambda1-lambda2*1",
"tu/tc-lambda1*1-lambda2*2",
*1意思是非线性的固定惩罚
"""


reward_method_list = ["u/c-lambda1",]
assignment_method_list =['pssa']
beta_list = [0.5,]
seed_list = [0,1]
C_base_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10, 50, 75, 100]
#seed_list = [1]
#C_base_list = [3]
for beta in beta_list:
    for seed in seed_list:
        for method in reward_method_list:
            for assignment in assignment_method_list:
                for C_base in C_base_list:
                    subprocess.run([
                        sys.executable,
                        str(THIS_DIR / "train.py"),
                        "--beta", str(beta),
                        "--reward_method", method,
                        "--seed", str(seed),
                        "--assignment_method", assignment,
                        "--C_base", str(C_base),
                        "--info_dir", INFO_DIR,
                        "--checkpoint_path", str(CHECKPOINT_PATH),
                    ], check=True, cwd=THIS_DIR)
