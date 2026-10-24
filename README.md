# SAGE：太阳能供能卫星通信调度实验

本仓库提供卫星通信调度的算法实现、实验输入数据和绘图代码，用于比较不同方法在卫星可见窗口、电池能量与通信费用约束下的传输表现。

仓库地址：[miniStewie/SAGE](https://github.com/miniStewie/SAGE)。

## 算法与实验内容

| 方法 | 实现方式 | 训练入口 |
| --- | --- | --- |
| SAGE | TD3 输出能量与费用预算，PSSA 完成数据包传输分配 | `TD3/train.py` |
| Greedy 分配对比 | 在 SAGE 训练入口中切换为 Greedy 分配求解器 | `TD3/train.py --assignment_method greedy` |
| PureTD3 | TD3 直接输出数据包、卫星与时隙选择 | `PureTD3/train.py` |
| PurePPO | PPO 直接输出离散的数据包、卫星与时隙选择 | `PurePPO/train.py` |

实验记录包括 reward、传输效用、通信费用、发送量、电池状态及约束违反次数。输入数据包含北京和海南东方两个场景。

**首次运行前需要配置数据路径。** 当前训练配置、批量脚本和部分绘图脚本仍保留开发机器的绝对路径。下文命令均从仓库根目录执行；示例参数用于说明运行方法，不代表论文的全部实验设置。

## 目录结构

```text
SAGE/
├── dataset/                    # TLE、可见窗口、太阳辐射等输入数据
├── TD3/
│   ├── train.py                # SAGE 训练入口
│   ├── train_config.py         # 场景、物理参数与训练配置
│   ├── TD3Agent.py             # TD3 网络、经验回放与参数更新
│   ├── env.py                  # 通信与能量仿真环境
│   ├── AssignmentSolver.py     # PSSA、Greedy 等分配求解器
│   ├── reward.py               # 奖励函数
│   ├── checkpoints/            # 模型权重
│   └── utils/                  # 卫星参数计算、加速实现、日志与运行工具
├── PureTD3/                    # 直接动作 TD3 对比方法
├── PurePPO/                    # 直接动作 PPO 对比方法
├── baseline_environment.py     # 对比方法共用的仿真规则
├── plot/                       # 图表生成脚本与实验图片
├── requirements.txt            # Python 依赖
└── README.md
```

`baseline_environment.py` 与 `TD3/utils/` 是算法运行所需的公共代码。PureTD3 和 PurePPO 通过兼容模块加载其中的环境规则、卫星参数计算、日志与加速函数。

## 环境安装

原开发环境使用 Python 3.10.11。建议使用 Python 3.10 创建独立虚拟环境。以下为 Windows PowerShell 命令：

```powershell
git clone https://github.com/miniStewie/SAGE.git
cd SAGE
py -3.10 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install --upgrade pip
& ./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

主要依赖为 PyTorch、NumPy、Numba、Pandas、Skyfield 和 Matplotlib。`.xlsx` 与 `.xls` 文件分别需要 OpenPyXL 与 xlrd。

依赖文件固定了 `torch==2.8.0+cu128`，安装的是 CUDA 12.8 构建版本；代码默认使用 CPU。若使用 CPU 专用构建，应相应调整依赖文件中的 PyTorch 版本与安装索引。使用 GPU 时需要兼容的驱动，并设置 `TrainConfig.device_str`。

Numba 函数第一次执行时会编译，首次运行耗时可能较长。虚拟环境和编译缓存不纳入版本管理。

## 数据准备与场景设置

### 输入文件

| 文件（位于 `dataset/`） | 用途 |
| --- | --- |
| `satellite_data_50.json` | 50 颗卫星的 TLE 及相关记录 |
| `satellite_visibility_beijing_2025_05.xlsx` | 北京场景卫星可见窗口 |
| `satellite_visibility_hainan_2025_10.xlsx` | 海南东方场景卫星可见窗口 |
| `北京25年5月每时太阳辐射强度.xls` | 北京逐小时太阳辐射 |
| `海南省东方市25年10月每时太阳辐射强度.xls` | 海南东方逐小时太阳辐射 |

### 设置路径

先修改 `TD3/train_config.py` 中的 `sat_json_path`、`visibility_table_path`、`solar_radiation_table_path`；保存模型时还需检查 `checkpoint_path`。

建议在文件顶部定义项目根目录：

```python
from pathlib import Path

SAGE_ROOT = Path(__file__).resolve().parent.parent
```

然后将 `TrainConfig` 类内对应字段替换为以下形式。海南东方场景示例：

```python
sat_json_path: str = str(SAGE_ROOT / "dataset" / "satellite_data_50.json")
visibility_table_path: str = str(
    SAGE_ROOT / "dataset" / "satellite_visibility_hainan_2025_10.xlsx"
)
solar_radiation_table_path: str = str(
    SAGE_ROOT / "dataset" / "海南省东方市25年10月每时太阳辐射强度.xls"
)
checkpoint_path: str = str(SAGE_ROOT / "TD3" / "checkpoints" / "td3_final.pt")
```

以上示例需要手动应用到配置文件。PureTD3 与 PurePPO 继承 `TrainConfig`，使用相同的公共输入路径。

### 切换场景

| 场景 | `ground_lat` | `ground_lon` | `base_time`（UTC） |
| --- | --- | --- | --- |
| 北京 | 39.9 | 116.4 | `datetime(2025, 5, 1, 0, 0, 0)` |
| 海南东方 | 19.1 | 108.6 | `datetime(2025, 10, 1, 0, 0, 0)` |

坐标、起始日期、可见窗口表和太阳辐射表需要配套切换。当前默认使用海南东方场景，会话长度为 30 分钟，`max_steps=1488`，`total_episode=100`。

更换 TLE、地点或时间范围后，应重新生成匹配的可见窗口表。生成工具为 `TD3/utils/build_visibility_table.py.py`；使用前需将 `TD3` 加入 Python 模块搜索路径，并检查该工具的输出路径。

## 运行实验

以下命令以已完成数据路径配置为前提。

### SAGE 与 Greedy 分配对比

```powershell
& ./.venv/Scripts/python.exe TD3/train.py --seed 0 --beta 0.4 --C_base 3 --assignment_method pssa --reward_method u/c-lambda1 --info_dir ./results/sage_dongfang_seed0_c3
```

将 `--assignment_method pssa` 改为 `--assignment_method greedy` 可切换分配求解器。训练轮数、仿真时间范围、预热次数、设备及 `pssa_backend` 在 `TrainConfig` 中配置。

当前 SAGE 入口没有 `--total_episode` 或 `--max_steps` 命令行选项。小规模流程检查可暂时设置 `total_episode=1`、`max_steps=48`、`warmup_steps=10`，检查后恢复正式实验配置。环境从第 8 小时开始推进，默认步长下起始会话索引为 16，`max_steps` 应大于该索引。

### PureTD3 与 PurePPO

```powershell
& ./.venv/Scripts/python.exe PureTD3/train.py --seed 0 --beta 0.4 --C_base 3 --info_dir ./results/pure_td3_dongfang_seed0_c3
& ./.venv/Scripts/python.exe PurePPO/train.py --seed 0 --beta 0.4 --C_base 3 --info_dir ./results/pure_ppo_dongfang_seed0_c3
```

这两个入口支持命令行设置小规模运行参数：

```powershell
& ./.venv/Scripts/python.exe PureTD3/train.py --seed 0 --total_episode 1 --max_steps 48 --warmup_steps 10 --info_dir ./results/smoke_pure_td3
& ./.venv/Scripts/python.exe PurePPO/train.py --seed 0 --total_episode 1 --max_steps 48 --ppo_epochs 1 --minibatch_size 16 --info_dir ./results/smoke_pure_ppo
```

两者默认使用 Numba 加速，可通过 `--direct_backend python` 切换为 Python 实现。PurePPO 的无效动作处理方式可设为 `--invalid_action_strategy discard` 或 `cycle`，当前默认值为 `discard`。动作维度根据配置和传输机会确定。

所有训练入口均支持 `--help` 查询参数。

### 批量运行与模型保存

批量脚本中的种子、费用参数、奖励函数和输出目录需按目标实验设置。目前仍有以下路径需要调整：

- `TD3/utils/run_all.py` 按自身目录查找 `train.py` 和 `checkpoints/`，应改为指向上一级 `TD3` 目录。
- `baseline_environment.py` 中的 `comparison_grid()` 仍引用旧的 `TD3/run_all.py`，调用前需更新该路径。
- PureTD3 / PurePPO 批量脚本的绝对输出路径需改为当前机器上的位置。

完成调整前，可以使用上述单次实验命令。

SAGE 默认 `save_checkpoint=False`。需要保存权重时，在 `TrainConfig` 中开启该选项，并按需传入 `--checkpoint_path`。只指定路径不会开启权重保存。不同场景和种子应使用独立文件名。

## 日志与图表复现

训练结束后，程序在 `--info_dir` 中生成按编号递增的 `infoN.xlsx`：

| 工作表 | 内容 |
| --- | --- |
| `step` | 按记录间隔保存的逐步仿真信息 |
| `avg` | 每个 episode 的 reward、utility、cost、发送量、违约次数等统计 |
| `sheet3` | 本次运行的配置参数 |

`avg` 表末尾的奖励方法说明行不参与统计。建议每组实验使用独立目录，并记录文件名、算法、场景、种子与参数的对应关系。

**完整 info 日志未上传到 GitHub。** 本地日志由 `.gitignore` 排除。已有图片可供查看，但重新生成训练曲线需要对应日志或重新运行实验；仅有少量样例日志不足以复现全部多种子统计图。

主要绘图脚本位于 `plot/`：

| 脚本 | 图表 |
| --- | --- |
| `plot_beijing_dongfang_reward_violation.py` | 两地 reward 与违约次数 |
| `plot_beijing_dongfang_packets_utility.py` | 两地发送量与效用 |
| `plot_beijing_dongfang_satellite_solar_mixed.py` | 两地卫星可见性与太阳能采集 |
| `PPO_VS_TD3/plot_beijing_dongfang_pure_rl_reward_compare.py` | PureTD3 / PurePPO 在两地的 reward 对比 |

绘图前需更新脚本顶部的输入与输出目录，例如 `BEIJING_DATA_DIR`、`DONGFANG_DATA_DIR`、`SAVE_DIR`，或 `methods` 中的 `data_dir`。部分脚本通过文件编号识别种子和参数组合，应同时核对种子数、参数顺序与起始编号。

```powershell
& ./.venv/Scripts/python.exe plot/plot_beijing_dongfang_reward_violation.py
```

当前两地 reward / violation 合并图使用 10 个种子的均值与样本标准差（`ddof=1`）。本机缺少指定字体时，可在脚本中选择已安装字体。

## 复现记录与引用

比较不同方法时，应统一并记录数据版本、场景、种子、会话长度、训练轮数、业务到达强度、电池与太阳能参数、费用参数、奖励函数、计算实现及无效动作处理方式。固定种子仍可能因硬件和依赖版本差异产生数值差异，应保留实际环境版本与逐种子结果。

当前仓库尚未提供正式论文引用条目和许可证文件。论文题目、作者、DOI / BibTeX、代码许可证及数据来源与再分发说明将在确认后补充。
