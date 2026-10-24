from dataclasses import dataclass, field
from datetime import datetime
from skyfield.api import wgs84
import random
import numpy as np
import torch


def set_random_seed(seed: int, strict_determinism: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass()
class TrainConfig:
    # ==================== 随机种子配置 ====================
    # 统一控制所有随机数生成器的种子，确保训练结果可复现
    # 影响范围：
    #   - 网络权重初始化 (TD3Agent.py)
    #   - 经验池随机采样 (TD3Agent.py:73)
    #   - 探索噪声生成 (TD3Agent.py:121, train.py:83)
    #   - 目标策略平滑噪声 (TD3Agent.py:132)
    #   - 数据包生成(泊松分布、Gamma分布) (env.py:364, 369, 371)
    #   - 卫星参数扰动 (compute_sats_info.py:90, 98)
    seed: int = 666
    C_base = 3
    satellite_cost_scale: float = 1.0  # 单包卫星费用倍率；
    # ==================== 训练循环参数 ====================
    max_steps: int = 1488  # 31*24
    warmup_steps: int = 100  # 预热经验池的轮数
    total_episode: int = 100  # 训练轮数,经验表示150开始会过拟合不稳定
    update_every: int = 10  # 每 2 步更新一次网络
    explore_noise: float = 0.1  # 探索噪声
    max_buffer_size: int = 50000  # 经验池大小
    action_dim: int = 2  # 动作维度(e,c)
    print_info_cnt: int = 1  # 数据收集间隔
    runs: int = 1  # 主程序运行次数

    # 时间与会话参数
    slot: int = 1  # 1slot = 分钟
    session_coefficient: int = 30  # 每次会话60分钟（slot=1分钟）
    # session 是每次会话的分钟数；缩短会话时也需按相同比例增大 max_steps
    # base_time: datetime = datetime(2025, 5, 1, 0, 0, 0)  # 北京日照表启示时间
    base_time: datetime = datetime(2025, 10, 1, 0, 0, 0)  # UTC 海南日照表启示时间

    # # 地面站与可见性参数（北京）
    # elevation_angle: float = 10  # 最低可通信仰角仰角阈值（度）
    # ground_lat: float = 39.9
    # ground_lon: float = 116.4
    # ground_elev_m: float = 0.0

    # 地面站与可见性参数（海南省东方市）
    elevation_angle: float = 10  # 最低可通信仰角仰角阈值（度）
    ground_lat: float = 19.1
    ground_lon: float = 108.6
    ground_elev_m: float = 0.0


    sat_json_path: str = "C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\dataset\satellite_data_50.json"  # 卫星TLE数据json

    visibility_table_path: str = "C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\dataset\satellite_visibility_hainan_2025_10.xlsx"  # 海南可见窗口表（预先打表）
    solar_radiation_table_path: str = "C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\dataset\海南省东方市25年10月每时太阳辐射强度.xls"  # 海南日照辐射表

    # visibility_table_path: str = "C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\dataset\satellite_visibility_beijing_2025_05.xlsx"  # 北京可见窗口表（预先打表）
    # solar_radiation_table_path: str = "C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\dataset\北京25年5月每时太阳辐射强度.xls"  # 海南日照辐射表

    # 环境参数（SAGEEnv）
    start_link_rate: float = 20 * 1000    # 基准kbps；
    base_energy_cost_wh: float = 0  #已弃用。若 >0：直接指定单包Wh；若=0：P0×(当前斜距/天顶参考高度)^2×发送秒数/3600，
    tx_power_w: float = 4.06  # 逐卫星天顶参考发送功率(W)，不是恒定实际功率
    power_reference_mode: str = 'per_satellite_min_radial_zenith'  # 每星固定d_ref=min_t(卫星地心半径-站点地心半径)
    real_battery: float = 10.0  # 电池容量(Wh)，即36 kJ
    standby_power_w: float = 0  # 常开待机功耗（W），0 为关闭；20 表示每小时耗电 20 Wh
    solar_array_area: float = 0.5  # 面板面积(m²)，满足用户指定的0.5m²下限，约70.7cm×70.7cm
    conversion_efficiency: float = 0.18  # 太阳能转换效率
    assignment_method: str = "pssa"  #


    # 组合优化算法
    pssa_backend: str = "numba"  # pssa实现：numba编译加速；python为原版，算法参数不变
    reward_method: str = "u/c-lambda1"  # 奖励函数算法
    """
    "u/c-lambda1-lambda2+n"
    "u/c-lambda1"
    "u/c-lambda1-lambda2"
    "tu/tc-lambda1"
    "tu/tc-lambda1-lambda2"
    "tu/tc-lambda1-lambda2+n"
    "tu/tc-lambda1-lambda2 * 1"
    """


    max_sats: int = 5  # 网络中输入的当前可见卫星个数，与输入维度有关，使用50颗卫星时，多数时刻可见卫星不足5
    e_init: float = 1.0  # 初始电量
    alpha: float = 0.25  # 要求能量中性的a系数（公式8）
    eta_init: float = 1.0  # 初始eta
    beta: float = 0.4  # β值（公式9）
    lambda_1: float = 5.0  # 惩罚系数λ1，惩罚违反能量中性
    lambda_2: float = 5.0  # 惩罚系数λ2,惩罚有资源不发送数据包
    eps_uc: float = 1e-6  # 防止分母为0的极小值
    eta_max: float = 100  # eta上限
    pkt_lmt: int = 200  # 是否启用数据包积压上限，为0则不启用
    packet_arrival_multiplier: float = 4.0  # 每小时业务生成倍率；缩短会话不会增加每小时业务量
    max_slots_per_satellite: int = 30  # 每颗卫星每次会话的PSSA时隙上限
    # 缓存溢出策略：
    #   "fifo"    ：保持原逻辑，优先丢弃最早进入缓存的数据包
    #   "utility" ：优先丢弃效用最低的数据包
    #   "hybrid"  ：综合效用(70%)和等待时间(30%)，优先丢弃保留分数最低的数据包
    buffer_overflow_policy: str = "utility"  # 当前数据包delay固定为1，效用排序等价于权重排序，优先剔除低权重包
    packet_size: int = 8 * 1000 * 10    # 单包大小,单位kb，8 * 1000 * 10也就是10MB

    # TD3 参数
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005  # 网络参数软更新比率
    policy_noise: float = 0.2
    noise_clip: float = 0.5  # 噪声上下限
    policy_delay: int = 2  # 策略网络延迟更新
    batch_size: int = 128
    device_str: str = "cpu"
    save_checkpoint: bool = False  # 是否在训练结束时保存权重文件；False 时仅保存实验日志
    checkpoint_path: str = "C:\\Adisk\\project\\PyCode\\BaseEnv\\RFL\\SAGE\\TD3\\checkpoints\\td3_final.pt"

    # 算法对比数据收集
    # 格式: [(轮次, 步数), ...] 例如 [(1, 10), (1, 50)] 表示收集第1轮第10步和第50步
    # 留空则不收集
    collect_points: list = field(default_factory=lambda: [])

    @property
    def session(self) -> int:
        # 1个session中有多少slot，session = session_coefficient * slot
        return self.session_coefficient * self.slot

    @property
    def ground_station(self):
        # 地面站位置（wgs84）
        return wgs84.latlon(
            self.ground_lat,
            self.ground_lon,
            elevation_m=self.ground_elev_m
        )

    @property
    def device(self):
        # 训练设备选择
        if self.device_str == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_str)
