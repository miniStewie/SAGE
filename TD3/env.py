from typing import List, Optional
import numpy as np
import math
import pandas as pd
import time
from AssignmentSolver import *
from datetime import datetime
from datetime import timedelta
from reward import _reward
import random

class Satellite:
    """卫星类"""
    # 类变量：所有 Satellite 实例共享的可见窗口表
    VISIBILITY_WINDOWS = None  # {sat_name: [(start_dt, end_dt), ...]}

    def __init__(self, name, rate, power_cost, delay, cost):
        self.name = name
        self.rate = rate  # 每分钟最多可接收包数，乘以剩余可见窗口时长可得可用总容量
        self.power_cost = power_cost  # 每个包的能耗百分比（0-1 范围内的时隙消耗）
        self.delay = delay  # 发包的固定延迟（时隙数）
        self.cost = cost  # 每包费用
        self.session_minutes = 60.0
        self.energy_profile = None

    def update_energy(self, base_time, t):
        """同步当前会话能耗；预计算后这里只做数组索引，所有求解器共用。"""
        profile = self.energy_profile
        if profile is None:
            return  # 直接指定Wh，或使用人工构造的静态卫星。
        if base_time != profile['base_time'] or self.session_minutes != profile['session_minutes']:
            raise ValueError(f'{self.name}: geometry cache does not match simulation time grid')
        if int(t) != t or not 0 <= t < len(profile['power_cost']):
            raise ValueError(f'{self.name}: missing geometry at session {t}')
        self.power_cost = float(profile['power_cost'][int(t)])
        self.performance.update({key: float(profile[key][int(t)]) for key in (
            'distance_km', 'reference_distance_km', 'energy_link_factor',
            'effective_tx_power_w', 'packet_energy_wh', 'power_cost')})
        self.performance['energy_session_index'] = int(t)

    def energy_info(self):
        """返回当期数值副本；下一会话更新卫星对象后，历史日志仍保持正确。"""
        data = getattr(self, 'performance', {})
        return {key: data.get(key) for key in (
            'energy_model', 'power_reference_mode', 'base_tx_power_w',
            'distance_km', 'reference_distance_km', 'energy_link_factor',
            'effective_tx_power_w', 'packet_energy_wh', 'energy_session_index')}

    # ========= 只在程序开始 / 环境初始化时调用一次 =========
    @classmethod
    def load_visibility_table(cls, excel_path: str):
        """
        从 Excel 中读取所有卫星的可见窗口，存入类变量 VISIBILITY_WINDOWS。
        """
        df = pd.read_excel(excel_path)

        visibility_dict = {}

        for col in df.columns:
            sat_name = str(col).strip()
            intervals = []

            for cell in df[col].dropna():
                # 统一转成字符串处理
                text = str(cell).strip()
                if not text:
                    continue

                # 按逗号拆分，左右可能有多余空格
                parts = [p.strip() for p in text.split(',')]
                if len(parts) != 2:
                    # 格式不对就跳过
                    continue

                start_str, end_str = parts
                # 这里假定格式为 "YYYY-MM-DD HH:MM:SS"
                start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
                end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")

                # 防御：确保 start < end
                if start_dt < end_dt:
                    intervals.append((start_dt, end_dt))

            # 按开始时间排序，方便后面查找
            intervals.sort(key=lambda x: x[0])
            visibility_dict[sat_name] = intervals

        cls.VISIBILITY_WINDOWS = visibility_dict

    def is_visible(self, base_time, t):
        """
        base_time: datetime，基准时间（比如 2025-05-01 08:00）
        t: 会话索引；实际分钟偏移量为 t * session_minutes

        返回:
            (visible: bool, remaining_visible_window: float)
            - 如果当前不可见，返回 (False, 0.0)
            - 如果可见，返回 (True, 当前时间到本次窗口结束的剩余分钟数)
        """

        check_time = base_time + timedelta(minutes=t * self.session_minutes)

        intervals = self.VISIBILITY_WINDOWS.get(self.name, [])
        # 简单线性扫描（窗口数量一般不多，性能够用）
        for (start_dt, end_dt) in intervals:
            if start_dt <= check_time <= end_dt:
                # 在某一个可见窗口内
                remaining_seconds = (end_dt - check_time).total_seconds()
                remaining_minutes = max(0.0, remaining_seconds / 60.0)
                return True, min(remaining_minutes, self.session_minutes)

        # 不在任何可见窗口内
        return False, 0.0


class Packet:
    """包类"""
    from train_config import TrainConfig
    packet_size = TrainConfig().packet_size  # 从配置读取，单位 kb

    def __init__(self, id, weight, delay):
        self.id = id
        self.weight = weight
        self.delay = delay
        self.utility = self.weight / (self.delay + 1e-6)  # 此效益未加卫星的delay，非真正效益


class SAGEEnv:
    HYBRID_UTILITY_WEIGHT = 0.7
    HYBRID_AGE_WEIGHT = 0.3
    BUFFER_OVERFLOW_POLICIES = {"fifo", "utility", "hybrid"}

    def __init__(
            self,
            packet_size: int,
            real_battery: int,
            solar_array_area: float,
            conversion_efficiency: float,
            slot: float,  # 时隙时长
            session: int,  # 会话时长
            satellites: List[Satellite],
            assignment_method: str,
            reward_method: str,
            max_sats: int,
            e_init: float,  # 初始的电池电量，百分比,初始设置为100%
            alpha: float,  # 要求能量中性的a系数（公式8）
            eta_init: float,  # 初始η值
            beta: float,  # β值（公式9）
            lambda_1: float,  # 惩罚系数λ（公式8）
            lambda_2: float,
            eps_uc: float,  # 防止（公式8）分母为0的极小值
            max_steps: int,  # 一轮循环中的最大会话次数
            eta_max: float,
            seed: Optional[int],  # 种子,不设置为随机种子
            base_time: datetime,
            pkt_lmt,
            buffer_overflow_policy: str,
            solar_radiation_table_path,
            C_base: float,
            standby_power_w: float = 0.0,
            packet_arrival_multiplier: float = 1.0,
            max_slots_per_satellite: int = 15,
            pssa_backend: str = 'numba',
    ):
        # 注意：随机种子统一由 train_config.set_random_seed() 在训练开始时设置
        # seed 参数保留用于文档记录，但不在此处重复设置
        self.pkt_lmt = pkt_lmt
        self.buffer_overflow_policy = str(buffer_overflow_policy).strip().lower()
        if self.buffer_overflow_policy not in self.BUFFER_OVERFLOW_POLICIES:
            supported = ", ".join(sorted(self.BUFFER_OVERFLOW_POLICIES))
            raise ValueError(
                f"Unknown buffer_overflow_policy={buffer_overflow_policy!r}; "
                f"expected one of: {supported}"
            )
        self.packet_size = packet_size
        if not math.isfinite(packet_arrival_multiplier) or packet_arrival_multiplier <= 0:
            raise ValueError("packet_arrival_multiplier must be positive and finite")
        if max_slots_per_satellite < 1 or int(max_slots_per_satellite) != max_slots_per_satellite:
            raise ValueError("max_slots_per_satellite must be a positive integer")
        self.packet_arrival_multiplier = packet_arrival_multiplier
        self.max_slots_per_satellite = int(max_slots_per_satellite)
        self.C_base = C_base

        self.real_battery = real_battery
        self.standby_power_w = float(standby_power_w)
        if not math.isfinite(self.standby_power_w) or self.standby_power_w < 0:
            raise ValueError("standby_power_w must be finite and non-negative")
        self.solar_array_area = solar_array_area
        self.conversion_efficiency = conversion_efficiency

        self.slot = slot
        self.session = session
        if not math.isfinite(session) or session <= 0:
            raise ValueError("session duration in minutes must be positive and finite")
        self.base_time = base_time
        # 预读数据
        self.solar_df = pd.read_excel(
            solar_radiation_table_path
        )

        self.satellites = satellites
        for satellite in satellites:
            satellite.session_minutes = session
        self.solver = AssignmentSolver(method=assignment_method,
                                       max_slots_per_satellite=self.max_slots_per_satellite,
                                       pssa_backend=pssa_backend)
        self.reward_method = reward_method

        self.e_init = e_init
        self.alpha = alpha
        self.e_t = self.e_init  # 动态电量

        self.eta = eta_init
        self.beta = beta

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.eps_uc = eps_uc
        self.max_steps = max_steps

        self.t = 0  # 当前轮的时隙
        self.eta_max = eta_max  # 最大eta值约束，防止eta过大
        self.backlog: List[Packet] = []
        self.last_u = 0.0
        self.last_c = 0.0
        self.last_power = 0.0
        self.max_sats = max_sats  # 网络中输入的最大卫星数

        self.total_c = 0.0
        self.total_u = 0.0

        # 数据收集功能
        self.collect_points = set()  # {(episode, step), ...}
        self.current_episode = 0
        self.current_step = 0
        self.reset_timing_stats()

    def reset_timing_stats(self):
        self.solver_call_time = 0.0
        self.solver_call_count = 0
        if hasattr(self, "solver") and hasattr(self.solver, "reset_timing_stats"):
            self.solver.reset_timing_stats()

    def get_timing_stats(self):
        stats = {}
        if hasattr(self, "solver") and hasattr(self.solver, "get_timing_stats"):
            stats.update(self.solver.get_timing_stats())
        stats.update({
            "solver_call_time": self.solver_call_time,
            "solver_call_count": self.solver_call_count,
        })
        return stats

    def set_collect_points(self, points):
        """设置要收集数据的点，格式: [(episode, step), ...]"""
        self.collect_points = set(points)

    def set_episode(self, ep):
        """在每轮开始时调用，设置当前轮次"""
        self.current_episode = ep

    def _packet_birth_t(self, packet: Packet) -> int:
        """从数据包编号中读取生成时隙；无法解析时按当前时隙处理。"""
        try:
            return int(str(packet.id).split("-", 1)[0])
        except (TypeError, ValueError):
            return int(self.t)

    def _trim_backlog(self) -> List[Packet]:
        """按配置的溢出策略裁剪缓存，并保持剩余数据包原有顺序。"""
        if self.pkt_lmt <= 0:
            return []

        overflow = len(self.backlog) - self.pkt_lmt
        if overflow <= 0:
            return []

        if self.buffer_overflow_policy == "fifo":
            drop_indices = set(range(overflow))
        elif self.buffer_overflow_policy == "utility":
            ranked_indices = sorted(
                range(len(self.backlog)),
                key=lambda index: (
                    float(self.backlog[index].utility),
                    self._packet_birth_t(self.backlog[index]),
                    index,
                ),
            )
            drop_indices = set(ranked_indices[:overflow])
        else:
            utilities = np.asarray(
                [float(packet.utility) for packet in self.backlog],
                dtype=float,
            )
            utility_min = float(np.min(utilities))
            utility_span = float(np.max(utilities) - utility_min)
            if utility_span > 1e-12:
                normalized_utilities = (utilities - utility_min) / utility_span
            else:
                normalized_utilities = np.ones_like(utilities)

            ages = np.asarray(
                [
                    max(0.0, float(self.t - self._packet_birth_t(packet)))
                    for packet in self.backlog
                ],
                dtype=float,
            )
            max_age = float(np.max(ages))
            normalized_ages = ages / max_age if max_age > 1e-12 else np.zeros_like(ages)
            retention_scores = (
                self.HYBRID_UTILITY_WEIGHT * normalized_utilities
                + self.HYBRID_AGE_WEIGHT * normalized_ages
            )
            ranked_indices = sorted(
                range(len(self.backlog)),
                key=lambda index: (
                    float(retention_scores[index]),
                    float(utilities[index]),
                    self._packet_birth_t(self.backlog[index]),
                    index,
                ),
            )
            drop_indices = set(ranked_indices[:overflow])

        dropped_packets = [
            packet
            for index, packet in enumerate(self.backlog)
            if index in drop_indices
        ]
        self.backlog = [
            packet
            for index, packet in enumerate(self.backlog)
            if index not in drop_indices
        ]
        return dropped_packets

    def reset(self, start_t=None):
        Packet.packet_size = self.packet_size
        self.total_c = 0
        self.total_u = 0
        self.t = int(8 * 60 / self.session) if start_t is None else int(start_t)
        self.e_t = self.e_init
        self.eta = max(0.0, self.eta)  # 非负
        self.backlog = self.packet_gen(self.t)  # 初始数据包
        self.last_u = 0.0
        self.last_c = 0.0
        self.last_power = 0.0
        self.current_step = 0  # 重置步数计数
        self.reset_timing_stats()
        return self.make_state(0)

    def step(self, action):
        self._update_link_energy()
        # e,c经由tanh激活函数输出范围为[-1,1],先将其映射到[0,1]
        e,c = action


        e_action = e
        c_action = c

        # session 已是分钟数；待机能量 = 功率(W) * 时长(h)，再除以电池容量(Wh)。
        # 常开模型：即使没有可见卫星或不发包，也预留并计入待机耗电。
        standby_energy_wh = self.standby_power_w * self.session / 60.0
        standby_energy = standby_energy_wh / self.real_battery
        available_tx_energy = max(0.0, self.e_t - standby_energy)
        e_frac = min(max((e + 1) * 0.5, 0.0), 1.0)
        e_budget = e_frac * available_tx_energy

        # 粗略算出一个花费上限，系数乘以上限才是真正的预算
        visible = [s for s in self.satellites if s.is_visible(self.base_time, self.t)[0]]

        # 与 PSSA 使用同一个每星时隙上限。
        c_max = 0.0
        for s in visible:
            remaining_minutes = s.is_visible(self.base_time, self.t)[1]
            tx_time_s = Packet.packet_size / s.rate
            num_slots = min(int(remaining_minutes * 60 / tx_time_s), self.max_slots_per_satellite)
            c_max += s.cost * num_slots

        c_frac = min(max((c + 1) * 0.5, 0.0), 1.0)
        c_budget = c_frac * c_max

        # 数据收集：如果当前点在收集列表中，保存数据
        self.current_step += 1
        if (self.current_episode, self.current_step) in self.collect_points:
            from utils.pssa_vs_greedy import save_solver_input
            import os
            os.makedirs("../dataset/compare_data", exist_ok=True)
            save_solver_input(
                f"compare_data/ep{self.current_episode}_step{self.current_step}.json",
                self.backlog, self.satellites, e_budget, c_budget, self.eta, self.base_time, self.t
            )

        solver_t0 = time.perf_counter()
        energy_floor_budget = max(0.0, available_tx_energy - self.alpha * self.e_init)
        selects, utility, power_used, cost = self.solver.solve(
            packets=self.backlog,
            satellites=self.satellites,
            e_budget=e_budget,
            c_budget=c_budget,
            eta=self.eta,
            t=self.t,
            base_time=self.base_time,
            energy_floor_budget=energy_floor_budget
        )
        self.solver_call_time += time.perf_counter() - solver_t0
        self.solver_call_count += 1
        self.total_c += cost
        self.total_u += utility
        # 维护挤压的数据包
        send_pks = [select[1].id for select in selects]
        if send_pks:
            self.backlog = [p for p in self.backlog if p.id not in send_pks]
            self.total_c += self.C_base

        # 维护能量
        total_energy_used = power_used + standby_energy
        energy_after = self.e_t - total_energy_used
        harvest = self.harvest()
        self.e_t = min(max(energy_after, 0.0) + harvest, self.e_init)

        # 违约计数
        violate1 = (energy_after < self.alpha * self.e_init)
        violate2 = (len(send_pks) == 0) and (self.e_t > self.alpha) and (len(self.backlog) > 0) and (len(visible) > 0)
        # 奖励
        reward = _reward(self.reward_method,
                         utility, cost, violate1, self.lambda_1, self.alpha, energy_after,
                         len(visible), len(self.backlog), len(send_pks), self.e_t, self.e_init, self.lambda_2,
                         self.total_c, self.total_u, self.eta, self.C_base
                         )

        # Robbins–Monro更新eta
        #eta_hat = (1 - self.beta) * self.eta + self.beta * (self.total_u / (self.total_c + self.C_base))
        eta_hat = (1 - self.beta) * self.eta + self.beta * (utility / (cost + self.C_base))
        self.eta = eta_hat
        # 下一个状态
        self.t += 1

        generated_packets = self.packet_gen(self.t)
        self.backlog.extend(generated_packets)
        dropped_packets = 0
        dropped_packet_items = []
        high_utility_cutoff = 0.0
        if self.pkt_lmt > 0:
            if len(self.backlog) > self.pkt_lmt:
                backlog_utilities_before_trim = np.asarray(
                    [packet.utility for packet in self.backlog],
                    dtype=float,
                )
                high_utility_cutoff = float(
                    np.quantile(backlog_utilities_before_trim, 0.75)
                )
            dropped_packet_items = self._trim_backlog()
            dropped_packets = len(dropped_packet_items)

        generated_utility = float(sum(packet.utility for packet in generated_packets))
        dropped_utility = float(sum(packet.utility for packet in dropped_packet_items))
        dropped_high_utility_items = [
            packet
            for packet in dropped_packet_items
            if packet.utility >= high_utility_cutoff
        ] if dropped_packet_items else []
        backlog_utility = float(sum(packet.utility for packet in self.backlog))

        self.last_u = float(utility)
        self.last_c = float(cost)
        self.last_power = float(total_energy_used)

        done = (self.t >= self.max_steps)

        info = {
            "reward": float(reward),
            "utility": float(utility),
            "cost": float(cost + (self.C_base if selects else 0.0)),
            "satellite_cost": float(cost),
            "base_cost_applied": float(self.C_base if selects else 0.0),
            "session_active": bool(selects),

            "eta": float(self.eta),
            "eta_hat": float(eta_hat),
            "violate1": bool(violate1),
            "violate2": bool(violate2),
            "send_pks": len(selects),
            "backlog": len(self.backlog),
            "generated_packets": len(generated_packets),
            "generated_utility": generated_utility,
            "dropped_packets": int(dropped_packets),
            "dropped_utility": dropped_utility,
            "dropped_high_utility_packets": len(dropped_high_utility_items),
            "dropped_high_utility": float(sum(
                packet.utility for packet in dropped_high_utility_items
            )),
            "high_utility_cutoff": high_utility_cutoff,
            "buffer_overflow_policy": self.buffer_overflow_policy,
            "dropped_packet_ids": [
                str(packet.id) for packet in dropped_packet_items
            ],
            "backlog_utility": backlog_utility,
            "average_backlog_utility": (
                backlog_utility / len(self.backlog) if self.backlog else 0.0
            ),
            "backlog_capacity": int(self.pkt_lmt) if self.pkt_lmt > 0 else None,
            "buffer_utilization": (
                float(len(self.backlog) / self.pkt_lmt)
                if self.pkt_lmt > 0 else 0.0
            ),
            "backlog_wait_hours": [
                float(max(0, self.t - int(str(packet.id).split("-", 1)[0]))) * self.session / 60.0
                for packet in self.backlog
            ],

            "e_action": float(e_action),
            "c_action": float(c_action),

            "e_budget": float(e_budget),
            "c_budget": float(c_budget),

            "e_frac": float(e_frac),
            "c_frac": float(c_frac),
            "c_max": float(c_max),

            # 延续原字段名，power/avg_power 表示总耗能（电池容量比例），并非瓦特。
            "power": float(total_energy_used),
            "power_tx": float(power_used),
            "power_standby": float(standby_energy),
            "standby_energy_wh": float(standby_energy_wh),
            "standby_power_w": float(self.standby_power_w),
            "power_harvest": float(harvest),
            "power_remain": float(self.e_t),
            # 实际剩余电量比例：耗电后、采能前；耗尽时记0，不显示负电量。
            "power_remain_before_harvest": float(max(0.0, energy_after)),
            "real_battery_wh": float(self.real_battery),
            "solar_array_area_m2": float(self.solar_array_area),
            "energy_floor": float(self.alpha * self.e_init),

            "visible_num": len(visible),
            "selected_satellite_count": len(self.satellites),
            "min_visible_cost": float(min((s.cost for s in visible), default=0.0)),
            "min_visible_power": float(min((s.power_cost for s in visible), default=0.0)),
            "visible_satellites": [
                {
                    "name": s.name,
                    "remaining_minutes": float(s.is_visible(self.base_time, self.t - 1)[1]),
                    **s.energy_info(),
                }
                for s in visible
            ],
            "assignments": [
                {
                    "packet_id": str(select[1].id),
                    "satellite": str(select[2].name),
                    "utility": float(select[3]),
                    "score": float(select[0]),
                    "target": float(select[4]),
                    "cost": float(select[2].cost),
                    "energy": float(select[2].power_cost),
                    **select[2].energy_info(),
                    "wait_hours": float(max(0, self.t - int(str(select[1].id).split("-", 1)[0]))) * self.session / 60.0,
                }
                for select in selects
            ],
        }

        # 先保存本会话的能耗日志，再推进几何，避免select中的卫星引用被覆盖。
        next_state = self.make_state(c_max)
        return next_state, reward, done, info

    def _update_link_energy(self):
        for satellite in self.satellites:
            satellite.update_energy(self.base_time, self.t)

    def make_state(self, c_max) -> np.ndarray:
        self._update_link_energy()  # 包括reset初态和下一会话观测。
        # 归一化基准
        rate_ref = 6000.0
        delay_ref = 10.0
        cost_ref = 3.0
        p_cost_ref = 0.01
        u_ref = 20.0  #
        c_max_ref = 40
        backlog_ref = float(self.pkt_lmt) if getattr(self, "pkt_lmt", 0) and self.pkt_lmt > 0 else 100.0

        def clip01(x: float) -> float:
            return float(np.clip(x, 0.0, 1.0))

        def log_norm(x: float, ref: float) -> float:
            # 0..1 的 log 压缩归一化
            return clip01(np.log1p(max(x, 0.0)) / np.log1p(ref))

        # ------- 1) 能量 / 阈值 / eta /c_max-------
        e_norm = clip01(self.e_t / self.e_init)
        alpha = clip01(self.alpha)  # 如果 alpha 本来就 <1，这里不会改变
        eta_norm = clip01((self.eta / self.eta_max))
        c_max_norm = clip01(c_max / c_max_ref)

        # ------- 2) backlog + utility统计
        n_backlog = len(self.backlog)
        b_norm = clip01(n_backlog / max(backlog_ref, 1.0))

        if n_backlog > 0:
            utils = [p.utility for p in self.backlog]
            u_max = float(np.max(utils))
            u_q1 = float(np.percentile(utils, 25))
            u_avg = float(np.mean(utils))
            u_q3 = float(np.percentile(utils, 75))
            u_min = float(np.min(utils))
        else:
            u_max = u_q1 = u_avg = u_q3 = u_min = 0.0

        u_max_n = log_norm(u_max, u_ref)
        u_q1_n = log_norm(u_q1, u_ref)
        u_avg_n = log_norm(u_avg, u_ref)
        u_q3_n = log_norm(u_q3, u_ref)
        u_min_n = log_norm(u_min, u_ref)

        state = [e_norm, alpha, c_max_norm, eta_norm, b_norm, u_max_n, u_q1_n, u_avg_n, u_q3_n, u_min_n]

        # ------- 3) 上一轮实际的 u/c/power
        last_u_n = log_norm(float(self.last_u), u_ref)
        last_c_n = clip01(float(self.last_c) / cost_ref)
        last_p_n = clip01(float(self.last_power) / max(self.e_init, 1e-6))
        state.extend([last_u_n, last_c_n, last_p_n])

        # ------- 4) 卫星特征
        vis = [s for s in self.satellites if s.is_visible(self.base_time, self.t)[0]]

        for i in range(self.max_sats):
            if i < len(vis):
                s = vis[i]
                rate_n = clip01(float(s.rate) / rate_ref)
                # 有界单调压缩；大能耗不再被统一截成1，仍能区分链路。
                pcost = float(s.power_cost)
                pcost_n = pcost / (pcost + p_cost_ref)
                delay_n = clip01(float(s.delay) / delay_ref)
                cost_n = clip01(float(s.cost) / cost_ref)
                state.extend([rate_n, pcost_n, delay_n, cost_n])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(state, dtype=np.float32)

    def packet_gen(self, t) -> List[Packet]:
        # 泊松分布模拟一天中的发包数量
        # base = 5 + 5 * (0.5 + 0.5 * math.sin(2.0 * math.pi * ((t % 24) - 6) / 24.0))
        # 小时内保持同一生成强度，半小时两次的期望和等于原来一小时。
        hour = math.floor(t * self.session / 60.0)
        base = 8 + 8 * (0.5 + 0.5 * math.sin(2.0 * math.pi * ((hour % 24) - 6) / 24.0))
        n = np.random.poisson(lam=max(1.0, base) * self.packet_arrival_multiplier * self.session / 60.0)

        pkts = []
        for i in range(int(n)):
            # 用 Gamma 分布制造更长尾的权重：均值≈3，方差≈9
            w = float(np.random.gamma(shape=2.0, scale=1.5)) + 0.1
            #d = int(np.random.randint(0, 5))

            d = 1
            pkts.append(Packet(id=f"{t}-{i}", weight=w, delay=d))
        return pkts

    def harvest(self):
        # 每到一整小时，进行一次能量收集
        pwm_efficiency = 0.64  # pwm 组件转换效率

        # session 已是分钟。逐小时辐射视为小时内均匀，按会话覆盖比例积分。
        start_hour = self.t * self.session / 60.0
        end_hour = (self.t + 1) * self.session / 60.0
        if start_hour == 0:
            return 0.0  # 保留原环境在 t=0 的处理；正常实验从第8小时开始
        harvested = 0.0
        for hour in range(math.floor(start_hour), math.ceil(end_hour)):
            fraction = min(end_hour, hour + 1) - max(start_hour, hour)
            solar_radiation = self.solar_df.iloc[min(hour, len(self.solar_df)-1), 3]
            solar_pane_output = (self.solar_array_area * self.conversion_efficiency
                                * solar_radiation * pwm_efficiency / 3600)
            harvested += solar_pane_output / self.real_battery * fraction
        return harvested
