"""Baseline physical rules matching the existing SAGE experiment (minutes, kbit/s, Wh)."""
import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

class Satellite:
    """卫星类"""
    # 类变量：所有 Satellite 实例共享的可见窗口表
    VISIBILITY_WINDOWS = None  # {sat_name: [(start_dt, end_dt), ...]}

    def __init__(self, name, rate, power_cost, delay, cost):
        self.name = name
        self.rate = rate  # kbit/s；包大小(kbit) / 速率得到发送秒数
        self.power_cost = power_cost  # 每个包的能耗百分比（0-1 范围内的时隙消耗）
        self.delay = delay  # 发包的固定延迟（时隙数）
        self.cost = cost  # 每包费用
        self.session_minutes = 60.0
        self.energy_profile = None
        self.performance = {}

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

class EnvironmentPhysics:
    HYBRID_UTILITY_WEIGHT = 0.7
    HYBRID_AGE_WEIGHT = 0.3
    def _packet_birth_t(self, packet) -> int:
        """从数据包编号中读取生成时隙；无法解析时按当前时隙处理。"""
        try:
            return int(str(packet.id).split("-", 1)[0])
        except (TypeError, ValueError):
            return int(self.t)

    def _trim_backlog(self):
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

    def _update_link_energy(self):
        for satellite in self.satellites:
            satellite.update_energy(self.base_time, self.t)

    def packet_gen(self, t):
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
            pkts.append(self.packet_type(id=f"{t}-{i}", weight=w, delay=d))
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

    def _visible_sats_with_windows(self):
        visible = []
        for sat in self.satellites:
            is_visible, minutes = sat.is_visible(self.base_time, self.t)
            if is_visible and minutes > 0:
                visible.append((sat, minutes))
        return visible

    def _slot_counts(self, visible):
        return [max(0, min(int(minutes * 60.0 / (self.packet_size / sat.rate)),
                           self.max_slots_per_satellite)) for sat, minutes in visible]

    def _compute_c_max(self, visible=None):
        if visible is None:
            visible = self._visible_sats_with_windows()
        return sum(s.cost * n for (s, _), n in zip(visible, self._slot_counts(visible)))


def align_direct_dimensions(cfg, satellites):
    """Cover all opportunities available to SAGE; resolve after run/CLI overrides."""
    if cfg.pkt_lmt <= 0:
        raise ValueError('Direct policies require a finite pkt_lmt for their action heads')
    first = int(8 * 60 / cfg.session)
    peak_visible = max((sum(s.is_visible(cfg.base_time, t)[0] for s in satellites)
                        for t in range(first, cfg.max_steps + 1)), default=0)
    required = {'max_packets': cfg.pkt_lmt,
                'max_visible_sats': max(cfg.max_sats, peak_visible, 1),
                'max_direct_slots': cfg.max_slots_per_satellite}
    for name, value in required.items():
        current = getattr(cfg, name)
        if current is None:
            setattr(cfg, name, value)
        elif current < value:
            raise ValueError(f'{name}={current} truncates SAGE opportunities; need >= {value}')
    # Direct slots must implement the same physical cap, even with oversized heads.
    if cfg.max_direct_slots != cfg.max_slots_per_satellite:
        raise ValueError('max_direct_slots must equal max_slots_per_satellite')
    combos = min(cfg.pkt_lmt, cfg.max_visible_sats * cfg.max_direct_slots)
    if cfg.max_action_combos is None:
        cfg.max_action_combos = combos
    elif cfg.max_action_combos < combos:
        raise ValueError(f'max_action_combos must be >= {combos} for a fair comparison')
    cfg.action_dim = cfg.max_action_combos * 3


def comparison_grid():
    """Read literal experiment grids from SAGE's launcher without launching training."""
    import ast
    from pathlib import Path
    names = {'reward_method_list', 'beta_list', 'seed_list', 'C_base_list'}
    tree = ast.parse((Path(__file__).parent / 'TD3/run_all.py').read_text(encoding='utf-8-sig'))
    grid = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    grid[target.id] = ast.literal_eval(node.value)
    if grid.keys() != names:
        raise ValueError('SAGE run_all.py must define literal lists for the comparison grid')
    return grid
