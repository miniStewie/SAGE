import json
import math
import random

from skyfield.api import EarthSatellite, load
import numpy as np
from datetime import timezone

ts = load.timescale()

def compute_visibility_window(sat, elevation_angle, ground_station, t0, t1):
    try:
        satellite = EarthSatellite(sat["LINE1"], sat["LINE2"], sat["OBJECT_NAME"])
    except Exception as e:
        print(f"无法解析卫星 {sat.get('OBJECT_NAME', '未知')} 的 TLE：{e}")
        return None
    try:
        times, events = satellite.find_events(ground_station, t0, t1, altitude_degrees=elevation_angle)
    except Exception as e:
        print(f"{sat.get('OBJECT_NAME', '未知')} 计算可见性时出错：{e}")
        return None

    if len(events) >= 3:
        for i in range(len(events) - 2):
            if events[i] == 0 and events[i + 2] == 2:
                rise_time = times[i]

                set_time = times[i + 2]
                duration = (set_time.utc_datetime() - rise_time.utc_datetime()).total_seconds() / 60.0
                rise_dt = rise_time.utc_datetime()
                base_dt = rise_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                first_visible = (rise_dt - base_dt).total_seconds() / 60.0
                return {
                    "window": round(duration, 2),
                    "first_visible": round(first_visible, 2)
                }
    return None

def compute_latency(sat, ground_station, elevation_angle, t0, t1, samples=200):
    """在可见窗口内均匀采样并返回 (mean_delay_ms, mean_distance_km)，若不可见或出错则返回 None"""
    try:
        satellite = EarthSatellite(sat["LINE1"], sat["LINE2"], sat["OBJECT_NAME"])
    except Exception as e:
        # 无法解析 TLE
        # print(f"compute_latency: 无法解析卫星 {sat.get('OBJECT_NAME', '未知')} 的 TLE：{e}")
        return None

    try:
        times, events = satellite.find_events(ground_station, t0, t1, altitude_degrees=elevation_angle)
    except Exception:
        return None

    visible_intervals = []
    for i in range(len(events) - 2):
        if events[i] == 0 and events[i+2] == 2:
            t_rise = times[i]
            t_set  = times[i+2]
            visible_intervals.append((t_rise, t_set))

    if not visible_intervals:
        return None

    distances = []
    for (t_rise, t_set) in visible_intervals:
        for k in range(samples):
            # 线性插值时间，注意 samples 个点时 k/samples 的取法
            frac = k / float(samples)
            t = t_rise + (t_set - t_rise) * frac
            sat_pos = satellite.at(t)
            gs_pos = ground_station.at(t)
            diff = sat_pos - gs_pos
            distances.append(diff.distance().km)

    if len(distances) == 0:
        return None

    distances = np.array(distances)
    delays_ms = distances * 1000.0 / 3e8 * 1000.0  # km -> m -> propagation time -> ms

    return round(float(delays_ms.mean()), 3), round(float(distances.mean()), 3)

def compute_cost(sat, rng, scale=1.0):
    """单包模拟费用；先按原公式定价，再缩放，保持原随机数消耗。"""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("satellite_cost_scale must be positive and finite")
    apogee = sat.get("APOGEE", 0)
    perigee = sat.get("PERIGEE", 0)
    base_cost = round((apogee + perigee) / 3000.0, 3)  # 原始计算结果
    random_offset = rng.random() * 2  # 生成 (0.5,1) 区间的随机浮点数
    original_cost = round(base_cost + random_offset, 3)
    return original_cost * scale

def compute_rate(start_link_rate, rng):
    # 每颗卫星初始化时抽样一次；倍率0.5+Beta(2,5)，范围(0.5,1.5)，期望11/14。
    noise = 0.5 + rng.betavariate(2.0, 5.0)
    rate = start_link_rate * noise

    if rate < 0:
        rate = 0.1
    return round(rate, 3)

def compute_link_energy_factor(distance_km, reference_distance_km):
    """Friis相对自由空间损耗：(当前斜距/参考斜距)^2，无经验项或任意截断。

    参考距离必须由调用方明确给出。一般参考链路允许K<1；生产环境采用
    逐星固定最低径向天顶参考，此几何约定自然保证采样时刻K>=1。
    """
    for name, value in [('distance_km', distance_km),
                        ('reference_distance_km', reference_distance_km)]:
        if value is None or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    return (distance_km / reference_distance_km) ** 2


def compute_energy_per_packet(
    base_energy_cost_wh, real_battery, *,
    packet_size_kbit=None, rate_kbps=None, tx_power_w=None,
    distance_km=None, reference_distance_km=None,
):
    """返回单包能耗占电池容量的比例。正值直接作为Wh，0启用功率×时长。

    发送时长(s) = 包大小(kbit) / 链路速率(kbit/s)。
    自动模式能耗(Wh) = 基准功率 * 链路修正系数 * 发送时长 / 3600。
    tx_power_w为参考条件功率，系数=(d/d_ref)^2；直接指定Wh时不再乘系数。
    待机耗电由环境单独计算。
    """
    if not math.isfinite(base_energy_cost_wh) or base_energy_cost_wh < 0:
        raise ValueError("base_energy_cost_wh must be finite and non-negative")
    if not math.isfinite(real_battery) or real_battery <= 0:
        raise ValueError("real_battery must be finite and positive")

    energy_wh = float(base_energy_cost_wh)
    if base_energy_cost_wh == 0:
        for name, value in (
            ("packet_size_kbit", packet_size_kbit),
            ("rate_kbps", rate_kbps),
            ("tx_power_w", tx_power_w),
        ):
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive in automatic energy mode")
        tx_time_s = packet_size_kbit / rate_kbps
        factor = compute_link_energy_factor(distance_km, reference_distance_km)
        energy_wh = tx_power_w * factor * tx_time_s / 3600.0

    # 保留浮点精度，避免小包的归一化能耗被 round(..., 5) 舍入为零。
    return energy_wh / real_battery


def precompute_energy_profile(sat, cfg, rate_kbps, times, station_xyz_km):
    """按会话起点批量传播TLE；训练循环及求解器只读缓存，不重复传播。

    先算h_i(t)=|r_sat(t)|-|r_station(t)|，再将本次模拟网格上的min h_i(t)
    作为该卫星固定参考斜距。固定参考保证同一链路的目标接收功率不随t漂移；
    逐星各自参考P0的约定，不代表不同卫星具有共同目标接收功率。
    """
    name = sat.get('OBJECT_NAME', '<unnamed>')
    try:
        orbit = EarthSatellite(sat['LINE1'], sat['LINE2'], name, ts)
        satellite_xyz = orbit.at(times).position.km
        distances = np.linalg.norm(satellite_xyz - station_xyz_km, axis=0)
        radial_heights = np.linalg.norm(satellite_xyz, axis=0) - np.linalg.norm(station_xyz_km, axis=0)
        if (not np.all(np.isfinite(distances)) or not np.all(np.isfinite(radial_heights))
                or np.any(distances <= 0) or np.any(radial_heights <= 0)):
            raise ValueError('non-finite geometry or satellite at/below station radius')
        references = np.full_like(distances, np.min(radial_heights))
        factors = np.square(distances / references)
        if np.any(factors < 1 - 1e-10):
            raise ValueError('invalid radial-zenith reference geometry')
        # Also validates power, packet size, rate, and battery in the scalar path.
        compute_energy_per_packet(0, cfg.real_battery, packet_size_kbit=cfg.packet_size,
                                  rate_kbps=rate_kbps, tx_power_w=cfg.tx_power_w,
                                  distance_km=distances[0], reference_distance_km=references[0])
        powers = cfg.tx_power_w * factors
        packet_wh = powers * (cfg.packet_size / rate_kbps) / 3600
        if not np.all(np.isfinite(packet_wh)):
            raise ValueError('non-finite packet energy')
    except Exception as exc:
        # Fail explicitly instead of turning missing geometry into a cheap link.
        raise ValueError(f'Cannot calculate free-space energy for {name}: {exc}') from exc
    return dict(base_time=cfg.base_time, session_minutes=cfg.session,
                reference_mode=cfg.power_reference_mode, base_tx_power_w=cfg.tx_power_w,
                distance_km=distances, reference_distance_km=references,
                energy_link_factor=factors, effective_tx_power_w=powers,
                packet_energy_wh=packet_wh, power_cost=packet_wh / cfg.real_battery)

def satellite_info_dict(cfg):
    with open(cfg.sat_json_path, "r", encoding="utf-8") as f:
        sat_list = json.load(f)

    sat_list = sorted(sat_list, key=lambda s: s.get("OBJECT_NAME", ""))

    print(f"共载入 {len(sat_list)} 颗卫星数据")
    print("仰角阈值 =", cfg.elevation_angle, "°\n")

    result = {}

    rng = random.Random(cfg.seed + 999)

    # 固定计算窗口起点：不要用 ts.now()，否则每次运行都会有微小变化
    base_dt_utc = (cfg.base_time.replace(tzinfo=timezone.utc) if cfg.base_time.tzinfo is None
                   else cfg.base_time.astimezone(timezone.utc))
    t0 = ts.from_datetime(base_dt_utc)
    t1 = t0 + 1.0
    ground_station = cfg.ground_station
    if cfg.base_energy_cost_wh == 0:
        if cfg.power_reference_mode != 'per_satellite_min_radial_zenith':
            raise ValueError(f'Unsupported power reference: {cfg.power_reference_mode}')
        # Include reset at physical hour 8 and the terminal observation.
        last_step = max(cfg.max_steps, math.ceil(8 * 60 / cfg.session))
        times = t0 + np.arange(last_step + 1) * cfg.session / 1440.0
        station_xyz = ground_station.at(times).position.km

    for sat in sat_list:
        name = sat.get("OBJECT_NAME", "未知卫星")

        visibility = compute_visibility_window(sat, cfg.elevation_angle, cfg.ground_station, t0, t1)

        latency_tuple = compute_latency(sat, cfg.ground_station, cfg.elevation_angle, t0, t1)
        if latency_tuple:
            latency_ms, mean_distance_km = latency_tuple
        else:
            latency_ms, mean_distance_km = None, None

        cost = compute_cost(sat, rng, scale=cfg.satellite_cost_scale)
        rate = compute_rate(cfg.start_link_rate, rng)
        profile = (precompute_energy_profile(sat, cfg, rate, times, station_xyz)
                   if cfg.base_energy_cost_wh == 0 else None)
        power_cost = (float(profile['power_cost'][0]) if profile is not None else
                      compute_energy_per_packet(cfg.base_energy_cost_wh, cfg.real_battery))

        entry = {
            **(visibility if visibility else {}),
            "period": sat.get("PERIOD"),
            "delay": latency_ms,
            "mean_distance_km": mean_distance_km,
            "cost": cost,
            "rate": rate,
            "power_cost": power_cost,
            "energy_model": 'friis_session_start' if profile is not None else 'fixed_packet_wh',
            "power_reference_mode": cfg.power_reference_mode if profile is not None else None,
            "base_tx_power_w": cfg.tx_power_w,
            "packet_energy_wh": power_cost * cfg.real_battery,
            "_energy_profile": profile,
        }

        result[name] = entry

        if visibility is None:
            print(f"{name}: 无可见窗口")
    return result
