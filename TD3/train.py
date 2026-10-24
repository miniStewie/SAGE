import numpy as np
import time
import torch
import os
from dataclasses import asdict

from train_config import TrainConfig, set_random_seed
from utils.save_info import print_info_to_excel, save_to_excel, print_avg_info_to_excel
from env import SAGEEnv, Satellite
from TD3Agent import TD3Agent, TD3config, ReplayBuffer
from utils.compute_sats_info import satellite_info_dict
import argparse


def build_env(cfg: TrainConfig) -> SAGEEnv:
    # 加载可见窗口打表结果
    Satellite.load_visibility_table(cfg.visibility_table_path)
    # 读取json + 计算卫星信息，实例化 Satellite 列表

    sat_info = satellite_info_dict(cfg)

    sats = []
    for k in sorted(sat_info.keys()):  # 关键：保证卫星顺序固定
        v = dict(sat_info[k])
        profile = v.pop('_energy_profile', None)
        sat = Satellite(name=k, rate=v["rate"], power_cost=v["power_cost"], delay=v["delay"], cost=v["cost"])
        sat.performance = dict(v)
        sat.energy_profile = profile
        sats.append(sat)

        print(k, v)

    # 构建环境
    env = SAGEEnv(
        real_battery=cfg.real_battery,  # 电池实际瓦数，单位w
        standby_power_w=cfg.standby_power_w,
        solar_array_area=cfg.solar_array_area,  # 太阳能板面积
        conversion_efficiency=cfg.conversion_efficiency,  # 太阳能转换效率
        slot=cfg.slot,  # 1slot = 分钟
        session=cfg.session,  # 1个session中有多少slot -> session = session_coefficient * slot
        satellites=sats,
        assignment_method=cfg.assignment_method,  # 组合优化算法
        pssa_backend=cfg.pssa_backend,
        reward_method=cfg.reward_method,
        e_init=cfg.e_init,  # 初始电量
        alpha=cfg.alpha,  # 要求能量中性的a系数（公式8）
        eta_init=cfg.eta_init,  # 初始eta
        beta=cfg.beta,  # β值（公式9）
        lambda_1=cfg.lambda_1,  # 惩罚系数λ1，惩罚违反能量中性
        lambda_2=cfg.lambda_2,  # 惩罚系数λ2,惩罚有资源不发送数据包
        eps_uc=cfg.eps_uc,  # 防止分母为0的极小值
        max_steps=cfg.max_steps,  # 一轮循环中的最大会话次数
        seed=cfg.seed,
        eta_max=cfg.eta_max,  # eta上限
        base_time=cfg.base_time,  # 开始计时的初始时间
        pkt_lmt=cfg.pkt_lmt,  # 是否启用数据包积压上限，为0则不启用，如果使用enum算法则要开启
        packet_arrival_multiplier=cfg.packet_arrival_multiplier,
        max_slots_per_satellite=cfg.max_slots_per_satellite,
        buffer_overflow_policy=cfg.buffer_overflow_policy,
        packet_size=cfg.packet_size,  # 单包大小
        solar_radiation_table_path=cfg.solar_radiation_table_path,  # 太阳辐射时间表
        max_sats=cfg.max_sats,
        C_base=cfg.C_base,
    )
    return env


def build_agent_cfg(cfg: TrainConfig, state_dim: int) -> TD3config:
    # TD3参数配置
    return TD3config(
        state_dim=state_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        tau=cfg.tau,
        policy_noise=cfg.policy_noise,
        noise_clip=cfg.noise_clip,
        policy_delay=cfg.policy_delay,
        batch_size=cfg.batch_size,
        device=cfg.device,
    )


from datetime import datetime  # 确保文件顶部已导入


def main(cfg: TrainConfig):
    # ===== 全局计时 =====
    start_dt = datetime.now()
    start_time = time.time()
    print(f"Training start time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    nominal_tx_s = cfg.packet_size / cfg.start_link_rate
    nominal_packet_wh = (cfg.base_energy_cost_wh if cfg.base_energy_cost_wh > 0
                         else cfg.tx_power_w * nominal_tx_s / 3600.0)
    print(
        f"[Energy config] battery={cfg.real_battery:g} Wh ({cfg.real_battery * 3.6:g} kJ), "
        f"base_tx_power={cfg.tx_power_w:g} W, solar_area={cfg.solar_array_area:g} m^2, "
        f"standby={cfg.standby_power_w:g} W, base_energy_cost={cfg.base_energy_cost_wh:g} Wh; "
        f"reference_packet_before_link_correction={nominal_packet_wh:g} Wh ({nominal_packet_wh * 3600:g} J), "
        f"packet_size={cfg.packet_size:g} kbit, rate={cfg.start_link_rate:g} kbit/s"
    )

    if cfg.base_energy_cost_wh == 0:
        print("[Energy model] Friis power control: K(t)=(d(t)/d_ref_i)^2; "
              f"reference={cfg.power_reference_mode}; geometry at UTC session start; "
              "no power cap, atmosphere loss or circuit power.")

    # 环境与智能体初始化
    env = build_env(cfg)
    if cfg.assignment_method == 'pssa':
        print(f"[PSSA] backend={env.solver.pssa_backend}; "
              "Numba首次调用可能包含编译耗时，后续复用缓存。")

    # ===== 数据收集设置（在 train_config.py 中配置 collect_points）=====
    if cfg.collect_points:
        env.set_collect_points(cfg.collect_points)
        print(f"[数据收集] 将在以下点收集数据: {cfg.collect_points}")

    state_dim = env.reset().shape[0]

    agent = TD3Agent(build_agent_cfg(cfg, state_dim))
    replay_buffer = ReplayBuffer(state_dim, cfg.action_dim, max_size=cfg.max_buffer_size)

    # ===== warmup 计时 =====
    warmup_t0 = time.perf_counter()
    state = env.reset()

    warmup_rng = np.random.default_rng(cfg.seed + 1000)
    warmup_push_cnt = 0
    warmup_step_cnt = 0
    for i in range(cfg.warmup_steps):
        action = warmup_rng.uniform(-1.0, 1.0, size=(cfg.action_dim,)).astype(np.float32)
        next_state, reward, done, info = env.step(action)
        warmup_step_cnt += 1
        if info["visible_num"] > 0 or done:
            replay_buffer.push(state, action, reward, next_state, done)
            warmup_push_cnt += 1
        state = next_state
        if done:
            state = env.reset()
            done = False

    warmup_elapsed = time.perf_counter() - warmup_t0
    print(
        f"[Warmup] steps={warmup_step_cnt}, pushed={warmup_push_cnt}, "
        f"time={warmup_elapsed:.2f}s ({warmup_elapsed / max(warmup_step_cnt, 1) * 1000:.2f}ms/step)"
    )

    # ===== 主训练循环 =====
    for ep in range(cfg.total_episode):
        # episode 总计时
        ep_t0 = time.perf_counter()

        env.set_episode(ep + 1)  # 设置当前轮次（用于数据收集）
        state = env.reset()

        total_reward, total_utility, total_cost, total_power, total_eta, total_send_pks = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        min_battery_percent = float(env.e_t) * 100.0
        total_violate1_cnt, total_violate2_cnt, total_backlog_num, total_e_budget, total_c_budget, done = 0, 0, 0, 0, 0, False

        current_noise = max(0.01, cfg.explore_noise * (1 - ep / cfg.total_episode))

        sum_critic_loss = 0.0
        sum_y_tgt = 0.0

        step_count = 0

        # 分项计时
        t_act = 0.0
        t_env = 0.0
        t_update = 0.0
        t_log = 0.0
        n_update = 0
        n_log = 0

        # 额外：统计 env.step / act / update 的调用次数（方便对齐）
        n_env_step = 0
        n_act = 0

        while not done:
            t0 = time.perf_counter()
            action = agent.act(state=state, noise_std=current_noise)
            t_act += time.perf_counter() - t0
            n_act += 1

            t0 = time.perf_counter()
            next_s, reward, done, info = env.step(action)
            t_env += time.perf_counter() - t0
            n_env_step += 1

            if info["visible_num"] > 0 or done:
                replay_buffer.push(state, action, reward, next_s, done)
            state = next_s

            loss_info = {"critic_loss": np.nan, "y_tgt": np.nan}
            step_count += 1

            if step_count % cfg.update_every == 0:
                t0 = time.perf_counter()
                loss_info = agent.update(replay_buffer)
                t_update += time.perf_counter() - t0
                n_update += 1
                sum_critic_loss += float(loss_info["critic_loss"])
                sum_y_tgt += float(loss_info["y_tgt"])

            if step_count % cfg.print_info_cnt == 0:
                merged_info = dict(info)
                merged_info.update(loss_info)
                t0 = time.perf_counter()
                print_info_to_excel(merged_info, ep + 1, step_count)
                t_log += time.perf_counter() - t0
                n_log += 1

            total_reward += reward
            total_utility += info["utility"]
            total_cost += info["cost"]
            total_power += info["power"]
            min_battery_percent = min(
                min_battery_percent, info["power_remain_before_harvest"] * 100.0
            )
            total_eta += info["eta"]
            total_violate1_cnt += info["violate1"]
            total_violate2_cnt += info["violate2"]
            total_backlog_num += info["backlog"]
            total_e_budget += info["e_budget"]
            total_c_budget += info["c_budget"]
            total_send_pks += info["send_pks"]

        # episode 总耗时
        ep_elapsed = time.perf_counter() - ep_t0
        timing_stats = env.get_timing_stats()

        avg_critic_loss = (sum_critic_loss / max(n_update, 1))

        # ===== 额外：估算 other 时间（除了 act/env/update/log 以外的剩余）=====
        t_known = t_act + t_env + t_update + t_log
        t_other = max(0.0, ep_elapsed - t_known)

        print_avg_info_to_excel(
            total_reward,
            total_utility, total_cost, total_power, total_eta,
            total_violate1_cnt, total_violate2_cnt, total_send_pks,
            total_backlog_num, total_e_budget, total_c_budget,
            ep + 1, step_count,
            avg_critic_loss,
            min_battery_percent=min_battery_percent,
        )

        print(
            f"[Episode {ep + 1}/{cfg.total_episode}] "
            f"time={ep_elapsed:.2f}s, steps={step_count}, updates={n_update}, logs={n_log}    "
            f"total_reward={total_reward:.3f}    "
            f"reward_tu/tc={round(total_utility / total_cost, 3) if total_cost != 0 else 0}    "
            f"total_utility={total_utility:.4f}    "
            f"total_cost={total_cost:.4f}    "
            f"avg_power={total_power / step_count:.4f}    "
            f"min_battery={min_battery_percent:.4f}%    "
            f"avg_eta={total_eta / step_count:.4f}    "
            f"total_violate_cnt={total_violate1_cnt}    "
            f"avg_backlog_num={total_backlog_num // step_count}    "
            f"avg_e_budget={total_e_budget / step_count:.4f}    "
            f"avg_c_budget={total_c_budget / step_count:.4f}    "
            f"total_send_pks={total_send_pks}    "
            f"avg_critic_loss={avg_critic_loss:.2f}"
        )

        # 分项耗时输出（每 step / 每 call）
        if step_count > 0:
            print(
                f"[Timing ep={ep + 1}] "
                f"act={t_act / max(n_act, 1) * 1000:.2f}ms/step, "
                f"env={t_env / max(n_env_step, 1) * 1000:.2f}ms/step, "
                f"assign_solver={timing_stats['solver_call_time'] / max(timing_stats['solver_call_count'], 1) * 1000:.2f}ms/call, "
                f"packet_loop={timing_stats['packet_loop_time'] / max(timing_stats['solver_call_count'], 1) * 1000:.2f}ms/call, "
                f"opt_loop={timing_stats['opt_loop_time'] / max(timing_stats['solver_call_count'], 1) * 1000:.2f}ms/call, "
                f"update={(t_update / max(n_update, 1)) * 1000:.2f}ms/call, "
                f"log={(t_log / max(n_log, 1)) * 1000:.2f}ms/call, "
                f"other={t_other / step_count * 1000:.2f}ms/step"
            )
            print()

    # ===== 保存耗时（也计时一下）=====
    t0 = time.perf_counter()
    # 使用实际运行的cfg，包含命令行覆盖值及动态添加的info_dir等参数。
    config_data = asdict(cfg)
    config_data.update(vars(cfg))
    config_data.update(
        C_base=cfg.C_base,  # C_base目前是类属性，不在dataclass字段中。
        session=cfg.session,
        device=str(agent.cfg.device),
        pssa_backend_actual=env.solver.pssa_backend,
        energy_model='friis_session_start' if cfg.base_energy_cost_wh == 0 else 'fixed_packet_wh',
        power_feature_encoding='x/(x+0.01)',
    )
    save_to_excel(cfg.reward_method, cfg.info_dir, config_data=config_data)
    save_elapsed = time.perf_counter() - t0
    print(f"[Save] time={save_elapsed:.2f}s")

    if cfg.save_checkpoint:
        checkpoint_path = agent.save(
            cfg.checkpoint_path,
            metadata={
                "seed": cfg.seed,
                "assignment_method": cfg.assignment_method,
                "pssa_backend": env.solver.pssa_backend,
                "reward_method": cfg.reward_method,
                "episodes": cfg.total_episode,
                "max_steps": cfg.max_steps,
                "real_battery_wh": cfg.real_battery,
                "tx_power_w": cfg.tx_power_w,
                "power_reference_mode": cfg.power_reference_mode,
                "energy_model": 'friis_session_start' if cfg.base_energy_cost_wh == 0 else 'fixed_packet_wh',
                "power_feature_encoding": 'x/(x+0.01)',
                "solar_array_area_m2": cfg.solar_array_area,
                "standby_power_w": cfg.standby_power_w,
                "base_energy_cost_wh": cfg.base_energy_cost_wh,
                "packet_size_kbit": cfg.packet_size,
                "start_link_rate_kbps": cfg.start_link_rate,
                "session_minutes": cfg.session,
                "packet_arrival_multiplier": cfg.packet_arrival_multiplier,
                "max_slots_per_satellite": cfg.max_slots_per_satellite,
                "backlog_limit": cfg.pkt_lmt,
            },
        )
        print(f"[Checkpoint] saved to {checkpoint_path}")
    else:
        print("[Checkpoint] saving disabled (save_checkpoint=False)")

    end_dt = datetime.now()
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60

    print(f"Training end time:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {minutes} minutes {seconds:.2f} seconds")


if __name__ == "__main__":
    # ====== 解析命令行参数（有默认值，可直接在PyCharm中点运行）======
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--reward_method", type=str, default="u/c-lambda1")
    parser.add_argument("--assignment_method", type=str)
    parser.add_argument("--C_base", type=float)
    parser.add_argument("--info_dir", type=str, default="./info")
    parser.add_argument("--checkpoint_path", type=str, default=None)

    # parser.add_argument("--lambda1", type=float)
    # parser.add_argument("--lambda2", type=float)
    args = parser.parse_args()

    # ====== 构建配置 ======
    cfg = TrainConfig()
    cfg.seed = args.seed
    cfg.beta = args.beta
    cfg.reward_method = args.reward_method
    if args.assignment_method is not None:
        cfg.assignment_method = args.assignment_method
    if args.C_base is not None:
        cfg.C_base = args.C_base
    cfg.info_dir = args.info_dir
    if args.checkpoint_path:
        cfg.checkpoint_path = args.checkpoint_path
    # cfg.lambda1 = args.lambda1
    # cfg.lambda2 = args.lambda2

    # ====== 固定随机性 ======
    set_random_seed(cfg.seed)

    main(cfg)
