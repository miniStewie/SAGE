import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
TD3_DIR = THIS_DIR.parent / "TD3"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from agent import ReplayBuffer, TD3Agent, build_agent_cfg
from config import PureTD3Config
from env import PureTD3Env, Satellite
from td3_compat import (
    print_avg_info_to_excel,
    print_info_to_excel,
    satellite_info_dict,
    align_direct_dimensions,
    save_to_excel,
    set_random_seed,
)


def build_env(cfg: PureTD3Config) -> PureTD3Env:
    Satellite.load_visibility_table(cfg.visibility_table_path)
    sat_info = satellite_info_dict(cfg)

    sats = []
    for name in sorted(sat_info.keys()):
        values = dict(sat_info[name])
        profile = values.pop('_energy_profile', None)
        satellite = Satellite(name=name, rate=values['rate'], power_cost=values['power_cost'],
                              delay=values['delay'], cost=values['cost'])
        satellite.performance = values
        satellite.energy_profile = profile
        satellite.session_minutes = cfg.session
        sats.append(satellite)
        print(name, values)

    align_direct_dimensions(cfg, sats)

    return PureTD3Env(
        packet_size=cfg.packet_size,
        direct_backend=cfg.direct_backend,
        standby_power_w=cfg.standby_power_w,
        packet_arrival_multiplier=cfg.packet_arrival_multiplier,
        buffer_overflow_policy=cfg.buffer_overflow_policy,
        real_battery=cfg.real_battery,
        solar_array_area=cfg.solar_array_area,
        conversion_efficiency=cfg.conversion_efficiency,
        slot=cfg.slot,
        session=cfg.session,
        satellites=sats,
        reward_method=cfg.reward_method,
        max_steps=cfg.max_steps,
        e_init=cfg.e_init,
        alpha=cfg.alpha,
        eta_init=cfg.eta_init,
        beta=cfg.beta,
        lambda_1=cfg.lambda_1,
        lambda_2=cfg.lambda_2,
        eps_uc=cfg.eps_uc,
        eta_max=cfg.eta_max,
        seed=cfg.seed,
        base_time=cfg.base_time,
        pkt_lmt=cfg.pkt_lmt,
        solar_radiation_table_path=cfg.solar_radiation_table_path,
        C_base=cfg.C_base,
        max_packets=cfg.max_packets,
        max_visible_sats=cfg.max_visible_sats,
        max_direct_slots=cfg.max_direct_slots,
        max_action_combos=cfg.max_action_combos,
    )


def main(cfg: PureTD3Config):
    start_dt = datetime.now()
    start_time = time.time()
    print(f"PureTD3 training start time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    env = build_env(cfg)
    print(
        f"[PureTD3] direct action_dim={cfg.action_dim}, "
        f"max_packets={cfg.max_packets}, max_visible_sats={cfg.max_visible_sats}, "
        f"max_slots={cfg.max_direct_slots}, max_action_combos={cfg.max_action_combos}"
    )

    print(f"[Direct backend] {env.direct_backend}; first compiled call includes JIT warmup")
    state = env.reset()
    state_dim = state.shape[0]
    expected_action_dim = cfg.max_action_combos * 3
    if cfg.action_dim != expected_action_dim:
        raise ValueError(f"cfg.action_dim={cfg.action_dim}, expected {expected_action_dim}")

    agent = TD3Agent(build_agent_cfg(cfg, state_dim))
    replay_buffer = ReplayBuffer(state_dim, cfg.action_dim, max_size=cfg.max_buffer_size)

    warmup_t0 = time.perf_counter()
    warmup_rng = np.random.default_rng(cfg.seed + 1000)
    warmup_push_cnt = 0
    state = env.reset()
    for _ in range(cfg.warmup_steps):
        action = warmup_rng.uniform(-1.0, 1.0, size=(cfg.action_dim,)).astype(np.float32)
        next_state, reward, done, info = env.step(action)
        replay_action = info.pop("_replay_action", action)
        if info["visible_num"] > 0 or done:
            replay_buffer.push(state, replay_action, reward, next_state, done)
            warmup_push_cnt += 1
        state = next_state
        if done:
            state = env.reset()
    warmup_elapsed = time.perf_counter() - warmup_t0
    print(
        f"[Warmup] steps={cfg.warmup_steps}, pushed={warmup_push_cnt}, "
        f"time={warmup_elapsed:.2f}s"
    )

    for ep in range(cfg.total_episode):
        ep_t0 = time.perf_counter()
        env.set_episode(ep + 1)
        state = env.reset()
        done = False
        step_count = 0

        total_reward = 0.0
        total_utility = 0.0
        total_cost = 0.0
        total_power = 0.0
        min_battery_percent = env.e_init * 100.0
        total_eta = 0.0
        total_send_pks = 0
        total_violate1_cnt = 0
        total_violate2_cnt = 0
        total_backlog_num = 0
        total_e_budget = 0.0
        total_c_budget = 0.0
        sum_critic_loss = 0.0
        n_update = 0
        n_log = 0

        t_act = 0.0
        t_env = 0.0
        t_update = 0.0
        t_log = 0.0
        current_noise = max(0.01, cfg.explore_noise * (1 - ep / cfg.total_episode))

        while not done:
            t0 = time.perf_counter()
            action = agent.act(state=state, noise_std=current_noise)
            t_act += time.perf_counter() - t0

            t0 = time.perf_counter()
            next_state, reward, done, info = env.step(action)
            replay_action = info.pop("_replay_action", action)
            t_env += time.perf_counter() - t0

            if info["visible_num"] > 0 or done:
                replay_buffer.push(state, replay_action, reward, next_state, done)
            state = next_state
            step_count += 1

            loss_info = {"critic_loss": np.nan, "y_tgt": np.nan}
            if step_count % cfg.update_every == 0 and replay_buffer.len >= cfg.batch_size:
                t0 = time.perf_counter()
                loss_info = agent.update(replay_buffer)
                t_update += time.perf_counter() - t0
                n_update += 1
                sum_critic_loss += float(loss_info["critic_loss"])

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
            min_battery_percent = min(min_battery_percent, info["power_remain_before_harvest"] * 100.0)
            total_eta += info["eta"]
            total_violate1_cnt += info["violate1"]
            total_violate2_cnt += info["violate2"]
            total_backlog_num += info["backlog"]
            total_e_budget += info["e_budget"]
            total_c_budget += info["c_budget"]
            total_send_pks += info["send_pks"]

        avg_critic_loss = sum_critic_loss / max(n_update, 1)
        print_avg_info_to_excel(
            total_reward,
            total_utility,
            total_cost,
            total_power,
            total_eta,
            total_violate1_cnt,
            total_violate2_cnt,
            total_send_pks,
            total_backlog_num,
            total_e_budget,
            total_c_budget,
            ep + 1,
            step_count,
            avg_critic_loss,
            min_battery_percent=min_battery_percent,
        )

        ep_elapsed = time.perf_counter() - ep_t0
        print(
            f"[PureTD3 Episode {ep + 1}/{cfg.total_episode}] "
            f"time={ep_elapsed:.2f}s, steps={step_count}, updates={n_update}, logs={n_log}    "
            f"total_reward={total_reward:.3f}    "
            f"reward_tu/tc={round(total_utility / total_cost, 3) if total_cost != 0 else 0}    "
            f"total_utility={total_utility:.4f}    "
            f"total_cost={total_cost:.4f}    "
            f"avg_power={total_power / step_count:.4f}    "
            f"avg_eta={total_eta / step_count:.4f}    "
            f"total_violate_cnt={total_violate1_cnt}    "
            f"avg_backlog_num={total_backlog_num // step_count}    "
            f"avg_e_budget={total_e_budget / step_count:.4f}    "
            f"avg_c_budget={total_c_budget / step_count:.4f}    "
            f"total_send_pks={total_send_pks}    "
            f"avg_critic_loss={avg_critic_loss:.2f}"
        )
        print(
            f"[PureTD3 Timing ep={ep + 1}] "
            f"act={t_act / max(step_count, 1) * 1000:.2f}ms/step, "
            f"env={t_env / max(step_count, 1) * 1000:.2f}ms/step, "
            f"update={(t_update / max(n_update, 1)) * 1000:.2f}ms/call, "
            f"log={(t_log / max(n_log, 1)) * 1000:.2f}ms/call"
        )
        print()

    t0 = time.perf_counter()
    from dataclasses import asdict
    config_data = asdict(cfg)
    config_data.update(direct_backend=env.direct_backend, algorithm="PureTD3", assignment_method="direct_action",
                       energy_constraint="hard_alpha_reserve",
                       traffic_rng="numpy_global_as_original_sage")
    save_to_excel(cfg.reward_method, cfg.info_dir, config_data=config_data)
    print(f"[Save] time={time.perf_counter() - t0:.2f}s")

    end_dt = datetime.now()
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60
    print(f"PureTD3 training end time:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PureTD3 total training time: {minutes} minutes {seconds:.2f} seconds")


if __name__ == "__main__":
    defaults = PureTD3Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct_backend", choices=["numba", "python"], default=defaults.direct_backend)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=defaults.beta)
    parser.add_argument("--reward_method", type=str, default=defaults.reward_method)
    parser.add_argument("--C_base", type=float, default=defaults.C_base)
    parser.add_argument("--info_dir", type=str, default="./pure_td3_info")
    parser.add_argument("--total_episode", type=int)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--print_info_cnt", type=int)
    args = parser.parse_args()

    cfg = PureTD3Config()
    cfg.direct_backend = args.direct_backend
    cfg.seed = args.seed
    cfg.beta = args.beta
    cfg.reward_method = args.reward_method
    cfg.C_base = args.C_base
    cfg.info_dir = args.info_dir
    if args.total_episode is not None:
        cfg.total_episode = args.total_episode
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.warmup_steps is not None:
        cfg.warmup_steps = args.warmup_steps
    if args.print_info_cnt is not None:
        cfg.print_info_cnt = args.print_info_cnt

    set_random_seed(cfg.seed)
    main(cfg)
