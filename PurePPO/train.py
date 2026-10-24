import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from agent import PPOAgent, RolloutBuffer, build_agent_cfg
from config import PurePPOConfig
from env import PurePPOEnv, Satellite
from ppo_compat import (
    print_avg_info_to_excel,
    print_info_to_excel,
    satellite_info_dict,
    align_direct_dimensions,
    save_to_excel,
    set_random_seed,
)


def build_env(cfg: PurePPOConfig) -> PurePPOEnv:
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

    return PurePPOEnv(
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
        invalid_action_strategy=cfg.invalid_action_strategy,
    )


def main(cfg: PurePPOConfig):
    start_dt = datetime.now()
    start_time = time.time()
    print(f"PurePPO training start time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    env = build_env(cfg)
    print(
        f"[PurePPO] logits_dim={cfg.logits_dim}, action_shape={cfg.action_shape}, "
        f"packet_classes={cfg.packet_action_dim}, sat_classes={cfg.sat_action_dim}, "
        f"slot_classes={cfg.slot_action_dim}, "
        f"invalid_action_strategy={cfg.invalid_action_strategy}"
    )

    print(f"[Direct backend] {env.direct_backend}; first compiled call includes JIT warmup")
    state = env.reset()
    state_dim = state.shape[0]
    expected_logits_dim = cfg.max_action_combos * (
        cfg.packet_action_dim + cfg.sat_action_dim + cfg.slot_action_dim
    )
    if cfg.logits_dim != expected_logits_dim:
        raise ValueError(f"cfg.logits_dim={cfg.logits_dim}, expected {expected_logits_dim}")

    agent = PPOAgent(build_agent_cfg(cfg, state_dim))

    for ep in range(cfg.total_episode):
        ep_t0 = time.perf_counter()
        env.set_episode(ep + 1)
        state = env.reset()
        rollout = RolloutBuffer()
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
        total_noop_combos = 0
        total_skipped_combos = 0

        t_act = 0.0
        t_env = 0.0
        t_update = 0.0
        t_log = 0.0
        n_log = 0

        while not done:
            t0 = time.perf_counter()
            action, logprob, value, entropy = agent.act(state)
            t_act += time.perf_counter() - t0

            t0 = time.perf_counter()
            next_state, reward, done, info = env.step(action)
            t_env += time.perf_counter() - t0

            rollout.add(state, action, logprob, reward, done, value)
            state = next_state
            step_count += 1

            if step_count % cfg.print_info_cnt == 0:
                merged_info = dict(info)
                merged_info.update(
                    {
                        "policy_loss": np.nan,
                        "value_loss": np.nan,
                        "entropy": float(entropy),
                        "approx_kl": np.nan,
                        "clipfrac": np.nan,
                        "action_logprob": float(logprob),
                        "state_value": float(value),
                    }
                )
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
            total_noop_combos += info["noop_combos"]
            total_skipped_combos += info["skipped_combos"]

        t0 = time.perf_counter()
        loss_info = agent.update(rollout)
        t_update += time.perf_counter() - t0

        update_info = {
            "ppo_update": True,
            "policy_loss": loss_info["policy_loss"],
            "value_loss": loss_info["value_loss"],
            "entropy": loss_info["entropy"],
            "approx_kl": loss_info["approx_kl"],
            "clipfrac": loss_info["clipfrac"],
        }
        print_info_to_excel(update_info, ep + 1, step_count)

        avg_value_loss = loss_info["value_loss"]
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
            avg_value_loss,
            min_battery_percent=min_battery_percent,
        )

        ep_elapsed = time.perf_counter() - ep_t0
        print(
            f"[PurePPO Episode {ep + 1}/{cfg.total_episode}] "
            f"time={ep_elapsed:.2f}s, steps={step_count}, updates=1, logs={n_log}    "
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
            f"avg_noop={total_noop_combos / step_count:.2f}    "
            f"avg_skip={total_skipped_combos / step_count:.2f}    "
            f"value_loss={loss_info['value_loss']:.2f}    "
            f"policy_loss={loss_info['policy_loss']:.4f}    "
            f"entropy={loss_info['entropy']:.2f}    "
            f"approx_kl={loss_info['approx_kl']:.6f}"
        )
        print(
            f"[PurePPO Timing ep={ep + 1}] "
            f"act={t_act / max(step_count, 1) * 1000:.2f}ms/step, "
            f"env={t_env / max(step_count, 1) * 1000:.2f}ms/step, "
            f"update={t_update * 1000:.2f}ms/call, "
            f"log={(t_log / max(n_log, 1)) * 1000:.2f}ms/call"
        )
        print()

    t0 = time.perf_counter()
    from dataclasses import asdict
    config_data = asdict(cfg)
    config_data.update(direct_backend=env.direct_backend, algorithm="PurePPO", assignment_method="direct_action",
                       energy_constraint="hard_alpha_reserve",
                       traffic_rng="numpy_global_as_original_sage")
    save_to_excel(cfg.reward_method, cfg.info_dir, config_data=config_data)
    print(f"[Save] time={time.perf_counter() - t0:.2f}s")

    end_dt = datetime.now()
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60
    print(f"PurePPO training end time:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PurePPO total training time: {minutes} minutes {seconds:.2f} seconds")


if __name__ == "__main__":
    defaults = PurePPOConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct_backend", choices=["numba", "python"], default=defaults.direct_backend)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=defaults.beta)
    parser.add_argument("--reward_method", type=str, default=defaults.reward_method)
    parser.add_argument("--C_base", type=float, default=defaults.C_base)
    parser.add_argument("--info_dir", type=str, default="./pure_ppo_info")
    parser.add_argument("--total_episode", type=int)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--print_info_cnt", type=int)
    parser.add_argument("--ppo_epochs", type=int)
    parser.add_argument("--minibatch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--invalid_action_strategy", choices=["cycle", "discard"])
    args = parser.parse_args()

    cfg = PurePPOConfig()
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
    if args.print_info_cnt is not None:
        cfg.print_info_cnt = args.print_info_cnt
    if args.ppo_epochs is not None:
        cfg.ppo_epochs = args.ppo_epochs
    if args.minibatch_size is not None:
        cfg.minibatch_size = args.minibatch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.invalid_action_strategy is not None:
        cfg.invalid_action_strategy = args.invalid_action_strategy

    set_random_seed(cfg.seed)
    main(cfg)
