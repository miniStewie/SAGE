from datetime import datetime, timedelta
from typing import List, Optional
import math
import warnings

import numpy as np
import pandas as pd

from td3_compat import reward_fn, Satellite, EnvironmentPhysics, select_actions, NUMBA_AVAILABLE, make_state_fast


class Packet:
    """Packet used by the direct-action baseline."""

    packet_size = 1024 * 8 * 8

    def __init__(self, id, weight, delay, birth_t=None):
        self.id = id
        self.weight = float(weight)
        self.delay = int(delay)
        self.birth_t = int(str(id).split("-", 1)[0]) if birth_t is None else int(birth_t)
        self.utility = self.weight / (self.delay + 1e-6)


class PureTD3Env(EnvironmentPhysics):
    packet_type = Packet
    """Direct-action TD3 environment with no assignment solver."""

    def __init__(
        self,
        packet_size: int,
        real_battery: int,
        solar_array_area: float,
        conversion_efficiency: float,
        slot: float,
        session: int,
        satellites: List[Satellite],
        reward_method: str,
        max_steps: int,
        e_init: float,
        alpha: float,
        eta_init: float,
        beta: float,
        lambda_1: float,
        lambda_2: float,
        eps_uc: float,
        eta_max: float,
        seed: Optional[int],
        base_time: datetime,
        pkt_lmt,
        solar_radiation_table_path,
        C_base: float,
        max_packets: int,
        max_visible_sats: int,
        max_direct_slots: int,
        max_action_combos: int,
        direct_backend: str = "numba",
        standby_power_w: float = 0.0,
        packet_arrival_multiplier: float = 1.0,
        buffer_overflow_policy: str = "utility",
    ):
        if direct_backend not in ("numba", "python"):
            raise ValueError("direct_backend must be numba or python")
        self.direct_backend = direct_backend
        if direct_backend == "numba" and not NUMBA_AVAILABLE:
            warnings.warn("Numba unavailable; using original Python direct backend", RuntimeWarning)
            self.direct_backend = "python"
        self.standby_power_w = standby_power_w
        self.packet_arrival_multiplier = packet_arrival_multiplier
        self.buffer_overflow_policy = buffer_overflow_policy
        for satellite in satellites:
            satellite.session_minutes = session
        self.packet_size = packet_size
        Packet.packet_size = packet_size
        self.real_battery = real_battery
        self.solar_array_area = solar_array_area
        self.conversion_efficiency = conversion_efficiency
        self.slot = slot
        self.session = session
        self.satellites = satellites
        self.reward_method = reward_method
        self.max_steps = max_steps
        self.e_init = e_init
        self.alpha = alpha
        self.eta = eta_init
        self.beta = beta
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.eps_uc = eps_uc
        self.eta_max = eta_max
        self.seed = seed
        self.base_time = base_time
        self.pkt_lmt = pkt_lmt
        self.C_base = C_base
        self.max_packets = max_packets
        self.max_visible_sats = max_visible_sats
        self.max_direct_slots = max_direct_slots
        self.max_slots_per_satellite = max_direct_slots
        self.max_action_combos = max_action_combos

        self.solar_df = pd.read_excel(solar_radiation_table_path)

        self.t = 0
        self.e_t = self.e_init
        self.backlog: List[Packet] = []
        self.last_u = 0.0
        self.last_c = 0.0
        self.last_power = 0.0
        self.total_c = 0.0
        self.total_u = 0.0
        self.current_episode = 0
        self.current_step = 0

    def set_episode(self, ep):
        self.current_episode = ep

    def reset_timing_stats(self):
        return None

    def get_timing_stats(self):
        return {
            "solver_call_time": 0.0,
            "solver_call_count": 0,
            "packet_loop_time": 0.0,
            "packet_loop_count": 0,
            "opt_loop_time": 0.0,
            "opt_loop_count": 0,
        }

    def reset(self):
        Packet.packet_size = self.packet_size
        self.total_c = 0.0
        self.total_u = 0.0
        self.t = int(8 * 60 / self.session)
        self.e_t = self.e_init
        self.eta = max(0.0, self.eta)
        self.backlog = self.packet_gen(self.t)
        self.last_u = 0.0
        self.last_c = 0.0
        self.last_power = 0.0
        self.current_step = 0
        return self.make_state(0)

    def step(self, action):
        self._update_link_energy()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != self.max_action_combos * 3:
            raise ValueError(
                f"PureTD3 action_dim must be {self.max_action_combos * 3}, got {action.size}"
            )

        self.current_step += 1
        visible_all = self._visible_sats_with_windows()
        visible = visible_all[: self.max_visible_sats]
        slot_counts = self._slot_counts(visible)
        c_max = self._compute_c_max(visible_all)

        active_packets = self.backlog[: self.max_packets]
        standby_energy_wh = self.standby_power_w * self.session / 60.0
        standby_energy = standby_energy_wh / self.real_battery
        # Direct baselines intentionally retain the hard alpha reserve.
        safe_energy_budget = max(0.0, self.e_t - standby_energy - self.alpha * self.e_init)

        if self.direct_backend == "numba":
            result = select_actions(self, action, active_packets, visible, slot_counts,
                                    safe_energy_budget, continuous=True)
        else:
            result = self._select_actions_python(action, active_packets, visible, slot_counts, safe_energy_budget)
        selections, total_utility, total_cost, total_power, noop_combos, skipped_combos, replay_action = result

        self.total_c += total_cost
        self.total_u += total_utility

        sent_ids = {packet.id for _, packet, _, _, _ in selections}
        if sent_ids:
            self.backlog = [p for p in self.backlog if p.id not in sent_ids]
            self.total_c += self.C_base

        total_energy_used = total_power + standby_energy
        energy_after = self.e_t - total_energy_used
        harvest = self.harvest()
        self.e_t = min(max(energy_after, 0.0) + harvest, self.e_init)

        violate1 = energy_after < self.alpha * self.e_init
        violate2 = (
            len(sent_ids) == 0
            and self.e_t > self.alpha
            and len(self.backlog) > 0
            and len(visible_all) > 0
        )

        reward = reward_fn(
            self.reward_method,
            total_utility,
            total_cost,
            violate1,
            self.lambda_1,
            self.alpha,
            energy_after,
            len(visible_all),
            len(self.backlog),
            len(sent_ids),
            self.e_t,
            self.e_init,
            self.lambda_2,
            self.total_c,
            self.total_u,
            self.eta,
            self.C_base,
        )

        eta_hat = (1 - self.beta) * self.eta + self.beta * (
            total_utility / (total_cost + self.C_base)
        )
        self.eta = eta_hat

        self.t += 1
        generated_packets = self.packet_gen(self.t)
        self.backlog.extend(generated_packets)
        dropped_packets = self._trim_backlog()

        self.last_u = float(total_utility)
        self.last_c = float(total_cost)
        self.last_power = float(total_energy_used)

        done = self.t >= self.max_steps

        info = {
            "reward": float(reward),
            "utility": float(total_utility),
            "cost": float(total_cost + (self.C_base if selections else 0.0)),
            "satellite_cost": float(total_cost),
            "base_cost_applied": float(self.C_base if selections else 0.0),
            "session_active": bool(selections),
            "eta": float(self.eta),
            "eta_hat": float(eta_hat),
            "violate1": bool(violate1),
            "violate2": bool(violate2),
            "send_pks": len(selections),
            "backlog": len(self.backlog),
            "e_action": float(total_power),
            "c_action": float(total_cost),
            "e_budget": float(safe_energy_budget),
            "c_budget": float(c_max),
            "e_frac": float(total_power / max(safe_energy_budget, 1e-9)),
            "c_frac": float(total_cost / max(c_max, 1e-9)),
            "c_max": float(c_max),
            "power": float(total_energy_used),
            "power_tx": float(total_power),
            "power_standby": float(standby_energy),
            "standby_energy_wh": float(standby_energy_wh),
            "power_remain_before_harvest": float(max(0.0, energy_after)),
            "real_battery_wh": float(self.real_battery),
            "energy_floor": float(self.alpha * self.e_init),
            "generated_packets": len(generated_packets),
            "generated_utility": float(sum(p.utility for p in generated_packets)),
            "dropped_packets": len(dropped_packets),
            "dropped_utility": float(sum(p.utility for p in dropped_packets)),
            "buffer_overflow_policy": self.buffer_overflow_policy,
            "visible_satellites": [dict(name=s.name, remaining_minutes=w, **s.energy_info()) for s, w in visible_all],
            "assignments": [dict(packet_id=p.id, satellite=s.name, energy=float(s.power_cost), utility=float(u), cost=float(s.cost), **s.energy_info()) for _, p, s, _, u in selections],
            "power_harvest": float(harvest),
            "power_remain": float(self.e_t),
            "visible_num": len(visible_all),
            "visible_used_num": len(visible),
            "decoded_combos": len(selections),
            "noop_combos": noop_combos,
            "skipped_combos": skipped_combos,
            "_replay_action": replay_action,
        }

        # Preserve the completed SAGE experiment: this feature carries the previous c_max.
        next_state = self.make_state(c_max)
        return next_state, reward, done, info

    def _select_actions_python(self, action, active_packets, visible, slot_counts, safe_energy_budget):
        used_packet_indices = set()
        used_slots = set()
        selections = []
        noop_combos = skipped_combos = 0
        total_utility = total_cost = total_power = 0.0
        replay_action = np.empty_like(action, dtype=np.float32)
        for combo_idx in range(self.max_action_combos):
            raw_packet, raw_sat, raw_slot = action[combo_idx * 3 : combo_idx * 3 + 3]
            packet_idx = self._decode_index(raw_packet, self.max_packets)
            if packet_idx >= self.max_packets:
                replay_action[combo_idx * 3 : combo_idx * 3 + 3] = self._encode_noop_combo()
                noop_combos += 1
                continue

            if not active_packets or not visible:
                replay_action[combo_idx * 3 : combo_idx * 3 + 3] = self._encode_noop_combo()
                skipped_combos += 1
                continue

            packet_idx = self._find_packet_index(packet_idx, active_packets, used_packet_indices)
            if packet_idx is None:
                replay_action[combo_idx * 3 : combo_idx * 3 + 3] = self._encode_noop_combo()
                skipped_combos += 1
                continue

            sat_idx = self._decode_index(raw_sat, self.max_visible_sats - 1)
            desired_slot = self._decode_index(raw_slot, self.max_direct_slots - 1)
            sat_slot = self._find_sat_slot(
                sat_idx,
                desired_slot,
                visible,
                slot_counts,
                used_slots,
                safe_energy_budget - total_power,
            )
            if sat_slot is None:
                replay_action[combo_idx * 3 : combo_idx * 3 + 3] = self._encode_noop_combo()
                skipped_combos += 1
                continue
            sat_idx, slot_idx = sat_slot

            sat = visible[sat_idx][0]
            packet = active_packets[packet_idx]

            used_packet_indices.add(packet_idx)
            used_slots.add((sat_idx, slot_idx))
            utility = packet.weight
            total_utility += utility
            total_cost += sat.cost
            total_power += sat.power_cost
            selections.append((packet_idx, packet, sat, slot_idx, utility))
            replay_action[combo_idx * 3 : combo_idx * 3 + 3] = self._encode_combo(
                packet_idx, sat_idx, slot_idx
            )

        return selections, total_utility, total_cost, total_power, noop_combos, skipped_combos, replay_action

    def make_state(self, c_max) -> np.ndarray:
        if self.direct_backend == "python":
            return self._make_state_python(c_max)
        return self._make_state_fast(c_max)

    def _make_state_fast(self, c_max) -> np.ndarray:
        return make_state_fast(self, c_max)

    def _make_state_python(self, c_max) -> np.ndarray:
        self._update_link_energy()
        rate_ref = 6000.0
        delay_ref = 10.0
        cost_ref = 3.0
        p_cost_ref = 0.01
        u_ref = 20.0
        c_max_ref = 40.0
        backlog_ref = float(self.pkt_lmt) if self.pkt_lmt and self.pkt_lmt > 0 else 100.0

        def clip01(x: float) -> float:
            return float(np.clip(x, 0.0, 1.0))

        def log_norm(x: float, ref: float) -> float:
            return clip01(np.log1p(max(x, 0.0)) / np.log1p(ref))

        n_backlog = len(self.backlog)
        utils = [p.utility for p in self.backlog]
        if utils:
            u_max = float(np.max(utils))
            u_q1 = float(np.percentile(utils, 25))
            u_avg = float(np.mean(utils))
            u_q3 = float(np.percentile(utils, 75))
            u_min = float(np.min(utils))
        else:
            u_max = u_q1 = u_avg = u_q3 = u_min = 0.0

        state = [
            clip01(self.e_t / self.e_init),
            clip01(self.alpha),
            clip01(c_max / c_max_ref),
            clip01(self.eta / self.eta_max),
            clip01(n_backlog / max(backlog_ref, 1.0)),
            log_norm(u_max, u_ref),
            log_norm(u_q1, u_ref),
            log_norm(u_avg, u_ref),
            log_norm(u_q3, u_ref),
            log_norm(u_min, u_ref),
            log_norm(float(self.last_u), u_ref),
            clip01(float(self.last_c) / cost_ref),
            clip01(float(self.last_power) / max(self.e_init, 1e-6)),
        ]

        active_packets = self.backlog[: self.max_packets]
        for i in range(self.max_packets):
            if i < len(active_packets):
                p = active_packets[i]
                age = max(0.0, float(self.t - p.birth_t))
                state.extend(
                    [
                        1.0,
                        log_norm(p.utility, u_ref),
                        log_norm(p.weight, u_ref),
                        clip01(age / max(float(self.max_steps), 1.0)),
                    ]
                )
            else:
                state.extend([0.0, 0.0, 0.0, 0.0])

        visible = self._visible_sats_with_windows()[: self.max_visible_sats]
        slot_counts = self._slot_counts(visible)
        for i in range(self.max_visible_sats):
            if i < len(visible):
                s, win_len = visible[i]
                state.extend(
                    [
                        1.0,
                        clip01(float(s.rate) / rate_ref),
                        float(s.power_cost) / (float(s.power_cost) + p_cost_ref),
                        clip01(float(s.delay) / delay_ref),
                        clip01(float(s.cost) / cost_ref),
                        clip01(float(win_len) / 120.0),
                        clip01(float(slot_counts[i]) / max(float(self.max_direct_slots), 1.0)),
                    ]
                )
            else:
                state.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(state, dtype=np.float32)

    @staticmethod
    def _decode_index(value, max_index):
        value = float(np.clip(value, -1.0, 1.0))
        return int(np.rint((value + 1.0) * 0.5 * max_index))

    @staticmethod
    def _encode_index(index, max_index):
        if max_index <= 0:
            return -1.0
        index = int(np.clip(index, 0, max_index))
        return float(index / max_index * 2.0 - 1.0)

    def _encode_combo(self, packet_idx, sat_idx, slot_idx):
        return np.array(
            [
                self._encode_index(packet_idx, self.max_packets),
                self._encode_index(sat_idx, self.max_visible_sats - 1),
                self._encode_index(slot_idx, self.max_direct_slots - 1),
            ],
            dtype=np.float32,
        )

    def _encode_noop_combo(self):
        return self._encode_combo(self.max_packets, 0, 0)

    @staticmethod
    def _find_packet_index(start_idx, active_packets, used_packet_indices):
        n_packets = len(active_packets)
        if n_packets == 0 or len(used_packet_indices) >= n_packets:
            return None

        start = int(start_idx) % n_packets
        for offset in range(n_packets):
            idx = (start + offset) % n_packets
            if idx not in used_packet_indices:
                return idx
        return None

    def _find_sat_slot(
        self,
        start_sat_idx,
        start_slot_idx,
        visible,
        slot_counts,
        used_slots,
        remaining_energy,
    ):
        n_sats = len(visible)
        if n_sats == 0:
            return None

        sat_start = int(start_sat_idx) % n_sats
        for sat_offset in range(n_sats):
            sat_idx = (sat_start + sat_offset) % n_sats
            sat = visible[sat_idx][0]
            if sat.power_cost > remaining_energy + 1e-12:
                continue

            slot_count = int(slot_counts[sat_idx])
            if slot_count <= 0:
                continue

            slot_start = int(start_slot_idx) % slot_count
            for slot_offset in range(slot_count):
                slot_idx = (slot_start + slot_offset) % slot_count
                if (sat_idx, slot_idx) not in used_slots:
                    return sat_idx, slot_idx

        return None

PureTD3 = PureTD3Env
