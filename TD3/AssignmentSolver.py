# 传入DRL的e_budget, c_budget,包序列,卫星序列
import math
import time
import warnings
import numpy as np
class AssignmentSolver:
    def __init__(self, method='greedy', t=0, max_slots_per_satellite=15, pssa_backend='numba', global_greedy_backend='fast'):
        if global_greedy_backend not in ("fast", "python"):
            raise ValueError("global_greedy_backend must be fast or python")
        self.global_greedy_backend = global_greedy_backend
        self.method = method
        self.max_slots_per_satellite = max_slots_per_satellite
        self.t = t
        if pssa_backend not in ('numba', 'python'):
            raise ValueError(f'Unknown PSSA backend: {pssa_backend}')
        self.pssa_backend = pssa_backend
        self._compiled_pssa = None
        if method == 'pssa' and pssa_backend == 'numba':
            try:
                from utils.pssa_compiled import solve
            except ImportError as exc:
                warnings.warn(f'Numba PSSA unavailable ({exc}); using slow Python PSSA. '
                              'Install SAGE/requirements.txt in the training interpreter.',
                              RuntimeWarning, stacklevel=2)
                self.pssa_backend = 'python'
            else:
                self._compiled_pssa = solve
        self.reset_timing_stats()

    def reset_timing_stats(self):
        self.timing_stats = {
            "packet_loop_time": 0.0,
            "packet_loop_count": 0,
            "opt_loop_time": 0.0,
            "opt_loop_count": 0,
        }

    def _record_timing(self, name, elapsed):
        self.timing_stats[f"{name}_time"] += elapsed
        self.timing_stats[f"{name}_count"] += 1

    def get_timing_stats(self):
        return dict(self.timing_stats)

    # 包序列[]，卫星序列[]，能量预算，花费预算，eta参数，当前时隙
    def solve(self, packets, satellites, e_budget, c_budget, eta, base_time, t, energy_floor_budget=None):
        # 返回：所选卫星的包传输序列列表，总效益，总能量消耗，总花费消耗
        if self.method == 'greedy':
            return self._solve_greedy(packets, satellites, e_budget, c_budget, eta, base_time, t)
        elif self.method == 'pssa':
            if self._compiled_pssa is not None:
                return self._compiled_pssa(
                    packets, satellites, e_budget, c_budget, eta, base_time, t,
                    max_slots=self.max_slots_per_satellite, record_timing=self._record_timing)
            return self._solve_pssa(packets, satellites, e_budget, c_budget, eta,  base_time, t)
        elif self.method in ('global_greedy', 'advanced_greedy', 'standalone_greedy', 'unbudgeted_greedy'):
            return self._global_greedy(
                packets, satellites, e_budget, c_budget, eta, base_time, t, energy_floor_budget
            )
        elif self.method == 'greedy_vec':
            return self._solve_greedy_vec(packets, satellites, e_budget, c_budget, eta, base_time, t)
        elif self.method == 'pssa_vec':
            return self._solve_pssa_vec(packets, satellites, e_budget, c_budget, eta, base_time, t)
        else:
            raise ValueError(f"Unknown solver method: {self.method}")

    def _solve_greedy(self, packets, satellites, e_budget, c_budget, eta, base_time, t):
        """
        1. 找所有可见卫星，得到它们的可见窗口长度。
        2. 对每颗卫星，根据可见窗口长度和单个包的发送时长，计算 slot 数。
        3. 列举所有 (packet, satellite, slot) 组合，计算 score = u - η c。
        4. 按 score 从大到小排序，在不超能量 / 费用预算的前提下贪心选：
           - 每个包最多被选一次；
           - 每个 slot 最多被用一次；
        5. 返回选中的 (包, 卫星) 组合列表及其总效益、总能量、总费用。
        """
        if not packets:
            return [], 0.0, 0.0, 0.0

        # 先筛出所有可见卫星，并计算每颗卫星的 slot 数和每个 slot 的持续时间
        visible_sats = []
        sat_slot_count = {}  # sat.name -> int #slots
        sat_slot_duration = {}  # sat.name -> float slot_length

        for s in satellites:
            is_vis, win_len = s.is_visible(base_time, t)
            if not is_vis or win_len <= 0:
                continue

            # 计算单个包的传输时间 tx_time：
            from env import Packet
            tx_time = Packet.packet_size / s.rate

            # slot 数量 = 窗口长度 / 单 slot 时长
            num_slots = max(1, int(win_len / tx_time))

            visible_sats.append(s)
            sat_slot_count[s.name] = num_slots
            # 用 win_len / num_slots 重新算一遍，避免整除截断带来的误差
            sat_slot_duration[s.name] = win_len / num_slots

        if not visible_sats:
            return [], 0.0, 0.0, 0.0

        #    列举所有 (packet, satellite, slot) 组合，并计算 score = u - η c
        candidates = []  # 每个元素是 (score, p_idx, sat_obj, slot_idx, utility)

        loop_t0 = time.perf_counter()
        for p_idx, p in enumerate(packets):
            for s in visible_sats:
                num_slots = sat_slot_count[s.name]
                slot_len = sat_slot_duration[s.name]

                for k in range(num_slots):
                    utility = p.weight / 1.0
                    target = utility - eta * s.cost
                    #是否保留小于0的目标
                    # if target < 0:
                    #     continue
                    score = utility - eta * s.cost

                    candidates.append((score, p_idx, s, k, utility, target))
        self._record_timing("packet_loop", time.perf_counter() - loop_t0)

        # 按 score 从大到小排序
        candidates.sort(key=lambda x: x[0], reverse=True)

        assigned_packets = set()  # 已经选过的 packet 下标
        used_slots = set()  # 已经占用的 slot (sat_name, slot_idx)
        total_cost = 0.0
        total_power = 0.0
        total_utility = 0.0
        plan = []  # [(score, p, s, utility), ...]

        # 4) 贪心选取
        for score, p_idx, s, slot_idx, utility, target in candidates:

            # 是否保留目标值小于0的目标
            # if score < 0:
            #     continue

            # 一个包只能发一次
            if p_idx in assigned_packets:
                continue

            # 一个 slot 只能使用一次
            slot_key = (s.name, slot_idx)
            if slot_key in used_slots:
                continue

            # 预算检查
            next_cost = total_cost + s.cost
            next_power = total_power + s.power_cost
            if next_cost > c_budget or next_power > e_budget:
                continue

            # 通过所有检查，可以选择这个组合
            assigned_packets.add(p_idx)
            used_slots.add(slot_key)
            total_cost = next_cost
            total_power = next_power
            total_utility += utility


            p = packets[p_idx]
            plan.append((score, p, s, utility, target))

        return plan, total_utility, total_power, total_cost

    def _global_greedy(self, packets, satellites, e_budget, c_budget, eta, base_time, t,
                       energy_floor_budget=None):
        if self.global_greedy_backend == "python":
            return self._global_greedy_python(packets, satellites, e_budget, c_budget, eta,
                                              base_time, t, energy_floor_budget)
        if not packets:
            return [], 0.0, 0.0, 0.0
        visible, counts, _ = self._build_visible_sats_vec(satellites, base_time, t)
        # Slots of the same satellite have identical scores, cost and power.
        # Keep per-satellite stable score ordering (including floating-point ties),
        # but avoid constructing and sorting packet x satellite x slot duplicates.
        visible.sort(key=lambda s: (s.cost, s.name))
        used = set()
        plan = []
        total_u = total_e = total_c = 0.0
        budget = float("inf") if energy_floor_budget is None else max(0.0, energy_floor_budget)
        loop_t0 = time.perf_counter()
        for sat in visible:
            ranked = [(p.weight / 1.0 - eta * sat.cost, i, p.weight / 1.0)
                      for i, p in enumerate(packets)]
            ranked.sort(key=lambda row: row[0], reverse=True)
            remaining_slots = counts[sat.name]
            for target, i, utility in ranked:
                if len(used) >= len(packets) or remaining_slots == 0:
                    break
                if i in used or total_e + sat.power_cost > budget + 1e-12:
                    continue
                used.add(i)
                remaining_slots -= 1
                total_c += sat.cost
                total_e += sat.power_cost
                total_u += utility
                plan.append((target, packets[i], sat, utility, target))
            if len(used) >= len(packets):
                break
        self._record_timing("packet_loop", time.perf_counter() - loop_t0)
        return plan, total_u, total_e, total_c

    def _global_greedy_python(self, packets, satellites, e_budget, c_budget, eta, base_time, t,
                       energy_floor_budget=None):
        """
        Budget-free greedy baseline.

        DRL still passes energy/cost budgets through the normal interface, but this
        solver ignores those action budgets. It only respects energy_floor_budget,
        which is the hard alpha floor from the environment, and never uses c_budget.
        It builds all packet/satellite/slot candidates, ranks candidates by
        u - eta * c, then visits visible satellites from low cost to high cost and
        fills their slots until slots, backlog, or alpha-safe energy is exhausted.
        """
        if not packets:
            return [], 0.0, 0.0, 0.0

        visible_sats, sat_slot_count, _ = self._build_visible_sats_vec(
            satellites, base_time, t
        )
        if not visible_sats:
            return [], 0.0, 0.0, 0.0

        candidates = []

        loop_t0 = time.perf_counter()
        for p_idx, p in enumerate(packets):
            for s in visible_sats:
                num_slots = sat_slot_count[s.name]

                for k in range(num_slots):
                    utility = p.weight / 1.0
                    target = utility - eta * s.cost
                    candidates.append((target, p_idx, s, k, utility, target))
        self._record_timing("packet_loop", time.perf_counter() - loop_t0)

        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates_by_sat = {}
        for candidate in candidates:
            candidates_by_sat.setdefault(candidate[2].name, []).append(candidate)

        visible_sats.sort(key=lambda s: (s.cost, s.name))

        assigned_packets = set()
        used_slots = set()
        total_cost = 0.0
        total_power = 0.0
        total_utility = 0.0
        plan = []
        safe_energy_budget = float("inf") if energy_floor_budget is None else max(0.0, energy_floor_budget)

        for s in visible_sats:
            sat_candidates = candidates_by_sat.get(s.name, [])
            used_sat_slots = 0
            max_sat_slots = sat_slot_count[s.name]

            for score, p_idx, sat_obj, slot_idx, utility, target in sat_candidates:
                if len(assigned_packets) >= len(packets):
                    break
                if used_sat_slots >= max_sat_slots:
                    break
                if p_idx in assigned_packets:
                    continue

                slot_key = (sat_obj.name, slot_idx)
                if slot_key in used_slots:
                    continue
                if total_power + sat_obj.power_cost > safe_energy_budget + 1e-12:
                    continue

                assigned_packets.add(p_idx)
                used_slots.add(slot_key)
                used_sat_slots += 1
                total_cost += sat_obj.cost
                total_power += sat_obj.power_cost
                total_utility += utility
                plan.append((score, packets[p_idx], sat_obj, utility, target))

            if len(assigned_packets) >= len(packets):
                break

        return plan, total_utility, total_power, total_cost

    def _solve_pssa(self, packets, satellites, e_budget, c_budget, eta, base_time, t, eps=0.2, sigma=1.5  ):
        #eps控制区间划分，eps越小，划区间越小，精度越高，同时还控制着迭代次数，r log(1/ε)
        #sigma控制算法停止时机，越小求解越优，只要当前解达到 OPT 的 1/σ，就停止，同时delta(x)中也有σ参与
        if not packets or e_budget <= 0 or c_budget <= 0:
            return [], 0.0, 0.0, 0.0

        # 1) 找所有可见卫星并计算 slot
        visible_sats = []
        sat_slot_count = {}
        sat_slot_duration = {}

        for s in satellites:
            is_vis, win_len = s.is_visible(base_time, t)
            if not is_vis or win_len <= 0:
                continue

            from env import Packet
            tx_time = Packet.packet_size / s.rate
            #print("packet_size=",Packet.packet_size, "s.rate=", s.rate)
            num_slots = min(int(win_len * 60 / tx_time), self.max_slots_per_satellite)
            if num_slots == 0:
                continue
            #print("num_slots=", num_slots,end=" ")

            visible_sats.append(s)
            sat_slot_count[s.name] = num_slots
            sat_slot_duration[s.name] = win_len / num_slots

        if not visible_sats:
            return [], 0.0, 0.0, 0.0

        # 输出时隙信息
        total_slots = sum(sat_slot_count.values())
        #print(f"可见卫星数={len(visible_sats)}, 总时隙数={total_slots}, 各卫星时隙: {dict(sat_slot_count)}")
        # 2) 构造 ground set Ω：仅保留 ν(x) > 0 的候选
        candidates = []
        min_power = float("inf")
        min_cost = float("inf")

        loop_t0 = time.perf_counter()
        for p_idx, p in enumerate(packets):
            for s in visible_sats:
                min_power = min(min_power, s.power_cost)
                min_cost = min(min_cost, s.cost)

                num_slots = sat_slot_count[s.name]
                for k in range(num_slots):
                    utility = p.weight
                    target = utility - eta * s.cost
                    if target <= 0:
                        continue

                    candidates.append(
                        {
                            "p_idx": p_idx,
                            "s": s,
                            "slot_idx": k,
                            "utility": utility,
                            "target": target,  # ν(x)
                        }
                    )
        self._record_timing("packet_loop", time.perf_counter() - loop_t0)
        if not candidates or min_power == float("inf") or min_cost == float("inf"):
            return [], 0.0, 0.0, 0.0

        # 3) 预计算 r（解的最大可能规模）

        r_e = e_budget / max(min_power, 1e-9)
        r_c = c_budget / max(min_cost, 1e-9)
        r = max(1, int(min(r_e, r_c)))

        # 4) Guess-OPT，生成候选最优值集合
        nu_hat = max(x["target"] for x in candidates)
        if nu_hat <= 0:
            return [], 0.0, 0.0, 0.0

        gopt = [] #(2到20之间)

        log_base = math.log(1.0 + eps)
        i_start = math.floor(math.log(nu_hat) / log_base)
        i_end = math.ceil(math.log(r * nu_hat) / log_base)
        for i in range(i_start, i_end + 1):
            gopt.append((1.0 + eps) ** i)

        #print("len(gopt)=", len(gopt),"len(packets)=", len(packets) ,"len(candidates)=", len(candidates))
        # 5) 遍历候选 OPT，用梯度驱动的 swap 更新
        best_X = []
        best_value = float("-inf")

        # 预建索引：便于通过 (p_idx)/(slot) 快速找到 X 中的冲突元素
        loop_t0 = time.perf_counter()
        for OPT in gopt:
            X = []
            nX = 0
            by_packet = {}
            by_slot = {}
            total_target = 0.0
            total_cost = 0.0
            total_power = 0.0

            # while: 达到 1/sigma 的目标值 或 迭代次数达上限
            max_iter = max(1, int(r * math.log(1.0 / eps)))
            while total_target < (OPT / max(sigma, 1e-9)) and nX < max_iter:
                best_y = None
                best_swap = None
                best_gain = float("-inf")

                # q(X) 与 f(X) 用于梯度计算
                ge_X = total_power / e_budget
                gc_X = total_cost / c_budget
                q_X = 2.0 - (ge_X + gc_X)
                if q_X <= 0:
                    break

                # 在 Ω \ X 中挑选一个最优 swap
                for y in candidates:
                    p_idx = y["p_idx"]
                    s = y["s"]
                    slot_idx = y["slot_idx"]

                    # 跳过已经在 X 内的同元素
                    if by_packet.get(p_idx) == y and by_slot.get((s.name, slot_idx)) == y:
                        continue

                    # matroid 交换集合：冲突包/冲突 slot
                    swap_set = []
                    conflict_p = by_packet.get(p_idx)
                    conflict_s = by_slot.get((s.name, slot_idx))
                    if conflict_p is not None:
                        swap_set.append(conflict_p)
                    if conflict_s is not None and conflict_s is not conflict_p:
                        swap_set.append(conflict_s)

                    # 预算检查（swap 后）
                    new_cost = total_cost + s.cost
                    new_power = total_power + s.power_cost
                    for z in swap_set:
                        new_cost -= z["s"].cost
                        new_power -= z["s"].power_cost
                    if new_cost > c_budget or new_power > e_budget:
                        continue

                    # 计算梯度
                    eps_y = s.power_cost / e_budget
                    rho_y = s.cost / c_budget
                    delta_y = sigma * q_X * y["target"] - (OPT - sigma * total_target) * (eps_y + rho_y)

                    delta_swap = 0.0
                    for z in swap_set:
                        eps_z = z["s"].power_cost / e_budget
                        rho_z = z["s"].cost / c_budget
                        delta_z = sigma * q_X * z["target"] - (OPT - sigma * total_target) * (eps_z + rho_z)
                        delta_swap += delta_z

                    gain = delta_y - delta_swap
                    if gain > best_gain:
                        best_gain = gain
                        best_y = y
                        best_swap = swap_set

                # 如果找不到可更新的元素，退出
                if best_y is None:
                    break

                # 执行 swap 更新
                if best_swap:
                    for z in best_swap:
                        X.remove(z)
                        total_target -= z["target"]
                        total_cost -= z["s"].cost
                        total_power -= z["s"].power_cost
                        by_packet.pop(z["p_idx"], None)
                        by_slot.pop((z["s"].name, z["slot_idx"]), None)

                X.append(best_y)
                total_target += best_y["target"]
                total_cost += best_y["s"].cost
                total_power += best_y["s"].power_cost
                by_packet[best_y["p_idx"]] = best_y
                by_slot[(best_y["s"].name, best_y["slot_idx"])] = best_y
                nX += 1

                # 删除负梯度元素
                ge_X = total_power / e_budget
                gc_X = total_cost / c_budget
                q_X = 2.0 - (ge_X + gc_X)
                if q_X <= 0:
                    break

                to_remove = []
                for z in X:
                    eps_z = z["s"].power_cost / e_budget
                    rho_z = z["s"].cost / c_budget
                    delta_z = sigma * q_X * z["target"] - (OPT - sigma * total_target) * (eps_z + rho_z)
                    if delta_z <= 0:
                        to_remove.append(z)
                if to_remove:
                    for z in to_remove:
                        X.remove(z)
                        total_target -= z["target"]
                        total_cost -= z["s"].cost
                        total_power -= z["s"].power_cost
                        by_packet.pop(z["p_idx"], None)
                        by_slot.pop((z["s"].name, z["slot_idx"]), None)


            # 更新全局最好解
            if total_target > best_value:
                best_value = total_target
                best_X = list(X)
        self._record_timing("opt_loop", time.perf_counter() - loop_t0)
        # 6) 返回
        assigned_packets = set()
        used_slots = set()
        total_cost = 0.0
        total_power = 0.0
        total_utility = 0.0
        plan = []

        for x in best_X:
            p_idx = x["p_idx"]
            s = x["s"]
            slot_idx = x["slot_idx"]
            utility = x["utility"]
            target = x["target"]

            if p_idx in assigned_packets:
                continue
            slot_key = (s.name, slot_idx)
            if slot_key in used_slots:
                continue

            assigned_packets.add(p_idx)
            used_slots.add(slot_key)
            total_cost += s.cost
            total_power += s.power_cost
            total_utility += utility
            plan.append((target, packets[p_idx], s, utility, target))

        # print("使用的参数")
        # print("eps=", eps, "sigma=", sigma)
        # print("len(plan)=",len(plan))
        return plan, total_utility, total_power, total_cost

    # ══════════════════════════════════════════════════════════════════
    #  numpy 向量化共用工具（greedy_vec / pssa_vec 专用）
    # ══════════════════════════════════════════════════════════════════

    def _build_visible_sats_vec(self, satellites, base_time, t, max_slots=None):
        """找可见卫星，计算时隙数和时隙时长。max_slots 可限制每星最大时隙数。"""
        from env import Packet
        visible_sats, sat_slot_count, sat_slot_duration = [], {}, {}
        for s in satellites:
            is_vis, win_len = s.is_visible(base_time, t)
            if not is_vis or win_len <= 0:
                continue
            tx_time = Packet.packet_size / s.rate
            num_slots = min(int(win_len * 60.0 / tx_time), self.max_slots_per_satellite)
            if num_slots <= 0:
                continue
            if max_slots is not None:
                num_slots = min(num_slots, max_slots)
            visible_sats.append(s)
            sat_slot_count[s.name] = num_slots
            sat_slot_duration[s.name] = win_len / num_slots
        return visible_sats, sat_slot_count, sat_slot_duration
