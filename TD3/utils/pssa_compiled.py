"""Compiled scalar PSSA: preserve candidate/swap order and strict comparisons.

Do not enable fastmath or reorder sums: rounding affects budget checks and ties.
The Python reference remains AssignmentSolver._solve_pssa.
"""
import math
import time
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _search(pids, slots, targets, costs, powers, guesses, eb, cb, r, npackets, nslots,
            eps, sigma):
    max_iter = max(1, int(r * math.log(1.0 / eps)))
    best_value = -np.inf
    best = np.empty(0, dtype=np.int64)
    for opt in guesses:
        chosen = np.empty(len(targets), dtype=np.int64)
        length = 0
        by_p = np.full(npackets, -1, dtype=np.int64)
        by_s = np.full(nslots, -1, dtype=np.int64)
        total = cost = power = 0.0
        iteration = 0
        while total < opt / max(sigma, 1e-9) and iteration < max_iter:
            q = 2.0 - (power / eb + cost / cb)
            if q <= 0:
                break
            best_y = -1
            best_gain = -np.inf
            for y in range(len(targets)):
                cp = by_p[pids[y]]
                cs = by_s[slots[y]]
                if cp == y and cs == y:
                    continue
                if cs == cp:
                    cs = -1
                nc = cost + costs[y]
                ne = power + powers[y]
                if cp >= 0:
                    nc -= costs[cp]
                    ne -= powers[cp]
                if cs >= 0:
                    nc -= costs[cs]
                    ne -= powers[cs]
                if nc > cb or ne > eb:
                    continue
                dy = sigma * q * targets[y] - (opt - sigma * total) * (powers[y] / eb + costs[y] / cb)
                ds = 0.0
                if cp >= 0:
                    ds += sigma * q * targets[cp] - (opt - sigma * total) * (powers[cp] / eb + costs[cp] / cb)
                if cs >= 0:
                    ds += sigma * q * targets[cs] - (opt - sigma * total) * (powers[cs] / eb + costs[cs] / cb)
                gain = dy - ds
                if gain > best_gain:
                    best_gain, best_y = gain, y
            if best_y < 0:
                break
            cp = by_p[pids[best_y]]
            cs = by_s[slots[best_y]]
            if cs == cp:
                cs = -1
            for z in (cp, cs):
                if z < 0:
                    continue
                for j in range(length):
                    if chosen[j] == z:
                        for k in range(j, length - 1):
                            chosen[k] = chosen[k + 1]
                        length -= 1
                        break
                total -= targets[z]
                cost -= costs[z]
                power -= powers[z]
                by_p[pids[z]] = -1
                by_s[slots[z]] = -1
            chosen[length] = best_y
            length += 1
            total += targets[best_y]
            cost += costs[best_y]
            power += powers[best_y]
            by_p[pids[best_y]] = best_y
            by_s[slots[best_y]] = best_y
            iteration += 1
            q = 2.0 - (power / eb + cost / cb)
            if q <= 0:
                break
            removals = np.empty(length, dtype=np.int64)
            nr = 0
            for j in range(length):
                z = chosen[j]
                delta = sigma * q * targets[z] - (opt - sigma * total) * (powers[z] / eb + costs[z] / cb)
                if delta <= 0:
                    removals[nr] = z
                    nr += 1
            for j in range(nr):
                z = removals[j]
                for k in range(length):
                    if chosen[k] == z:
                        for h in range(k, length - 1):
                            chosen[h] = chosen[h + 1]
                        length -= 1
                        break
                total -= targets[z]
                cost -= costs[z]
                power -= powers[z]
                by_p[pids[z]] = -1
                by_s[slots[z]] = -1
        if total > best_value:
            best_value = total
            best = chosen[:length].copy()
    return best


def solve(packets, satellites, e_budget, c_budget, eta, base_time, t, max_slots=15,
          eps=0.2, sigma=1.5, record_timing=None):
    from env import Packet
    if not packets or e_budget <= 0 or c_budget <= 0:
        return [], 0.0, 0.0, 0.0
    visible = []
    for sat in satellites:
        vis, minutes = sat.is_visible(base_time, t)
        if not vis or minutes <= 0:
            continue
        count = min(int(minutes * 60 / (Packet.packet_size / sat.rate)), max_slots)
        if count:
            visible.append((sat, count))
    if not visible:
        return [], 0.0, 0.0, 0.0
    start = time.perf_counter()
    rows = []
    # Match the original packet/satellite/slot nesting exactly.
    for pi, packet in enumerate(packets):
        for si, (sat, count) in enumerate(visible):
            target = packet.weight - eta * sat.cost
            if target > 0:
                rows.extend((pi, si * max_slots + k, target, sat.cost, sat.power_cost, si) for k in range(count))
    if record_timing is not None:
        record_timing('packet_loop', time.perf_counter() - start)
    if not rows:
        return [], 0.0, 0.0, 0.0
    a = np.array(rows, dtype=np.float64)
    r = max(1, int(min(e_budget / max(min(s.power_cost for s, _ in visible), 1e-9),
                       c_budget / max(min(s.cost for s, _ in visible), 1e-9))))
    nu = max(a[:, 2])
    log_base = math.log(1.0 + eps)
    guesses = np.array([(1.0 + eps) ** i for i in range(math.floor(math.log(nu) / log_base),
                                               math.ceil(math.log(r * nu) / log_base) + 1)])
    start = time.perf_counter()
    indices = _search(a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2], a[:, 3], a[:, 4],
                      guesses, float(e_budget), float(c_budget), r, len(packets), len(visible) * max_slots,
                      eps, sigma)
    if record_timing is not None:
        record_timing('opt_loop', time.perf_counter() - start)
    plan = []
    utility = power = cost = 0.0
    for i in indices:
        p = packets[int(a[i, 0])]
        s = visible[int(a[i, 5])][0]
        target = float(a[i, 2])
        plan.append((target, p, s, p.weight, target))
        utility += p.weight
        power += s.power_cost
        cost += s.cost
    return plan, utility, power, cost
