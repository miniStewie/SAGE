"""Compiled direct-action decoding. Preserve order, rounding and budget checks."""
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(**kwargs):
        return lambda function: function


@njit(cache=True, fastmath=False)
def select_combos(actions, weights, costs, powers, slots, energy_budget,
                  max_packets, max_sats, max_slots, continuous, cycle):
    """Return one executed triple per head; -1 means no transmission."""
    n_packets, n_sats = len(weights), len(costs)
    used_packets = np.zeros(n_packets, dtype=np.bool_)
    used_slots = np.zeros((n_sats, max_slots), dtype=np.bool_)
    executed = np.full((len(actions), 3), -1, dtype=np.int64)
    replay = np.empty((len(actions), 3), dtype=np.float32)
    utility = cost = power = 0.0
    noops = skipped = 0
    for k in range(len(actions)):
        if continuous:
            # Match Python float(np.clip(float32, ...)) then np.rint (ties to even).
            p = int(np.rint((min(max(actions[k, 0], -1.0), 1.0) + 1.0) * 0.5 * max_packets))
            s = int(np.rint((min(max(actions[k, 1], -1.0), 1.0) + 1.0) * 0.5 * (max_sats - 1)))
            slot = int(np.rint((min(max(actions[k, 2], -1.0), 1.0) + 1.0) * 0.5 * (max_slots - 1)))
        else:
            p = int(min(max(actions[k, 0], 0), max_packets))
            s = int(min(max(actions[k, 1], 0), max_sats - 1))
            slot = int(min(max(actions[k, 2], 0), max_slots - 1))
        replay[k, 0], replay[k, 1], replay[k, 2] = 1.0, -1.0, -1.0
        if p >= max_packets:
            noops += 1
            continue
        if n_packets == 0 or n_sats == 0:
            skipped += 1
            continue
        remaining = energy_budget - power
        if cycle:
            start = p % n_packets
            p = -1
            for offset in range(n_packets):
                candidate = (start + offset) % n_packets
                if not used_packets[candidate]:
                    p = candidate
                    break
            found_s = found_slot = -1
            if p >= 0:
                start_s = s % n_sats
                for offset in range(n_sats):
                    candidate_s = (start_s + offset) % n_sats
                    if powers[candidate_s] > remaining + 1e-12 or slots[candidate_s] <= 0:
                        continue
                    start_slot = slot % slots[candidate_s]
                    for offset_slot in range(slots[candidate_s]):
                        candidate_slot = (start_slot + offset_slot) % slots[candidate_s]
                        if not used_slots[candidate_s, candidate_slot]:
                            found_s, found_slot = candidate_s, candidate_slot
                            break
                    if found_s >= 0:
                        break
            if p < 0 or found_s < 0:
                skipped += 1
                continue
            s, slot = found_s, found_slot
        else:
            if (p >= n_packets or used_packets[p] or s >= n_sats
                    or slot >= slots[s] or used_slots[s, slot]
                    or powers[s] > remaining + 1e-12):
                skipped += 1
                continue
        used_packets[p] = True
        used_slots[s, slot] = True
        executed[k, 0], executed[k, 1], executed[k, 2] = p, s, slot
        utility += weights[p]
        cost += costs[s]
        power += powers[s]
        if continuous:
            replay[k, 0] = p / max_packets * 2.0 - 1.0
            replay[k, 1] = s / (max_sats - 1) * 2.0 - 1.0 if max_sats > 1 else -1.0
            replay[k, 2] = slot / (max_slots - 1) * 2.0 - 1.0 if max_slots > 1 else -1.0
    return executed, replay, utility, cost, power, noops, skipped


def select_actions(env, actions, packets, visible, slots, budget, continuous):
    # Float32 actions are promoted before arithmetic to reproduce scalar decoding.
    actions = np.asarray(actions, dtype=np.float64).reshape(-1, 3)
    result = select_combos(
        actions, np.asarray([p.weight for p in packets], dtype=np.float64),
        np.asarray([s.cost for s, _ in visible], dtype=np.float64),
        np.asarray([s.power_cost for s, _ in visible], dtype=np.float64),
        np.asarray(slots, dtype=np.int64), float(budget), env.max_packets,
        env.max_visible_sats, env.max_direct_slots, continuous,
        continuous or env.invalid_action_strategy == 'cycle',
    )
    executed, replay, utility, cost, power, noops, skipped = result
    selections = [(int(p), packets[p], visible[s][0], int(slot), packets[p].weight)
                  for p, s, slot in executed if p >= 0]
    return selections, utility, cost, power, noops, skipped, replay.reshape(-1)


def make_state(env, c_max):
    """Batch numpy normalization; retain the reference feature order and float32 output."""
    env._update_link_energy()
    utilities = np.asarray([p.utility for p in env.backlog], dtype=np.float64)
    if len(utilities):
        summaries = [np.max(utilities), np.percentile(utilities, 25), np.mean(utilities),
                     np.percentile(utilities, 75), np.min(utilities)]
    else:
        summaries = [0.] * 5
    backlog_ref = float(env.pkt_lmt) if env.pkt_lmt and env.pkt_lmt > 0 else 100.
    state = np.zeros(13 + 4 * env.max_packets + 7 * env.max_visible_sats, dtype=np.float64)
    state[:5] = np.clip([env.e_t / env.e_init, env.alpha, c_max / 40.,
                         env.eta / env.eta_max, len(env.backlog) / max(backlog_ref, 1.)], 0., 1.)
    denominator = np.log1p(20.)
    state[5:10] = np.clip(np.log1p(np.maximum(summaries, 0.)) / denominator, 0., 1.)
    state[10:13] = np.clip([np.log1p(max(float(env.last_u), 0.)) / denominator,
                             float(env.last_c) / 3., float(env.last_power) / max(env.e_init, 1e-6)], 0., 1.)
    packets = env.backlog[:env.max_packets]
    packet_features = state[13:13 + 4 * env.max_packets].reshape(-1, 4)
    if packets:
        packet_features[:len(packets), 0] = 1.
        values = np.asarray([(p.utility, p.weight) for p in packets], dtype=np.float64)
        packet_features[:len(packets), 1:3] = np.clip(np.log1p(np.maximum(values, 0.)) / denominator, 0., 1.)
        ages = np.asarray([max(0., float(env.t - p.birth_t)) for p in packets])
        packet_features[:len(packets), 3] = np.clip(ages / max(float(env.max_steps), 1.), 0., 1.)
    visible = env._visible_sats_with_windows()[:env.max_visible_sats]
    counts = env._slot_counts(visible)
    satellite_features = state[13 + 4 * env.max_packets:].reshape(-1, 7)
    for i, (sat, minutes) in enumerate(visible):
        satellite_features[i] = [1., *np.clip([float(sat.rate) / 6000.], 0., 1.),
                                 float(sat.power_cost) / (float(sat.power_cost) + .01),
                                 *np.clip([float(sat.delay) / 10., float(sat.cost) / 3.,
                                           float(minutes) / 120., float(counts[i]) / max(float(env.max_direct_slots), 1.)], 0., 1.)]
    return state.astype(np.float32)
