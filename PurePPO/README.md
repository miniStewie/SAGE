# PurePPO direct discrete-action baseline

This folder contains the direct-action PPO baseline for SAGE.

The policy emits a fixed set of independent categorical choices. Each hourly
session has up to 15 `(packet_idx, sat_idx, slot_idx)` tuples:

- `packet_idx`: 51 classes, where `0-49` select backlog packet slots and `50`
  is no-op.
- `sat_idx`: 5 classes for the current visible satellite slots.
- `slot_idx`: 15 classes for receiver slots.

The actor therefore produces `15 * (51 + 5 + 15) = 1065` logits. The sampled
integer action has shape `(15, 3)`.

The environment reuses the same fixed state representation, satellite data,
visibility table, solar-energy table, eta update, and reward function as the
direct TD3 baseline. Invalid integer choices are deterministically projected by
cycling from the sampled index to the next usable packet or satellite-slot
choice. No objective-based ranking or repair search is added.

Invalid action handling is controlled by `invalid_action_strategy`:

- `cycle`: the default behavior, cycling from an invalid sampled index to the
  next usable packet or satellite-slot.
- `discard`: strict behavior, discarding invalid, duplicate, unavailable, or
  energy-infeasible combinations without searching for replacements.

Example:

```powershell
cd C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\PurePPO
python train.py --seed 42 --C_base 1.5 --reward_method u/c-lambda1 --info_dir ./pure_ppo_info
```

Strict invalid-action discard mode:

```powershell
python train.py --seed 42 --C_base 1.5 --invalid_action_strategy discard --info_dir ./pure_ppo_discard_info
```

Short smoke test:

```powershell
python train.py --seed 1 --C_base 1.5 --total_episode 2 --max_steps 24 --ppo_epochs 2 --minibatch_size 16 --print_info_cnt 100 --info_dir ./smoke_info
```
