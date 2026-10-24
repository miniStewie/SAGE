# PureTD3 direct-action baseline

This folder contains the direct-action TD3 baseline for SAGE.

PureTD3 does not call `AssignmentSolver`, `pssa`, or `greedy`. The actor outputs
a fixed 45-dimensional continuous action. Every three values are decoded into
one `(packet_idx, sat_idx, slot_idx)` tuple, so each hourly session can emit up
to 15 packet-satellite-slot combinations.

The environment reuses the same satellite data, visibility table, solar-energy
table, packet generation style, eta update, and reward function used by the
existing TD3/SAGE experiment. Invalid or out-of-range direct actions are
deterministically projected to legal indices without objective-based sorting or
combinatorial search.

Example:

```powershell
cd C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\PureTD3
python train.py --seed 42 --C_base 1.5 --reward_method u/c-lambda1 --info_dir ./pure_td3_info
```

Short smoke test:

```powershell
python train.py --seed 1 --C_base 1.5 --total_episode 2 --max_steps 24 --warmup_steps 10 --print_info_cnt 100 --info_dir ./smoke_info
```
