import os
import json
import pandas as pd
from datetime import datetime, timedelta
from skyfield.api import EarthSatellite, load

from train_config import TrainConfig

# 输入输出配置（统一来自 TrainConfig）
cfg = TrainConfig()

JSON_PATH = cfg.sat_json_path                  # 卫星TLE数据json
EXCEL_PATH = "satellite_visibility_hainan_2025_10.xlsx"         # 可见窗口表（预先打表）
ground_station = cfg.ground_station            # 地面站位置（wgs84.latlon / Topos 都可用）
ELEVATION_DEG = float(cfg.elevation_angle)     # 仰角阈值（度）

# 时间范围（UTC）
# 默认：从 cfg.base_time 所在日期的 00:00:00 开始，覆盖 max_steps 个 step，
# 每个 step 对应 session 分钟，所以总跨度 = max_steps * session 分钟
start_date = datetime(cfg.base_time.year, cfg.base_time.month, cfg.base_time.day, 0, 0, 0)
end_date = start_date + timedelta(minutes=cfg.max_steps * cfg.session)

ts = load.timescale()

# 1. 读取卫星 JSON
with open(JSON_PATH, "r", encoding="utf-8") as f:
    sat_data = json.load(f)

# 2. 读取已有 Excel（若存在）
if os.path.exists(EXCEL_PATH):
    df_existing = pd.read_excel(EXCEL_PATH)
else:
    df_existing = pd.DataFrame()

existing_names = set(df_existing.columns)  # 已存在的卫星列名（按 OBJECT_NAME）

# 3. 计算“新增卫星”的可见窗口
new_cols = {}  # name -> list[str]

for sat in sat_data:
    name = sat.get("OBJECT_NAME")
    line1 = sat.get("LINE1")
    line2 = sat.get("LINE2")

    if not name or not line1 or not line2:
        continue  # 数据不完整，跳过

    # 已存在就跳过
    if name in existing_names:
        continue

    satellite = EarthSatellite(line1, line2, name, ts)

    rise_set_list = []
    d = start_date
    while d < end_date:
        # 当天 UTC 00:00:00 ~ 23:59:59
        t0 = ts.utc(d.year, d.month, d.day, 0, 0, 0)
        t1 = ts.utc(d.year, d.month, d.day, 23, 59, 59)

        times, events = satellite.find_events(
            ground_station, t0, t1, altitude_degrees=ELEVATION_DEG
        )

        # 如果一开始就处于可见（没有 rise 事件），需要把 t0 作为“rise_time”
        rise_time = None
        try:
            alt0, _, _ = (satellite - ground_station).at(t0).altaz()
            if alt0.degrees >= ELEVATION_DEG:
                rise_time = t0.utc_strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            # altaz 计算失败也不影响主流程，按原逻辑继续
            pass

        for t, e in zip(times, events):
            if e == 0:  # 升起
                rise_time = t.utc_strftime("%Y-%m-%d %H:%M:%S")
            elif e == 2:  # 落下
                if rise_time is not None:
                    set_time = t.utc_strftime("%Y-%m-%d %H:%M:%S")
                    rise_set_list.append(f"{rise_time}  ,  {set_time}")
                    rise_time = None

        d += timedelta(days=1)

    new_cols[name] = rise_set_list

# 4. 将新增列拼到原 Excel，并写回
if new_cols:
    # 新增列 DataFrame
    df_new = pd.DataFrame({k: pd.Series(v) for k, v in new_cols.items()})

    # 横向拼接：按 index 对齐，自动补 NaN
    df_out = pd.concat([df_existing, df_new], axis=1)
else:
    df_out = df_existing

df_out.to_excel(EXCEL_PATH, index=False)

print(f"完成：新增 {len(new_cols)} 颗卫星；总列数 {df_out.shape[1]}；输出 -> {EXCEL_PATH}")
if new_cols:
    print("本次新增卫星：", list(new_cols.keys())[:10], ("..." if len(new_cols) > 10 else ""))
