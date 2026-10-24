import numpy as np
import pandas as pd
import re
import os
import json
from datetime import date, datetime
from pathlib import Path

# 创建一个全局 DataFrame，用于存储所有数据
step_output_data = []
avg_output_data = []

def print_info_to_excel(info, episode, step):
    formatted_info = {}
    for key, value in info.items():
        if isinstance(value, (float, np.floating)):
            formatted_info[key] = round(float(value), 4)
        else:
            formatted_info[key] = value
    formatted_info["episode"] = episode
    formatted_info["step"] = step

    step_output_data.append(formatted_info)


def print_avg_info_to_excel(
        total_reward, total_utility, total_cost, total_power, total_eta,
        total_violate1_cnt, total_violate2_cnt ,total_send_pks, total_backlog_num, total_e_budget, total_c_budget,
        episode, step_count,
        avg_critic_loss, min_battery_percent=np.nan
):
    avg_info = {
        "total_reward": total_reward,
        "episode": episode,
        "reward_tu/tc": round(total_utility / total_cost, 3) if total_cost != 0 else 0,
        "total_utility": total_utility,
        "total_cost": total_cost,
        "eta": round(total_eta / step_count, 4),
        "violate1": total_violate1_cnt,
        "violate2": total_violate2_cnt,
        "total_send_pks": total_send_pks,
        "avg_backlog": total_backlog_num // step_count,
        "avg_power": round(total_power / step_count, 4),
        # 百分数（0–100）：本轮每个会话耗电后、采能前的最低电量。
        "min_battery_percent": round(float(min_battery_percent), 4),
        "avg_e_budget": round(total_e_budget / step_count, 4),
        "avg_c_budget": round(total_c_budget / step_count, 4),

        "avg_critic_loss": round(float(avg_critic_loss), 2) if not np.isnan(avg_critic_loss) else np.nan,
    }
    avg_output_data.append(avg_info)




def _config_excel_value(value):
    """保留数值类型，将日期、列表等配置转换为Excel可读的单元格值。"""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if value is None:
        return ""
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def save_to_excel(reward_method, info_dir="./info", config_data=None):
    avg_output_data.append({"reward_method":reward_method})
    folder = info_dir

    os.makedirs(folder, exist_ok=True)

    # 找出 infoN.xlsx 的最大 N
    max_n = 0
    for name in os.listdir(folder):
        m = re.fullmatch(r"info(\d+)\.xlsx", name)
        if m:
            max_n = max(max_n, int(m.group(1)))

    # 新文件名 = 最大编号 + 1
    filename = os.path.join(folder, f"info{max_n + 1}.xlsx")

    df1 = pd.DataFrame(step_output_data)
    df2 = pd.DataFrame(avg_output_data)

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df1.to_excel(writer, sheet_name="step", index=False)
        df2.to_excel(writer, sheet_name="avg", index=False)
        if config_data is not None:
            config_rows = [
                {"parameter": name, "value": _config_excel_value(value)}
                for name, value in config_data.items()
            ]
            pd.DataFrame(config_rows, columns=["parameter", "value"]).to_excel(
                writer, sheet_name="sheet3", index=False
            )
            sheet = writer.sheets["sheet3"]
            sheet.freeze_panes = "A2"
            sheet.column_dimensions["A"].width = 32
            sheet.column_dimensions["B"].width = 80
    step_output_data.clear()
    avg_output_data.clear()
    print("Saved: info",max_n + 1)
