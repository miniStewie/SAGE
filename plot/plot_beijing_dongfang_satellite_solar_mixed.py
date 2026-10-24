"""北京、东方卫星可见性与太阳能采集：2×2 混合图（卫星柱状图、集能折线图）。

运行后输出组合 PNG/PDF；原有四个绘图脚本及其输出不受影响。
卫星按整个会话内曾经可见的不同卫星数统计，太阳能按每小时 Wh 统计。
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TD3_ROOT = next(
    (parent for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents)
     if (parent / "train_config.py").is_file() and (parent / "env.py").is_file()),
    None,
)
if TD3_ROOT is None:
    raise FileNotFoundError(f"无法从 {SCRIPT_DIR} 向上找到 TD3 源码目录")
sys.path.insert(0, str(TD3_ROOT))

from train_config import TrainConfig
from env import Satellite


# ============================================================
# 1. 输入、输出路径（硬编码，集中在这里修改）
# ============================================================
SAT_JSON_PATH = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\dataset\satellite_data_50.json"
BEIJING_VISIBILITY_PATH = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\dataset\satellite_visibility_beijing_2025_05.xlsx"
HAINAN_VISIBILITY_PATH = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\dataset\satellite_visibility_hainan_2025_10.xlsx"
BEIJING_SOLAR_PATH = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\dataset\北京25年5月每时太阳辐射强度.xls"
HAINAN_SOLAR_PATH = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\dataset\海南省东方市25年10月每时太阳辐射强度.xls"
OUTPUT_DIR = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\TD3\plot\plot_result_combined"
OUTPUT_NAME = "beijing-dongfang-satellite-solar-mixed"

# ============================================================
# 2. 计算参数：None 表示沿用 TrainConfig，可以填数值独立调整
# ============================================================
SESSION_MINUTES = None       # 当前配置为 30 分钟；填 60 可绘制一小时会话
SOLAR_ARRAY_AREA = None      # 当前配置为 0.5 m²
CONVERSION_EFFICIENCY = None # 当前配置为 0.18
PWM_EFFICIENCY = 0.64

# 地点和月份独立设置，不受 TrainConfig.base_time/max_steps 影响。
SCENARIOS = [
    {
        "name": "Beijing",
        "start": datetime(2025, 5, 1),
        "end": datetime(2025, 6, 1),
        "visibility_path": BEIJING_VISIBILITY_PATH,
        "solar_path": BEIJING_SOLAR_PATH,
    },
    {
        "name": "Dongfang",
        "start": datetime(2025, 10, 1),
        "end": datetime(2025, 11, 1),
        "visibility_path": HAINAN_VISIBILITY_PATH,
        "solar_path": HAINAN_SOLAR_PATH,
    },
]

# ============================================================
# 3. 布局与样式：左北京、右东方；上卫星、下太阳能
# ============================================================
FIG_SIZE = (3.5, 4.0)       # 恢复第一版整体尺寸；仅让子图伸展以利用中间留白
FONT_FAMILY = "Helvetica"
BASE_FONT_SIZE = 7.5
TITLE_FONT_SIZE = 7.5
LABEL_FONT_SIZE = 7.5
TICK_FONT_SIZE = 6.5
AXIS_LINEWIDTH = 0.6
SAVE_DPI = 600

SUBPLOT_LEFT = 0.10
SUBPLOT_RIGHT = 0.985
SUBPLOT_BOTTOM = 0.055
SUBPLOT_TOP = 0.95
SUBPLOT_WSPACE = 0.08
SUBPLOT_HSPACE = 0.22
X_TICK_DAYS = [1, 15, 31]

SHARE_Y_SCALE = True        # 同一行使用相同 y 轴范围，方便对比两地
AUTO_Y_AXIS = True
SATELLITE_Y_LIM = (0, 21)   # AUTO_Y_AXIS=False 时使用
SOLAR_Y_LIM = (0, 50)
Y_MARGIN_RATIO = 0.20        # 顶部留白，给图内右上角的子图标注预留空间

SATELLITE_COLOR = "#6BAED6"
SATELLITE_EDGE_COLOR = "#2171B5"
SATELLITE_BAR_WIDTH_RATIO = 0.8
SOLAR_COLOR = "#E07A72"
SOLAR_LINE_WIDTH = 0.55
SOLAR_LINE_ALPHA = 0.9
SHOW_GRID = False


def read_satellite_counts(scenario, sat_names, session_minutes):
    """与原卫星脚本相同：窗口和会话存在正时长交集就计数，每颗星计一次。"""
    Satellite.load_visibility_table(scenario["visibility_path"])
    windows = Satellite.VISIBILITY_WINDOWS
    missing = [name for name in sat_names if name not in windows]
    if missing:
        raise ValueError(f"{scenario['name']} 可见性表缺少卫星：{missing}")

    start, end = scenario["start"], scenario["end"]
    if end <= start:
        raise ValueError("绘图结束时间必须晚于开始时间")
    count = int(np.ceil((end - start).total_seconds() / (session_minutes * 60)))
    timestamps = [start + timedelta(minutes=i * session_minutes) for i in range(count)]
    values = []
    for session_start in timestamps:
        session_end = min(session_start + timedelta(minutes=session_minutes), end)
        values.append(sum(
            any(a < session_end and b > session_start for a, b in windows[name])
            for name in sat_names
        ))
    return timestamps, np.asarray(values, dtype=int)


def read_solar_energy(scenario, area, efficiency):
    """保留原太阳能脚本口径：第 7 列 UTC+8 时间，第 4 列小时辐射量 J/m²。"""
    frame = pd.read_excel(scenario["solar_path"])
    times = pd.to_datetime(frame.iloc[:, 6])
    radiation = frame.iloc[:, 3].to_numpy(dtype=float)
    if times.isna().any() or not np.isfinite(radiation).all():
        raise ValueError(f"{scenario['name']} 日照表存在无效时间或辐射值")
    order = np.argsort(times.to_numpy())
    energy = area * efficiency * PWM_EFFICIENCY * radiation / 3600.0
    return times.iloc[order].reset_index(drop=True), energy[order]


def english_date(x, _pos):
    date = mdates.num2date(x)
    months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    return f"{months[date.month - 1]} {date.day:02d}"


def main():
    cfg = TrainConfig()
    minutes = float(cfg.session if SESSION_MINUTES is None else SESSION_MINUTES)
    area = float(cfg.solar_array_area if SOLAR_ARRAY_AREA is None else SOLAR_ARRAY_AREA)
    efficiency = float(cfg.conversion_efficiency if CONVERSION_EFFICIENCY is None else CONVERSION_EFFICIENCY)
    if not all(np.isfinite(v) and v > 0 for v in (minutes, area, efficiency)):
        raise ValueError("会话间隔、太阳能板面积和转换效率必须是有限正数")

    with open(SAT_JSON_PATH, encoding="utf-8") as stream:
        names = list(dict.fromkeys(
            str(sat["OBJECT_NAME"]).strip() for sat in json.load(stream) if sat.get("OBJECT_NAME")
        ))
    if not names:
        raise ValueError("卫星列表为空")

    data = []
    for scenario in SCENARIOS:
        sat_times, sat_values = read_satellite_counts(scenario, names, minutes)
        solar_times, solar_values = read_solar_energy(scenario, area, efficiency)
        data.append(((sat_times, sat_values), (solar_times, solar_values)))
        print(f"{scenario['name']}: {len(sat_values)} sessions, "
              f"mean satellites={sat_values.mean():.4f}, "
              f"solar peak={solar_values.max():.4f} Wh")
    print(f"Session={minutes:g} min; solar area={area:g} m²; efficiency={efficiency:g}")

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "Arial", "DejaVu Sans"],
        "font.size": BASE_FONT_SIZE,
        "axes.linewidth": AXIS_LINEWIDTH,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE)
    for row in range(2):
        shared_peak = max(data[col][row][1].max() for col in range(2))
        for col, scenario in enumerate(SCENARIOS):
            ax = axes[row, col]
            times, values = data[col][row]
            satellite_panel = row == 0
            if satellite_panel:
                # 卫星保留原柱状图样式。
                ax.bar(
                    times, values,
                    width=pd.Timedelta(minutes=minutes * SATELLITE_BAR_WIDTH_RATIO),
                    color=SATELLITE_COLOR, edgecolor=SATELLITE_EDGE_COLOR,
                    linewidth=0.12, alpha=0.9,
                )
            else:
                # 集能使用原始逐小时数据的折线，不平滑、不重采样。
                ax.plot(
                    times, values, color=SOLAR_COLOR,
                    linewidth=SOLAR_LINE_WIDTH, alpha=SOLAR_LINE_ALPHA,
                )
            if AUTO_Y_AXIS:
                peak = shared_peak if SHARE_Y_SCALE else values.max()
                ax.set_ylim(0, max(1.0, peak * (1 + Y_MARGIN_RATIO)))
            else:
                ax.set_ylim(SATELLITE_Y_LIM if satellite_panel else SOLAR_Y_LIM)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=satellite_panel))
            # 只标注目标月份，避免日照表末尾的次月 1 日与本月 31 日标签重叠。
            ax.set_xticks([
                scenario["start"].replace(day=day) for day in X_TICK_DAYS
            ])
            ax.xaxis.set_major_formatter(FuncFormatter(english_date))
            ax.set_xlim(scenario["start"], times[-1] if satellite_panel else times.iloc[-1])
            # 首尾日期向图内对齐，缩小列间距后也不会在中间挤在一起。
            date_labels = ax.get_xticklabels()
            date_labels[0].set_horizontalalignment("left")
            date_labels[-1].set_horizontalalignment("right")
            ax.tick_params(direction="in", length=2, width=0.6, labelsize=TICK_FONT_SIZE)
            if SHOW_GRID:
                ax.grid(True, axis="y", alpha=0.2)
            else:
                ax.grid(False)
            ax.text(
                0.96, 0.96, scenario["name"],
                transform=ax.transAxes, ha="right", va="top",
                fontsize=TITLE_FONT_SIZE,
            )
            if col == 0:
                ax.set_ylabel("Satellite count" if satellite_panel else "Harvested energy (Wh)", fontsize=LABEL_FONT_SIZE)
            elif SHARE_Y_SCALE:
                ax.tick_params(labelleft=False)

    fig.subplots_adjust(
        left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT,
        bottom=SUBPLOT_BOTTOM, top=SUBPLOT_TOP,
        wspace=SUBPLOT_WSPACE, hspace=SUBPLOT_HSPACE,
    )
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        target = output_dir / f"{OUTPUT_NAME}.{extension}"
        fig.savefig(target, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.04)
        print(f"Saved: {target}")
    plt.close(fig)
    return fig, axes, data


if __name__ == "__main__":
    main()
