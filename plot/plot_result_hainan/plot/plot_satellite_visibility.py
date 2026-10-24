import sys
import os
import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 按源码文件定位 TD3，脚本移动目录后也能正确导入训练配置和环境。
TD3_ROOT = next(
    (parent for parent in (Path(SCRIPT_DIR), *Path(SCRIPT_DIR).parents)
     if (parent / "train_config.py").is_file() and (parent / "env.py").is_file()),
    None,
)
if TD3_ROOT is None:
    raise FileNotFoundError(f"无法从 {SCRIPT_DIR} 向上找到 train_config.py 和 env.py")
TD3_ROOT = str(TD3_ROOT)

if TD3_ROOT not in sys.path:
    sys.path.insert(0, TD3_ROOT)

from train_config import TrainConfig
from env import Satellite


# ============================================================
# 0. 调试入口参数
# ============================================================

# ---------- 输入、输出路径：直接修改这里 ----------
VISIBILITY_TABLE_PATH = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\dataset\satellite_visibility_hainan_2025_10.xlsx"
SAT_JSON_PATH = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\dataset\satellite_data_50.json"
PLOT_DIR = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\TD3\plot\plot_result_hainan\sat-and-sun"
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------- 海南数据的绘图时段（与可见性表一致，结束时间不包含在内） ----------
PLOT_START_TIME = datetime(2025, 10, 1)
PLOT_END_TIME = datetime(2025, 11, 1)

# ---------- 输出 ----------
OUTPUT_PNG_FILENAME = "satellite_visibility.png"
OUTPUT_PDF_FILENAME = "satellite_visibility.pdf"
SAVE_DPI = 300

# ---------- 图尺寸 ----------
FIG_WIDTH = 10
FIG_HEIGHT = 4

# ---------- 字体 ----------
FONT_FAMILY = "Helvetica"
BASE_FONT_SIZE = 16
LABEL_FONT_SIZE = 17
TICK_FONT_SIZE = 17

# ---------- 坐标轴 ----------
AXES_LINEWIDTH = 1.3
TICK_LENGTH = 5
TICK_WIDTH = 1.2

# y 轴刻度：0, 2, 4, 6, 8 ...
Y_TICK_STEP = 3
Y_TOP_MARGIN = 0.8

# x 轴英文日期显示
X_DAY_INTERVAL = 5
X_DATE_FORMAT = "%b %d"     # 例如：May 01, May 06

# ---------- 柱状图样式 ----------
BAR_WIDTH_RATIO = 0.8
BAR_COLOR = "#6BAED6"
BAR_EDGE_COLOR = "#2171B5"
BAR_LINEWIDTH = 0.25
BAR_ALPHA = 0.85

# ---------- 是否显示网格 ----------
SHOW_GRID = False


# ============================================================
# 1. 全局绘图设置
# ============================================================

def setup_matplotlib():
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}

    if FONT_FAMILY not in available_fonts:
        print(f"Warning: {FONT_FAMILY} is not installed or not detected by Matplotlib.")
        print("Matplotlib may fall back to another font.")

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY],
        "font.size": BASE_FONT_SIZE,

        "axes.linewidth": AXES_LINEWIDTH,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
        "xtick.major.width": TICK_WIDTH,
        "ytick.major.width": TICK_WIDTH,

        "figure.dpi": 150,

        # 保存 PDF 时尽量保留可编辑字体
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ============================================================
# 2. 参数配置
# ============================================================

cfg = TrainConfig()

base_time = PLOT_START_TIME
SESSION_MINUTES = cfg.session
if not np.isfinite(SESSION_MINUTES) or SESSION_MINUTES <= 0:
    raise ValueError("会话间隔必须是正数（分钟）")
if PLOT_END_TIME <= PLOT_START_TIME:
    raise ValueError("绘图结束时间必须晚于开始时间")
# 不使用训练配置中的月份或总步数；30 分钟会话对应十月的 1488 个时段。
N_SESSIONS = int(np.ceil(
    (PLOT_END_TIME - base_time).total_seconds() / (60 * SESSION_MINUTES)
))


# ============================================================
# 3. 计算卫星可见性数据
# ============================================================

print("加载卫星可见性表 ...")
Satellite.load_visibility_table(VISIBILITY_TABLE_PATH)

vis_windows = Satellite.VISIBILITY_WINDOWS

with open(SAT_JSON_PATH, "r", encoding="utf-8") as f:
    sat_names = [
        str(sat["OBJECT_NAME"]).strip()
        for sat in json.load(f)
        if sat.get("OBJECT_NAME")
    ]

missing_sats = [name for name in sat_names if name not in vis_windows]
if missing_sats:
    raise ValueError(f"Visibility table is missing satellites: {missing_sats}")

print(f"  Using {len(sat_names)} satellites from {SAT_JSON_PATH}")


def is_visible_during_session(sat_name, session_start, session_end):
    """Return True if a satellite is visible for any positive duration in a session."""
    for start_dt, end_dt in vis_windows.get(sat_name, []):
        if start_dt >= session_end:
            break
        if start_dt < session_end and end_dt > session_start:
            return True
    return False


vis_counts = []
timestamps = []

for t in range(N_SESSIONS):
    session_start = base_time + timedelta(minutes=t * SESSION_MINUTES)
    session_end = min(session_start + timedelta(minutes=SESSION_MINUTES), PLOT_END_TIME)

    cnt = sum(
        1
        for sat_name in sat_names
        if is_visible_during_session(sat_name, session_start, session_end)
    )

    vis_counts.append(cnt)
    timestamps.append(session_start)

vis_counts = np.array(vis_counts)
avg_vis = vis_counts.mean()

print(f"  Average visible satellites per session = {avg_vis:.4f}")


# ============================================================
# 4. 绘制卫星可见性柱状图
# ============================================================

setup_matplotlib()

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

ax.bar(
    timestamps,
    vis_counts,
    width=pd.Timedelta(minutes=SESSION_MINUTES * BAR_WIDTH_RATIO),
    color=BAR_COLOR,
    edgecolor=BAR_EDGE_COLOR,
    linewidth=BAR_LINEWIDTH,
    alpha=BAR_ALPHA,
)

# 坐标轴标题
ax.set_ylabel(
    "satellite count",
    fontsize=LABEL_FONT_SIZE,
    labelpad=8
)

# 如果需要 x 轴标题，可以取消下面这段注释
# ax.set_xlabel(
#     "Date",
#     fontsize=LABEL_FONT_SIZE,
#     labelpad=8
# )

# x 轴范围
ax.set_xlim(timestamps[0], timestamps[-1])

# y 轴：0, 2, 4, 6, 8 ...
y_max = int(vis_counts.max())
y_top = y_max + Y_TOP_MARGIN

# 保证顶部刻度是 2 的整数倍
y_tick_max = int(np.ceil(y_top / Y_TICK_STEP) * Y_TICK_STEP)

ax.set_ylim(0, y_tick_max)
ax.set_yticks(np.arange(0, y_tick_max + 1, Y_TICK_STEP))

# x 轴英文日期，比如 May 01, May 06
ax.xaxis.set_major_formatter(mdates.DateFormatter(X_DATE_FORMAT))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=X_DAY_INTERVAL))

# 字体大小
ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)

# 去掉背景网格
if SHOW_GRID:
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
else:
    ax.grid(False)

# 去掉上边框和右边框
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(AXES_LINEWIDTH)

plt.tight_layout()

# ============================================================
# 5. 保存图片：PNG + PDF
# ============================================================

png_path = os.path.join(PLOT_DIR, OUTPUT_PNG_FILENAME)
pdf_path = os.path.join(PLOT_DIR, OUTPUT_PDF_FILENAME)

plt.savefig(
    png_path,
    bbox_inches="tight",
    dpi=SAVE_DPI
)

plt.savefig(
    pdf_path,
    bbox_inches="tight"
)

print(f"Saved PNG: {png_path}")
print(f"Saved PDF: {pdf_path}")

plt.close()
