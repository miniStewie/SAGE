import sys
import os
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TD3_ROOT = next(
    (parent for parent in Path(SCRIPT_DIR).parents
     if (parent / "train_config.py").is_file() and (parent / "env.py").is_file()),
    None,
)
if TD3_ROOT is None:
    raise FileNotFoundError(f"无法从 {SCRIPT_DIR} 向上找到 train_config.py 和 env.py")
TD3_ROOT = str(TD3_ROOT)

if TD3_ROOT not in sys.path:
    sys.path.insert(0, TD3_ROOT)

from train_config import TrainConfig


# ============================================================
# 0. 调试入口：常用参数都放这里
# ============================================================

# ---------- 输入、输出路径：直接修改这里 ----------
SOLAR_RADIATION_TABLE_PATH = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\dataset\北京25年5月每时太阳辐射强度.xls"
PLOT_DIR = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\TD3\plot\plot_result_beijing\plot2"

# ---------- 输出文件名 ----------
SAVE_NAME = "solar_harvesting"

# ---------- 图像尺寸 ----------
FIG_WIDTH = 10
FIG_HEIGHT = 4

# ---------- 字体大小 ----------
BASE_FONT_SIZE = 16
LABEL_FONT_SIZE = 19
TICK_FONT_SIZE = 19

# ---------- 坐标轴线宽和刻度 ----------
AXES_LINE_WIDTH = 1.4
TICK_MAJOR_SIZE = 5
TICK_MAJOR_WIDTH = 1.2

# ---------- 柱状图样式 ----------
BAR_COLOR = "#E07A72"
BAR_EDGE_COLOR = "#C95C55"
BAR_ALPHA = 0.9
BAR_WIDTH_HOURS = 1

# ---------- 横坐标刻度间隔 ----------
X_TICK_DAY_INTERVAL = 5

# ---------- 右侧归一化坐标轴 ----------
SHOW_RIGHT_AXIS = False

# ---------- 图片保存参数 ----------
SAVE_DPI = 300


# ============================================================
# 1. 全局绘图设置
# ============================================================

def setup_matplotlib():
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}

    if "Helvetica" not in available_fonts:
        print("Warning: Helvetica is not installed or not detected by Matplotlib.")
        print("Matplotlib may fall back to another font.")

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],

        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": LABEL_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,

        "axes.linewidth": AXES_LINE_WIDTH,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": TICK_MAJOR_SIZE,
        "ytick.major.size": TICK_MAJOR_SIZE,
        "xtick.major.width": TICK_MAJOR_WIDTH,
        "ytick.major.width": TICK_MAJOR_WIDTH,

        "figure.dpi": 150,

        # 保存 PDF 时尽量保留可编辑字体
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def english_date_formatter(x, pos):
    """
    将 Matplotlib 日期数字转换成英文日期格式，例如 May 01。
    不依赖系统语言环境，避免中文系统下显示成 5月。
    """
    dt = mdates.num2date(x)

    month_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }

    return f"{month_map[dt.month]} {dt.day:02d}"


# ============================================================
# 2. 主程序
# ============================================================

def main():
    setup_matplotlib()
    os.makedirs(PLOT_DIR, exist_ok=True)

    cfg = TrainConfig()

    A = cfg.solar_array_area
    eta_s = cfg.conversion_efficiency
    eta_pwm = 0.64
    E_0_wh = cfg.real_battery

    # ========================================================
    # 2.1 读取并计算太阳能采集数据
    # ========================================================

    print("加载太阳辐射数据 ...")
    solar_df = pd.read_excel(SOLAR_RADIATION_TABLE_PATH)

    utc8_col = solar_df.columns[6]
    rad_col = solar_df.columns[3]

    solar_times = pd.to_datetime(solar_df[utc8_col])
    solar_rad = solar_df[rad_col].values.astype(float)

    harvested_wh = A * eta_s * eta_pwm * solar_rad / 3600.0
    peak_harvest_wh = harvested_wh.max()

    print(f"  峰值采集能量 = {peak_harvest_wh:.4f} Wh")

    # ========================================================
    # 2.2 绘制太阳能采集柱状图
    # ========================================================

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.bar(
        solar_times,
        harvested_wh,
        width=pd.Timedelta(hours=BAR_WIDTH_HOURS),
        color=BAR_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        alpha=BAR_ALPHA,
    )

    # 不再绘制峰值红线
    # ax.axhline(...)

    ax.set_ylabel("harvested energy(Wh)", fontsize=LABEL_FONT_SIZE)
    #ax.set_xlabel("Date", fontsize=LABEL_FONT_SIZE)

    ax.set_xlim(solar_times.iloc[0], solar_times.iloc[-1])

    # 横坐标改成英文日期，而不是 05/01 这种数字格式
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=X_TICK_DAY_INTERVAL))
    ax.xaxis.set_major_formatter(FuncFormatter(english_date_formatter))

    # 去掉背景网格
    ax.grid(False)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FONT_SIZE,
        width=TICK_MAJOR_WIDTH,
        length=TICK_MAJOR_SIZE,
        direction="in"
    )

    # 右轴：归一化能量
    if SHOW_RIGHT_AXIS:
        ax_r = ax.twinx()
        y0, y1 = ax.get_ylim()
        ax_r.set_ylim(y0 / E_0_wh, y1 / E_0_wh)
        ax_r.set_ylabel(f"Normalized Energy ($E_0$ = {E_0_wh} Wh)", fontsize=LABEL_FONT_SIZE)

        ax_r.tick_params(
            axis="both",
            which="major",
            labelsize=TICK_FONT_SIZE,
            width=TICK_MAJOR_WIDTH,
            length=TICK_MAJOR_SIZE,
            direction="in"
        )

    fig.tight_layout()

    # ========================================================
    # 2.3 保存 PNG 和 PDF
    # ========================================================

    png_path = os.path.join(PLOT_DIR, f"{SAVE_NAME}.png")
    pdf_path = os.path.join(PLOT_DIR, f"{SAVE_NAME}.pdf")

    fig.savefig(png_path, bbox_inches="tight", dpi=SAVE_DPI)
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
