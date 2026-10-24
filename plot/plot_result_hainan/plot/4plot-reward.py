import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm

# ===================== 路径设置 =====================
data_dir = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\TD3\info_hainan"
save_dir = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\TD3\plot_result_hainan\reward-and-violation"
os.makedirs(save_dir, exist_ok=True)

# ===================== 你的实验参数顺序 =====================
seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

C_base_list = [
    1.0, 2.0, 3.0, 4.0, 5.0,
    6.0, 7.0, 8.0, 9.0, 10,
    50, 75, 100
]
Y_SCALE = 100.0
# ===================== 文件编号起点 =====================
# SAGE:   info1.xlsx   ~ info130.xlsx
# greedy: info131.xlsx ~ info260.xlsx
method_file_start = {
    "SAGE": 1,
    "greedy": 131,
}

# ===================== 批量绘图设置 =====================
target_method_list = ["SAGE"]

# 2×2 排列顺序
target_C_base_layout = [5.0, 7.0, 1.0, 3.0]

# 每个 metric 输出一张 2×2 图
target_metric_list = [
    # "eta",
    "total_reward",
    # "total_utility",
    # "total_send_pks",
    #violate1",
]

metric_axis_labels = {
    "eta": r"$\eta$",
    "total_reward": r"reward ($\times 100$)",
    "total_utility": "utility",
    "total_send_pks": "sent packets",
    "violate1": "violation",
    "reward_tu/tc": "TU/TC",
}

# ===================== 图形控制参数 =====================
show_std_band = True

# 每个小图宽度，单位 inch
single_subplot_width = 3.8
single_subplot_height = single_subplot_width * 0.618

# 左列横坐标：多显示一个 0
fixed_x_ticks_left = [0, 25, 50, 75, 100]

# 右列横坐标：依旧只显示 25, 50, 75, 100
fixed_x_ticks_right = [25, 50, 75, 100]

# 横坐标显示范围
x_axis_min = 0
x_axis_max = 100

# y 轴 tick 数量
y_tick_count = 5

# PNG 分辨率
png_dpi = 600

# ===================== 坐标轴标题距离控制 =====================
# x 轴 title 越大越靠近图
supxlabel_y = 0.09

# y 轴 title 越大越靠近图
supylabel_x = 0.12

# 子图整体边距
plot_area_fraction = 0.72
subplot_left = 0.18
subplot_bottom = 0.18
subplot_right = subplot_left + plot_area_fraction
subplot_top = subplot_bottom + plot_area_fraction

# ===================== 字体设置 =====================
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.sans-serif"] = ["Helvetica"]

# 让 PDF 尽量保存为可编辑文字，而不是 Type 3 字体
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

try:
    helvetica_path = fm.findfont(
        fm.FontProperties(family="Helvetica"),
        fallback_to_default=False
    )
    print("Helvetica 已找到：", helvetica_path)
except Exception as e:
    print("Helvetica 没找到，当前环境没有正确安装 Helvetica。")
    print(e)

plt.rcParams["axes.unicode_minus"] = False

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.default"] = "it"

BASE_FONT_SIZE = 16
LABEL_FONT_SIZE = 11.5
TICK_FONT_SIZE = 11

plt.rcParams["font.size"] = BASE_FONT_SIZE
plt.rcParams["axes.labelsize"] = LABEL_FONT_SIZE
plt.rcParams["xtick.labelsize"] = TICK_FONT_SIZE
plt.rcParams["ytick.labelsize"] = TICK_FONT_SIZE

method_colors = {
    "SAGE": "#1f77b4",
    #"SAGE": "#AE0D02",
    "greedy": "#d62728",
}


def make_safe_filename(name):
    """
    Windows 文件名中不能包含这些字符：
    \\ / : * ? " < > |
    这里统一替换成下划线。
    """
    safe_name = re.sub(r'[\\/:*?"<>|]+', "_", str(name))
    safe_name = safe_name.strip().strip(".")
    return safe_name


def format_number(x):
    """
    把 1.0 显示成 1，把 5.0 显示成 5。
    """
    if abs(float(x) - int(float(x))) < 1e-9:
        return str(int(float(x)))
    return f"{float(x):g}"


def get_file_index(method, seed_idx, target_C_base):
    """
    根据运行顺序计算文件编号。
    """
    if method not in method_file_start:
        raise ValueError(f"未知方法：{method}")

    if target_C_base not in C_base_list:
        raise ValueError(f"C_base={target_C_base} 不在 C_base_list 中")

    start_idx = method_file_start[method]
    c_idx = C_base_list.index(target_C_base)

    file_idx = start_idx + seed_idx * len(C_base_list) + c_idx
    return file_idx


def read_seed_curves(method, target_C_base, target_metric):
    """
    读取某个 method、某个 C_base、某个 metric 在 10 个 seed 下的 episode 曲线。
    """
    all_seed_curves = []
    used_files = []
    episode_x = None

    for seed_idx, seed in enumerate(seed_list):
        file_idx = get_file_index(
            method=method,
            seed_idx=seed_idx,
            target_C_base=target_C_base
        )

        filename = f"info{file_idx}.xlsx"
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到文件：{filepath}")

        df = pd.read_excel(filepath, sheet_name="avg", engine="openpyxl")

        if target_metric not in df.columns:
            raise ValueError(f"{filename} 的 avg sheet 中找不到字段：{target_metric}")

        if episode_x is None:
            if "epsolid" in df.columns:
                episode_x = df["epsolid"].values
            elif "episode" in df.columns:
                episode_x = df["episode"].values
            else:
                episode_x = np.arange(1, len(df) + 1)

        y = df[target_metric].values
        y = y / Y_SCALE
        if len(y) != len(episode_x):
            raise ValueError(f"{filename} 的 episode 数量和前面的文件不一致")

        all_seed_curves.append(y)
        used_files.append(filename)

    all_seed_curves = np.array(all_seed_curves, dtype=float)

    mean_curve = np.mean(all_seed_curves, axis=0)

    if all_seed_curves.shape[0] > 1:
        std_curve = np.std(all_seed_curves, axis=0, ddof=1)
    else:
        std_curve = np.zeros_like(mean_curve)

    return episode_x, mean_curve, std_curve, used_files


def beautify_axes(ax):
    """
    统一处理坐标轴外观。

    重点：
    1. 不手动移动 tick label，保证横坐标数字和真实刻度位置一致。
    2. spine 使用 butt 线帽，避免边框端点突出。
    """
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_capstyle("butt")
        spine.set_joinstyle("miter")
        spine.set_clip_on(True)

    # y 轴刻度线
    ax.tick_params(
        axis="y",
        direction="in",
        length=2.5,
        width=0.8,
        pad=2,
        labelsize=TICK_FONT_SIZE
    )

    # x 轴刻度线
    ax.tick_params(
        axis="x",
        direction="in",
        length=2.5,
        width=0.8,
        pad=2,
        labelsize=TICK_FONT_SIZE
    )


def plot_one_metric_grid(method, target_metric):
    """
    对一个 metric，绘制一张 2×2 图。
    四个子图分别是 target_C_base_layout 中的四个 C_base。
    同时保存 PNG 和 PDF。
    """
    metric_label = metric_axis_labels.get(target_metric, target_metric)
    color = method_colors.get(method, "#1f77b4")

    # ===================== 先读取所有 C_base 的曲线 =====================
    curve_data = {}
    all_lower = []
    all_upper = []
    episode_x_ref = None

    for C_base in target_C_base_layout:
        episode_x, mean_curve, std_curve, used_files = read_seed_curves(
            method=method,
            target_C_base=C_base,
            target_metric=target_metric
        )

        if episode_x_ref is None:
            episode_x_ref = episode_x
        else:
            if len(episode_x) != len(episode_x_ref):
                raise ValueError("不同 C_base 的 episode 数量不一致")

        curve_data[C_base] = {
            "episode_x": episode_x,
            "mean": mean_curve,
            "std": std_curve,
            "used_files": used_files,
        }

        all_lower.append(mean_curve - std_curve)
        all_upper.append(mean_curve + std_curve)

    all_lower = np.concatenate(all_lower)
    all_upper = np.concatenate(all_upper)

    y_min = float(np.nanmin(all_lower))
    y_max = float(np.nanmax(all_upper))

    if abs(y_max - y_min) < 1e-12:
        y_margin = 1.0
    else:
        y_margin = 0.06 * (y_max - y_min)

    y_lim = (y_min - y_margin, y_max + y_margin)

    # ===================== 创建 2×2 图 =====================
    fig_width = 2 * single_subplot_width
    fig_height = 2 * single_subplot_height

    # 注意：
    # 这里不要使用 sharex=True。
    # 因为左下角和右下角的 x 轴标签不完全一样：
    # 左下角要显示 0, 25, 50, 75, 100；
    # 右下角只显示 25, 50, 75, 100。
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(fig_width, fig_height),
        sharex=False,
        sharey=True,
        gridspec_kw={
            "wspace": 0.0,
            "hspace": 0.0,
        }
    )

    axes_flat = axes.flatten()

    for idx, (ax, C_base) in enumerate(zip(axes_flat, target_C_base_layout)):
        data = curve_data[C_base]
        episode_x = data["episode_x"]
        mean_curve = data["mean"]
        std_curve = data["std"]

        row_idx = idx // 2
        col_idx = idx % 2

        # 标准差阴影
        if show_std_band:
            ax.fill_between(
                episode_x,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color=color,
                alpha=0.13,
                linewidth=0
            )

        # 均值曲线
        ax.plot(
            episode_x,
            mean_curve,
            color=color,
            linewidth=1.3,
            marker="o",
            markersize=2.2,
            markevery=max(1, len(episode_x) // 8),
            markerfacecolor="white",
            markeredgewidth=0.8
        )

        # 小图右上角标注 c_0
        ax.text(
            0.93,
            0.1,
            rf"$c_0={format_number(C_base)}$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=LABEL_FONT_SIZE,
            fontweight="bold"
        )

        # 不绘制网格
        ax.grid(False)

        # 坐标轴范围
        ax.set_xlim(x_axis_min, x_axis_max)
        ax.set_ylim(y_lim)

        # ===================== 横坐标刻度单独控制 =====================
        # 左列子图：显示 0, 25, 50, 75, 100
        if col_idx == 0:
            ax.set_xticks(fixed_x_ticks_left)
            ax.set_xticklabels([format_number(v) for v in fixed_x_ticks_left])

        # 右列子图：依旧只显示 25, 50, 75, 100
        else:
            ax.set_xticks(fixed_x_ticks_right)
            ax.set_xticklabels([format_number(v) for v in fixed_x_ticks_right])

        # y 轴自动控制 tick 数量
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_tick_count))

        # 坐标轴样式处理
        beautify_axes(ax)

    # 顶部两个子图不显示 x tick label
    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)

    # 右侧两个子图不显示 y tick label
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)

    # ===================== 共享坐标轴标签 =====================
    fig.supxlabel(
        "episode",
        x=0.55,
        y=supxlabel_y,
        fontsize=LABEL_FONT_SIZE,
        fontweight="bold"
    )

    fig.supylabel(
        metric_label,
        x=supylabel_x,
        y=0.54,
        fontsize=LABEL_FONT_SIZE,
        fontweight="bold"
    )

    fig.subplots_adjust(
        left=subplot_left,
        right=subplot_right,
        bottom=subplot_bottom,
        top=subplot_top,
        wspace=0.0,
        hspace=0.0
    )

    # ===================== 保存 PNG + PDF =====================
    safe_metric_name = make_safe_filename(target_metric)
    base_save_name = f"{method}-{safe_metric_name}-under-different-base-fees"

    png_save_path = os.path.join(save_dir, base_save_name + ".png")
    pdf_save_path = os.path.join(save_dir, base_save_name + ".pdf")

    # 保存 PNG
    plt.savefig(
        png_save_path,
        format="png",
        dpi=png_dpi,
        bbox_inches="tight",
        pad_inches=0.01
    )

    # 保存 PDF
    plt.savefig(
        pdf_save_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.01
    )

    plt.close(fig)

    print("=" * 80)
    print(f"PNG 图像已保存到：{png_save_path}")
    print(f"PDF 图像已保存到：{pdf_save_path}")
    print(f"Method: {method}")
    print(f"Metric: {target_metric}")
    print(f"C_base layout: {target_C_base_layout}")


# ===================== 批量运行 =====================
for method in target_method_list:
    for target_metric in target_metric_list:
        plot_one_metric_grid(
            method=method,
            target_metric=target_metric
        )

print("全部 PNG 和 PDF 图片绘制完成。")