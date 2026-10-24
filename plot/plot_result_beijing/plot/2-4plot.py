import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm

# ===================== 路径设置 =====================
data_dir = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\TD3\info_beijing"
save_dir = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\TD3\plot_result_beijing\sage-send-packets-and-utility"
os.makedirs(save_dir, exist_ok=True)

# ===================== 你的实验参数顺序 =====================
seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
Y_SCALE = 1000.0
C_base_list = [
    1.0, 2.0, 3.0, 4.0, 5.0,
    6.0, 7.0, 8.0, 9.0, 10,
    50, 75, 100
]
# ===================== 文件编号起点 =====================
# sage:   info1.xlsx   ~ info130.xlsx
# greedy: info131.xlsx ~ info260.xlsx
method_file_start = {
    "sage": 1,
    "greedy": 131,
}

# ===================== 批量绘图设置 =====================
target_method_list = ["sage"]

# 分别画 4 张图
target_C_base_list = [1.0, 3.0, 5.0, 7.0]

# 每张图中上下两个子图
target_metric_list = [
    "total_send_pks",
    "total_utility",
]

metric_axis_labels = {
    "total_send_pks": r"sent packets ($\times 1000$)",
    "total_utility": r"utility ($\times 1000$)",
}

# ===================== y 轴范围设置 =====================
auto_y_axis = False  # True：各子图自适应（包含已显示的标准差阴影）；False：使用下方固定配置
auto_y_margin = 0.08  # 自适应时上下留白比例

# 仅在 auto_y_axis = False 时使用
fixed_y_ticks = {
    "total_send_pks": [5,10,15,20],
    "total_utility": [30,45,60,75],
}

fixed_y_lims = {
    "total_send_pks": (3, 22),
    "total_utility": (20, 85),
}

# ===================== 图形控制参数 =====================
show_std_band = True

# 每个小图宽度，单位 inch
single_subplot_width = 3.8
single_subplot_height = single_subplot_width * 0.618

# 坐标轴 tick 数量
x_tick_count = 5
y_tick_count = 5

# PNG 分辨率
png_dpi = 600

# x 轴两端数字向内缩的距离，sage-send-packets-and-utility-c0-7.pdf单位 point
x_tick_inward_points = 0

# ===================== 坐标轴标题距离控制 =====================
supxlabel_y = 0.07

# 子图整体边距
plot_area_fraction = 0.74
subplot_left = 0.22
subplot_bottom = 0.18
subplot_right = subplot_left + plot_area_fraction
subplot_top = subplot_bottom + plot_area_fraction

# y 轴 label 距离
ylabel_pad = 4
ylabel_x = -0.15

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
LABEL_FONT_SIZE = 15
TICK_FONT_SIZE = 15

plt.rcParams["font.size"] = BASE_FONT_SIZE
plt.rcParams["axes.labelsize"] = LABEL_FONT_SIZE
plt.rcParams["xtick.labelsize"] = TICK_FONT_SIZE
plt.rcParams["ytick.labelsize"] = TICK_FONT_SIZE

method_colors = {
    "sage": "#1f77b4",
    "pssa": "#1f77b4",
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

        y = df[target_metric].values / Y_SCALE
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


def build_x_ticks(episode_x):
    """
    控制横轴 tick 数量。
    """
    x_max = int(np.ceil(np.nanmax(episode_x)))

    if x_tick_count <= 2:
        ticks = [0, x_max]
    else:
        ticks = np.linspace(0, x_max, x_tick_count)

    ticks = [int(round(v)) for v in ticks]
    ticks = sorted(list(set(ticks)))

    return ticks


def nudge_end_xticklabels(ax, fig, inward_points=8):
    """
    把 x 轴两端的 tick label 真正向图内平移。
    - 最左标签：向右挪 inward_points
    - 最右标签：向左挪 inward_points
    """
    labels = ax.get_xticklabels()
    if len(labels) == 0:
        return

    if inward_points <= 0:
        return

    for lab in labels:
        lab.set_horizontalalignment("center")

    if len(labels) >= 1:
        labels[0].set_horizontalalignment("left")
        trans_left = mtransforms.ScaledTranslation(
            inward_points / 72.0, 0, fig.dpi_scale_trans
        )
        labels[0].set_transform(labels[0].get_transform() + trans_left)

        labels[-1].set_horizontalalignment("right")
        trans_right = mtransforms.ScaledTranslation(
            -inward_points / 72.0, 0, fig.dpi_scale_trans
        )
        labels[-1].set_transform(labels[-1].get_transform() + trans_right)


def beautify_axes(ax):
    """
    统一处理坐标轴外观。
    """
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_capstyle("butt")
        spine.set_joinstyle("miter")
        spine.set_clip_on(True)

    # y 轴保留正常刻度线
    ax.tick_params(
        axis="y",
        direction="in",
        length=2.5,
        width=0.8,
        pad=2
    )

    # x 轴不画刻度小竖线，只保留刻度数字
    ax.tick_params(
        axis="x",
        direction="in",
        length=2.5,
        width=0.8,
        pad=2
    )


def apply_y_axis(ax, target_metric):
    """
    根据开关自适应 y 轴，或按 metric 使用固定刻度和范围。

    重点：
    tick 固定为你指定的数值；
    ylim 稍微放宽，避免 tick label 压在子图边界上。
    """
    if auto_y_axis:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_tick_count))
        ax.margins(y=auto_y_margin)
        ax.autoscale(enable=True, axis="y")
        return

    if target_metric in fixed_y_ticks:
        ticks = fixed_y_ticks[target_metric]
        ax.set_yticks(ticks)
        ax.set_yticklabels([format_number(v) for v in ticks])

        if target_metric in fixed_y_lims:
            ax.set_ylim(fixed_y_lims[target_metric])
        else:
            y_min = min(ticks)
            y_max = max(ticks)
            y_margin = 0.08 * (y_max - y_min)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_tick_count))


def plot_one_C_two_metrics(method, target_C_base):
    """
    对一个 C_base 绘制一张上下排列的图：
    上部：total_send_pks
    下部：total_utility

    每个子图比例仍然保持 0.618:1。
    上下子图之间没有空隙。
    同时保存 PNG 和 PDF。
    """
    color = method_colors.get(method, "#1f77b4")

    # ===================== 读取两个 metric 的曲线 =====================
    curve_data = {}
    episode_x_ref = None

    for target_metric in target_metric_list:
        episode_x, mean_curve, std_curve, used_files = read_seed_curves(
            method=method,
            target_C_base=target_C_base,
            target_metric=target_metric
        )

        if episode_x_ref is None:
            episode_x_ref = episode_x
        else:
            if len(episode_x) != len(episode_x_ref):
                raise ValueError("两个 metric 的 episode 数量不一致")

        curve_data[target_metric] = {
            "episode_x": episode_x,
            "mean": mean_curve,
            "std": std_curve,
            "used_files": used_files,
        }

    x_ticks = build_x_ticks(episode_x_ref)
    x_max = max(x_ticks)

    # 让 x 轴也不要刚好卡在端点
    # 这样 0 和 100 不会紧贴左右边框
    x_margin = 0.025 * x_max
    x_lim = (-x_margin, x_max + x_margin)

    # ===================== 创建 2×1 图 =====================
    fig_width = single_subplot_width
    fig_height = 2 * single_subplot_height

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=False,
        gridspec_kw={
            "hspace": 0.0,
        }
    )

    for row_idx, (ax, target_metric) in enumerate(zip(axes, target_metric_list)):
        data = curve_data[target_metric]
        episode_x = data["episode_x"]
        mean_curve = data["mean"]
        std_curve = data["std"]

        metric_label = metric_axis_labels.get(target_metric, target_metric)

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
        if row_idx == 0 or 1:
            ax.text(
                0.90,
                0.2,
                rf"$c_0={format_number(target_C_base)}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=LABEL_FONT_SIZE,
                fontweight="bold"
            )

        # y 轴标题分别对应上下两个 metric
        ax.set_ylabel(
            metric_label,
            fontsize=LABEL_FONT_SIZE,
            fontweight="bold",
            labelpad=ylabel_pad
        )
        ax.yaxis.set_label_coords(ylabel_x, 0.38)

        # 不绘制网格
        ax.grid(False)

        # x 轴范围不要刚好卡在端点
        ax.set_xlim(x_lim)

        # x tick 控制
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([format_number(v) for v in x_ticks])

        # 按开关选择自适应或固定 y 轴
        apply_y_axis(ax, target_metric)

        # 坐标轴样式处理
        beautify_axes(ax)

    # 上部子图不显示 x tick label
    axes[0].tick_params(labelbottom=False)

    # ===================== 共享 x 轴标题 =====================
    fig.supxlabel(
        "episode",
        x=(subplot_left + subplot_right) / 2,
        y=supxlabel_y,
        fontsize=LABEL_FONT_SIZE,
        fontweight="bold"
    )

    fig.subplots_adjust(
        left=subplot_left,
        right=subplot_right,
        bottom=subplot_bottom,
        top=subplot_top,
        hspace=0.0
    )

    # 先 draw 一次，确保 tick label 对象已经就位
    fig.canvas.draw()

    # 只需要调整底部子图的 x 轴两端数字
    nudge_end_xticklabels(
        axes[1],
        fig,
        inward_points=x_tick_inward_points
    )

    # ===================== 保存 PNG + PDF =====================
    safe_C_name = make_safe_filename(format_number(target_C_base))
    base_save_name = f"{method}-send-packets-and-utility-c0-{safe_C_name}"

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
    print(f"C_base: {target_C_base}")
    print(f"Metrics: {target_metric_list}")


# ===================== 批量运行 =====================
for method in target_method_list:
    for target_C_base in target_C_base_list:
        plot_one_C_two_metrics(
            method=method,
            target_C_base=target_C_base
        )

print("全部 PNG 和 PDF 图片绘制完成。")
