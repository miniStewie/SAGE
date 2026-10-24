"""合并北京和东方的 reward / violation，每项指标输出一张 2×2 图。

布局和字号沿用两地 4plot-reward.py、4plot-violation.py；曲线为 10 个种子的均值，
band 为均值 ± 样本标准差（ddof=1），不做额外平滑。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


# ===================== 硬编码输入、输出路径 =====================
BEIJING_DATA_DIR = Path(
    r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\TD3\info\info_beijing"
)
DONGFANG_DATA_DIR = Path(
    r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\TD3\info\info_hainan"
)
SAVE_DIR = Path(
    r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\TD3\plot\plot_result_combined\reward-and-violation"
)

# 文件按 seed 分块；这里只按块位置汇总全部 10 个种子，不依赖种子名称顺序。
# 北京已整理为 100 个文件，东方仍为 130 个文件。
SEED_COUNT = 10
REGIONS = {
    "Beijing": {
        "data_dir": BEIJING_DATA_DIR,
        "cbase_order": list(range(1, 11)),
        "file_start": 1,
        "color": "#1f77b4",
        "marker": "o",
    },
    "Dongfang": {
        "data_dir": DONGFANG_DATA_DIR,
        "cbase_order": list(range(1, 11)) + [50, 75, 100],
        "file_start": 1,
        "color": "#009E73",
        "marker": "s",
    },
}
target_C_base_layout = [5, 7, 1, 3]
target_metric_list = ["total_reward", "violate1"]
# reward 原图除以 100，violation 原图除以 10。
Y_SCALE = np.array([100.0, 10.0])
metric_axis_labels = {
    "total_reward": r"reward ($\times 100$)",
    "violate1": r"violation count ($\times 10$)",
}
METRIC_COLORS = {
    "total_reward": {"Beijing": "#1f77b4", "Dongfang": "#009E73"},
    "violate1": {"Beijing": "#AE0D02", "Dongfang": "#004978"},
}

# ===================== 原图布局及统一曲线样式 =====================
show_std_band = True
single_subplot_width = 3.8
single_subplot_height = single_subplot_width * 0.618
fixed_x_ticks_left = [0, 25, 50, 75, 100]
fixed_x_ticks_right = [25, 50, 75, 100]
x_axis_min = 0
x_axis_max = 100
y_tick_count = 5
png_dpi = 600
supxlabel_y = 0.09
supylabel_x = {"total_reward": 0.12, "violate1": 0.115}
subplot_left = 0.18
subplot_bottom = 0.18
subplot_right = 0.90
subplot_top = 0.90
BASE_FONT_SIZE = 16
LABEL_FONT_SIZE = 11.5
TICK_FONT_SIZE = 11
LINE_WIDTH = 1.3
MARKER_SIZE = 3.5  # 单位 point；原来为 2.2
MARKER_EDGE_WIDTH = 0.8
BAND_ALPHA = 0.13
Y_MARGIN = 0.06
show_legend = True
LEGEND_FONT_SIZE = LABEL_FONT_SIZE

def configure_fonts():
    plt.rcParams.update({
        "font.family": "Helvetica",
        "font.sans-serif": ["Helvetica"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
        "mathtext.default": "it",
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": LABEL_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
    })


def format_number(value):
    return f"{float(value):g}"


def read_region_curves(region_name, cbase):
    """一次读取两个指标，并检查种子之间的 episode 对齐。"""
    config = REGIONS[region_name]
    cbase_order = config["cbase_order"]
    cbase_index = cbase_order.index(cbase)
    curves = []
    episode_ref = None
    used_files = []
    for seed_index in range(SEED_COUNT):
        file_index = config["file_start"] + seed_index * len(cbase_order) + cbase_index
        path = config["data_dir"] / f"info{file_index}.xlsx"
        df = pd.read_excel(path, sheet_name="avg", engine="openpyxl")
        missing = set(target_metric_list) - set(df.columns)
        if missing:
            raise ValueError(f"{path}: avg 表缺少字段 {sorted(missing)}")
        episode_column = next(
            (name for name in ("epsolid", "episode") if name in df.columns), None
        )
        if episode_column is not None:
            # 表尾 reward_method 是说明行，没有 episode，也没有指标值。
            metadata_rows = (
                df[episode_column].isna()
                & df[target_metric_list].isna().all(axis=1)
            )
            df = df.loc[~metadata_rows]
            episodes = df[episode_column].to_numpy(dtype=float)
        else:
            df = df.loc[~df[target_metric_list].isna().all(axis=1)]
            episodes = np.arange(1, len(df) + 1, dtype=float)
        values = df[target_metric_list].to_numpy(dtype=float) / Y_SCALE
        if (
            len(episodes) == 0
            or not np.isfinite(episodes).all()
            or not np.isfinite(values).all()
            or np.any(np.diff(episodes) <= 0)
        ):
            raise ValueError(f"{path}: episode 或指标数据为空、缺失或顺序异常")
        if episode_ref is None:
            episode_ref = episodes
        elif not np.array_equal(episodes, episode_ref):
            raise ValueError(f"{path}: episode 与其他种子不一致")
        curves.append(values)
        used_files.append(path.name)
    stacked = np.stack(curves)
    print(f"{region_name}, c0={cbase}: {', '.join(used_files)}")
    return {
        "episode": episode_ref,
        "mean": np.mean(stacked, axis=0),
        "std": np.std(stacked, axis=0, ddof=1),
    }


def beautify_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_capstyle("butt")
        spine.set_joinstyle("miter")
        spine.set_clip_on(True)
    ax.tick_params(axis="both", direction="in", length=2.5, width=0.8, pad=2)


def make_figure(metric, curve_data):
    """每个 c0 子图均显示两地均值及各自的样本标准差带。"""
    metric_index = target_metric_list.index(metric)
    lower, upper = [], []
    for regions in curve_data.values():
        for data in regions.values():
            mean = data["mean"][:, metric_index]
            std = data["std"][:, metric_index] if show_std_band else 0.0
            lower.append(mean - std)
            upper.append(mean + std)
    y_min = float(np.min(np.concatenate(lower)))
    y_max = float(np.max(np.concatenate(upper)))
    margin = Y_MARGIN * (y_max - y_min) if y_max > y_min else 1.0
    y_lim = (y_min - margin, y_max + margin)

    fig, axes = plt.subplots(
        2, 2, figsize=(2 * single_subplot_width, 2 * single_subplot_height),
        sharex=False, sharey=True,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )
    for index, (ax, cbase) in enumerate(zip(axes.flat, target_C_base_layout)):
        for region, config in REGIONS.items():
            data = curve_data[cbase][region]
            episodes = data["episode"]
            mean = data["mean"][:, metric_index]
            std = data["std"][:, metric_index]
            color = METRIC_COLORS[metric][region]
            if show_std_band:
                ax.fill_between(
                    episodes, mean - std, mean + std,
                    color=color, alpha=BAND_ALPHA, linewidth=0,
                )
            ax.plot(
                episodes, mean, label=region, color=color,
                linewidth=LINE_WIDTH, marker=config["marker"],
                markersize=MARKER_SIZE,
                # 仅间隔显示形状标记；曲线仍使用全部 episode 数据。
                markevery=max(1, len(episodes) // 8),
                markerfacecolor="white", markeredgewidth=MARKER_EDGE_WIDTH,
            )
        ax.text(
            0.93, 0.1 if metric == "total_reward" else 0.85,
            rf"$c_0={format_number(cbase)}$",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=LABEL_FONT_SIZE, fontweight="bold",
        )
        ax.grid(False)
        ax.set_xlim(x_axis_min, x_axis_max)
        ax.set_ylim(y_lim)
        ticks = fixed_x_ticks_left if index % 2 == 0 else fixed_x_ticks_right
        ax.set_xticks(ticks)
        ax.set_xticklabels([format_number(v) for v in ticks])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_tick_count))
        beautify_axes(ax)
    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    # 共用图例位于原图顶部留白内，不遮挡曲线。
    if show_legend:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center",
            bbox_to_anchor=((subplot_left + subplot_right) / 2, subplot_top),
            ncol=2, fontsize=LEGEND_FONT_SIZE, frameon=False,
            handlelength=1.3, handletextpad=0.4, borderaxespad=0.25,
        )
    fig.supxlabel(
        "episode", x=0.55, y=supxlabel_y,
        fontsize=LABEL_FONT_SIZE, fontweight="bold",
    )
    fig.supylabel(
        metric_axis_labels[metric], x=supylabel_x[metric], y=0.54,
        fontsize=LABEL_FONT_SIZE, fontweight="bold",
    )
    fig.subplots_adjust(
        left=subplot_left, right=subplot_right,
        bottom=subplot_bottom, top=subplot_top,
        wspace=0.0, hspace=0.0,
    )
    return fig


def main():
    configure_fonts()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    curve_data = {
        cbase: {region: read_region_curves(region, cbase) for region in REGIONS}
        for cbase in target_C_base_layout
    }
    for metric in target_metric_list:
        fig = make_figure(metric, curve_data)
        name = f"beijing-dongfang-{metric}-under-different-base-fees"
        for extension in ("png", "pdf"):
            path = SAVE_DIR / f"{name}.{extension}"
            fig.savefig(
                path, format=extension, dpi=png_dpi,
                bbox_inches="tight", pad_inches=0.01,
            )
            print(f"Saved: {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
