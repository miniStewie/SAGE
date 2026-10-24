"""合并北京和东方的 SAGE 训练曲线，每个 c0 输出上下两子图的 PNG/PDF。

绘图参数沿用两地 plot/2-4plot.py；曲线为 10 个种子的均值，
band 为均值 ± 样本标准差（ddof=1），不做额外平滑。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
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
    r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\TD3\plot\plot_result_combined\sage-send-packets-and-utility"
)

# 文件按 seed 分块；这里只按块位置汇总全部 10 个种子，不依赖种子名称顺序。
# 北京、东方均已整理为 100 个文件，每个种子对应 cbase 1–10。
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
        "cbase_order": list(range(1, 11)),
        "file_start": 1,
        "color": "#009E73",
        "marker": "s",
    },
}
target_C_base_list = [1, 3, 5, 7]
target_metric_list = ["total_send_pks", "total_utility"]
Y_SCALE = 1000.0
metric_axis_labels = {
    "total_send_pks": r"sent packets ($\times 1000$)",
    "total_utility": r"utility ($\times 1000$)",
}

# ===================== 沿用原图配置 =====================
auto_y_axis = False
auto_y_margin = 0.08
# 合并后采用北京原图范围，涵盖东方原图的范围。
fixed_y_ticks = {
    "total_send_pks": [5, 10, 15, 20],
    "total_utility": [30, 45, 60, 75],
}
fixed_y_lims = {
    "total_send_pks": (3, 22),
    "total_utility": (20, 85),
}
show_std_band = True
single_subplot_width = 3.8
single_subplot_height = single_subplot_width * 0.618
x_tick_count = 5
y_tick_count = 5
png_dpi = 600
x_tick_inward_points = 0
supxlabel_y = 0.07
plot_area_fraction = 0.74
subplot_left = 0.22
subplot_bottom = 0.18
subplot_right = subplot_left + plot_area_fraction
subplot_top = subplot_bottom + plot_area_fraction
ylabel_pad = 4
ylabel_x = -0.15
BASE_FONT_SIZE = 16
LABEL_FONT_SIZE = 15
TICK_FONT_SIZE = 15
LINE_WIDTH = 1.3
MARKER_SIZE = 3.5  # 单位 point；原来为 2.2
MARKER_EDGE_WIDTH = 0.8
BAND_ALPHA = 0.13

# 新增地区图例；放在上部子图右上方的留白中。
show_legend = True
LEGEND_FONT_SIZE = LABEL_FONT_SIZE
LEGEND_LOCATION = "upper right"


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


def apply_y_axis(ax, metric):
    if auto_y_axis:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_tick_count))
        ax.margins(y=auto_y_margin)
        ax.autoscale(enable=True, axis="y")
    else:
        ax.set_yticks(fixed_y_ticks[metric])
        ax.set_yticklabels([format_number(v) for v in fixed_y_ticks[metric]])
        ax.set_ylim(fixed_y_lims[metric])


def nudge_end_xticklabels(ax, fig):
    labels = ax.get_xticklabels()
    if not labels or x_tick_inward_points <= 0:
        return
    for label, sign, alignment in (
        (labels[0], 1, "left"), (labels[-1], -1, "right")
    ):
        label.set_horizontalalignment(alignment)
        offset = mtransforms.ScaledTranslation(
            sign * x_tick_inward_points / 72.0, 0, fig.dpi_scale_trans
        )
        label.set_transform(label.get_transform() + offset)


def make_figure(cbase, curve_data):
    """创建图并返回 figure，供保存或检查使用。"""
    x_max = int(np.ceil(max(np.max(d["episode"]) for d in curve_data.values())))
    x_ticks = sorted(set(
        int(round(v)) for v in np.linspace(0, x_max, max(2, x_tick_count))
    ))
    x_margin = 0.025 * x_max
    fig, axes = plt.subplots(
        2, 1,
        figsize=(single_subplot_width, 2 * single_subplot_height),
        sharex=True, sharey=False,
        gridspec_kw={"hspace": 0.0},
    )
    for metric_index, (ax, metric) in enumerate(zip(axes, target_metric_list)):
        for region_name, config in REGIONS.items():
            data = curve_data[region_name]
            episodes = data["episode"]
            mean = data["mean"][:, metric_index]
            std = data["std"][:, metric_index]
            if show_std_band:
                ax.fill_between(
                    episodes, mean - std, mean + std,
                    color=config["color"], alpha=BAND_ALPHA, linewidth=0,
                )
            ax.plot(
                episodes, mean, label=region_name,
                color=config["color"], linewidth=LINE_WIDTH,
                marker=config["marker"], markersize=MARKER_SIZE,
                markevery=max(1, len(episodes) // 8),
                markerfacecolor="white", markeredgewidth=MARKER_EDGE_WIDTH,
            )
        # 仅 c0=1 的发送量子图上移标注，避开曲线并留在图例下方。
        c0_label_y = 0.60 if cbase == 1 and metric == "total_send_pks" else 0.2
        ax.text(
            0.90, c0_label_y, rf"$c_0={format_number(cbase)}$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=LABEL_FONT_SIZE, fontweight="bold",
        )
        ax.set_ylabel(
            metric_axis_labels[metric], fontsize=LABEL_FONT_SIZE,
            fontweight="bold", labelpad=ylabel_pad,
        )
        ax.yaxis.set_label_coords(ylabel_x, 0.38)
        ax.grid(False)
        ax.set_xlim(-x_margin, x_max + x_margin)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([format_number(v) for v in x_ticks])
        apply_y_axis(ax, metric)
        beautify_axes(ax)
    axes[0].tick_params(labelbottom=False)
    if show_legend:
        axes[0].legend(
            loc=LEGEND_LOCATION, fontsize=LEGEND_FONT_SIZE,
            frameon=False, handlelength=1.3, handletextpad=0.4,
            labelspacing=0.2, borderaxespad=0.4,
        )
    fig.supxlabel(
        "episode", x=(subplot_left + subplot_right) / 2,
        y=supxlabel_y, fontsize=LABEL_FONT_SIZE, fontweight="bold",
    )
    fig.subplots_adjust(
        left=subplot_left, right=subplot_right,
        bottom=subplot_bottom, top=subplot_top, hspace=0.0,
    )
    fig.canvas.draw()
    nudge_end_xticklabels(axes[1], fig)
    return fig


def main():
    configure_fonts()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    for cbase in target_C_base_list:
        curve_data = {
            region: read_region_curves(region, cbase) for region in REGIONS
        }
        fig = make_figure(cbase, curve_data)
        name = f"beijing-dongfang-send-packets-and-utility-c0-{format_number(cbase)}"
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
