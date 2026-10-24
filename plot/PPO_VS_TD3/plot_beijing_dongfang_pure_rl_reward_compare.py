import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm


# Four hardcoded inputs; original regional figures are kept.
save_dir = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\PPO_VS_TD3\plot_result\combined"
methods = {
    "Beijing-TD3": {
        "data_dir": r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\PureTD3\pure_td3_info_beijing",
        "label": "Beijing TD3", "color": "gray", "marker": "o", "linestyle": "-",
        "ground_lat": 39.9, "ground_lon": 116.4,
    },
    "Beijing-PPO": {
        "data_dir": r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\PurePPO\pure_ppo_info_beijing",
        "label": "Beijing PPO", "color": "#FF7F0E", "marker": "o", "linestyle": "-",
        "ground_lat": 39.9, "ground_lon": 116.4,
    },
    "Dongfang-TD3": {
        "data_dir": r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\PureTD3\pure_td3_info_hainan",
        "label": "Dongfang TD3", "color": "gray", "marker": "s", "linestyle": "--",
        "ground_lat": 19.1, "ground_lon": 108.6,
    },
    "Dongfang-PPO": {
        "data_dir": r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\PurePPO\pure_ppo_info_hainan",
        "label": "Dongfang PPO", "color": "#FF7F0E", "marker": "s", "linestyle": "--",
        "ground_lat": 19.1, "ground_lon": 108.6,
    },
}
MARKER_SIZE = 3.5
MARKER_EDGE_WIDTH = 0.8
BAND_ALPHA = 0.10
seed_list = list(range(10))
C_base_list = [1.0, 3.0, 5.0, 7.0]
target_C_base_layout = [5.0, 7.0, 1.0, 3.0]

sheet_name = "avg"
target_metric = "reward"
metric_aliases = {
    "reward": "total_reward",
}
Y_SCALE = 100.0

show_std_band = True
single_subplot_width = 2.65
single_subplot_height = 3.2  # Allow larger text while retaining the compact 1x4 layout

fixed_x_ticks_left = [0, 50, 100]
fixed_x_ticks_right = [0, 50, 100]
x_axis_min = 0
x_axis_max = 100
y_tick_count = 5
png_dpi = 600

supxlabel_y = 0.025
supylabel_x = 0.015
subplot_left = 0.10
subplot_bottom = 0.30
subplot_right = 0.99
subplot_top = 0.80
subplot_wspace = 0.22  # Separate adjacent 100/0 labels without shifting their tick positions

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.sans-serif"] = ["Helvetica"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.default"] = "it"

try:
    helvetica_path = fm.findfont(
        fm.FontProperties(family="Helvetica"),
        fallback_to_default=False,
    )
    print("Helvetica found:", helvetica_path)
except Exception as exc:
    print("Helvetica not found, matplotlib will use the fallback font.")
    print(exc)

BASE_FONT_SIZE = 22
LABEL_FONT_SIZE = 22
TICK_FONT_SIZE = 21
LEGEND_FONT_SIZE = 20

plt.rcParams["font.size"] = BASE_FONT_SIZE
plt.rcParams["axes.labelsize"] = LABEL_FONT_SIZE
plt.rcParams["xtick.labelsize"] = TICK_FONT_SIZE
plt.rcParams["ytick.labelsize"] = TICK_FONT_SIZE

metric_axis_labels = {
    "reward": r"reward ($\times 100$)",
    "total_reward": r"reward ($\times 100$)",
}


def make_safe_filename(name):
    safe_name = re.sub(r'[\\/:*?"<>|]+', "_", str(name))
    return safe_name.strip().strip(".")


def format_number(x):
    if abs(float(x) - int(float(x))) < 1e-9:
        return str(int(float(x)))
    return f"{float(x):g}"


def get_file_index(seed_idx, target_C_base):
    if target_C_base not in C_base_list:
        raise ValueError(f"C_base={target_C_base} is not in C_base_list")
    c_idx = C_base_list.index(target_C_base)
    return 1 + seed_idx * len(C_base_list) + c_idx


def resolve_metric_column(df, metric):
    if metric in df.columns:
        return metric

    alias = metric_aliases.get(metric)
    if alias is not None and alias in df.columns:
        return alias

    raise ValueError(
        f"Cannot find metric '{metric}' in sheet '{sheet_name}'. "
        f"Available columns: {list(df.columns)}"
    )


def read_seed_curves(data_dir, target_C_base, metric):
    all_seed_curves = []
    used_files = []
    episode_x = None

    for seed_idx, seed in enumerate(seed_list):
        file_idx = get_file_index(seed_idx=seed_idx, target_C_base=target_C_base)
        filename = f"info{file_idx}.xlsx"
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing file: {filepath}")

        df = pd.read_excel(filepath, sheet_name=sheet_name, engine="openpyxl")
        metric_column = resolve_metric_column(df, metric)

        # Drop only the trailing reward_method metadata row.
        if "episode" in df.columns:
            metadata = df["episode"].isna() & df[metric_column].isna()
            df = df.loc[~metadata]
            current_x = df["episode"].to_numpy(dtype=float)
        else:
            current_x = np.arange(1, len(df) + 1, dtype=float)
        y = df[metric_column].to_numpy(dtype=float) / Y_SCALE
        if (not len(y) or not np.isfinite(y).all()
                or not np.isfinite(current_x).all()
                or np.any(np.diff(current_x) <= 0)):
            raise ValueError(f"Invalid episode or metric values: {filepath}")
        if episode_x is None:
            episode_x = current_x
        elif not np.array_equal(episode_x, current_x):
            raise ValueError(f"Episode mismatch: {filepath}")

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
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_capstyle("butt")
        spine.set_joinstyle("miter")
        spine.set_clip_on(True)

    ax.tick_params(
        axis="y",
        direction="in",
        length=2.5,
        width=0.8,
        pad=2,
        labelsize=TICK_FONT_SIZE,
    )
    ax.tick_params(
        axis="x",
        direction="in",
        length=2.5,
        width=0.8,
        pad=2,
        labelsize=TICK_FONT_SIZE,
    )


def collect_curve_data():
    curve_data = {}
    episode_x_ref = None

    for method_name, method_cfg in methods.items():
        curve_data[method_name] = {}

        for C_base in target_C_base_layout:
            episode_x, mean_curve, std_curve, used_files = read_seed_curves(
                data_dir=method_cfg["data_dir"],
                target_C_base=C_base,
                metric=target_metric,
            )

            if episode_x_ref is None:
                episode_x_ref = episode_x
            elif not np.array_equal(episode_x, episode_x_ref):
                raise ValueError(
                    "Different methods or C_base groups have different episode counts"
                )

            curve_data[method_name][C_base] = {
                "episode_x": episode_x,
                "mean": mean_curve,
                "std": std_curve,
                "used_files": used_files,
            }

    return curve_data


def compute_common_y_limits(curve_data):
    # One common scale makes the four horizontally adjacent panels comparable.
    lower, upper = [], []
    for groups in curve_data.values():
        for data in groups.values():
            mean = data["mean"]
            std = data["std"] if show_std_band else 0.0
            lower.append(mean - std)
            upper.append(mean + std)
    y_min = float(np.min(np.concatenate(lower)))
    y_max = float(np.max(np.concatenate(upper)))
    margin = max(0.12, 0.08 * (y_max - y_min))
    return y_min - margin, y_max + margin


def choose_c0_label_position(curve_data, C_base, y_lim):
    tail_lowers = []
    tail_uppers = []

    for method_name in methods:
        data = curve_data[method_name][C_base]
        mean_curve = data["mean"]
        std_curve = data["std"] if show_std_band else 0.0

        tail_start = int(0.8 * len(mean_curve))
        tail_start = min(max(tail_start, 0), len(mean_curve) - 1)

        tail_lowers.append((mean_curve - std_curve)[tail_start:])
        tail_uppers.append((mean_curve + std_curve)[tail_start:])

    tail_low = float(np.nanmin(np.concatenate(tail_lowers)))
    tail_high = float(np.nanmax(np.concatenate(tail_uppers)))

    lower_gap = tail_low - y_lim[0]
    upper_gap = y_lim[1] - tail_high

    if upper_gap >= lower_gap:
        return 0.9, "top"

    return 0.1, "bottom"


def plot_reward_grid():
    metric_label = metric_axis_labels.get(target_metric, target_metric)

    curve_data = collect_curve_data()
    common_y_lim = compute_common_y_limits(curve_data)

    fig_width = 4 * single_subplot_width
    fig_height = single_subplot_height

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(fig_width, fig_height),
        sharex=False,
        sharey=True,
        gridspec_kw={
            "wspace": subplot_wspace,
            "hspace": 0.0,
        },
    )

    legend_handles = []
    legend_labels = []

    axes_flat = axes.flatten()

    for idx, (ax, C_base) in enumerate(zip(axes_flat, target_C_base_layout)):
        col_idx = idx

        for method_name, method_cfg in methods.items():
            data = curve_data[method_name][C_base]
            episode_x = data["episode_x"]
            mean_curve = data["mean"]
            std_curve = data["std"]

            if show_std_band:
                ax.fill_between(
                    episode_x,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    color=method_cfg["color"],
                    alpha=BAND_ALPHA,
                    linewidth=0,
                )

            line, = ax.plot(
                episode_x,
                mean_curve,
                color=method_cfg["color"],
                linestyle=method_cfg["linestyle"],
                linewidth=1.35,
                marker=method_cfg["marker"], markersize=MARKER_SIZE,
                markevery=max(1, len(episode_x) // 8),
                markerfacecolor="white", markeredgewidth=MARKER_EDGE_WIDTH,
            )

            if idx == 0:
                legend_handles.append(line)
                legend_labels.append(method_cfg["label"])

        y_lim = common_y_lim
        text_y, text_va = choose_c0_label_position(curve_data, C_base, y_lim)

        ax.text(
            0.93,
            text_y,
            rf"$c_0={format_number(C_base)}$",
            transform=ax.transAxes,
            ha="right",
            va=text_va,
            fontsize=LABEL_FONT_SIZE,
            fontweight="bold",
        )

        ax.grid(False)
        ax.set_xlim(x_axis_min, x_axis_max)
        ax.set_ylim(y_lim)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_tick_count))

        if col_idx == 0:
            ax.set_xticks(fixed_x_ticks_left)
            ax.set_xticklabels([format_number(v) for v in fixed_x_ticks_left])
        else:
            ax.set_xticks(fixed_x_ticks_right)
            ax.set_xticklabels([format_number(v) for v in fixed_x_ticks_right])

        beautify_axes(ax)

    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    fig.legend(
        legend_handles, legend_labels,
        loc="lower center",
        bbox_to_anchor=((subplot_left + subplot_right) / 2, subplot_top),
        ncol=4, frameon=False, fontsize=LEGEND_FONT_SIZE,
        handlelength=1.3, handletextpad=0.4, columnspacing=0.8,
        borderaxespad=0.25, labelspacing=0.25,
    )

    fig.supxlabel(
        "episode",
        x=(subplot_left + subplot_right) / 2,
        y=supxlabel_y,
        fontsize=LABEL_FONT_SIZE,
        fontweight="bold",
    )

    fig.supylabel(
        metric_label,
        x=supylabel_x,
        y=(subplot_bottom + subplot_top) / 2,
        fontsize=LABEL_FONT_SIZE,
        fontweight="bold",
    )

    fig.subplots_adjust(
        left=subplot_left,
        right=subplot_right,
        bottom=subplot_bottom,
        top=subplot_top,
        wspace=subplot_wspace,
        hspace=0.0,
    )

    safe_metric_name = make_safe_filename(target_metric)
    base_save_name = f"Beijing-Dongfang-PureTD3-vs-PurePPO-{safe_metric_name}-under-different-base-fees"

    os.makedirs(save_dir, exist_ok=True)
    png_save_path = os.path.join(save_dir, base_save_name + ".png")
    pdf_save_path = os.path.join(save_dir, base_save_name + ".pdf")

    plt.savefig(
        png_save_path,
        format="png",
        dpi=png_dpi,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.savefig(
        pdf_save_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close(fig)

    print("=" * 80)
    print(f"PNG saved to: {png_save_path}")
    print(f"PDF saved to: {pdf_save_path}")
    print(f"Sheet: {sheet_name}")
    print(f"Metric: {target_metric} -> {metric_aliases.get(target_metric, target_metric)}")
    print(f"C_base layout: {target_C_base_layout}")
    print(f"Seeds: {seed_list}")


if __name__ == "__main__":
    plot_reward_grid()
