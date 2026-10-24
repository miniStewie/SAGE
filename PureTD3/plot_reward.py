import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(THIS_DIR, "pure_td3_info")
save_dir = os.path.join(THIS_DIR, "plot_result", "reward")
os.makedirs(save_dir, exist_ok=True)

seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
C_base_list = [1.0, 3.0, 5.0, 7.0]
target_C_base_layout = [5.0, 7.0, 1.0, 3.0]

sheet_name = "avg"
target_metric = "reward"
metric_aliases = {
    "reward": "total_reward",
}
Y_SCALE = 100.0

show_std_band = True
single_subplot_width = 3.8
single_subplot_height = single_subplot_width * 0.618

fixed_x_ticks_left = [0, 50, 100, 150, 200]
fixed_x_ticks_right = [0, 50, 100, 150, 200]
x_axis_min = 0
x_axis_max = 200
fixed_y_ticks = [4, 6, 8, 10, 12]
fixed_y_lim = (4.0, 13.0)
png_dpi = 600

supxlabel_y = 0.09
supylabel_x = 0.12
plot_area_fraction = 0.72
subplot_left = 0.18
subplot_bottom = 0.18
subplot_right = subplot_left + plot_area_fraction
subplot_top = subplot_bottom + plot_area_fraction

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

BASE_FONT_SIZE = 16
LABEL_FONT_SIZE = 11.5
TICK_FONT_SIZE = 11

plt.rcParams["font.size"] = BASE_FONT_SIZE
plt.rcParams["axes.labelsize"] = LABEL_FONT_SIZE
plt.rcParams["xtick.labelsize"] = TICK_FONT_SIZE
plt.rcParams["ytick.labelsize"] = TICK_FONT_SIZE

method_name = "PureTD3"
method_color = "#1f77b4"

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


def read_seed_curves(target_C_base, metric):
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

        if episode_x is None:
            if "episode" in df.columns:
                episode_x = df["episode"].values
            else:
                episode_x = np.arange(1, len(df) + 1)

        y = df[metric_column].values.astype(float) / Y_SCALE
        if len(y) != len(episode_x):
            raise ValueError(f"{filename} has a different episode count")

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


def plot_reward_grid():
    metric_label = metric_axis_labels.get(target_metric, target_metric)
    curve_data = {}
    all_lower = []
    all_upper = []
    episode_x_ref = None

    for C_base in target_C_base_layout:
        episode_x, mean_curve, std_curve, used_files = read_seed_curves(
            target_C_base=C_base,
            metric=target_metric,
        )

        if episode_x_ref is None:
            episode_x_ref = episode_x
        elif len(episode_x) != len(episode_x_ref):
            raise ValueError("Different C_base groups have different episode counts")

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
    y_lim = fixed_y_lim

    fig_width = 2 * single_subplot_width
    fig_height = 2 * single_subplot_height
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(fig_width, fig_height),
        sharex=False,
        sharey=True,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )

    axes_flat = axes.flatten()
    for idx, (ax, C_base) in enumerate(zip(axes_flat, target_C_base_layout)):
        data = curve_data[C_base]
        episode_x = data["episode_x"]
        mean_curve = data["mean"]
        std_curve = data["std"]
        row_idx = idx // 2
        col_idx = idx % 2

        if show_std_band:
            ax.fill_between(
                episode_x,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color=method_color,
                alpha=0.13,
                linewidth=0,
            )

        ax.plot(
            episode_x,
            mean_curve,
            color=method_color,
            linewidth=1.3,
            marker="o",
            markersize=2.2,
            markevery=max(1, len(episode_x) // 8),
            markerfacecolor="white",
            markeredgewidth=0.8,
        )

        text_y = 0.9 if row_idx == 0 else 0.1
        text_va = "top" if row_idx == 0 else "bottom"
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

        if col_idx == 0:
            ax.set_xticks(fixed_x_ticks_left)
            ax.set_xticklabels([format_number(v) for v in fixed_x_ticks_left])
        else:
            ax.set_xticks(fixed_x_ticks_right)
            ax.set_xticklabels([format_number(v) for v in fixed_x_ticks_right])

        ax.set_yticks(fixed_y_ticks)
        beautify_axes(ax)

    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)

    fig.supxlabel(
        "episode",
        x=0.55,
        y=supxlabel_y,
        fontsize=LABEL_FONT_SIZE,
        fontweight="bold",
    )
    fig.supylabel(
        metric_label,
        x=supylabel_x,
        y=0.54,
        fontsize=LABEL_FONT_SIZE,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=subplot_left,
        right=subplot_right,
        bottom=subplot_bottom,
        top=subplot_top,
        wspace=0.0,
        hspace=0.0,
    )

    safe_metric_name = make_safe_filename(target_metric)
    base_save_name = f"{method_name}-{safe_metric_name}-under-different-base-fees"
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
