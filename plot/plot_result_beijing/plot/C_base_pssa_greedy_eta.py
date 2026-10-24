'''
输入一个字段名，输出 greedy 和 SAGE 的对比图。
'''

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ============================================================
# 0. 入口设置：常用参数都放这里，方便调试
# ============================================================

pssa_data_dir = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\TD3\info\info_beijing"
greedy_data_dir = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\TD3\info\info_beijing_greedy"
save_dir = r"C:\Adisk\project\PyCode\BaseEnv\RFL\SAGE\TD3\plot\plot_result_beijing\sage-vs-greedy"
os.makedirs(save_dir, exist_ok=True)

# 你要画哪个字段，就在这里改
selected_metric = "reward_tu/tc"

available_metrics = [
    "reward_tu/tc",
]

# 是否显示标准差阴影
show_std_band = True

# 图片尺寸
fig_size = (9.5, 5.8)

# 图片分辨率
figure_dpi = 120
save_dpi = 600

# 全局字体大小
global_font_size = 15
axis_label_size = 15
title_size = 16
legend_size = 19

BASE_FONT_SIZE = 16
LABEL_FONT_SIZE = 25
TICK_FONT_SIZE = 25

global_font_size = BASE_FONT_SIZE
axis_label_size = LABEL_FONT_SIZE
tick_label_size = TICK_FONT_SIZE


# ============================================================
# 1. Matplotlib 全局字体设置
# ============================================================

def setup_matplotlib():
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}

    if "Helvetica" not in available_fonts:
        print("Warning: Helvetica is not installed or not detected by Matplotlib.")
        print("Matplotlib may fall back to Arial or DejaVu Sans.")

    matplotlib.rcParams.update({
        # 普通文本统一使用 Helvetica
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],

        # 负号正常显示
        "axes.unicode_minus": False,

        # 字体大小
        "font.size": global_font_size,
        "axes.labelsize": axis_label_size,
        "axes.titlesize": title_size,
        "legend.fontsize": legend_size,
        "xtick.labelsize": tick_label_size,
        "ytick.labelsize": tick_label_size,

        # 坐标轴线宽
        "axes.linewidth": 1.0,

        # 不使用网格
        "axes.grid": False,

        # 注意：这里不把 mathtext 强行改成 Helvetica，
        # 这样 $C_0$、$\eta$ 这类 LaTeX 变量仍然保持数学字体。
    })


setup_matplotlib()


# ============================================================
# 2. 参数设置
# ============================================================

method_data_dirs = {
    "SAGE": pssa_data_dir,  # PSSA 数据，沿用图例名称 SAGE
    "greedy": greedy_data_dir,
}

seed_list = list(range(10))

# 两个目录均独立从 info1.xlsx 编号到 info100.xlsx：
# 外层 seed=0..9，内层 C_base=1..10；不再包含 50、75、100。
full_C_base_list = [
    1.0, 2.0, 3.0, 4.0, 5.0,
    6.0, 7.0, 8.0, 9.0, 10,
]

# 实际绘图只保留 C_0 = 1 到 10
plot_C0_list = [
    1.0, 2.0, 3.0, 4.0, 5.0,
    6.0, 7.0, 8.0, 9.0, 10
]

available_metrics = [
    "eta",
    "total_reward",
    "total_utility",
    "total_send_pks",
    "violate1",
    "reward_tu/tc",
]

metric_titles = {
    "eta": "Eta",
    "total_reward": "reward",
    "total_utility": "utility",
    "total_send_pks": "sent packets",
    "violate1": "Constraint Violation",
    "reward_tu/tc": r"$\eta$",
}

colors = {
    "SAGE": "#1f77b4",
    "greedy": "#d62728",
}

markers = {
    "SAGE": "o",
    "greedy": "s",
}

linestyles = {
    "SAGE": "-",
    "greedy": "--",
}


# ============================================================
# 3. 文件名安全处理函数
# ============================================================

def make_safe_filename(name):
    """
    Windows 文件名中不能包含这些字符：
    \ / : * ? " < > |
    这里统一替换成下划线，避免保存图片时报错。
    """
    safe_name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    safe_name = safe_name.strip().strip(".")
    return safe_name


# ============================================================
# 4. 字段合法性检查
# ============================================================

if selected_metric not in available_metrics:
    raise ValueError(
        f"字段名设置错误：{selected_metric}\n"
        f"可选字段为：{available_metrics}"
    )

metrics = [selected_metric]


# ============================================================
# 5. 读取并整理数据
# ============================================================

records = []

for method, data_dir in method_data_dirs.items():
    for seed_idx, seed in enumerate(seed_list):
        for c_idx, C_base in enumerate(full_C_base_list):
            if C_base not in plot_C0_list:
                continue

            file_idx = seed_idx * len(full_C_base_list) + c_idx + 1
            filename = f"info{file_idx}.xlsx"
            filepath = os.path.join(data_dir, filename)

            if not os.path.exists(filepath):
                raise FileNotFoundError(f"找不到文件：{filepath}")

            df = pd.read_excel(filepath, sheet_name="avg", engine="openpyxl")

            if selected_metric not in df.columns:
                raise ValueError(
                    f"{filename} 的 avg sheet 中缺少字段：{selected_metric}"
                )

            one_record = {
                "file": filename,
                "assignment_method": method,
                "seed": seed,
                "C0": C_base,
                selected_metric: df[selected_metric].mean()
            }

            records.append(one_record)

result_df = pd.DataFrame(records)


# ============================================================
# 6. 检查数据数量
# ============================================================

print("每种方法、每个 C0 对应的 seed 数量：")
count_check = result_df.groupby(["assignment_method", "C0"])["seed"].nunique()
print(count_check)


# ============================================================
# 7. 聚合数据
# ============================================================

mean_df = (
    result_df
    .groupby(["assignment_method", "C0"], sort=False)[metrics]
    .mean()
)

std_df = (
    result_df
    .groupby(["assignment_method", "C0"], sort=False)[metrics]
    .std()
)


# ============================================================
# 8. 横轴设置
# ============================================================

# 横轴等间距显示
x_pos = np.arange(len(plot_C0_list))
x_labels = [f"{x:g}" for x in plot_C0_list]


# ============================================================
# 9. 绘图
# ============================================================

fig, ax = plt.subplots(figsize=fig_size, dpi=figure_dpi)

for method in method_data_dirs:

    y_mean = np.array([
        mean_df.loc[(method, C0), selected_metric]
        for C0 in plot_C0_list
    ])

    y_std = np.array([
        std_df.loc[(method, C0), selected_metric]
        for C0 in plot_C0_list
    ])

    ax.plot(
        x_pos,
        y_mean,
        label=method,
        color=colors[method],
        linestyle=linestyles[method],
        marker=markers[method],
        markersize=6,
        linewidth=2.4,
        markerfacecolor="white",
        markeredgewidth=1.7,
    )

    if show_std_band:
        ax.fill_between(
            x_pos,
            y_mean - y_std,
            y_mean + y_std,
            color=colors[method],
            alpha=0.13,
            linewidth=0,
        )

title = metric_titles.get(selected_metric, selected_metric)

# ax.set_title(
#     rf"{title} Comparison between SAGE and greedy under Different $C_0$",
#     fontweight="bold",
#     pad=12
# )

ax.set_xlabel(r"$c_0$")
ax.set_ylabel(title)

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)

# 去掉网格
ax.grid(False)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_capstyle("butt")
    spine.set_joinstyle("miter")

ax.tick_params(axis="both", direction="in", length=4, width=1)

ax.legend(
    frameon=True,
    fancybox=True,
    framealpha=0.9,
    borderpad=0.6,

    # 关键：加长图例中的线段，避免虚线被截断得很难看
    handlelength=3.2,
    handletextpad=0.8,

    # 让图例里每条线只显示一个 marker
    numpoints=1
)

fig.tight_layout()


# ============================================================
# 10. 保存图像
# ============================================================

safe_metric_name = make_safe_filename(selected_metric)

save_path_png = os.path.join(
    save_dir,
    f"SAGE-vs-greedy-{safe_metric_name}.png"
)

save_path_pdf = os.path.join(
    save_dir,
    f"SAGE-vs-greedy-{safe_metric_name}.pdf"
)

plt.savefig(save_path_png, dpi=save_dpi, bbox_inches="tight")
plt.savefig(save_path_pdf, bbox_inches="tight")
plt.close()

print(f"PNG 图像已保存到：{save_path_png}")
