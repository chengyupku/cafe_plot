import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.ticker as ticker
from microbenchmark_results import *

# 柔和淡雅配色方案
colers_sets = [
    (251/255, 128/255, 114/255),  # 淡橙红
    (252/255, 205/255, 229/255),  # 淡粉色
    # (255/255, 190/255, 187/255),  # 淡珊瑚粉
    (141/255, 211/255, 199/255),  # 淡青绿
    (190/255, 184/255, 220/255),  # 淡紫色
    (253/255, 180/255, 98/255),   # 淡橙黄
    (179/255, 222/255, 105/255),  # 淡黄绿
    (128/255, 177/255, 211/255),  # 淡蓝色
]

# ]
# colers_sets = [
#     # nilu
#     (20 / 255, 54 / 255, 95 / 255),
#     (248 / 255, 231 / 255, 210 / 255),
#     # (118 / 255, 162 / 255, 185 / 255),
#     (191 / 255, 217 / 255, 229 / 255),
#     (214 / 255, 79 / 255, 56 / 255),
#     (112 / 255, 89 / 255, 146 / 255),
#     # dori
#     (214 / 255, 130 / 255, 148 / 255),
#     (169 / 255, 115 / 255, 153 / 255),
#     (248 / 255, 242 / 255, 236 / 255),
#     (214 / 255, 130 / 255, 148 / 255),
#     (243 / 255, 191 / 255, 202 / 255),
#     # (41/ 255, 31/ 255, 39/ 255),
#     # coller
#     # (72/ 255, 76/ 255, 35/ 255),
#     (124 / 255, 134 / 255, 65 / 255),
#     (185 / 255, 198 / 255, 122 / 255),
#     (248 / 255, 231 / 255, 210 / 255),
#     (182 / 255, 110 / 255, 151 / 255),
# ]
hatch_patterns = ["-", "+", "x", "\\", ".", "o", "O", "*"]


fig = plt.figure(figsize=(16, 8))

legend_items = {}

llm_legands = []
other_legands = []


def get_legend_item(label):
    if label not in legend_items:
        idx = len(legend_items)
        legend_items[label] = (
            colers_sets[idx % len(colers_sets)],
            hatch_patterns[idx % len(hatch_patterns)],
        )
    return legend_items[label]


# 设置网格布局
gs = gridspec.GridSpec(
    4, 24, figure=fig, height_ratios=[1, 1, 1, 1], wspace=3.2, hspace=0.4
)


#  Figure 1.1
ax1_1 = fig.add_subplot(gs[0, 0:12])
ax1_1.set_title("MHA/GQA on H100")
ax1_1.set_ylim(0, 2.5)  # 下面的图为0到5

ax1_1.spines["top"].set_visible(True)

times_data = mha_times_data_h100
providers = mha_providers
_1x_baseline = "TileSight"
_1x_baseline_times = dict(times_data)[_1x_baseline]

norm_time_data = []
for label, times in times_data:
    norm_time = [
        t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
    ]
    norm_time_data.append((label, norm_time))

x = np.arange(len(providers))

bar_width =0.2

ax1_1.axhline(y=1, color="black", linestyle="dashed")

for i, (label, norm_time) in enumerate(norm_time_data):
    print(label)
    if label not in llm_legands:
        llm_legands.append(label)

    rec = ax1_1.bar(
        x + i * bar_width,
        norm_time,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )
    for rect in rec:
        height = rect.get_height()
        if height == 0:
            warning_text = f"{label} Failed"
            ax1_1.text(
                rect.get_x() + rect.get_width() / 2 + 0.01,
                height + 0.05,
                warning_text,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color="red",
                weight="bold",
            )

ax1_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax1_1.set_xticklabels(providers, fontsize=10)
ax1_1.grid(False)
ax1_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax1_1.set_xticklabels(providers, fontsize=10)



#  Figure 1.2
ax1_2 = fig.add_subplot(gs[0, 12:24])
ax1_2.set_title("MHA/GQA on MI210")
ax1_2.set_ylim(0, 2.5)  # 下面的图为0到5

ax1_2.spines["top"].set_visible(True)

times_data = mha_times_data_mi210
providers = mha_providers
_1x_baseline = "TileSight"
_1x_baseline_times = dict(times_data)[_1x_baseline]

norm_time_data = []
for label, times in times_data:
    norm_time = [
        t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
    ]
    norm_time_data.append((label, norm_time))

x = np.arange(len(providers))

bar_width =0.2

ax1_2.axhline(y=1, color="black", linestyle="dashed")

for i, (label, norm_time) in enumerate(norm_time_data):
    print(label)
    if label not in llm_legands:
        llm_legands.append(label)

    rec = ax1_2.bar(
        x + i * bar_width,
        norm_time,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )
    for rect in rec:
        height = rect.get_height()
        if height == 0:
            warning_text = f"{label} Failed"
            ax1_2.text(
                rect.get_x() + rect.get_width() / 2 + 0.01,
                height + 0.05,
                warning_text,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color="red",
                weight="bold",
            )

ax1_2.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax1_2.set_xticklabels(providers, fontsize=10)
ax1_2.grid(False)
ax1_2.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax1_2.set_xticklabels(providers, fontsize=10)


#  Figure 2.1
ax2_1 = fig.add_subplot(gs[1, 0:12])
ax2_1.set_title("MLA on H100")
ax2_1.set_ylim(0, 12)  # 下面的图为0到5

ax2_1.spines["top"].set_visible(True)

times_data = mla_times_data_h100
providers = mla_providers
_1x_baseline = "TileSight"
_1x_baseline_times = dict(times_data)[_1x_baseline]

norm_time_data = []
for label, times in times_data:
    norm_time = [
        t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
    ]
    norm_time_data.append((label, norm_time))

x = np.arange(len(providers))

bar_width =0.2

ax2_1.axhline(y=1, color="black", linestyle="dashed")

for i, (label, norm_time) in enumerate(norm_time_data):
    print(label)
    if label not in llm_legands:
        llm_legands.append(label)

    rec = ax2_1.bar(
        x + i * bar_width,
        norm_time,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )
    for rect in rec:
        height = rect.get_height()
        if height == 0:
            warning_text = f"{label} Failed"
            ax2_1.text(
                rect.get_x() + rect.get_width() / 2 + 0.01,
                height + 0.05,
                warning_text,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color="red",
                weight="bold",
            )

ax2_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax2_1.set_xticklabels(providers, fontsize=10)
ax2_1.grid(False)
ax2_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax2_1.set_xticklabels(providers, fontsize=10)



#  Figure 2.2
gs_mla_mi210 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 12:24], hspace=0.25)
ax2_2_2 = fig.add_subplot(gs_mla_mi210[0])
ax2_2_1 = fig.add_subplot(gs_mla_mi210[1:])

# ax2_2_2 = fig.add_subplot(gs[1, 0:24])
# ax2_2_1 = fig.add_subplot(gs[2, 0:24])
# 设置两个Y轴的范围
ax2_2_2.set_ylim(28,40)  # 上面的图为10到最大值
ax2_2_1.set_ylim(0, 10)  # 下面的图为0到5
# Draw cublas as a horizontal dashed line
ax2_2_2.axhline(y=1, color="black", linestyle="dashed")
ax2_2_2.axhline(y=1, color="black", linestyle="dashed")
ax2_2_2.spines["bottom"].set_visible(False)
ax2_2_2.set_xticklabels([])
ax2_2_2.set_xticks([])
ax2_2_2.set_title("MLA on MI210")

ax2_2_1.spines["top"].set_visible(False)

times_data = mla_times_data_mi210
providers = mla_providers
_1x_baseline = "TileSight"
_1x_baseline_times = dict(times_data)[_1x_baseline]

norm_time_data = []
for label, times in times_data:
    # if label != _1x_baseline:
    norm_time = [
        t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
    ]
    norm_time_data.append((label, norm_time))
print(norm_time_data)
# Plotting
# fig, ax = plt.subplots(figsize=(6, 2))

# max_speedup = np.ceil(max([max(speedup) for _, speedup in speed_up_data]))

print(norm_time_data)
# Create an array for x-axis positions
x = np.arange(len(providers))

# Set the width of the bars
bar_width =0.2

# Draw cublas as a horizontal dashed line
ax2_2_1.axhline(y=1, color="black", linestyle="dashed")


# Create bars using a loop
print(len(x))
print(len(norm_time_data))
for i, (label, norm_time) in enumerate(norm_time_data):
    print(label)
    if label not in llm_legands:
        llm_legands.append(label)
    
    rec = ax2_2_2.bar(
        x + i * bar_width,
        norm_time,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )

    rec = ax2_2_1.bar(
        x + i * bar_width,
        norm_time,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )
    for rect in rec:
        height = rect.get_height()
        if height == 0:
            warning_text = f"{label} Failed"
            ax2_2_1.text(
                rect.get_x() + rect.get_width() / 2 + 0.01,
                height + 0.05,
                warning_text,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color="red",
                weight="bold",
            )

d = 0.01  # 斜线的长度
kwargs = dict(transform=ax2_2_2.transAxes, color="k", clip_on=False)
ax2_2_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-left diagonal
ax2_2_2.plot((-d, +d), (-d, +d), **kwargs)  # top-right diagonal


kwargs.update(transform=ax2_2_1.transAxes)  # switch to the bottom axes
ax2_2_1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2_2_1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

ax2_2_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax2_2_1.set_xticklabels(providers, fontsize=10)
ax2_2_1.grid(False)

ax2_2_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax2_2_1.set_xticklabels(providers, fontsize=10)



#  Figure 3.1
ax3_1 = fig.add_subplot(gs[2, 0:12])
ax3_1.set_title("GEMM on H100")
ax3_1.set_ylim(0, 3.5)  # 下面的图为0到5

ax3_1.spines["top"].set_visible(True)

times_data = gemm_times_data_h100
providers = gemm_providers
_1x_baseline = "TileSight"
_1x_baseline_times = dict(times_data)[_1x_baseline]

norm_time_data = []
for label, times in times_data:
    norm_time = [
        t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
    ]
    norm_time_data.append((label, norm_time))

x = np.arange(len(providers))

bar_width =0.2

ax3_1.axhline(y=1, color="black", linestyle="dashed")

for i, (label, norm_time) in enumerate(norm_time_data):
    print(label)
    if label not in llm_legands:
        llm_legands.append(label)

    rec = ax3_1.bar(
        x + i * bar_width,
        norm_time,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )
    for rect in rec:
        height = rect.get_height()
        if height == 0:
            warning_text = f"{label} Failed"
            ax3_1.text(
                rect.get_x() + rect.get_width() / 2 + 0.01,
                height + 0.05,
                warning_text,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color="red",
                weight="bold",
            )

ax3_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax3_1.set_xticklabels(providers, fontsize=10)
ax3_1.grid(False)
ax3_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax3_1.set_xticklabels(providers, fontsize=10)



#  Figure 3.2
ax3_2 = fig.add_subplot(gs[2, 12:24])
ax3_2.set_title("GEMM on MI210")
ax3_2.set_ylim(0, 3.5)  # 下面的图为0到5

ax3_2.spines["top"].set_visible(True)

times_data = gemm_times_data_mi210
providers = gemm_providers
_1x_baseline = "TileSight"
_1x_baseline_times = dict(times_data)[_1x_baseline]

norm_time_data = []
for label, times in times_data:
    norm_time = [
        t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
    ]
    norm_time_data.append((label, norm_time))

x = np.arange(len(providers))

bar_width =0.2

ax3_2.axhline(y=1, color="black", linestyle="dashed")

for i, (label, norm_time) in enumerate(norm_time_data):
    print(label)
    if label not in llm_legands:
        llm_legands.append(label)

    rec = ax3_2.bar(
        x + i * bar_width,
        norm_time,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )
    for rect in rec:
        height = rect.get_height()
        if height == 0:
            warning_text = f"{label} Failed"
            ax3_2.text(
                rect.get_x() + rect.get_width() / 2 + 0.01,
                height + 0.05,
                warning_text,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color="red",
                weight="bold",
            )

ax3_2.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax3_2.set_xticklabels(providers, fontsize=10)
ax3_2.grid(False)
ax3_2.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax3_2.set_xticklabels(providers, fontsize=10)


#  Figure 3.1
#  Figure 4.2
gs_dg_h100 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[3, 0:12], hspace=0.25)
ax4_1_2 = fig.add_subplot(gs_dg_h100[0])
ax4_1_1 = fig.add_subplot(gs_dg_h100[1:])

# ax4_1_2 = fig.add_subplot(gs[1, 0:24])
# ax4_1_1 = fig.add_subplot(gs[2, 0:24])
# 设置两个Y轴的范围
ax4_1_2.set_ylim(8,20)  # 上面的图为10到最大值
ax4_1_1.set_ylim(0, 2)  # 下面的图为0到5
# Draw cublas as a horizontal dashed line
ax4_1_2.axhline(y=1, color="black", linestyle="dashed")
ax4_1_2.axhline(y=1, color="black", linestyle="dashed")
ax4_1_2.spines["bottom"].set_visible(False)
ax4_1_2.set_xticklabels([])
ax4_1_2.set_xticks([])
ax4_1_2.set_title("Dequant GEMM on H100")

ax4_1_1.spines["top"].set_visible(False)

times_data = dequant_gemm_times_data_h100
providers = dequant_gemm_providers
_1x_baseline = "TileSight"
_1x_baseline_times = dict(times_data)[_1x_baseline]

norm_time_data = []
for label, times in times_data:
    # if label != _1x_baseline:
    norm_time = [
        t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
    ]
    norm_time_data.append((label, norm_time))
print(norm_time_data)
# Plotting
# fig, ax = plt.subplots(figsize=(6, 2))

# max_speedup = np.ceil(max([max(speedup) for _, speedup in speed_up_data]))

print(norm_time_data)
# Create an array for x-axis positions
x = np.arange(len(providers))

# Set the width of the bars
bar_width =0.2

# Draw cublas as a horizontal dashed line
ax4_1_1.axhline(y=1, color="black", linestyle="dashed")


# Create bars using a loop
print(len(x))
print(len(norm_time_data))
for i, (label, norm_time) in enumerate(norm_time_data):
    print(label)
    if label not in llm_legands:
        llm_legands.append(label)
    
    rec = ax4_1_2.bar(
        x + i * bar_width,
        norm_time,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )

    rec = ax4_1_1.bar(
        x + i * bar_width,
        norm_time,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )
    for rect in rec:
        height = rect.get_height()
        if height == 0:
            warning_text = f"{label} Failed"
            ax4_1_1.text(
                rect.get_x() + rect.get_width() / 2 + 0.01,
                height + 0.05,
                warning_text,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color="red",
                weight="bold",
            )

d = 0.01  # 斜线的长度
kwargs = dict(transform=ax4_1_2.transAxes, color="k", clip_on=False)
ax4_1_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-left diagonal
ax4_1_2.plot((-d, +d), (-d, +d), **kwargs)  # top-right diagonal


kwargs.update(transform=ax4_1_1.transAxes)  # switch to the bottom axes
ax4_1_1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax4_1_1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

ax4_1_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax4_1_1.set_xticklabels(providers, fontsize=10)
ax4_1_1.grid(False)

ax4_1_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax4_1_1.set_xticklabels(providers, fontsize=10)


#  Figure 3.2
ax4_2 = fig.add_subplot(gs[3, 12:24])
ax4_2.set_title("Dequant GEMM on MI210")
ax4_2.set_ylim(0, 1.5)  # 下面的图为0到5

ax4_2.spines["top"].set_visible(True)

times_data = dequant_gemm_times_data_mi210
providers = dequant_gemm_providers
_1x_baseline = "TileSight"
_1x_baseline_times = dict(times_data)[_1x_baseline]

norm_time_data = []
for label, times in times_data:
    norm_time = [
        t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
    ]
    norm_time_data.append((label, norm_time))

x = np.arange(len(providers))

bar_width =0.2

ax4_2.axhline(y=1, color="black", linestyle="dashed")

for i, (label, norm_time) in enumerate(norm_time_data):
    print(label)
    if label not in llm_legands:
        llm_legands.append(label)

    rec = ax4_2.bar(
        x + i * bar_width,
        norm_time,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )
    for rect in rec:
        height = rect.get_height()
        if height == 0:
            warning_text = f"{label} Failed"
            ax4_2.text(
                rect.get_x() + rect.get_width() / 2 + 0.01,
                height + 0.05,
                warning_text,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color="red",
                weight="bold",
            )

ax4_2.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax4_2.set_xticklabels(providers, fontsize=10)
ax4_2.grid(False)
ax4_2.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax4_2.set_xticklabels(providers, fontsize=10)



legend_fontsize = 6

handles_other = []
labels_other = []
handles_Ladder = []
labels_Ladder = []
for ax in [ax1_1, ax1_2, ax2_1, ax2_2_1, ax2_2_2, ax3_1, ax3_2, ax4_1_1, ax4_1_2, ax4_2]:
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label not in (labels_other + labels_Ladder):
            if "Ladder" in label:
                handles_Ladder.append(handle)
                labels_Ladder.append(label)
            else:
                handles_other.append(handle)
                labels_other.append(label)
        else:
            pass
handles_other.extend(handles_Ladder)
labels_other.extend(labels_Ladder)
print(handles_other)
# 调整图例位置和大小
legend_fontsize = 12
fig.legend(
    handles_other,
    labels_other,
    loc="upper center",           # 图例中心对齐
    bbox_to_anchor=(0.45, 0.980 + 0.0),   # 调整图例到图的顶部
    ncol=len(labels_other),       # 根据标签数量设置为水平排开
    fontsize=legend_fontsize,     # 字体大小
    frameon=True,                 # 是否显示边框
    facecolor='white',            # 边框背景色
    edgecolor='black'             # 边框颜色
)

fig.text(
    0.09,
    0.5,
    "Normalized latency Vs. TileSight (lower is better)",
    fontsize=15,
    rotation=90,
    va="center",
    ha="center",
)
plt.subplots_adjust(top=0.9, bottom=0.15, right=0.75)
# plt.show()

plt.savefig(
    "./microbenchmark_tilesight.pdf",
    bbox_inches="tight",
)