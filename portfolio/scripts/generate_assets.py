#!/usr/bin/env python3
"""Generate public-safe portfolio figures from verified repository evidence.

This script does not run model training or inference. Values are deliberately
hard-coded only when a source is listed in EVIDENCE.md; author-published values
whose raw logs are unavailable are labeled inside the corresponding assets.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

NAVY = "#17324D"
BLUE = "#2F6BFF"
CYAN = "#10A6A6"
TEAL = "#0E8A72"
ORANGE = "#E97932"
PALE_BLUE = "#EAF0FF"
PALE_CYAN = "#E7F7F7"
PALE_ORANGE = "#FFF0E6"
LIGHT = "#F5F7FA"
MID = "#6B7785"
GRID = "#DCE2E8"


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 22,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "svg.fonttype": "none",
        }
    )


def box(ax, xy, width, height, title, subtitle, facecolor, edgecolor):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=2.2,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    cx = xy[0] + width / 2
    ax.text(cx, xy[1] + height * 0.62, title, ha="center", va="center", fontsize=18, weight="bold", color=NAVY)
    ax.text(cx, xy[1] + height * 0.29, subtitle, ha="center", va="center", fontsize=14, color=MID)


def arrow(ax, start, end, color=NAVY, style="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=2.4,
            linestyle=style,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(18, 9), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.93, "Unified Multimodal BEV Architecture", fontsize=30, weight="bold", color=NAVY)
    ax.text(
        0.04,
        0.875,
        "Camera and LiDAR features meet under one canonical spatial contract.",
        fontsize=17,
        color=MID,
    )

    box(ax, (0.04, 0.59), 0.14, 0.15, "Camera", "6 views", LIGHT, BLUE)
    box(ax, (0.24, 0.59), 0.18, 0.15, "WidthFormer", "camera BEV", PALE_BLUE, BLUE)
    box(ax, (0.04, 0.29), 0.14, 0.15, "LiDAR", "point set", LIGHT, CYAN)
    box(ax, (0.24, 0.29), 0.18, 0.15, "DSVT", "LiDAR BEV", PALE_CYAN, CYAN)
    box(ax, (0.51, 0.44), 0.20, 0.17, "Depth-guided Fusion", "deformable attention", PALE_ORANGE, ORANGE)
    box(ax, (0.78, 0.44), 0.15, 0.17, "DAL Head", "decoupled tasks", "#ECF7F3", TEAL)

    arrow(ax, (0.18, 0.665), (0.24, 0.665), BLUE)
    arrow(ax, (0.18, 0.365), (0.24, 0.365), CYAN)
    arrow(ax, (0.42, 0.665), (0.51, 0.555), BLUE)
    arrow(ax, (0.42, 0.365), (0.51, 0.495), CYAN)
    arrow(ax, (0.71, 0.525), (0.78, 0.525), ORANGE)
    arrow(ax, (0.42, 0.335), (0.80, 0.435), CYAN, "--")
    ax.text(0.60, 0.335, "LiDAR-only box regression", ha="center", va="center", fontsize=14, color=CYAN)

    badge = FancyBboxPatch(
        (0.29, 0.08),
        0.42,
        0.105,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=NAVY,
        edgecolor=NAVY,
    )
    ax.add_patch(badge)
    ax.text(0.50, 0.145, "Canonical BEV coordinate convention", ha="center", va="center", fontsize=15, color="white")
    ax.text(0.50, 0.105, "[B, C, Y, X]   •   row = Y   •   column = X", ha="center", va="center", fontsize=17, weight="bold", color="white")

    fig.savefig(ASSETS / "01_architecture.png", dpi=100, bbox_inches="tight", facecolor="white")
    fig.savefig(ASSETS / "01_architecture.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def recovery() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=100, gridspec_kw={"width_ratios": [0.92, 1.35]})
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "DAL Target Layout Bug: Measured Recovery",
        x=0.055,
        y=0.965,
        ha="left",
        va="top",
        fontsize=29,
        weight="bold",
        color=NAVY,
    )
    fig.text(
        0.055,
        0.845,
        "A local target-layout fix restored spatial agreement and non-zero validation mAP.",
        va="top",
        fontsize=17,
        color=MID,
    )

    left = axes[0]
    left.set_title("Target occupancy alignment", loc="left", pad=22, color=NAVY, weight="bold")
    left.text(
        0.0,
        1.01,
        "val frames, GT box centers vs LiDAR occupancy grid",
        transform=left.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        color=MID,
    )
    lx = [0, 1]
    alignment = [43.2, 86.0]
    lbars = left.bar(lx, alignment, width=0.58, color=["#AAB3BD", TEAL], zorder=3)
    left.set_ylim(0, 100)
    left.set_ylabel("Aligned GT centers (%)")
    left.set_xticks(lx, ["Before fix\nrow = X, col = Y", "After fix\nrow = Y, col = X"])
    left.grid(axis="y", color=GRID, linewidth=1.2, zorder=0)
    left.spines[["top", "right"]].set_visible(False)
    left.spines[["left", "bottom"]].set_color(GRID)
    for bar, pct, count in zip(lbars, alignment, ["725 / 1,680", "1,444 / 1,680"]):
        left.text(
            bar.get_x() + bar.get_width() / 2,
            pct + 2.2,
            f"{pct:.1f}%\n{count}",
            ha="center",
            va="bottom",
            fontsize=17,
            weight="bold",
            color=NAVY if pct < 50 else TEAL,
            linespacing=1.25,
        )

    right = axes[1]
    right.set_title("Validation mAP across distinct training intervals", loc="left", pad=22, color=NAVY, weight="bold")
    x = [0, 1, 2]
    values = [0.0, 0.2045, 0.5281]
    bars = right.bar(x, values, width=0.62, color=["#AAB3BD", BLUE, TEAL], zorder=3)
    right.set_ylim(0, 0.62)
    right.set_xlim(-0.65, 2.65)
    right.set_ylabel("nuScenes mAP")
    right.set_xticks(
        x,
        [
            "Before fix\nmini 20 ep + full 2 ep",
            "After fix\npost-fix epoch 1",
            "Best run\npost-fix epoch 19",
        ],
    )
    right.grid(axis="y", color=GRID, linewidth=1.2, zorder=0)
    right.spines[["top", "right"]].set_visible(False)
    right.spines[["left", "bottom"]].set_color(GRID)
    for i, (bar, value) in enumerate(zip(bars, values)):
        right.text(
            bar.get_x() + bar.get_width() / 2,
            max(value + 0.018, 0.025),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=19,
            weight="bold",
            color=[NAVY, BLUE, TEAL][i],
        )
    right.text(
        2.0,
        0.41,
        "NDS 0.5132",
        ha="center",
        va="center",
        fontsize=17,
        color=NAVY,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=TEAL, linewidth=1.8),
    )
    fig.text(
        0.055,
        0.025,
        "Values as recorded on the author's training log page; raw logs not included in this repository.",
        fontsize=14,
        color=MID,
    )
    fig.subplots_adjust(left=0.055, right=0.97, top=0.70, bottom=0.19, wspace=0.18)
    fig.savefig(ASSETS / "02_target_bug_recovery.png", dpi=100, facecolor="white")
    fig.savefig(ASSETS / "02_target_bug_recovery.svg", facecolor="white")
    plt.close(fig)


def single_sweep_tradeoff() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=100)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Single-Sweep Latency Decision: Measured Velocity Cost",
        x=0.055,
        y=0.965,
        ha="left",
        va="top",
        fontsize=29,
        weight="bold",
        color=NAVY,
    )
    fig.text(
        0.055,
        0.845,
        "Same codebase, dataset, and evaluation; temporal LiDAR context is the controlled difference.",
        va="top",
        fontsize=17,
        color=MID,
    )

    labels = ["Single-sweep\nthis run", "Multi-sweep\nBEVFusion reference"]
    colors = [ORANGE, TEAL]

    left = axes[0]
    left.set_title("Velocity error (lower is better)", loc="left", pad=22, color=NAVY, weight="bold")
    mave = [1.132, 0.257]
    bars = left.bar([0, 1], mave, width=0.58, color=colors, zorder=3)
    left.set_ylim(0, 1.3)
    left.set_ylabel("mAVE (m/s)")
    left.set_xticks([0, 1], labels)
    left.grid(axis="y", color=GRID, linewidth=1.2, zorder=0)
    left.spines[["top", "right"]].set_visible(False)
    left.spines[["left", "bottom"]].set_color(GRID)
    for bar, value, color in zip(bars, mave, colors):
        left.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.3f}", ha="center", va="bottom", fontsize=20, weight="bold", color=color)

    right = axes[1]
    right.set_title("NDS velocity term (higher is better)", loc="left", pad=22, color=NAVY, weight="bold")
    velocity_term = [0.000, 0.743]
    bars = right.bar([0, 1], velocity_term, width=0.58, color=colors, zorder=3)
    right.set_ylim(0, 0.86)
    right.set_ylabel("Velocity term")
    right.set_xticks([0, 1], labels)
    right.grid(axis="y", color=GRID, linewidth=1.2, zorder=0)
    right.spines[["top", "right"]].set_visible(False)
    right.spines[["left", "bottom"]].set_color(GRID)
    for bar, value, color in zip(bars, velocity_term, colors):
        right.text(bar.get_x() + bar.get_width() / 2, max(value + 0.025, 0.025), f"{value:.3f}", ha="center", va="bottom", fontsize=20, weight="bold", color=color)

    message = FancyBboxPatch(
        (0.12, 0.04),
        0.76,
        0.07,
        transform=fig.transFigure,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=PALE_ORANGE,
        edgecolor=ORANGE,
        linewidth=2,
    )
    fig.patches.append(message)
    fig.text(
        0.50,
        0.075,
        "Single sweep was an intentional latency decision; measured cost was ~0.07 NDS from velocity alone.",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
        color=NAVY,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.70, bottom=0.22, wspace=0.24)
    fig.savefig(ASSETS / "03_single_sweep_tradeoff.png", dpi=100, facecolor="white")
    fig.savefig(ASSETS / "03_single_sweep_tradeoff.svg", facecolor="white")
    plt.close(fig)


def fp16_vs_fp32() -> None:
    fig, ax = plt.subplots(figsize=(18, 7.5), dpi=100)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.text(0.03, 0.92, "FP16 Instability vs FP32 Completion", fontsize=29, weight="bold", color=NAVY, transform=ax.transAxes)
    ax.text(0.03, 0.845, "The faster, smaller FP16 run failed near the cyclic learning-rate peak.", fontsize=17, color=MID, transform=ax.transAxes)

    columns = ["Precision", "Outcome", "Throughput", "GPU memory"]
    rows = [
        ["FP16", "NaN death at epoch 2, LR 9.2e-4\n(Hungarian assigner: matrix contains invalid numeric entries)", "0.60 s/iter", "12 GB"],
        ["FP32", "20 epochs completed incl. LR peak 1e-3", "0.86 s/iter", "19 GB"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        colWidths=[0.14, 0.50, 0.17, 0.17],
        cellLoc="left",
        colLoc="left",
        bbox=[0.03, 0.22, 0.94, 0.52],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(17)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(4)
        cell.PAD = 0.13
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row == 1:
            cell.set_facecolor(PALE_ORANGE)
            cell.get_text().set_color(NAVY)
            if col == 0:
                cell.get_text().set_weight("bold")
                cell.get_text().set_color(ORANGE)
        else:
            cell.set_facecolor(PALE_CYAN)
            cell.get_text().set_color(NAVY)
            if col == 0:
                cell.get_text().set_weight("bold")
                cell.get_text().set_color(TEAL)
    ax.text(
        0.03,
        0.10,
        "Decision: use FP32 for schedule completion; no new run was performed for this portfolio.",
        fontsize=16,
        weight="bold",
        color=NAVY,
        transform=ax.transAxes,
    )
    fig.savefig(ASSETS / "04_fp16_vs_fp32.svg", facecolor="white")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    configure()
    architecture()
    recovery()
    single_sweep_tradeoff()
    fp16_vs_fp32()


if __name__ == "__main__":
    main()
