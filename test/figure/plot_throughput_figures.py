#!/usr/bin/env python3
"""Plot throughput figures from manually curated JSON tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


LLADA_COLOR = "#8DA9D8"
SDAR_COLOR = "#ED9736"
TREND_GREEN = "#74c476"
BASELINE_BLUE = "#111111"
BAR_BLUE = LLADA_COLOR
BAR_GOLD = SDAR_COLOR
HATCH_GRAY = "#d4d4d4"
TREND_AXIS_LABEL_FONTSIZE = 33
TREND_TICK_FONTSIZE = 36
TREND_LEGEND_FONTSIZE = 30
TREND_ANNOTATION_FONTSIZE = 32
TREND_BASELINE_FONTSIZE = 36


TASK_LABELS = {
    "gsm8k": "GSM8K",
    "math": "MATH",
    "gpqa": "GPQA",
    "mmlu": "MMLU",
    "humaneval": "HumanEval",
    "sharegpt": "ShareGPT",
    "ruler_1k": "RULER-1K",
    "ruler_2k": "RULER-2K",
    "ruler_3k": "RULER-3K",
    "ruler_4k": "RULER-4K",
    "ruler_1_4k": "RULER-1-4K",
    "ruler_1~4k mix": "RULER-1-4K mix",
}

LONG_CONTEXT_LABELS = {
    "ruler_1k": "1K",
    "ruler_2k": "2K",
    "ruler_3k": "3K",
    "ruler_4k": "4K",
    "ruler_1_4k": "1-4K mix",
    "ruler_1~4k mix": "1-4K mix",
}

LONG_CONTEXT_ORDER = {
    "ruler_1k": 1,
    "ruler_2k": 2,
    "ruler_1_4k": 2.5,
    "ruler_1~4k mix": 2.5,
    "ruler_3k": 3,
    "ruler_4k": 4,
}
LONG_CONTEXT_EXCLUDED_TASKS = {"ruler_1_4k", "ruler_1~4k mix"}

TASK_NUMBER_ORDER = [
    "gsm8k",
    "math",
    "humaneval",
    "sharegpt",
    "mmlu",
    "gpqa",
    "ruler_1k",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def output_path(output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / name


def bubble_x(row: Mapping) -> float:
    return float(row["steps_per_block_cov_over_mean"])


def speedup_y(row: Mapping) -> float:
    return float(row["throughput_speedup"])


def linear_fit(rows: Sequence[Mapping]) -> tuple[np.ndarray, np.ndarray, float] | None:
    if len(rows) < 2:
        return None
    xs = np.array([bubble_x(r) for r in rows], dtype=float)
    ys = np.array([speedup_y(r) for r in rows], dtype=float)
    if np.allclose(xs, xs[0]):
        return None
    slope, intercept = np.polyfit(xs, ys, 1)
    fit_x = np.linspace(float(xs.min()), float(xs.max()), 100)
    fit_y = slope * fit_x + intercept
    r = float(np.corrcoef(xs, ys)[0, 1])
    return fit_x, fit_y, r


def log_x_linear_fit(rows: Sequence[Mapping]) -> tuple[np.ndarray, np.ndarray, float] | None:
    if len(rows) < 2:
        return None
    xs = np.array([bubble_x(r) for r in rows], dtype=float)
    ys = np.array([speedup_y(r) for r in rows], dtype=float)
    if np.any(xs <= 0):
        return None
    log_xs = np.log10(xs)
    if np.allclose(log_xs, log_xs[0]):
        return None
    slope, intercept = np.polyfit(log_xs, ys, 1)
    fit_x = np.logspace(float(log_xs.min()), float(log_xs.max()), 100)
    fit_y = slope * np.log10(fit_x) + intercept
    r = float(np.corrcoef(log_xs, ys)[0, 1])
    return fit_x, fit_y, r


def long_context_rows(data: Mapping) -> list[Mapping]:
    rows = [
        row
        for row in data.get("long_context_step_variability", [])
        if str(row.get("task")) not in LONG_CONTEXT_EXCLUDED_TASKS
    ]
    return sorted(rows, key=lambda r: LONG_CONTEXT_ORDER.get(str(r.get("task")), 99))


def task_rows(data: Mapping) -> list[Mapping]:
    rows = data.get("task_step_variability")
    if rows:
        return list(rows)
    return long_context_rows(data)


def short_long_context_label(task: str) -> str:
    return LONG_CONTEXT_LABELS.get(task, TASK_LABELS.get(task, task))


def task_label(task: str) -> str:
    return TASK_LABELS.get(task, task.upper())


def annotate_points(
    ax,
    rows: Sequence[Mapping],
    label_fn,
    offsets: Mapping[str, tuple] | None = None,
    fontsize: int = TREND_ANNOTATION_FONTSIZE,
) -> None:
    offsets = offsets or {}
    for row in rows:
        task = str(row["task"])
        offset = offsets.get(task, (8, 8))
        dx, dy = offset[:2]
        ha = offset[2] if len(offset) > 2 else ("right" if dx < 0 else "left")
        va = offset[3] if len(offset) > 3 else "center"
        ax.annotate(
            label_fn(task),
            xy=(bubble_x(row), speedup_y(row)),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=fontsize,
            fontweight="bold",
            ha=ha,
            va=va,
            color="#222222",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.68, pad=0.2),
            clip_on=False,
        )


def annotate_points_with_connectors(
    ax,
    rows: Sequence[Mapping],
    label_fn,
    label_positions: Mapping[str, tuple],
    color: str,
    fontsize: int = 28,
) -> None:
    for row in rows:
        task = str(row["task"])
        if task not in label_positions:
            continue

        position = label_positions[task]
        label_x, label_y = position[:2]
        ha = position[2] if len(position) > 2 else "center"
        va = position[3] if len(position) > 3 else "center"
        annotation = ax.annotate(
            label_fn(task),
            xy=(bubble_x(row), speedup_y(row)),
            xytext=(label_x, label_y),
            textcoords="data",
            fontsize=fontsize,
            fontweight="bold",
            ha=ha,
            va=va,
            color="#222222",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.74, pad=0.12),
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                linewidth=1.8,
                alpha=0.72,
                shrinkA=4,
                shrinkB=9,
            ),
            clip_on=False,
            zorder=6,
        )
        if annotation.arrow_patch is not None:
            annotation.arrow_patch.set_zorder(3)


def annotate_task_numbers(
    ax,
    rows: Sequence[Mapping],
    task_numbers: Mapping[str, int],
) -> None:
    for row in rows:
        task = str(row["task"])
        number = task_numbers.get(task)
        if number is None:
            continue
        ax.text(
            bubble_x(row),
            speedup_y(row) - 0.005,
            str(number),
            ha="center",
            va="center",
            fontsize=36,
            fontweight="bold",
            color="#111111",
            clip_on=False,
            zorder=8,
        )


def annotate_task_number_callouts(
    ax,
    rows: Sequence[Mapping],
    task_numbers: Mapping[str, int],
    offsets: Mapping[str, tuple[int, int]],
) -> None:
    for row in rows:
        task = str(row["task"])
        number = task_numbers.get(task)
        if number is None:
            continue
        dx, dy = offsets.get(task, (-36, 34))
        ax.annotate(
            str(number),
            xy=(bubble_x(row), speedup_y(row)),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=36,
            fontweight="bold",
            color="#111111",
            bbox=dict(
                boxstyle="circle,pad=0.22",
                facecolor=LLADA_COLOR,
                edgecolor="#111111",
                linewidth=1.2,
                alpha=0.98,
            ),
            arrowprops=dict(
                arrowstyle="<|-",
                color="#111111",
                linewidth=2,
                shrinkA=4,
                shrinkB=12,
            ),
            clip_on=False,
            zorder=10,
        )


def annotate_marker_labels(
    ax,
    rows: Sequence[Mapping],
    label_fn,
    fontsize: int = 30,
) -> None:
    for row in rows:
        ax.text(
            bubble_x(row),
            speedup_y(row) - 0.005,
            label_fn(str(row["task"])),
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="#111111",
            clip_on=False,
            zorder=8,
        )


def draw_direction_arrows(ax, rows: Sequence[Mapping]) -> None:
    for prev, curr in zip(rows, rows[1:]):
        ax.annotate(
            "",
            xy=(bubble_x(curr), speedup_y(curr)),
            xytext=(bubble_x(prev), speedup_y(prev)),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=2.0, shrinkA=13, shrinkB=13),
        )


def setup_bubble_axis(ax) -> None:
    ax.axhline(1.0, color=BASELINE_BLUE, linestyle=":", linewidth=3.4, alpha=0.9)
    ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.set_xlabel("Iteration per Block (CoV / Mean)", fontsize=TREND_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Throughput Improvement", fontsize=TREND_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TREND_TICK_FONTSIZE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_long_context_trend(llada: Mapping, sdar: Mapping, output: Path, dpi: int) -> None:
    llada_rows = long_context_rows(llada)
    sdar_rows = long_context_rows(sdar)

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    ax.set_xscale("log")

    ax.scatter(
        [bubble_x(r) for r in llada_rows],
        [speedup_y(r) for r in llada_rows],
        s=3000,
        color=LLADA_COLOR,
        edgecolor="black",
        linewidth=1.2,
        label="LLaDA2.0-mini",
        zorder=4,
    )
    ax.scatter(
        [bubble_x(r) for r in sdar_rows],
        [speedup_y(r) for r in sdar_rows],
        s=2600,
        marker="s",
        color=SDAR_COLOR,
        edgecolor="black",
        linewidth=1.2,
        label="SDAR-8B-Chat",
        zorder=4,
    )

    llada_fit = log_x_linear_fit(llada_rows)
    llada_trend_handle = None
    if llada_fit is not None:
        fx, fy, r = llada_fit
        (llada_trend_handle,) = ax.plot(
            fx,
            fy,
            color=LLADA_COLOR,
            linewidth=5,
            label=f"LLADA($r={abs(r):.2f}$)",
            zorder=2,
        )
    sdar_fit = log_x_linear_fit(sdar_rows)
    sdar_trend_handle = None
    if sdar_fit is not None:
        fx, fy, r = sdar_fit
        (sdar_trend_handle,) = ax.plot(
            fx,
            fy,
            color=SDAR_COLOR,
            linewidth=5,
            label=f"SDAR($r={abs(r):.2f}$)",
            zorder=2,
        )

    annotate_marker_labels(ax, llada_rows, short_long_context_label)
    annotate_marker_labels(ax, sdar_rows, short_long_context_label)

    setup_bubble_axis(ax)
    #ax.set_xlabel("Bubble opportunity score = Steps/Block CoV / mean", fontsize=TREND_AXIS_LABEL_FONTSIZE)
    #ax.set_ylabel("Throughput speedup", fontsize=TREND_AXIS_LABEL_FONTSIZE)
    ax.set_xlim(0.033, 0.145)
    ax.set_ylim(0.95, 1.84)
    ax.set_xticks([0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.14])
    ax.set_yticks([1.0, 1.2, 1.4, 1.6, 1.8])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.tick_params(axis="both", labelsize=TREND_TICK_FONTSIZE)
    from matplotlib.lines import Line2D

    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=LLADA_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=30,
            label="LLaDA2.0-mini",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=SDAR_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=30,
            label="SDAR-8B-Chat",
        ),
    ]
    trend_handles = [
        handle
        for handle in (llada_trend_handle, sdar_trend_handle)
        if handle is not None
    ]
    baseline_handle = Line2D(
        [0],
        [0],
        color=BASELINE_BLUE,
        linestyle=":",
        linewidth=3.4,
        alpha=0.9,
        label="Baseline",
    )
    leg = ax.legend(
        handles=model_handles + trend_handles + [baseline_handle],
        loc="upper right",
        bbox_to_anchor=(1.035, 1.03),
        frameon=True,
        fontsize=TREND_LEGEND_FONTSIZE,
        borderpad=0.15,
        labelspacing=0.02,
        handlelength=0.8,
        handleheight=0.0,
        markerscale=0.7
    )
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)
    leg.get_frame().set_facecolor((1, 1, 1, 0.7))
    leg.get_frame().set_edgecolor("#aaaaaa")
    leg.get_frame().set_linewidth(1.2)

    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.00)
    plt.close(fig)
    print(f"[plot] wrote {output}")


def plot_bubble_opportunity_trend(llada: Mapping, sdar: Mapping, output: Path, dpi: int) -> None:
    llada_rows = task_rows(llada)
    sdar_rows = task_rows(sdar)
    available_tasks = {str(row["task"]) for row in llada_rows + sdar_rows}
    ordered_tasks = [task for task in TASK_NUMBER_ORDER if task in available_tasks]
    ordered_tasks.extend(sorted(available_tasks - set(ordered_tasks)))
    task_numbers = {task: idx for idx, task in enumerate(ordered_tasks, start=1)}
    llada_callout_tasks = {"mmlu", "gpqa"}
    llada_inline_rows = [
        row for row in llada_rows
        if str(row["task"]) not in llada_callout_tasks
    ]
    llada_callout_rows = [
        row for row in llada_rows
        if str(row["task"]) in llada_callout_tasks
    ]

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    ax.set_xscale("log")

    ax.scatter(
        [bubble_x(r) for r in llada_inline_rows],
        [speedup_y(r) for r in llada_inline_rows],
        s=2300,
        color=LLADA_COLOR,
        edgecolor="black",
        linewidth=1.2,
        label="LLaDA2.0-mini",
        zorder=4
    )
    ax.scatter(
        [bubble_x(r) for r in llada_callout_rows],
        [speedup_y(r) for r in llada_callout_rows],
        s=500,
        color=LLADA_COLOR,
        edgecolor="black",
        linewidth=1.2,
        label="_nolegend_",
        zorder=4,
    )
    ax.scatter(
        [bubble_x(r) for r in sdar_rows],
        [speedup_y(r) for r in sdar_rows],
        s=1900,
        marker="s",
        color=SDAR_COLOR,
        edgecolor="black",
        linewidth=1.2,
        label="SDAR-8B-Chat",
        zorder=4
    )

    llada_fit = log_x_linear_fit(llada_rows)
    llada_trend_handle = None
    if llada_fit is not None:
        fx, fy, r = llada_fit
        (llada_trend_handle,) = ax.plot(
            fx,
            fy,
            color=LLADA_COLOR,
            linewidth=5,
            label=f"LLADA ($r={abs(r):.2f}$)",
            zorder=2,
        )
    sdar_fit = log_x_linear_fit(sdar_rows)
    sdar_trend_handle = None
    if sdar_fit is not None:
        fx, fy, r = sdar_fit
        (sdar_trend_handle,) = ax.plot(
            fx,
            fy,
            color=SDAR_COLOR,
            linewidth=5,
            label=f"SDAR ($r={abs(r):.2f}$)",
            zorder=2,
        )

    annotate_task_numbers(
        ax,
        llada_inline_rows,
        task_numbers,
    )
    annotate_task_numbers(
        ax,
        sdar_rows,
        task_numbers,
    )
    annotate_task_number_callouts(
        ax,
        llada_callout_rows,
        task_numbers,
        offsets={
            "mmlu": (0, 72),
            "gpqa": (-74, 20),
        },
    )

    setup_bubble_axis(ax)
    #ax.set_xlabel("Bubble opportunity score = Steps/Block CoV / mean", fontsize=TREND_AXIS_LABEL_FONTSIZE)
    #ax.set_ylabel("Throughput speedup", fontsize=TREND_AXIS_LABEL_FONTSIZE)
    ax.set_xlim(0.01, 0.23)
    ax.set_ylim(0.88, 2.62)
    ax.set_yticks([1, 1.5, 2.0, 2.5])
    ax.set_xticks([0.01, 0.02, 0.03, 0.05, 0.10, 0.20])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.tick_params(axis="both", labelsize=TREND_TICK_FONTSIZE)
    from matplotlib.lines import Line2D

    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=LLADA_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=30,
            label="LLaDA2.0-mini",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=SDAR_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=30,
            label="SDAR-8B-Chat",
        ),
    ]
    trend_handles = [
        handle
        for handle in (llada_trend_handle, sdar_trend_handle)
        if handle is not None
    ]
    baseline_handle = Line2D(
        [0],
        [0],
        color=BASELINE_BLUE,
        linestyle=":",
        linewidth=3.4,
        alpha=0.9,
        label="Baseline",
    )
    leg = ax.legend(
        handles=model_handles + trend_handles + [baseline_handle],
        loc="upper left",
        frameon=True,
        fontsize=TREND_LEGEND_FONTSIZE,
        borderpad=0.15,
        labelspacing=0.02,
        handlelength=0.8,
        handleheight=0.0,
        markerscale=0.7
    )
    leg.get_frame().set_facecolor((1, 1, 1, 0.7))
    leg.get_frame().set_edgecolor("#aaaaaa")
    leg.get_frame().set_linewidth(1.2)
    ax.add_artist(leg)

    from matplotlib.lines import Line2D

    task_handles = [
        Line2D([], [], linestyle="none", label=f"{task_numbers[task]}  {task_label(task)}")
        for task in ordered_tasks
    ]
    task_leg = ax.legend(
        handles=task_handles,
        loc="center left",
        bbox_to_anchor=(0.69, 0.37),
        frameon=True,
        fontsize=TREND_LEGEND_FONTSIZE,
        borderpad=0.15,
        labelspacing=0.02,
        handlelength=-0.5,
        handleheight=0,
    )
    task_leg.get_frame().set_facecolor((1, 1, 1, 0.78))
    task_leg.get_frame().set_edgecolor("#aaaaaa")
    task_leg.get_frame().set_linewidth(1.2)

    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.00)
    plt.close(fig)
    print(f"[plot] wrote {output}")


def effectiveness_percent(row: Mapping) -> float:
    tokens = float(row["tokens"])
    return float(row["baseline_effectiveness"]) / tokens * 100.0


_NAVY = "#003366"

def draw_effectiveness_panel(ax, rows: Sequence[Mapping], title: str, colors: list):
    labels = [str(r["token_block_batch"]).split("/")[-1] for r in rows]
    xs = np.arange(len(rows))
    values = [effectiveness_percent(r) for r in rows]
    scales = [float(r["effectiveness_scale_up"]) for r in rows]

    bars = ax.bar(xs, values, width=0.62, color=colors, edgecolor="#777777", linewidth=0.5)

    for x, value in zip(xs, values):
        ax.text(x, value + 0.45, f"{value:.1f}%", ha="center", va="bottom", fontsize=10)

    ax2 = ax.twinx()
    ax2.plot(xs, scales, color=_NAVY, marker="o", markersize=5.5, linewidth=1.5)
    for x, scale in zip(xs, scales):
        ax2.text(x, scale - 0.055, f"{scale:.2f}", color=_NAVY, ha="center", va="top", fontsize=9, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_xlabel("batch size", fontsize=18)
    ax.set_ylim(38, 55)
    ax.grid(True, axis="y", color="#dddddd", alpha=0.85, linewidth=0.8)
    ax.set_axisbelow(True)
    ax2.set_ylim(1.82, 2.40)
    ax2.tick_params(axis="y", colors=_NAVY, labelsize=14)
    ax2.spines["right"].set_color(_NAVY)
    for spine in ("top",):
        ax.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    return ax2


def plot_gsm8k_effectiveness(llada: Mapping, output: Path, dpi: int) -> None:
    rows = list(llada.get("throughput_scaling", []))
    # batch_size is the last field in "token_block_batch" (e.g. "512/32/16" → 16)
    batch_rows = [r for r in rows if r.get("group") == "purple"]
    batch_rows = sorted(batch_rows, key=lambda r: int(str(r["token_block_batch"]).split("/")[-1]))

    labels = [str(r["token_block_batch"]).split("/")[-1] for r in batch_rows]
    block_values = [effectiveness_percent(r) for r in batch_rows]
    scales = [float(r["effectiveness_scale_up"]) for r in batch_rows]
    iter_values = [b * s for b, s in zip(block_values, scales)]

    BLOCK_COLOR = LLADA_COLOR
    ITER_COLOR = SDAR_COLOR
    y_spacing = 0.24
    group_gap = 0.1
    y_block = np.arange(len(batch_rows)) * y_spacing
    y_iter = y_block + y_block[-1] + y_spacing + group_gap
    y = np.concatenate([y_block, y_iter])
    height = 0.22
    bar_hatches = ["", "", ""] * 2
    bar_alphas = [0.6, 0.8, 1] * 2

    fig, ax = plt.subplots(1, 1, figsize=(8, 2.8), constrained_layout=True)
    block_bars = ax.barh(y_block, block_values, height=height, color=BLOCK_COLOR,
                         edgecolor="#111111", linewidth=0.5, label="Block Level Batch")
    iter_bars = ax.barh(y_iter, iter_values, height=height, color=ITER_COLOR,
                        edgecolor="#111111", linewidth=0.5, label="Iteration Level Batch")
    for bar, hatch, alpha in zip([*block_bars, *iter_bars], bar_hatches, bar_alphas):
        bar.set_hatch(hatch)
        bar.set_alpha(alpha)

    for y_pos, scale, val in zip(y_iter, scales, iter_values):
        ax.text(val + 1.0, y_pos, f"{scale:.2f}x",
                ha="left", va="center", fontsize=16, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels + labels, fontsize=16)
    ax.set_xlabel("Batch Efficiency (%)", fontsize=22, labelpad=0)
    ax.set_ylabel("Batch Size", fontsize=22, labelpad=0)
    ax.tick_params(axis="both", labelsize=18)
    ymax = max(iter_values) * 1.15
    ax.set_xlim(0, ymax)
    ax.set_xticks(np.arange(0, ymax, 20))
    ax.grid(True, axis="x", color="#dddddd", alpha=0.85, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    leg = fig.legend(
        loc="lower right",
        ncol=1,
        frameon=True,
        fontsize=20,
        bbox_to_anchor=(0.98, 0.3),
        borderpad=0.1,
        labelspacing=0.1,
        columnspacing=0.6,
        handletextpad=0.4,
        borderaxespad=0.0,
        handlelength=1,
    )
    leg.get_frame().set_facecolor((1, 1, 1, 0.7))
    leg.get_frame().set_edgecolor("#aaaaaa")
    leg.get_frame().set_linewidth(1.2)
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)   
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.00)
    plt.close(fig)
    print(f"[plot] wrote {output}")


def plot_multigpu(multigpu: Mapping, output: Path, dpi: int) -> None:
    groups = list(multigpu.get("results", []))
    labels = [g["gpu_label"] for g in groups]
    baseline_name = "Ling-mini-2.0 (SGLang)"
    block_sglang_name = "LLaDA2.0-mini (SGLang)"
    block_shiftserve_name = "LLaDA2.0-mini (Ours)"

    sglang_ratios = []
    shiftserve_ratios = []
    for group in groups:
        by_name = {f"{s['model']} ({s['system']})": s for s in group["series"]}
        baseline = float(by_name[baseline_name]["throughput"])
        sglang_ratios.append(float(by_name[block_sglang_name]["throughput"]) / baseline)
        shiftserve_ratios.append(float(by_name[block_shiftserve_name]["throughput"]) / baseline)

    bar_values = sglang_ratios + shiftserve_ratios
    bar_colors = [LLADA_COLOR] * len(labels) + [SDAR_COLOR] * len(labels)
    bar_hatches = ["", "", ""] * 2
    bar_alphas = [0.6, 0.8, 1] * 2
    y_spacing = 0.25
    group_gap = 0.1
    y_sglang = np.arange(len(labels)) * y_spacing
    y_shiftserve = y_sglang + y_sglang[-1] + y_spacing + group_gap
    y = np.concatenate([y_sglang, y_shiftserve])
    y_labels = labels + labels
    height = 0.22
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

    def ratio_label(ratio: float) -> str:
        pct = int(round((ratio - 1.0) * 100))
        direction = "Up" if pct >= 0 else "Down"
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct}%"
        #return f"{sign}{pct}% {direction}"

    bars = ax.barh(
        y,
        bar_values,
        height=height,
        color=bar_colors,
        edgecolor="#111111",
        linewidth=1.0,
    )
    for bar, hatch, alpha in zip(bars, bar_hatches, bar_alphas):
        bar.set_hatch(hatch)
        bar.set_alpha(alpha)
    for bar, ratio in zip(bars, bar_values):
        ax.text(
            ratio + 0.02,
            bar.get_y() + bar.get_height() / 2,
            ratio_label(ratio),
            ha="left",
            va="center",
            fontsize=24,
            fontweight="bold",
            linespacing=0.3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, fontsize=28, va="center")
    ax.set_xlabel("Relative Throughput to AR LLM", fontsize=28)
    ax.set_ylabel("Number of GPU", fontsize=28)
    ax.axvline(1.0, color="#4d4d4d", linestyle="--", linewidth=2.4)
    ax.grid(True, color="#dddddd", alpha=0.85, linewidth=0.8)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=LLADA_COLOR, edgecolor="#111111", label="SGLang"),
        Patch(facecolor=SDAR_COLOR, edgecolor="#111111", label="ShiftServe"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="lower right",
        #bbox_to_anchor=(-0.013, 1.03),
        frameon=True,
        fontsize=28,
        labelspacing=0.0,
        borderpad=0.1,
        handletextpad=0.2,
        handlelength=1

    )
    leg.get_frame().set_facecolor((1, 1, 1, 0.7))
    leg.get_frame().set_edgecolor("#aaaaaa")
    leg.get_frame().set_linewidth(1.2)
    xmax = max(max(sglang_ratios), max(shiftserve_ratios)) + 0.2
    ax.set_xlim(0.0, xmax)
    ax.set_xticks(np.arange(0.0, xmax, 0.5))
    ax.tick_params(axis="x", labelsize=22)
    ax.tick_params(axis="y", labelsize=26)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.00)
    plt.close(fig)
    print(f"[plot] wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot throughput figures from JSON tables.")
    here = Path(__file__).resolve().parent
    parser.add_argument("--llada-json", type=Path, default=here / "throughput_llada2_results.json")
    parser.add_argument("--sdar-json", type=Path, default=here / "throughput_sdar_results.json")
    parser.add_argument("--multigpu-json", type=Path, default=here / "throughput_multigpu_results.json")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/dlm_sched_comparison"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    llada = load_json(args.llada_json)
    sdar = load_json(args.sdar_json)
    multigpu = load_json(args.multigpu_json)

    plot_long_context_trend(
        llada,
        sdar,
        output_path(args.output_dir, "throughput_long_context_trend.pdf"),
        args.dpi,
    )
    plot_bubble_opportunity_trend(
        llada,
        sdar,
        output_path(args.output_dir, "throughput_bubble_opportunity_trend.pdf"),
        args.dpi,
    )
    plot_gsm8k_effectiveness(
        llada,
        output_path(args.output_dir, "throughput_gsm8k_effectiveness.pdf"),
        args.dpi,
    )
    plot_multigpu(
        multigpu,
        output_path(args.output_dir, "throughput_multigpu.pdf"),
        args.dpi,
    )


if __name__ == "__main__":
    main()
