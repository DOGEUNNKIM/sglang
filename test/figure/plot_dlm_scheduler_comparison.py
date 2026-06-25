#!/usr/bin/env python3
"""
Plot scheduler SLO attainment curves from run_dlm_scheduler_comparison.sh output.

The expected input schema is:
{
  "SCHEDULER": {
    "REQUEST_RATE": {
      "task": {
        "strict_ttfb": 0.0,
        "strict_tpob": 0.0,
        "strict_all": 0.0,
        "relaxed_ttfb": 0.0,
        "relaxed_tpob": 0.0,
        "relaxed_all": 0.0,
        "p99_ttfb_ms": 0.0,
        "p99_tpob_ms": 0.0
      }
    }
  }
}

Example:
    python ~/sglang/test/plot_dlm_scheduler_comparison.py \
        --summary /tmp/dlm_sched_comparison/slo_summary.json

    python ~/sglang/test/plot_dlm_scheduler_comparison.py \
        --metrics all ttfb tpob
"""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt


DEFAULT_SCHEDULER_ORDER = ["TTFB", "DECODE", "LST", "SOLA", "FCFS", "PREFILL"]

STYLE = {
    "DECODE": {"label": "FasterTransformer", "color": "#DD8452", "marker": "s", "linestyle": "-"},
    "LST": {"label": "ShiftServe", "color": "#8172B2", "marker": "^", "linestyle": "-"},
    "SOLA": {"label": "SOLA", "color": "#C44E52", "marker": "s", "linestyle": "-"},
    "FCFS": {"label": "FCFS", "color": "#55A868", "marker": "D", "linestyle": "-"},
    "PREFILL": {"label": "vLLM", "color": "#64B5CD", "marker": "v", "linestyle": "-"},
    "TTFB": {"label": "TTFB", "color": "#E377C2", "marker": "o", "linestyle": "-"},
}

METRIC_SUFFIX = {
    "all": "all",
    "ttfb": "ttfb",
    "tpob": "tpob",
}

METRIC_LABEL = {
    "all": "End-to-end",
    "ttfb": "TTFB",
    "tpob": "TPOB",
}

Summary = Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]]


def load_summary(path: Path) -> Summary:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_experiment_label(path: Path) -> str:
    name = path.parent.name
    upper_name = name.upper()
    if "LLADA2" in upper_name:
        return "LLaDA2.0\n-mini"
    if "SDAR" in upper_name:
        return "SDAR\n-8B-Chat"
    return name


def rate_key(rate: str) -> float:
    try:
        return float(rate)
    except ValueError:
        return float("inf")


def sorted_schedulers(summary: Summary, requested: Optional[List[str]]) -> List[str]:
    available = set(summary.keys())
    if requested:
        return [s for s in requested if s in available]

    ordered = [s for s in DEFAULT_SCHEDULER_ORDER if s in available]
    return ordered


def sorted_tasks(summary: Summary, requested: Optional[List[str]]) -> List[str]:
    tasks = set()
    for scheduler_data in summary.values():
        for rate_data in scheduler_data.values():
            tasks.update(rate_data.keys())

    if requested:
        return [task for task in requested if task in tasks]
    return sorted(tasks)


def sorted_tasks_for_summaries(
    summaries: Iterable[Summary],
    requested: Optional[List[str]],
) -> List[str]:
    tasks = set()
    for summary in summaries:
        for scheduler_data in summary.values():
            for rate_data in scheduler_data.values():
                tasks.update(rate_data.keys())

    if requested:
        return [task for task in requested if task in tasks]
    return sorted(tasks)


def sorted_schedulers_for_summaries(
    summaries: Iterable[Summary],
    requested: Optional[List[str]],
) -> List[str]:
    available = set()
    for summary in summaries:
        available.update(summary.keys())

    if requested:
        return [scheduler for scheduler in requested if scheduler in available]

    return [scheduler for scheduler in DEFAULT_SCHEDULER_ORDER if scheduler in available]


def collect_series(
    summary: Summary,
    scheduler: str,
    task: str,
    field: str,
) -> Tuple[List[float], List[float]]:
    xs = []
    ys = []
    for rate, rate_data in summary.get(scheduler, {}).items():
        metrics = rate_data.get(task)
        if not metrics or metrics.get(field) is None:
            continue
        xs.append(rate_key(rate))
        ys.append(float(metrics[field]))

    pairs = sorted(zip(xs, ys), key=lambda pair: pair[0])
    if not pairs:
        return [], []
    return [p[0] for p in pairs], [p[1] for p in pairs]


def collect_p99_max_series(
    summary: Summary,
    scheduler: str,
    task: str,
) -> Tuple[List[float], List[float]]:
    xs = []
    ys = []
    for rate, rate_data in summary.get(scheduler, {}).items():
        metrics = rate_data.get(task)
        if not metrics:
            continue
        ttfb = metrics.get("p99_ttfb_ms")
        tpob = metrics.get("p99_tpob_ms")
        if ttfb is None or tpob is None:
            continue
        xs.append(rate_key(rate))
        ys.append(max(float(ttfb), float(tpob)))

    pairs = sorted(zip(xs, ys), key=lambda pair: pair[0])
    if not pairs:
        return [], []
    return [p[0] for p in pairs], [p[1] for p in pairs]


def p99_value(metrics: Mapping[str, float], field: str) -> Optional[float]:
    if field == "p99_max_ms":
        ttfb = metrics.get("p99_ttfb_ms")
        tpob = metrics.get("p99_tpob_ms")
        if ttfb is None or tpob is None:
            return None
        return max(float(ttfb), float(tpob))

    value = metrics.get(field)
    if value is None:
        return None
    return float(value)


def collect_normalized_p99_series(
    summary: Summary,
    scheduler: str,
    task: str,
    field: str,
    baseline_scheduler: str,
) -> Tuple[List[float], List[float]]:
    xs = []
    ys = []
    scheduler_rates = summary.get(scheduler, {})
    baseline_rates = summary.get(baseline_scheduler, {})

    for rate, rate_data in scheduler_rates.items():
        metrics = rate_data.get(task)
        baseline_metrics = baseline_rates.get(rate, {}).get(task)
        if not metrics or not baseline_metrics:
            continue

        if field == "p99_max_ms":
            ttfb = metrics.get("p99_ttfb_ms")
            tpob = metrics.get("p99_tpob_ms")
            baseline_ttfb = baseline_metrics.get("p99_ttfb_ms")
            baseline_tpob = baseline_metrics.get("p99_tpob_ms")
            if (
                ttfb is None
                or tpob is None
                or baseline_ttfb is None
                or baseline_tpob is None
            ):
                continue
            baseline_ttfb = float(baseline_ttfb)
            baseline_tpob = float(baseline_tpob)
            if baseline_ttfb <= 0 or baseline_tpob <= 0:
                continue
            xs.append(rate_key(rate))
            ys.append(max(float(ttfb) / baseline_ttfb, float(tpob) / baseline_tpob))
        else:
            value = p99_value(metrics, field)
            baseline = p99_value(baseline_metrics, field)
            if value is None or baseline is None or baseline <= 0:
                continue
            xs.append(rate_key(rate))
            ys.append(value / baseline)

    pairs = sorted(zip(xs, ys), key=lambda pair: pair[0])
    if not pairs:
        return [], []
    return [p[0] for p in pairs], [p[1] for p in pairs]


def nice_rate_labels(rates: Iterable[float]) -> List[str]:
    labels = []
    for rate in rates:
        if rate.is_integer():
            labels.append(str(int(rate)))
        else:
            labels.append(f"{rate:g}")
    return labels


TASK_LABELS = {
    "gpqa": "GPQA",
    "gsm8k": "GSM8K",
    "humaneval": "HumanEval",
    "math": "MATH",
    "mmlu": "MMLU",
    "ruler_4k": "RULER-4K",
    "sharegpt": "ShareGPT",
}


SMALL_SUBPLOT_TITLE_FONTSIZE = 11
SUMMARY_SUBPLOT_TITLE_FONTSIZE = 30
SCATTER_SUBPLOT_TITLE_FONTSIZE = 22
COMBINED_SCATTER_TITLE_FONTSIZE = 18
LEGEND_BORDERPAD = 0.05
LEGEND_LABELSPACING = 0.05
LEGEND_HANDLEHEIGHT = 0.55
LEGEND_HANDLE_TEXT_PAD = 0.35
RIGHT_Y_LABEL_FONTSIZE = 24


PRESENTATION_STYLE = {
    "TTFB": {"label": "TTFB", "short": "TB", "color": "#CC79C2", "marker": "o", "alpha": 0.45},
    "DECODE": {"label": "FasterTransformer", "short": "FT", "color": "#D55E00", "marker": "s", "alpha": 0.50},
    "LST": {"label": "ShiftServe", "short": "SS", "color": "#E41A1C", "marker": "^", "alpha": 1.0},
    "SOLA": {"label": "SOLA", "short": "SL", "color": "#8D79C6", "marker": "P", "alpha": 0.48},
    "FCFS": {"label": "FCFS", "short": "FF", "color": "#5AA469", "marker": "D", "alpha": 0.44},
    "PREFILL": {"label": "vLLM", "short": "VL", "color": "#6EC6DF", "marker": "v", "alpha": 0.48},
}


def task_label(task: str) -> str:
    return TASK_LABELS.get(task, task.upper())


def presentation_style(scheduler: str) -> Dict[str, object]:
    style = dict(STYLE.get(scheduler, {}))
    style.update(PRESENTATION_STYLE.get(scheduler, {}))
    return style


def add_right_ylabel(fig, label: str, x: float = 0.985, y: float = 0.45) -> None:
    fig.text(
        x,
        y,
        label,
        rotation=90,
        va="center",
        ha="center",
        fontsize=RIGHT_Y_LABEL_FONTSIZE,
        clip_on=False,
    )


def plot_summary(
    summary: Summary,
    output: Path,
    metrics: List[str],
    tasks: List[str],
    schedulers: List[str],
    title: Optional[str],
    dpi: int,
) -> None:
    if not tasks:
        raise ValueError("No matching tasks found in the summary.")
    if not schedulers:
        raise ValueError("No matching schedulers found in the summary.")

    rows = []
    for metric in metrics:
        rows.extend(
            [
                (f"strict_{METRIC_SUFFIX[metric]}", f"{METRIC_LABEL[metric]}\nTight SLO\nAttainment"),
                (f"relaxed_{METRIC_SUFFIX[metric]}", f"{METRIC_LABEL[metric]}\nLoose SLO\nAttainment"),
            ]
        )

    fig, axes = plt.subplots(
        len(rows),
        len(tasks),
        figsize=(28, 10),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )

    for col, task in enumerate(tasks):
        for row, (field, row_label) in enumerate(rows):
            ax = axes[row][col]

            all_rates = set()
            for scheduler in schedulers:
                xs, ys = collect_series(summary, scheduler, task, field)
                if not xs:
                    continue
                all_rates.update(xs)
                style = STYLE.get(scheduler, {})
                ax.plot(
                    xs,
                    ys,
                    label=style.get("label", scheduler),
                    color=style.get("color"),
                    marker=style.get("marker", "o"),
                    linestyle=style.get("linestyle", "-"),
                    linewidth=2.5,
                    markersize=8,
                )

            ax.set_ylim(-0.03, 1.05)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.grid(True, axis="both", color="#dddddd", linewidth=0.8, alpha=0.85)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=9)

            if all_rates:
                sorted_rates = sorted(all_rates)
                ax.set_xticks(sorted_rates)
                ax.set_xticklabels(nice_rate_labels(sorted_rates))
                xmin, xmax = min(sorted_rates), max(sorted_rates)
                pad = max((xmax - xmin) * 0.07, 0.1)
                ax.set_xlim(xmin - pad, xmax + pad)

            if row == 0:
                ax.set_title(task.upper(), fontsize=SMALL_SUBPLOT_TITLE_FONTSIZE, fontweight="bold")
            if row == len(rows) - 1:
                ax.set_xlabel("Requests/s", fontsize=11)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=10, fontweight="bold")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        leg = fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 6),
            frameon=True,
            bbox_to_anchor=(0.5, 0.99),
            fontsize=SMALL_SUBPLOT_TITLE_FONTSIZE,
        )
        leg.get_frame().set_facecolor((1, 1, 1, 0.7))
        leg.get_frame().set_edgecolor("#aaaaaa")
        leg.get_frame().set_linewidth(1.2)

    metric_label = " / ".join(METRIC_LABEL[metric] for metric in metrics)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.00)
    plt.close(fig)
    print(f"[plot] wrote {output}")


def plot_summary_multi(
    experiments: List[Tuple[str, Summary]],
    output: Path,
    metrics: List[str],
    tasks: List[str],
    schedulers: List[str],
    title: Optional[str],
    dpi: int,
) -> None:
    if not tasks:
        raise ValueError("No matching tasks found in the summaries.")
    if not schedulers:
        raise ValueError("No matching schedulers found in the summaries.")

    _TASK_YLIM = {
        "gpqa":      (1, 2**6),
        "sharegpt":  (1, 2**6),
        "gsm8k":     (1, 2**3),
        "math":      (1, 2**3),
        "mmlu":      (1, 2**3),
        "humaneval": (1, 2**2),
        "ruler_4k":  (1, 2**2),
    }
    _GROUP_PRIORITY = [(1, 2**6), (1, 2**3), (1, 2**2)]
    tasks = sorted(tasks, key=lambda t: next(
        (i for i, g in enumerate(_GROUP_PRIORITY) if _TASK_YLIM.get(t) == g),
        len(_GROUP_PRIORITY),
    ))

    fig, axes = plt.subplots(
        len(experiments),
        len(tasks),
        figsize=(28, 5),
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.14, right=0.9, top=0.8, bottom=0.1, hspace=0.28, wspace=0.08)
    for exp_idx, (exp_label, exp_summary) in enumerate(experiments):
        for col, task in enumerate(tasks):
            ax = axes[exp_idx][col]

            all_rates = set()
            for scheduler in schedulers:
                xs, ys = collect_series(exp_summary, scheduler, task, "strict_all")
                if not xs:
                    continue
                all_rates.update(xs)
                style = presentation_style(scheduler)
                is_shiftserve = scheduler == "LST"
                ax.plot(
                    xs,
                    ys,
                    label=str(style.get("label", scheduler)),
                    color=str(style.get("color", "#888888")),
                    marker=str(style.get("marker", "o")),
                    linestyle="-",
                    linewidth=2.5,
                    markersize=8,
                    alpha=1 if is_shiftserve else 0.4,
                    markeredgewidth=1,
                    zorder=5 if is_shiftserve else 3,
                )

            ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.9, alpha=0.75, zorder=1)
            ax.set_ylim(0.1, 1.05)
            ax.set_yticks([0.1, 0.4, 0.7, 1.0])
            ax.grid(True, axis="both", color="#dddddd", linewidth=0.8, alpha=0.85)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=22)

            if all_rates:
                sorted_rates = sorted(all_rates)
                ax.set_xticks(sorted_rates)
                ax.set_xticklabels(nice_rate_labels(sorted_rates))
                xmin, xmax = min(sorted_rates), max(sorted_rates)
                pad = max((xmax - xmin) * 0.06, 0.08)
                ax.set_xlim(xmin - pad, xmax + pad)

            if exp_idx == 0:
                ax.set_title(task_label(task), fontsize=SUMMARY_SUBPLOT_TITLE_FONTSIZE, fontweight="bold")
            if exp_idx == len(experiments) - 1:
                ax.set_xlabel("Request rate(r/s)", fontsize=23)

    for exp_idx, (exp_label, _) in enumerate(experiments):
        pos = axes[exp_idx][0].get_position()
        y_center = (pos.y0 + pos.y1) / 2
        axes[exp_idx][0].set_ylabel(f"{exp_label}", fontsize=26, fontweight="bold", labelpad=25)
        add_right_ylabel(fig, "SLO Attain.", x=0.113, y=y_center)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        leg = fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 6),
            frameon=True,
            bbox_to_anchor=(0.5, 0.99),
            bbox_transform=fig.transFigure,
            fontsize=SUMMARY_SUBPLOT_TITLE_FONTSIZE,
            columnspacing=1.5,
            handlelength=2,
            handletextpad=LEGEND_HANDLE_TEXT_PAD,
            handleheight=LEGEND_HANDLEHEIGHT,
            borderpad=LEGEND_BORDERPAD,
            labelspacing=LEGEND_LABELSPACING,
            borderaxespad=0.0,
            markerscale=2.0,
        )
        for h in leg.legend_handles:
            h.set_linewidth(3.5)
            h.set_alpha(1.0)
        leg.get_frame().set_facecolor((1, 1, 1, 0.7))
        leg.get_frame().set_edgecolor("#aaaaaa")
        leg.get_frame().set_linewidth(1.2)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.00)
    plt.close(fig)
    print(f"[plot] wrote {output}")


def default_p99_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}_p99_latency{output.suffix}")


def has_any_field(summary: Summary, fields: Iterable[str]) -> bool:
    wanted = set(fields)
    for scheduler_data in summary.values():
        for rate_data in scheduler_data.values():
            for metrics in rate_data.values():
                if wanted.intersection(metrics.keys()):
                    return True
    return False


def plot_p99_summary(
    summary: Summary,
    output: Path,
    tasks: List[str],
    schedulers: List[str],
    title: Optional[str],
    dpi: int,
    log_scale: bool,
    log_base: int,
    normalize_baseline: Optional[str],
) -> None:
    if not has_any_field(summary, ["p99_ttfb_ms", "p99_tpob_ms"]):
        print("[plot] no p99_ttfb_ms/p99_tpob_ms fields found; skipped p99 latency plot")
        return

    if normalize_baseline and normalize_baseline not in summary:
        raise ValueError(f"Cannot normalize p99 latency: baseline scheduler {normalize_baseline!r} not found.")

    unit_label = f"normalized to {normalize_baseline}" if normalize_baseline else "ms"
    rows = [
        ("p99_ttfb_ms", f"p99 TTFB\n({unit_label})"),
        ("p99_tpob_ms", f"p99 TPOB\n({unit_label})"),
        ("p99_max_ms", f"max(p99 TTFB,\np99 TPOB)\n({unit_label})"),
    ]

    fig, axes = plt.subplots(
        len(rows),
        len(tasks),
        figsize=(28, 10),
        sharey=False,
        squeeze=False,
        constrained_layout=True,
    )

    for col, task in enumerate(tasks):
        for row, (field, row_label) in enumerate(rows):
            ax = axes[row][col]
            all_rates = set()

            for scheduler in schedulers:
                if normalize_baseline:
                    xs, ys = collect_normalized_p99_series(
                        summary, scheduler, task, field, normalize_baseline
                    )
                elif field == "p99_max_ms":
                    xs, ys = collect_p99_max_series(summary, scheduler, task)
                else:
                    xs, ys = collect_series(summary, scheduler, task, field)
                if not xs:
                    continue
                all_rates.update(xs)
                style = STYLE.get(scheduler, {})
                ax.plot(
                    xs,
                    ys,
                    label=style.get("label", scheduler),
                    color=style.get("color"),
                    marker=style.get("marker", "o"),
                    linestyle=style.get("linestyle", "-"),
                    linewidth=2.5,
                    markersize=8,
                )

            if log_scale:
                ax.set_yscale("log", base=log_base)
            if normalize_baseline:
                ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.1, alpha=0.75)
            ax.grid(True, axis="both", color="#dddddd", linewidth=0.8, alpha=0.85)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=12)

            if all_rates:
                sorted_rates = sorted(all_rates)
                ax.set_xticks(sorted_rates)
                ax.set_xticklabels(nice_rate_labels(sorted_rates))
                xmin, xmax = min(sorted_rates), max(sorted_rates)
                pad = max((xmax - xmin) * 0.07, 0.1)
                ax.set_xlim(xmin - pad, xmax + pad)

            if row == 0:
                ax.set_title(task.upper(), fontsize=SMALL_SUBPLOT_TITLE_FONTSIZE, fontweight="bold")
            if row == len(rows) - 1:
                ax.set_xlabel("Requests/s", fontsize=11)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=10, fontweight="bold")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        leg = fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 6),
            frameon=True,
            bbox_to_anchor=(0.5, 0.99),
            fontsize=SMALL_SUBPLOT_TITLE_FONTSIZE,
        )
        leg.get_frame().set_facecolor((1, 1, 1, 0.7))
        leg.get_frame().set_edgecolor("#aaaaaa")
        leg.get_frame().set_linewidth(1.2)

    if log_scale:
        scale_label = f"log2 scale" if log_base == 2 else f"log base {log_base} scale"
    else:
        scale_label = "linear scale"
    norm_label = f", normalized to {normalize_baseline}" if normalize_baseline else ""

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.00)
    plt.close(fig)
    print(f"[plot] wrote {output}")


def collect_worst_tail_values(
    summary: Summary,
    scheduler: str,
    task: str,
    normalize_baseline: Optional[str],
) -> List[float]:
    values: List[float] = []
    scheduler_rates = summary.get(scheduler, {})
    baseline_rates = summary.get(normalize_baseline, {}) if normalize_baseline else {}

    for rate, rate_data in scheduler_rates.items():
        metrics = rate_data.get(task)
        if not metrics:
            continue
        value = p99_value(metrics, "p99_max_ms")
        if value is None:
            continue

        if normalize_baseline:
            baseline_metrics = baseline_rates.get(rate, {}).get(task)
            if not baseline_metrics:
                continue
            baseline = p99_value(baseline_metrics, "p99_max_ms")
            if baseline is None or baseline <= 0:
                continue
            value = value / baseline

        values.append(value)

    return values


def plot_p99_summary_multi(
    experiments: List[Tuple[str, Summary]],
    output: Path,
    tasks: List[str],
    schedulers: List[str],
    title: Optional[str],
    dpi: int,
    log_scale: bool,
    log_base: int,
    normalize_baseline: Optional[str],
) -> None:
    if not any(has_any_field(summary, ["p99_ttfb_ms", "p99_tpob_ms"]) for _, summary in experiments):
        print("[plot] no p99_ttfb_ms/p99_tpob_ms fields found; skipped p99 latency plot")
        return

    if normalize_baseline:
        missing = [label for label, summary in experiments if normalize_baseline not in summary]
        if missing:
            labels = ", ".join(missing)
            raise ValueError(
                f"Cannot normalize p99 latency: baseline scheduler {normalize_baseline!r} "
                f"not found in {labels}."
            )

    box_schedulers = [
        scheduler for scheduler in schedulers
        if not normalize_baseline or scheduler != normalize_baseline
    ]
    if not box_schedulers:
        print("[plot] no non-baseline schedulers found; skipped p99 latency plot")
        return

    _TASK_YLIM = {
        "gpqa":      (1, 2**6),
        "sharegpt":  (1, 2**6),
        "gsm8k":     (1, 2**3),
        "math":      (1, 2**3),
        "mmlu":      (1, 2**3),
        "humaneval": (1, 2**2),
        "ruler_4k":  (1, 2**2),
    }
    _GROUP_PRIORITY = [(1, 2**6), (1, 2**3), (1, 2**2)]
    _BASELINE_AXIS_FRACTION = 0.20
    tasks = sorted(tasks, key=lambda t: next(
        (i for i, g in enumerate(_GROUP_PRIORITY) if _TASK_YLIM.get(t) == g),
        len(_GROUP_PRIORITY),
    ))

    fig, axes = plt.subplots(
        len(experiments),
        len(tasks),
        figsize=(28, 5.4),
        sharey=False,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.14, right=0.976, top=0.8, bottom=0.1, hspace=0.15, wspace=0.14)

    for exp_idx, (exp_label, exp_summary) in enumerate(experiments):
        for col, task in enumerate(tasks):
            ax = axes[exp_idx][col]
            plot_data: List[List[float]] = []
            plot_positions: List[int] = []
            tick_labels: List[str] = []
            box_colors: List[str] = []

            for pos, scheduler in enumerate(box_schedulers, start=1):
                values = collect_worst_tail_values(
                    exp_summary,
                    scheduler,
                    task,
                    normalize_baseline,
                )
                if not values:
                    continue
                style = presentation_style(scheduler)
                plot_data.append(values)
                plot_positions.append(pos)
                tick_labels.append(str(style.get("short") or style.get("label", scheduler)))
                box_colors.append(str(style.get("color", "#888888")))

            if plot_data:
                bp = ax.boxplot(
                    plot_data,
                    positions=plot_positions,
                    widths=0.55,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "#222222", "linewidth": 1.0},
                    whiskerprops={"color": "#555555", "linewidth": 0.8},
                    capprops={"color": "#555555", "linewidth": 0.8},
                )
                for patch, color in zip(bp["boxes"], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.62)
                    patch.set_edgecolor("#333333")
                    patch.set_linewidth(0.8)

                for pos, values in zip(plot_positions, plot_data):
                    mean_val = sum(values) / len(values)
                    ax.scatter(pos, mean_val, marker="D", s=20, color="black", zorder=5)

                all_values = [value for values in plot_data for value in values]
                if log_scale:
                    ax.set_yscale("log", base=log_base)
                if normalize_baseline:
                    ax.axhline(1.0, color="#E41A1C", linestyle="--", linewidth=2.4, alpha=0.9)
                if all_values:
                    if log_scale:
                        ymin_auto = max(min(all_values + ([1.0] if normalize_baseline else [])) * 0.82, 1e-3)
                        ymax_auto = max(max(all_values + ([1.0] if normalize_baseline else [])) * 1.22, 1.12)
                        if normalize_baseline:
                            upper_exp = max(math.log(ymax_auto, log_base), 1e-9)
                            lower_exp_required = math.log(ymin_auto, log_base)
                            if lower_exp_required < 0:
                                upper_exp = max(
                                    upper_exp,
                                    -((1.0 - _BASELINE_AXIS_FRACTION) / _BASELINE_AXIS_FRACTION) * lower_exp_required,
                                )
                            lower_exp = -(_BASELINE_AXIS_FRACTION / (1.0 - _BASELINE_AXIS_FRACTION)) * upper_exp
                            ymin = log_base ** lower_exp
                            ymax = log_base ** upper_exp
                        else:
                            ymin = ymin_auto
                            ymax = ymax_auto
                    else:
                        ymin = min(0.0, min(all_values) * 0.92)
                        ymax = max(all_values + ([1.0] if normalize_baseline else [])) * 1.15
                    ax.set_ylim(ymin, ymax)

            ax.set_xlim(0.5, len(box_schedulers) + 0.5)
            ax.set_xticks(plot_positions if plot_positions else range(1, len(box_schedulers) + 1))
            if exp_idx == len(experiments) - 1:
                ax.set_xticklabels(tick_labels if tick_labels else [
                    str(presentation_style(s).get("short") or presentation_style(s).get("label", s))
                    for s in box_schedulers
                ])
            else:
                ax.set_xticklabels([])
            ax.grid(True, axis="y", color="#dddddd", linewidth=0.8, alpha=0.85)
            ax.grid(True, axis="x", color="#eeeeee", linewidth=0.55, alpha=0.45)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=26)
            if log_scale:
                from matplotlib.ticker import FuncFormatter
                if normalize_baseline:
                    ymin, ymax = ax.get_ylim()
                    yticks = [tick for tick in ax.get_yticks() if ymin <= tick <= ymax]
                    if not any(math.isclose(tick, 1.0, rel_tol=0.0, abs_tol=1e-12) for tick in yticks):
                        ax.set_yticks(sorted(yticks + [1.0]))
                ax.yaxis.set_major_formatter(FuncFormatter(
                    lambda y, _: str(int(round(math.log2(y)))) if y > 0 else ""
                ))

            if exp_idx == 0:
                ax.set_title(task_label(task), fontsize=SUMMARY_SUBPLOT_TITLE_FONTSIZE, fontweight="bold")

    for exp_idx, (exp_label, _) in enumerate(experiments):
        pos = axes[exp_idx][0].get_position()
        y_center = (pos.y0 + pos.y1) / 2
        axes[exp_idx][0].set_ylabel(f"{exp_label}", fontsize=26, fontweight="bold",labelpad=25)
        add_right_ylabel(fig, "Latency"+"($2^y$)", x=0.123, y=y_center)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(
            facecolor=str(presentation_style(scheduler).get("color", "#888888")),
            edgecolor="#333333",
            alpha=0.62,
            label=(
                f"{presentation_style(scheduler).get('short')}: {presentation_style(scheduler).get('label', scheduler)}"
                if presentation_style(scheduler).get("short")
                else str(presentation_style(scheduler).get("label", scheduler))
            ),
        )
        for scheduler in box_schedulers
    ]
    if normalize_baseline:
        baseline_label = str(presentation_style(normalize_baseline).get("label", normalize_baseline))
        legend_handles.append(
            Line2D([0], [0], color="#E41A1C", linestyle="--", linewidth=2.4, label=f"{baseline_label}")
        )

    leg = fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=min(len(legend_handles), 6),
        frameon=True,
        bbox_to_anchor=(0.55, 0.98),
        bbox_transform=fig.transFigure,
        fontsize=SUMMARY_SUBPLOT_TITLE_FONTSIZE,
        columnspacing=1.5,
        handlelength=2,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        handleheight=LEGEND_HANDLEHEIGHT,
        borderpad=LEGEND_BORDERPAD,
        labelspacing=LEGEND_LABELSPACING,
        borderaxespad=0.0,
    )
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    leg.get_frame().set_facecolor((1, 1, 1, 0.7))
    leg.get_frame().set_edgecolor("#aaaaaa")
    leg.get_frame().set_linewidth(1.2)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.00)
    plt.close(fig)
    print(f"[plot] wrote {output}")


def default_bar_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}_bar_p99{output.suffix}")


def default_scatter_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}_scatter{output.suffix}")


def _read_request_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _read_scatter_request_records(run_dir: Path, task: str) -> List[Dict]:
    for filename in (f"request_latency_{task}.jsonl", f"dlm_request_latency_{task}.jsonl"):
        records = _read_request_jsonl(run_dir / filename)
        if records:
            return records
    return []


def _lighten_color(hex_color: str, amount: float = 0.55) -> str:
    """Blend hex_color toward white by `amount` (0=original, 1=white)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02X}{g:02X}{b:02X}"


def _split_by_slo(
    xs: List, ys: List, slo: Optional[Dict]
) -> Tuple[List, List, List, List]:
    """Split (x, y) lists into (met_x, met_y, viol_x, viol_y) using SLO thresholds."""
    if not slo:
        return xs, ys, [], []
    ttfb_lim = slo.get("ttfb_ms")
    tpob_lim = slo.get("tpob_ms")
    met_x, met_y, viol_x, viol_y = [], [], [], []
    for x, y in zip(xs, ys):
        ttfb_ok = (ttfb_lim is None) or (x is not None and x <= ttfb_lim)
        tpob_ok = (tpob_lim is None) or (y is None) or (y <= tpob_lim)
        if ttfb_ok and tpob_ok:
            met_x.append(x); met_y.append(y)
        else:
            viol_x.append(x); viol_y.append(y)
    return met_x, met_y, viol_x, viol_y


def _split_records(records: List[Dict]) -> Tuple[List, List, List, List, List, List]:
    strict_x, strict_y, release_x, release_y, other_x, other_y = [], [], [], [], [], []
    for r in records:
        if "ttfb_ms" not in r or "tpob_ms" not in r:
            continue
        slo_type = r.get("slo_type")
        if slo_type == "strict":
            strict_x.append(r["ttfb_ms"]); strict_y.append(r["tpob_ms"])
        elif slo_type == "release":
            release_x.append(r["ttfb_ms"]); release_y.append(r["tpob_ms"])
        else:
            other_x.append(r["ttfb_ms"]); other_y.append(r["tpob_ms"])
    return strict_x, strict_y, release_x, release_y, other_x, other_y


def _draw_slo_lines(ax, has_strict: bool, has_release: bool, has_other: bool, slo_strict, slo_relaxed, single_mode: bool = False) -> None:
    # slo_type 없는 레코드(other)는 단독일 때 반대 tier가 없으면 해당 SLO 선을 그림
    draw_strict = has_strict or (has_other and not has_release)
    draw_release = has_release or (has_other and not has_strict)
    if single_mode:
        slo = slo_strict if (draw_strict and slo_strict) else (slo_relaxed if (draw_release and slo_relaxed) else None)
        if slo:
            ax.axvline(slo["ttfb_ms"], color="#111111", linestyle="--", linewidth=2.8, alpha=1)
            ax.axhline(slo["tpob_ms"], color="#111111", linestyle=":",  linewidth=2.8, alpha=1)
    else:
        if draw_strict and slo_strict:
            ax.axvline(slo_strict["ttfb_ms"], color="#C44E52", linestyle="--", linewidth=2.8, alpha=1)
            ax.axhline(slo_strict["tpob_ms"], color="#C44E52", linestyle="--", linewidth=2.8, alpha=1)
        if draw_release and slo_relaxed:
            ax.axvline(slo_relaxed["ttfb_ms"], color="#4472C4", linestyle="--", linewidth=2.8, alpha=1)
            ax.axhline(slo_relaxed["tpob_ms"], color="#4472C4", linestyle="--", linewidth=2.8, alpha=1)


def _load_scatter_slo_from_summaries(
    latency_root: Path,
    task: str,
    rate: str,
    schedulers: List[str],
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Recover strict/release SLOs from dlm_benchmark.py summary JSON files."""
    for scheduler in schedulers:
        sched_dir = latency_root / f"scheduler_{scheduler}" / f"request_rate_{rate}" / task
        for summary_path in sorted(sched_dir.glob("summary_*.json")):
            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

            task_slo = data.get("slo_data", {}).get(task, {})
            strict_ttfb = task_slo.get("strict_ttfb_slo_ms")
            strict_tpob = task_slo.get("strict_tpob_slo_ms")
            release_ttfb = task_slo.get("release_ttfb_slo_ms")
            release_tpob = task_slo.get("release_tpob_slo_ms")
            if strict_ttfb is None or strict_tpob is None:
                continue

            strict = {"ttfb_ms": float(strict_ttfb), "tpob_ms": float(strict_tpob)}
            relaxed = None
            if release_ttfb is not None and release_tpob is not None:
                relaxed = {"ttfb_ms": float(release_ttfb), "tpob_ms": float(release_tpob)}
            return strict, relaxed

    return None, None


def plot_scatter_ttfb_tpob(
    latency_root: Path,
    output: Path,
    task: str,
    rate: str,
    schedulers: List[str],
    slo_config_path: Optional[Path],
    title: Optional[str],
    dpi: int,
    no_combined: bool = False,
) -> None:
    slo_strict: Optional[Dict] = None
    slo_relaxed: Optional[Dict] = None
    cfg_path = slo_config_path or (latency_root / "slo_config.json")
    if cfg_path.exists():
        with cfg_path.open() as f:
            task_slo = json.load(f).get(task, {})
        slo_strict = task_slo.get("strict")
        slo_relaxed = task_slo.get("relaxed")
    if not slo_strict:
        inferred_strict, inferred_relaxed = _load_scatter_slo_from_summaries(
            latency_root=latency_root,
            task=task,
            rate=rate,
            schedulers=schedulers,
        )
        slo_strict = inferred_strict
        slo_relaxed = inferred_relaxed

    # Load and split records per scheduler
    sched_split: List[Tuple[str, str, List, List, List, List, List, List]] = []
    for scheduler in schedulers:
        run_dir = latency_root / f"scheduler_{scheduler}" / f"request_rate_{rate}" / task
        records = _read_scatter_request_records(run_dir, task)
        if not records:
            continue
        label = presentation_style(scheduler).get("label", scheduler)
        color = str(presentation_style(scheduler).get("color", "#888888"))
        sx, sy, rx, ry, ox, oy = _split_records(records)
        sched_split.append((label, color, sx, sy, rx, ry, ox, oy))

    if not sched_split:
        print(f"[plot] no latency records for task={task!r} rate={rate!r}; skipped scatter")
        return

    has_any_strict = any(s[2] for s in sched_split)
    has_any_release = any(s[4] for s in sched_split)
    single_type = has_any_strict != has_any_release  # all-strict or all-release
    # When both SLO tiers are present, color by tier (strict=red, release=blue)
    # instead of by scheduler so the tier split is immediately visible.
    color_by_tier = has_any_strict and has_any_release
    _STRICT_COLOR  = "#C44E52"  # matches strict SLO line color
    _RELEASE_COLOR = "#4472C4"  # matches release SLO line color

    # ── Per-scheduler subplots ─────────────────────────────────────────────────
    n = len(sched_split)
    fig, axes = plt.subplots(1, n, figsize=(20, 3.2), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.74, bottom=0.1, wspace=0.16)

    from matplotlib.ticker import FuncFormatter
    _ms_to_s = FuncFormatter(lambda v, _: f"{int(round(v / 1000))}")

    for col, (label, sched_color, sx, sy, rx, ry, ox, oy) in enumerate(sched_split):
        ax = axes[0][col]
        dot_s = _STRICT_COLOR  if color_by_tier else sched_color
        dot_r = _RELEASE_COLOR if color_by_tier else sched_color
        dot_o = sched_color
        if sx:
            s_met_x, s_met_y, s_viol_x, s_viol_y = _split_by_slo(sx, sy, slo_strict)
            if s_met_x:
                ax.scatter(s_met_x, s_met_y, s=5, alpha=0.55, color=_lighten_color(dot_s), label="Strict (Attainment)", rasterized=True)
            if s_viol_x:
                ax.scatter(s_viol_x, s_viol_y, s=5, alpha=0.95, color=dot_s, label="Strict (Violated)", rasterized=True)
        if rx:
            r_met_x, r_met_y, r_viol_x, r_viol_y = _split_by_slo(rx, ry, slo_relaxed)
            if r_met_x:
                ax.scatter(r_met_x, r_met_y, s=5, alpha=0.55, color=_lighten_color(dot_r), label="Release (Attainment)", rasterized=True)
            if r_viol_x:
                ax.scatter(r_viol_x, r_viol_y, s=5, alpha=0.95, color=dot_r, label="Release (Violated)", rasterized=True)
        if ox:
            ax.scatter(ox, oy, s=5, alpha=0.6, color=dot_o, label="other", rasterized=True)
        _draw_slo_lines(ax, bool(sx), bool(rx), bool(ox), slo_strict, slo_relaxed, single_mode=single_type)
        ax.xaxis.set_major_formatter(_ms_to_s)
        ax.yaxis.set_major_formatter(_ms_to_s)
        ax.set_xlabel("TTFB (s)", fontsize=20)
        ax.set_title(label, fontsize=SCATTER_SUBPLOT_TITLE_FONTSIZE, fontweight="bold")
        if col == 0:
            ax.set_ylabel("TPOB (s)", fontsize=20)
        ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=16)

    # shared legend at top
    from matplotlib.lines import Line2D
    handles, labels_leg = axes[0][0].get_legend_handles_labels()
    if single_type:
        handles.append(Line2D([0], [0], color="#111111", linestyle="--", linewidth=3.0))
        labels_leg.append("strict TTFB")
        handles.append(Line2D([0], [0], color="#111111", linestyle=":", linewidth=3.0))
        labels_leg.append("strict TPOB")
    else:
        if slo_strict:
            handles.append(Line2D([0], [0], color="#A82432", linestyle="--", linewidth=3.0))
            labels_leg.append("strict SLO")
        if slo_relaxed:
            handles.append(Line2D([0], [0], color="#4472C4", linestyle="--", linewidth=2.2))
            labels_leg.append("release SLO")
    leg = fig.legend(handles, labels_leg, loc="upper center", ncol=len(handles),
                     fontsize=SCATTER_SUBPLOT_TITLE_FONTSIZE, frameon=True, markerscale=4,
                     bbox_to_anchor=(0.52, 0.99), bbox_transform=fig.transFigure,
                     handletextpad=LEGEND_HANDLE_TEXT_PAD,
                     handlelength=1.2,
                     handleheight=LEGEND_HANDLEHEIGHT,
                     borderpad=LEGEND_BORDERPAD,
                     labelspacing=LEGEND_LABELSPACING,
                     borderaxespad=0.0,
                     columnspacing=0.5)
    for handle, label in zip(leg.legend_handles, labels_leg):
        if label.startswith("strict"):
            handle.set_alpha(1.0)
            if hasattr(handle, "set_linewidth"):
                handle.set_linewidth(3.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.00)
    plt.close(fig)
    print(f"[plot] wrote {output}")

    # ── Combined figure ────────────────────────────────────────────────────────
    # color_by_tier: dots colored by SLO tier (strict=red, release=blue),
    #                marker shape distinguishes scheduler.
    # single_type:   dots colored by scheduler (existing behaviour).
    if no_combined or (not single_type and not color_by_tier):
        return

    has_any_other = any(s[6] for s in sched_split)
    fig2, ax2 = plt.subplots(figsize=(6.0, 5.0))

    if color_by_tier:
        _MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
        for i, (label, sched_color, sx, sy, rx, ry, ox, oy) in enumerate(sched_split):
            mkr = _MARKERS[i % len(_MARKERS)]
            if sx:
                s_met_x, s_met_y, s_viol_x, s_viol_y = _split_by_slo(sx, sy, slo_strict)
                if s_met_x:
                    ax2.scatter(s_met_x, s_met_y, s=10, alpha=0.55, color=_lighten_color(_STRICT_COLOR),
                                marker=mkr, label=f"{label} strict (met)", rasterized=True)
                if s_viol_x:
                    ax2.scatter(s_viol_x, s_viol_y, s=10, alpha=0.95,
                                color=_STRICT_COLOR, marker=mkr,
                                label=f"{label} strict (viol)", rasterized=True)
            if rx:
                r_met_x, r_met_y, r_viol_x, r_viol_y = _split_by_slo(rx, ry, slo_relaxed)
                if r_met_x:
                    ax2.scatter(r_met_x, r_met_y, s=10, alpha=0.55, color=_lighten_color(_RELEASE_COLOR),
                                marker=mkr, label=f"{label} release (met)", rasterized=True)
                if r_viol_x:
                    ax2.scatter(r_viol_x, r_viol_y, s=10, alpha=0.95,
                                color=_RELEASE_COLOR, marker=mkr,
                                label=f"{label} release (viol)", rasterized=True)
            if ox:
                ax2.scatter(ox, oy, s=10, alpha=0.6, color=sched_color,
                            marker=mkr, label=label, rasterized=True)
        slo_label = "strict+release"
    else:
        slo_for_combined = slo_strict if has_any_strict else slo_relaxed
        for label, sched_color, sx, sy, rx, ry, ox, oy in sched_split:
            xs = sx if sx else (rx if rx else ox)
            ys = sy if sy else (ry if ry else oy)
            if not xs:
                continue
            met_x, met_y, viol_x, viol_y = _split_by_slo(xs, ys, slo_for_combined)
            if met_x:
                ax2.scatter(met_x, met_y, s=5, alpha=0.55, color=_lighten_color(sched_color), label=label, rasterized=True)
            if viol_x:
                ax2.scatter(viol_x, viol_y, s=5, alpha=0.95, color=sched_color,
                            label=f"{label} (violated)", rasterized=True)
        slo_label = "strict" if has_any_strict else "relaxed"

    _draw_slo_lines(ax2, has_any_strict, has_any_release, has_any_other, slo_strict, slo_relaxed)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v / 1000))}"))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v / 1000))}"))
    ax2.set_xlabel("TTFB (s)", fontsize=16)
    ax2.set_ylabel("TPOB (s)", fontsize=16)
    ax2.legend(fontsize=COMBINED_SCATTER_TITLE_FONTSIZE, frameon=False, markerscale=2)
    ax2.grid(True, color="#dddddd", linewidth=0.8, alpha=0.85)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="both", labelsize=13)
    ax2.set_title(
        title or f"TTFB vs TPOB all schedulers ({slo_label}) — {task.upper()} @ {rate} req/s",
        fontsize=COMBINED_SCATTER_TITLE_FONTSIZE, fontweight="bold",
    )
    fig2.tight_layout()
    combined_output = output.with_name(f"{output.stem}_combined{output.suffix}")
    fig2.savefig(combined_output, dpi=dpi, bbox_inches="tight", pad_inches=0.00)
    plt.close(fig2)
    print(f"[plot] wrote {combined_output}")


def default_sr_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}_strict_release{output.suffix}")


def plot_strict_release_bar(
    summary: Summary,
    output: Path,
    task: str,
    rate: str,
    schedulers: List[str],
    title: Optional[str],
    dpi: int,
) -> None:
    labels, strict_vals, relaxed_vals = [], [], []
    for scheduler in schedulers:
        metrics = summary.get(scheduler, {}).get(rate, {}).get(task)
        if not metrics:
            continue
        s_all = metrics.get("strict_all")
        r_all = metrics.get("relaxed_all")
        if s_all is None or r_all is None:
            continue
        labels.append(STYLE.get(scheduler, {}).get("label", scheduler))
        strict_vals.append(float(s_all))
        relaxed_vals.append(float(r_all))

    if not labels:
        print(f"[plot] no strict/relaxed data for task={task!r} rate={rate!r}; skipped strict-release chart")
        return

    width = 0.35
    xs = list(range(len(labels)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(4.5, 1.6 * len(labels)) * 2, 4.8))

    # Panel 1: strict vs relaxed attainment per scheduler
    ax1.bar([i - width / 2 for i in xs], strict_vals, width, label="Strict", color="#C44E52")
    ax1.bar([i + width / 2 for i in xs], relaxed_vals, width, label="Relaxed", color="#4472C4")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("SLO Attainment", fontsize=11)
    ax1.set_ylim(-0.03, 1.05)
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.grid(True, axis="y", color="#dddddd", linewidth=0.8, alpha=0.85)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=10, frameon=False)
    ax1.set_title("Strict vs Relaxed Attainment", fontsize=11, fontweight="bold")

    # Panel 2: gap (relaxed - strict) — smaller is better discrimination
    gaps = [r - s for r, s in zip(relaxed_vals, strict_vals)]
    ax2.bar(xs, gaps, color="#8172B2")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Gap  (Relaxed − Strict)", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.grid(True, axis="y", color="#dddddd", linewidth=0.8, alpha=0.85)
    ax2.set_axisbelow(True)
    ax2.set_title("Attainment Gap  (↓ = better tier discrimination)", fontsize=11, fontweight="bold")

    fig.suptitle(
        title or f"Strict vs Relaxed SLO Discrimination — {task.upper()} @ {rate} req/s",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.00)
    plt.close(fig)
    print(f"[plot] wrote {output}")


def plot_bar_p99(
    summary: Summary,
    output: Path,
    task: str,
    rate: str,
    schedulers: List[str],
    slo_ttfb_ms: Optional[float],
    slo_tpob_ms: Optional[float],
    title: Optional[str],
    dpi: int,
) -> None:
    labels, ttfb_vals, tpob_vals = [], [], []
    for scheduler in schedulers:
        metrics = summary.get(scheduler, {}).get(rate, {}).get(task)
        if not metrics:
            continue
        ttfb = metrics.get("p99_ttfb_ms")
        tpob = metrics.get("p99_tpob_ms")
        if ttfb is None or tpob is None:
            continue
        labels.append(STYLE.get(scheduler, {}).get("label", scheduler))
        ttfb_vals.append(float(ttfb))
        tpob_vals.append(float(tpob))

    if not labels:
        print(f"[plot] no p99 data for task={task!r} rate={rate!r}; skipped bar chart")
        return

    width = 0.35
    xs = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(5.0, 1.8 * len(labels)), 5.0))
    ax.bar([i - width / 2 for i in xs], ttfb_vals, width, label="TTFB p99", color="#E8A838")
    ax.bar([i + width / 2 for i in xs], tpob_vals, width, label="TPOB p99", color="#4472C4")

    if slo_ttfb_ms is not None:
        ax.axhline(slo_ttfb_ms, color="#E8A838", linestyle="--", linewidth=1.5,
                   label=f"TTFB SLO ({slo_ttfb_ms:.0f} ms)")
    if slo_tpob_ms is not None:
        ax.axhline(slo_tpob_ms, color="#4472C4", linestyle="--", linewidth=1.5,
                   label=f"TPOB SLO ({slo_tpob_ms:.0f} ms)")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(fontsize=10, frameon=False)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.00)
    plt.close(fig)
    print(f"[plot] wrote {output}")


def default_breakdown_bar_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}_breakdown_bar{output.suffix}")


def default_block_unmask_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}_block_unmask_steps{output.suffix}")


def percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("percentile() requires at least one value")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _load_block_unmask_points_from_bellman_log(
    path: Path,
) -> Tuple[List[Tuple[int, int]], Optional[int]]:
    points: List[Tuple[int, int]] = []
    block_size: Optional[int] = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            traj = rec.get("traj")
            if not isinstance(traj, list) or not traj:
                continue
            try:
                int_traj = [int(v) for v in traj]
            except (TypeError, ValueError):
                continue
            total_steps = len(int_traj) - 1  # last element is 0 (done)
            if total_steps <= 0:
                continue
            for i, masked in enumerate(int_traj[:-1]):  # exclude final 0
                if masked <= 0:
                    continue
                remaining = total_steps - i
                points.append((masked, remaining))
            table = rec.get("table")
            if block_size is None and isinstance(table, list) and table:
                block_size = len(table) - 1

    return points, block_size


def _summary_request_records(data: Mapping, task: str) -> List[Mapping]:
    latency_data = data.get("latency_data")
    if isinstance(latency_data, Mapping):
        task_data = latency_data.get(task)
        if isinstance(task_data, Mapping):
            records = task_data.get("request")
            if isinstance(records, list):
                return [r for r in records if isinstance(r, Mapping)]

        # If the caller omitted or renamed the task, use the only task with
        # request records. This keeps the helper useful for single-task runs.
        candidates = []
        for maybe_task_data in latency_data.values():
            if isinstance(maybe_task_data, Mapping):
                records = maybe_task_data.get("request")
                if isinstance(records, list):
                    candidates.append(records)
        if len(candidates) == 1:
            return [r for r in candidates[0] if isinstance(r, Mapping)]

    records = data.get("request")
    if isinstance(records, list):
        return [r for r in records if isinstance(r, Mapping)]
    return []


def _first_block_mask_count(input_len: Optional[float], block_size: int) -> int:
    if input_len is None:
        return block_size
    try:
        partial = int(input_len) % block_size
    except (TypeError, ValueError):
        return block_size
    return block_size if partial == 0 else block_size - partial


def _load_block_unmask_points_from_summary(
    path: Path,
    task: str,
) -> Tuple[List[Tuple[int, int]], Optional[int], Optional[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records = _summary_request_records(data, task)
    points: List[Tuple[int, int]] = []
    inferred_block_size: Optional[int] = None

    for rec in records:
        steps_list = rec.get("block_steps_list")
        if not isinstance(steps_list, list) or not steps_list:
            continue

        block_size_raw = rec.get("block_size", inferred_block_size or 32)
        try:
            block_size = int(block_size_raw)
        except (TypeError, ValueError):
            block_size = inferred_block_size or 32
        if block_size <= 0:
            continue
        inferred_block_size = block_size

        first_masks = _first_block_mask_count(rec.get("input_len"), block_size)
        for idx, steps in enumerate(steps_list):
            try:
                forwards_to_finish = int(steps)
            except (TypeError, ValueError):
                continue
            if forwards_to_finish <= 0:
                continue
            masked_at_start = first_masks if idx == 0 else block_size
            if masked_at_start <= 0:
                continue
            points.append((masked_at_start, forwards_to_finish))

    model = data.get("model")
    return points, inferred_block_size, str(model) if model else None


def plot_block_unmask_steps(
    summary_path: Optional[Path],
    bellman_log_path: Optional[Path],
    output: Path,
    task: str,
    model_label: Optional[str],
    dpi: int,
) -> None:
    points: List[Tuple[int, int]] = []
    block_size: Optional[int] = None
    source_model: Optional[str] = None

    if bellman_log_path is not None and bellman_log_path.exists():
        points, block_size = _load_block_unmask_points_from_bellman_log(bellman_log_path)
    elif summary_path is not None and summary_path.exists():
        points, block_size, source_model = _load_block_unmask_points_from_summary(summary_path, task)
    else:
        print("[plot] no block-unmask source found; skipped block-unmask plot")
        return

    if not points:
        print("[plot] no block_steps_list/traj data found; skipped block-unmask plot")
        return

    bs = block_size or max(p[0] for p in points)
    points = [(m, s) for m, s in points if m != bs]
    if not points:
        print("[plot] no data after excluding block_size; skipped block-unmask plot")
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    max_x = max(xs)

    grouped: Dict[int, List[int]] = {}
    for masked_count, steps in points:
        grouped.setdefault(masked_count, []).append(steps)
    grouped_x = sorted(grouped)
    means = [statistics.mean(grouped[x]) for x in grouped_x]
    medians = [statistics.median(grouped[x]) for x in grouped_x]
    lower_band = [percentile([float(v) for v in grouped[x]], 10) for x in grouped_x]
    upper_band = [percentile([float(v) for v in grouped[x]], 90) for x in grouped_x]

    model_text = model_label or source_model or "DLM"
    fig, ax = plt.subplots(figsize=(14.5, 5), constrained_layout=True)
    ax.fill_between(
        grouped_x,
        lower_band,
        upper_band,
        color="#8DA9D8",
        alpha=0.22,
        linewidth=0,
        label="10-90% Range",
        zorder=1,
    )
    ax.scatter(
        xs,
        ys,
        s=75,
        color="#4C72B0",
        alpha=0.45,
        edgecolors="none",
        label="Per-Block",
        rasterized=True,
        zorder=2,
    )
    ax.plot(
        grouped_x,
        means,
        color="#E41A1C",
        linestyle="--",
        marker="o",
        markersize=8,
        linewidth=3.0,
        label="Mean",
        zorder=4,
    )
    ax.plot(
        grouped_x,
        medians,
        color="#444444",
        linestyle=":",
        linewidth=2.5,
        label="Median",
        zorder=3,
    )

    #ax.set_title(
    #    f"Block Unmask Steps based on Masked Tokens",
    #    fontsize=30,
    #    pad=14,
    #)
    ax.set_xlabel("Masked Tokens", fontsize=36)
    ax.set_ylabel("Unmask Steps", fontsize=36)
    ax.set_xlim(0.4, max_x + 0.6)
    ax.set_xticks(list(range(0, max_x + 1, 8)))

    y_upper = max(ys)
    ax.set_ylim(0.5, y_upper + max(1.0, y_upper * 0.08))
    ax.set_yticks(list(range(0, int(y_upper) + 2, 8)))
    ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=30)
    leg = ax.legend(loc="upper left", 
                    bbox_to_anchor=(0.02, 0.98),
                    fontsize=30, 
                    frameon=True, 
                    labelspacing=0.02,
                    handlelength=0.9,
                    borderaxespad=0.2,
                    handletextpad=0.5,
                    handleheight=0.6,
                    markerscale=1.5,
                    borderpad=0.2,
                    )
    leg.get_frame().set_facecolor((1, 1, 1, 0.7))
    leg.get_frame().set_edgecolor("#aaaaaa")
    leg.get_frame().set_linewidth(1.2)
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)
        handle.set_linewidth(3.0)

    #stats_text = (
    #    f"n={len(ys)}  mean={statistics.mean(ys):.1f}  p95={percentile([float(y) for y in ys], 95):.1f}"
    #)
    #ax.text(
    #    0.98,
    #    0.97,
    #    stats_text,
    #    transform=ax.transAxes,
    #    ha="right",
    #    va="top",
    #    fontsize=20,
    #    color="#444444",
    #)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.00)
    plt.close(fig)
    print(f"[plot] wrote {output}")


def _breakdown_components(
    p99_total: float,
    mean_total: Optional[float],
    mean_parts: List[Optional[float]],
) -> List[float]:
    """Scale mean component proportions to sum to p99_total.

    If any component is missing, falls back to uniform split of the remainder.
    Returns a list of heights that sum exactly to p99_total.
    """
    if mean_total is None or mean_total <= 0:
        n = len(mean_parts)
        return [p99_total / n] * n

    scale = p99_total / mean_total
    parts = [(v or 0.0) * scale for v in mean_parts]

    # Clamp negatives and normalise to preserve total
    parts = [max(0.0, v) for v in parts]
    total_parts = sum(parts)
    if total_parts <= 0:
        n = len(parts)
        return [p99_total / n] * n

    factor = p99_total / total_parts
    return [v * factor for v in parts]


def plot_breakdown_bar(
    summary: Summary,
    output: Path,
    task: str,
    rate: str,
    schedulers: List[str],
    slo_ttfb_ms: Optional[float],
    slo_tpob_ms: Optional[float],
    title: Optional[str],
    dpi: int,
) -> None:
    def _float_metric(metrics: Mapping[str, float], key: str) -> Optional[float]:
        value = metrics.get(key)
        if value is None:
            return None
        return float(value)

    def _parts_for_ttfb(scheduler: str, metrics: Mapping[str, float]) -> List[Tuple[str, float]]:
        forward = _float_metric(metrics, "p95_ttfb_selected_forward_ms")
        if forward is None:
            forward = _float_metric(metrics, "p95_ideal_ttfb_ms") or 0.0
        sched_wait = _float_metric(metrics, "p95_ttfb_selected_sched_wait_ms")
        if sched_wait is None:
            sched_wait = _float_metric(metrics, "p95_sched_wait_ms") or 0.0
        pd_gap = _float_metric(metrics, "p95_ttfb_selected_first_unmask_gap_ms")
        if pd_gap is None:
            pd_gap = _float_metric(metrics, "p95_first_unmask_gap_ms") or 0.0
        decode_wait = _float_metric(metrics, "p95_ttfb_selected_first_block_decode_wait_ms") or 0.0
        other = _float_metric(metrics, "p95_ttfb_selected_other_ms") or 0.0

        # SOLA/PREFILL expose the first-block delay mostly as prefill-decode coupling.
        if scheduler in {"SOLA", "PREFILL"}:
            pd_gap += sched_wait
            sched_wait = 0.0

        return [
            ("forward", forward),
            ("prefill", sched_wait),
            ("pd", pd_gap),
            ("decode", decode_wait),
            ("other", other),
        ]

    def _parts_for_tpob(metrics: Mapping[str, float]) -> List[Tuple[str, float]]:
        forward = _float_metric(metrics, "p95_tpob_selected_forward_ms")
        if forward is None:
            forward = _float_metric(metrics, "p95_ideal_tpob_ms") or 0.0
        decode_wait = _float_metric(metrics, "p95_tpob_selected_decode_wait_ms")
        if decode_wait is None:
            decode_wait = _float_metric(metrics, "p95_decode_wait_ms") or 0.0
        other = _float_metric(metrics, "p95_tpob_selected_other_ms") or 0.0
        return [
            ("forward", forward),
            ("decode", decode_wait),
            ("other", other),
        ]

    preferred = ["TTFB", "LST", "SOLA", "DECODE", "FCFS", "PREFILL"]
    ordered_schedulers = [scheduler for scheduler in preferred if scheduler in schedulers]
    ordered_schedulers.extend([scheduler for scheduler in schedulers if scheduler not in ordered_schedulers])

    rows = []
    for scheduler in ordered_schedulers:
        metrics = summary.get(scheduler, {}).get(rate, {}).get(task)
        if not metrics:
            continue
        p95_ttfb = _float_metric(metrics, "p95_ttfb_selected_total_ms")
        if p95_ttfb is None:
            p95_ttfb = _float_metric(metrics, "p95_ttfb_ms")
        p95_tpob = _float_metric(metrics, "p95_tpob_selected_total_ms")
        if p95_tpob is None:
            p95_tpob = _float_metric(metrics, "p95_tpob_ms")
        if p95_ttfb is None or p95_tpob is None:
            continue

        rows.append({
            "scheduler": scheduler,
            "ttfb_total": p95_ttfb,
            "tpob_total": p95_tpob,
            "ttfb_parts": _parts_for_ttfb(scheduler, metrics),
            "tpob_parts": _parts_for_tpob(metrics),
        })

    if not rows:
        print(f"[plot] no breakdown data for task={task!r} rate={rate!r}; skipped breakdown bar")
        return

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import matplotlib.colors as mcolors

    comp_style = {
        "prefill": {"label": "Server  Queue", "color": "#ED9736", "hatch": None},
        "pd": {"label": "Prefill    Queue", "color": "#6AA982", "hatch": None},
        "decode": {"label": "Decode Queue", "color": "#8DA9D8", "hatch": None},
        "forward": {"label": "Forward", "color": "#9E9E9E", "hatch": None},
        "other": {"label": "Others", "color": "#9E9E9E", "hatch": "**"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    fig.subplots_adjust(left=0, right=1, top=0.77, bottom=0, wspace=0.01)
    ax_ttfb, ax_tpob = axes
    y_spacing = 0.4
    y_positions = [i * y_spacing for i in range(len(rows))]
    bar_height = 0.38
    x_max = max(max(row["ttfb_total"], row["tpob_total"]) for row in rows) * 1.15

    def _draw_background(ax, slo_value: Optional[float]) -> None:
        import matplotlib.ticker as ticker
        if slo_value is not None:
            ax.axvspan(0, slo_value, color="#5AA469", alpha=0.1, zorder=-10)
            ax.axvspan(slo_value, x_max*1.15, color="#FCE9D6", alpha=0.0, zorder=-10)
            ax.axvline(slo_value, color="#444444", linestyle="--", linewidth=1.6, zorder=3)
        ax.patch.set_alpha(0)
        ax.set_xlim(0, x_max*1.15)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}"))
        ax.grid(True, axis="x", color="#dddddd", linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=30)
        #ax.set_xlabel("Latency (s)", fontsize=30)

    def _draw_split_bar(ax, y: float, parts: List[Tuple[str, float]], slo_value: Optional[float]) -> None:
        left = 0.0
        for comp_key, value in parts:
            value = max(0.0, float(value))
            if value <= 0:
                continue
            right = left + value
            style = comp_style[comp_key]
            color = style["color"]
            hatch = style["hatch"] or ""
            if slo_value is not None and left < slo_value < right:
                ax.barh(
                    y,
                    slo_value - left,
                    left=left,
                    height=bar_height,
                    color=color,
                    alpha=0.60,
                    hatch=hatch,
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=2,
                )
                ax.barh(
                    y,
                    right - slo_value,
                    left=slo_value,
                    height=bar_height,
                    color=color,
                    alpha=1,
                    hatch=hatch,
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=2,
                )
            else:
                satisfied = slo_value is None or right <= slo_value
                ax.barh(
                    y,
                    value,
                    left=left,
                    height=bar_height,
                    color=color,
                    alpha=0.60 if satisfied else 1,
                    hatch=hatch,
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=2,
                )
            left = right

    def _annotate_total(ax, y: float, total: float, slo_value: Optional[float]) -> None:
        violated = slo_value is not None and total > slo_value
        if violated:
            delta_s = (total - slo_value) / 1000
            if delta_s < 0.1:
                return
            label = f"+{delta_s:.1f}s"
        else:
            #label = f"{total / 1000:.1f} s"
            label = ""
        ax.text(
            total,
            y,
            label,
            ha="left",
            va="center",
            fontsize=33,
            fontweight="bold" if violated else "normal",
            color="#003366",
            clip_on=False,
            #rotation=-20,
        )

    for ax, metric_key, total_key, slo_value, panel_title in [
        (ax_ttfb, "ttfb_parts", "ttfb_total", slo_ttfb_ms, "(a) TTFB Breakdown (s)"),
        (ax_tpob, "tpob_parts", "tpob_total", slo_tpob_ms, "(b) TPOB Breakdown (s)"),
    ]:
        _draw_background(ax, slo_value)

        for y, row in zip(y_positions, rows):
            _draw_split_bar(ax, y, row[metric_key], slo_value)
            _annotate_total(ax, y, float(row[total_key]), slo_value)
            ax.set_xlabel(panel_title, fontsize=44)
        #ax.set_title(panel_title, fontsize=30, pad=10)

    def _scheduler_axis_label(scheduler: str) -> str:
        style = presentation_style(scheduler)
        return str(style.get("short") or style.get("label", scheduler))

    def _scheduler_mapping_label(scheduler: str) -> str:
        style = presentation_style(scheduler)
        short = str(style.get("short") or style.get("label", scheduler))
        label = str(style.get("label", scheduler))
        return r"$\mathbf{" + short + r"}$:" + f"{label}"

    ax_ttfb.set_yticks(y_positions)
    ax_ttfb.set_yticklabels(
        [
            _scheduler_axis_label(str(row["scheduler"]))
            for row in rows
        ],
        fontsize=34,
        fontweight="bold",
    )
    ax_ttfb.invert_yaxis()

    region_handles = [
        Patch(facecolor=mcolors.to_rgba("#5AA469", 0.1), edgecolor="#111111", label="SLO Attainment"),
        #Patch(facecolor="white", edgecolor="#111111", alpha=1, label="SLO Violation"),
    ]
    comp_handles = [
        Patch(
            facecolor=style["color"],
            hatch=style["hatch"] or "",
            edgecolor="white",
            label=style["label"],
        )
        for key, style in comp_style.items()
        if key != "other"
    ]

    all_handles = region_handles + comp_handles
    leg = ax_tpob.legend(
        handles=all_handles,
        loc="lower right",
        bbox_to_anchor=(1.02, -0.03),
        ncol=1,
        frameon=True,
        fontsize=38,
        columnspacing=0.05,
        handletextpad=0.3,
        handlelength=1.0,
        labelspacing=0.03,
        borderpad=0.1,

    )
    leg.get_frame().set_facecolor((1, 1, 1, 0.85))
    leg.get_frame().set_edgecolor("#aaaaaa")
    leg.get_frame().set_linewidth(1.2)

    scheduler_mapping_handles = []
    seen_scheduler_labels = set()
    for row in rows:
        mapping_label = _scheduler_mapping_label(str(row["scheduler"]))
        if mapping_label in seen_scheduler_labels:
            continue
        seen_scheduler_labels.add(mapping_label)
        scheduler_mapping_handles.append(
            Line2D([], [], linestyle="none", label=mapping_label)
        )
    if scheduler_mapping_handles:
        map_leg = fig.legend(
            handles=scheduler_mapping_handles,
            loc="upper center",
            bbox_to_anchor=(0.49, 0.97),
            #ncol=len(scheduler_mapping_handles),
            ncol=3,
            frameon=True,
            fontsize=32,
            handlelength=0,
            handletextpad=0,
            columnspacing=2.1,
            borderpad=0.01,
            labelspacing=0.02,
        )
        map_leg.get_frame().set_facecolor((1, 1, 1, 0.85))
        map_leg.get_frame().set_edgecolor("#111111")
        map_leg.get_frame().set_linewidth(1.2)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.00)
    plt.close(fig)
    print(f"[plot] wrote {output}")


def default_user_experiment_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}_user_experiment{output.suffix}")


def collect_p95_pair_series(
    summary: Summary,
    scheduler: str,
    task: str,
) -> Tuple[List[float], List[float], List[float]]:
    rates: List[float] = []
    ttfb_vals: List[float] = []
    tpob_vals: List[float] = []

    for rate, rate_data in summary.get(scheduler, {}).items():
        metrics = rate_data.get(task)
        if not metrics:
            continue
        ttfb = metrics.get("p95_ttfb_ms")
        tpob = metrics.get("p95_tpob_ms")
        if ttfb is None or tpob is None:
            continue
        rates.append(rate_key(rate))
        ttfb_vals.append(float(ttfb))
        tpob_vals.append(float(tpob))

    triples = sorted(zip(rates, ttfb_vals, tpob_vals), key=lambda triple: triple[0])
    if not triples:
        return [], [], []
    return [t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples]


def _draw_user_trajectory(
    ax,
    summary: Summary,
    task: str,
    schedulers: List[str],
    slo_ttfb_ms: Optional[float],
    slo_tpob_ms: Optional[float],
) -> Tuple[List, List[str]]:
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D

    series: List[Tuple[str, List[float], List[float], List[float]]] = []
    all_rates: List[float] = []
    all_xs: List[float] = []
    all_ys: List[float] = []

    for scheduler in schedulers:
        rates, xs, ys = collect_p95_pair_series(summary, scheduler, task)
        if not xs:
            continue
        xs = [x / 1000.0 for x in xs]
        ys = [y / 1000.0 for y in ys]
        series.append((scheduler, rates, xs, ys))
        all_rates.extend(rates)
        all_xs.extend(xs)
        all_ys.extend(ys)

    if not series:
        raise ValueError(f"No p95 TTFB/TPOB data found for task={task!r}.")

    slo_ttfb_s = slo_ttfb_ms / 1000.0 if slo_ttfb_ms is not None else None
    slo_tpob_s = slo_tpob_ms / 1000.0 if slo_tpob_ms is not None else None
    max_x = max(all_xs + ([slo_ttfb_s] if slo_ttfb_s is not None else []))
    max_y = max(all_ys + ([slo_tpob_s] if slo_tpob_s is not None else []))
    x_max = max_x * 1.04
    y_max = max_y * 1.066
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.5, y_max)

    if slo_ttfb_s is not None and slo_tpob_s is not None:
        y_frac = min(max(slo_tpob_s / y_max, 0.0), 1.0)
        ax.axvspan(0, slo_ttfb_s, ymin=0, ymax=y_frac, facecolor="#EAF4E9", alpha=0, zorder=0)
        ax.axvspan(0, slo_ttfb_s, ymin=y_frac, ymax=1, facecolor="#EAF1FB", alpha=1, zorder=0)
        ax.axvspan(slo_ttfb_s, x_max, ymin=0, ymax=y_frac, facecolor="#FCE9D6", alpha=0.7, zorder=0)
        ax.axvspan(slo_ttfb_s, x_max, ymin=y_frac, ymax=1, facecolor="#F2F2F2", alpha=0.7, zorder=0)
        ax.axvline(slo_ttfb_s, color="#444444", linestyle="--", linewidth=3, alpha=1, zorder=2)
        ax.axhline(slo_tpob_s, color="#444444", linestyle="--", linewidth=3, alpha=1, zorder=2)

        ql_kw = dict(ha="center", va="center", fontsize=30, fontweight="bold", color="#111111", zorder=10)
        #ax.text(slo_ttfb_s * 0.5, slo_tpob_s * 0.5,
        #        "No Violation", **ql_kw)
        ax.text(slo_ttfb_s * 0.655, (slo_tpob_s + y_max) * 0.7,
                "Inter-Block\nViolation", **ql_kw)
        ax.text((slo_ttfb_s + x_max) * 0.63, slo_tpob_s * 0.8,
                "First-Block Violation", **ql_kw)
        ax.text((slo_ttfb_s + x_max) * 0.295, (slo_tpob_s + y_max) * 0.7,
                "Both\nViolation", **ql_kw)

    #if all_rates:
    #    sorted_rates = sorted(set(all_rates))
    #    start_label, end_label = nice_rate_labels([sorted_rates[0], sorted_rates[-1]])
    #    ax.text(
    #        0.8,
    #        0.8,
    #        f"↑ load: {start_label} → {end_label} req/s",
    #        transform=ax.transAxes,
    #        ha="left",
    #        va="top",
    #        fontsize=18,
    #        color="#444444",
    #    )

    # Offsets (points) for end-of-trajectory labels, tuned to avoid overlap
    label_offsets = {
        "TTFB":      (14,   0),
        "ShiftServe":(14,   0),
        "SOLA":      (14,   8),
    }
    # DECODE/FCFS/PREFILL은 공통 x 좌표에 center 정렬로 별도 처리
    _right_group = {"DECODE", "FCFS", "PREFILL","TTFB","SOLA","LST" }
    _right_y_offsets = {"DECODE": 3.2, "FCFS": 2.0, "PREFILL": 0.8}

    handles = []
    labels: List[str] = []

    def draw_scheduler_path(scheduler: str, xs: List[float], ys: List[float], color: str) -> None:
        if scheduler not in ("DECODE", "FCFS"):
            ax.plot(
                xs,
                ys,
                color=mcolors.to_rgba(color, 0.3),
                linewidth=60,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=3,
            )

        for x0, y0, x1, y1 in zip(xs, ys, xs[1:], ys[1:]):
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops={
                    "arrowstyle": "-" if scheduler in ("FCFS", "PREFILL") else "-|>",
                    "color": color,
                    "lw": 4,
                    "alpha": 1,
                    "shrinkA": 10,
                    "shrinkB": 16,
                    "mutation_scale": 32,
                },
                zorder=6,
            )

    for scheduler, rates, xs, ys in series:
        if scheduler in ("FCFS", "PREFILL"):
            style = PRESENTATION_STYLE.get(scheduler, STYLE.get(scheduler, {}))
            color = str(style.get("color", "#888888"))
            draw_scheduler_path(scheduler, xs, ys, color)

    for scheduler, rates, xs, ys in series:
        style = PRESENTATION_STYLE.get(scheduler, STYLE.get(scheduler, {}))
        color = str(style.get("color", "#888888"))
        line_alpha = 0.5 if scheduler in ("DECODE", "FCFS", "PREFILL", "SOLA", "TTFB") else 1.0
        line_color = mcolors.to_rgba(color, line_alpha)
        label = style.get("label", scheduler)
        label = "First Block Priority" if scheduler == "TTFB" else label
        ax.scatter(
            xs,
            ys,
            s=800,
            marker=style.get("marker", "o"),
            facecolor=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=8,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=line_color,
                marker=style.get("marker", "o"),
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=1.0,
                linestyle="none",
                linewidth=0,
                markersize=18,
                label=label,
            )
        )
        labels.append(label)

        if scheduler not in ("FCFS", "PREFILL"):
            draw_scheduler_path(scheduler, xs, ys, color)

        if scheduler in _right_group:
            continue  # 아래에서 공통 x로 일괄 처리

        dx, dy = label_offsets.get(scheduler, (10, 10))
        ax.annotate(
            label,
            xy=(xs[-1], ys[-1]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
            va="center" if dy == 0 else ("bottom" if dy > 0 else "top"),
            fontsize=25,
            color="#444444",
            fontweight="bold",
        )

    # DECODE/FCFS/PREFILL: 공통 anchor_x에 ha="center"로 수직 정렬
    _right_series = {s: (xs, ys) for s, _, xs, ys in series if s in _right_group}
    if _right_series:
        anchor_x = max(xs[-1] for xs, ys in _right_series.values()) + x_max * 0.01
        base_y = min(ys[-1] for xs, ys in _right_series.values())
        for sched in ("DECODE", "FCFS", "PREFILL"):
            if sched not in _right_series:
                continue
            style = PRESENTATION_STYLE.get(sched, STYLE.get(sched, {}))
            #ax.text(
            #    anchor_x,
            #    base_y + _right_y_offsets[sched],
            #    style.get("label", sched),
            #    ha="center",
            #    va="center",
            #    fontsize=18,
            #    color="#003366",
            #    fontweight="bold",
            #)

    ax.set_xlabel("First Block Delay (s)", fontsize=38)
    ax.set_ylabel("Inter Block Delay (s)", fontsize=38)
    ax.grid(True, axis="both", color="#dddddd", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    import numpy as np
    ax.set_xticks(np.arange(0, x_max + 1, 4))
    ax.set_yticks(np.arange(0, y_max + 1, 4))
    ax.tick_params(axis="both", labelsize=26)
    #ax.set_title(
    #    "User-Perceived Latency Trajectory",
    #    fontsize=35,
    #    pad=16,
    #)
    leg = ax.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        ncol=1,
        frameon=True,
        fontsize=32,
        handlelength=1.5,
        columnspacing=0.7,
        labelspacing=0.07,
        markerscale=1.3,
        handletextpad=0.2,
        borderpad=0.2,


    )
    leg.get_frame().set_facecolor((1, 1, 1, 0.7))
    leg.get_frame().set_edgecolor("#aaaaaa")
    leg.get_frame().set_linewidth(1.2)
    for lh in leg.legend_handles:
        lh.set_alpha(0.75)
    return handles, labels


def _draw_user_rate_panels(
    axes,
    summary: Summary,
    task: str,
    schedulers: List[str],
    slo_ttfb_ms: Optional[float],
    slo_tpob_ms: Optional[float],
) -> Tuple[List, List[str]]:
    ax_ttfb, ax_tpob = axes
    panels = [
        (ax_ttfb, "p95_ttfb_ms", "p95 Time to First Token (ms)\n(Response Speed)", slo_ttfb_ms),
        (ax_tpob, "p95_tpob_ms", "p95 Max Inter-Block Gap (ms)\n(Response Continuity)", slo_tpob_ms),
    ]

    for ax, field, ylabel, slo_val in panels:
        all_rates = set()
        for scheduler in schedulers:
            xs, ys = collect_series(summary, scheduler, task, field)
            if not xs:
                continue
            all_rates.update(xs)
            style = STYLE.get(scheduler, {})
            ax.plot(
                xs,
                ys,
                label=style.get("label", scheduler),
                color=style.get("color"),
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                linewidth=2.0,
                markersize=5.0,
            )

        if slo_val is not None:
            ax.axhline(
                slo_val,
                color="#444444",
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
            )
            ax.text(
                0.02,
                slo_val,
                f"SLO\n{slo_val:.0f}",
                transform=ax.get_yaxis_transform(),
                va="bottom",
                ha="left",
                fontsize=9,
                color="#444444",
            )

        if all_rates:
            sorted_rates = sorted(all_rates)
            ax.set_xticks(sorted_rates)
            ax.set_xticklabels(nice_rate_labels(sorted_rates), fontsize=9)
            xmin, xmax = min(sorted_rates), max(sorted_rates)
            pad = max((xmax - xmin) * 0.07, 0.1)
            ax.set_xlim(xmin - pad, xmax + pad)

        ax.set_xlabel(f"{task.upper()} Req/s", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, axis="both", color="#dddddd", linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9)

    handles, labels = ax_ttfb.get_legend_handles_labels()
    if not handles:
        handles, labels = ax_tpob.get_legend_handles_labels()
    return handles, labels


def plot_user_experiment(
    summary: Summary,
    output: Path,
    task: str,
    schedulers: List[str],
    slo_ttfb_ms: Optional[float],
    slo_tpob_ms: Optional[float],
    title: Optional[str],
    dpi: int,
) -> None:
    """Single user-experiment trajectory figure."""
    fig, ax_trajectory = plt.subplots(figsize=(14.5, 7.5), constrained_layout=True)

    _draw_user_trajectory(
        ax=ax_trajectory,
        summary=summary,
        task=task,
        schedulers=schedulers,
        slo_ttfb_ms=slo_ttfb_ms,
        slo_tpob_ms=slo_tpob_ms,
    )

    if title:
        fig.suptitle(title, fontsize=15, y=0.995)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.00)
    plt.close(fig)
    print(f"[plot] wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot DLM scheduler SLO attainment from slo_summary.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("/tmp/dlm_sched_comparison/slo_summary.json"),
        help="Path to slo_summary.json.",
    )
    parser.add_argument(
        "--summary-label",
        type=str,
        default=None,
        help="Label for --summary in combined figures. Defaults to the summary parent directory.",
    )
    parser.add_argument(
        "--summary2",
        type=Path,
        default=None,
        help="Optional second slo_summary.json. When set, SLO and p99 figures show both summaries.",
    )
    parser.add_argument(
        "--summary2-label",
        type=str,
        default=None,
        help="Label for --summary2 in combined figures. Defaults to the summary parent directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/dlm_sched_comparison/slo_attainment_comparison.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--p99-output",
        type=Path,
        default=None,
        help="Output image path for p99_ttfb_ms/p99_tpob_ms. Defaults to '<output_stem>_p99_latency.<ext>'.",
    )
    parser.add_argument(
        "--no-p99",
        action="store_true",
        help="Do not draw the extra p99 latency figure.",
    )
    parser.add_argument(
        "--p99-linear",
        action="store_true",
        help="Use a linear y-axis for p99 latency instead of the default log scale.",
    )
    parser.add_argument(
        "--p99-log-base",
        type=int,
        default=2,
        help="Log base for the p99 latency figure.",
    )
    parser.add_argument(
        "--p99-normalize-baseline",
        type=str,
        default="LST",
        help="Normalize p99 latency by this scheduler at the same task/request rate. Use '' for raw ms.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_SUFFIX.keys()),
        default=None,
        help="Backward-compatible alias for plotting one metric.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(METRIC_SUFFIX.keys()),
        default=["all", "ttfb", "tpob"],
        help="Attainment metrics to plot. Defaults to all, TTFB, and TPOB.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task order to plot. Defaults to all tasks in alphabetical order.",
    )
    parser.add_argument(
        "--schedulers",
        nargs="+",
        default=None,
        help="Scheduler order to plot. Defaults to known order plus any extras.",
    )
    parser.add_argument("--title", type=str, default=None, help="Figure title.")
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI.")
    parser.add_argument("--no-breakdown", action="store_true",
                        help="Do not draw the stacked breakdown bar figure.")
    parser.add_argument("--breakdown-output", type=Path, default=None,
                        help="Output path for breakdown bar. Defaults to '<output_stem>_breakdown_bar.<ext>'.")
    parser.add_argument("--no-user-experiment", action="store_true",
                        help="Do not draw the user-experiment (Korean label) figure.")
    parser.add_argument("--user-experiment-output", type=Path, default=None,
                        help="Output path for user-experiment figure. Defaults to '<output_stem>_user_experiment.<ext>'.")
    parser.add_argument("--user-experiment-task", type=str, default=None,
                        help="Task to show in the user-experiment figure. Defaults to --bar-task.")
    parser.add_argument("--ue-slo-ttfb-ms", type=float, default=None,
                        help="TTFB SLO in ms for user-experiment figure (horizontal dashed line).")
    parser.add_argument("--ue-slo-tpob-ms", type=float, default=None,
                        help="TPOB SLO in ms for user-experiment figure (horizontal dashed line).")
    parser.add_argument("--no-bar", action="store_true", help="Do not draw the bar p99 figure.")
    parser.add_argument("--bar-output", type=Path, default=None,
                        help="Output path for bar chart. Defaults to '<output_stem>_bar_p99.<ext>'.")
    parser.add_argument("--bar-task", type=str, default="humaneval",
                        help="Task to show in the bar chart.")
    parser.add_argument("--bar-rate", type=str, default="18",
                        help="Request rate to show in the bar chart.")
    parser.add_argument("--bar-slo-ttfb-ms", type=float, default=None,
                        help="TTFB SLO in ms; drawn as a dashed line.")
    parser.add_argument("--bar-slo-tpob-ms", type=float, default=None,
                        help="TPOB SLO in ms; drawn as a dashed line.")
    parser.add_argument("--no-sr", action="store_true",
                        help="Do not draw the strict-vs-relaxed discrimination chart.")
    parser.add_argument("--sr-output", type=Path, default=None,
                        help="Output path for strict-release chart. Defaults to '<output_stem>_strict_release.<ext>'.")
    parser.add_argument("--no-scatter", action="store_true",
                        help="Do not draw the per-request TTFB vs TPOB scatter chart.")
    parser.add_argument("--no-scatter-combined", action="store_true",
                        help="Do not draw the combined scatter figure.")
    parser.add_argument("--scatter-root", type=Path, default=None,
                        help="Root directory containing scheduler_*/request_rate_*/<task> latency jsonl files.")
    parser.add_argument("--scatter-output", type=Path, default=None,
                        help="Output path for scatter chart. Defaults to '<output_stem>_scatter.<ext>'.")
    parser.add_argument("--scatter-task", type=str, default=None,
                        help="Task to show in the scatter chart. Defaults to --bar-task.")
    parser.add_argument("--scatter-rate", type=str, default=None,
                        help="Request rate to show in the scatter chart. Defaults to --bar-rate.")
    parser.add_argument("--scatter-slo-config", type=Path, default=None,
                        help="slo_config.json path for scatter SLO lines. Defaults to <scatter-root>/slo_config.json, then summary slo_data.")
    parser.add_argument("--single-slo-scatter-output", type=Path, default=None,
                        help="Also draw the original single-SLO scatter from --summary parent, --bar-task, and --bar-rate.")
    parser.add_argument("--slo-config", type=Path, default=None,
                        help="slo_config.json path for scatter SLO lines. Defaults to <output_dir>/slo_config.json.")
    parser.add_argument("--no-block-unmask", action="store_true",
                        help="Do not draw block unmask steps by masked-token count.")
    parser.add_argument("--block-unmask-summary", type=Path, default=None,
                        help="dlm_benchmark.py summary JSON containing latency_data request records.")
    parser.add_argument("--block-unmask-bellman-log", type=Path, default=None,
                        help="Optional bellman_log_<task>.jsonl; when present, traj is used before summary data.")
    parser.add_argument("--block-unmask-output", type=Path, default=None,
                        help="Output path for block-unmask plot. Defaults to '<output_stem>_block_unmask_steps.<ext>'.")
    parser.add_argument("--block-unmask-task", type=str, default=None,
                        help="Task to read from --block-unmask-summary. Defaults to --bar-task.")
    parser.add_argument("--block-unmask-model-label", type=str, default=None,
                        help="Model label for block-unmask plot title. Defaults to model field in summary.")
    args = parser.parse_args()

    summary = load_summary(args.summary)
    experiments: List[Tuple[str, Summary]] = [
        (args.summary_label or infer_experiment_label(args.summary), summary)
    ]
    if args.summary2:
        summary2 = load_summary(args.summary2)
        experiments.append((args.summary2_label or infer_experiment_label(args.summary2), summary2))

    tasks = sorted_tasks_for_summaries([exp_summary for _, exp_summary in experiments], args.tasks)
    schedulers = sorted_schedulers_for_summaries(
        [exp_summary for _, exp_summary in experiments],
        args.schedulers,
    )
    metrics = [args.metric] if args.metric else args.metrics
    if len(experiments) > 1:
        plot_summary_multi(
            experiments=experiments,
            output=args.output,
            metrics=metrics,
            tasks=tasks,
            schedulers=schedulers,
            title=args.title,
            dpi=args.dpi,
        )
    else:
        plot_summary(
            summary=summary,
            output=args.output,
            metrics=metrics,
            tasks=tasks,
            schedulers=schedulers,
            title=args.title,
            dpi=args.dpi,
        )
    if not args.no_p99:
        normalize_baseline = args.p99_normalize_baseline or None
        if len(experiments) > 1:
            plot_p99_summary_multi(
                experiments=experiments,
                output=args.p99_output or default_p99_output(args.output),
                tasks=tasks,
                schedulers=schedulers,
                title=None,
                dpi=args.dpi,
                log_scale=not args.p99_linear,
                log_base=args.p99_log_base,
                normalize_baseline=normalize_baseline,
            )
        else:
            plot_p99_summary(
                summary=summary,
                output=args.p99_output or default_p99_output(args.output),
                tasks=tasks,
                schedulers=schedulers,
                title=None,
                dpi=args.dpi,
                log_scale=not args.p99_linear,
                log_base=args.p99_log_base,
                normalize_baseline=normalize_baseline,
            )
    if not args.no_scatter:
        scatter_root = args.scatter_root or args.summary.parent
        scatter_slo_config = args.scatter_slo_config
        if scatter_slo_config is None and args.scatter_root is None:
            scatter_slo_config = args.slo_config
        plot_scatter_ttfb_tpob(
            latency_root=scatter_root,
            output=args.scatter_output or default_scatter_output(args.output),
            task=args.scatter_task or args.bar_task,
            rate=args.scatter_rate or args.bar_rate,
            schedulers=schedulers,
            slo_config_path=scatter_slo_config,
            title=args.title,
            dpi=args.dpi,
            no_combined=args.no_scatter_combined,
        )
    if args.single_slo_scatter_output is not None:
        plot_scatter_ttfb_tpob(
            latency_root=args.summary.parent,
            output=args.single_slo_scatter_output,
            task=args.bar_task,
            rate=args.bar_rate,
            schedulers=schedulers,
            slo_config_path=args.slo_config,
            title=args.title,
            dpi=args.dpi,
            no_combined=args.no_scatter_combined,
        )
    if not args.no_sr:
        plot_strict_release_bar(
            summary=summary,
            output=args.sr_output or default_sr_output(args.output),
            task=args.bar_task,
            rate=args.bar_rate,
            schedulers=schedulers,
            title=args.title,
            dpi=args.dpi,
        )
    if not args.no_bar:
        bar_slo_ttfb = args.bar_slo_ttfb_ms
        bar_slo_tpob = args.bar_slo_tpob_ms
        if (bar_slo_ttfb is None or bar_slo_tpob is None) and args.bar_task:
            cfg_path = args.slo_config or (args.summary.parent / "slo_config.json")
            if cfg_path.exists():
                import json as _json
                task_slo = _json.loads(cfg_path.read_text()).get(args.bar_task, {})
                strict = task_slo.get("strict", {})
                if bar_slo_ttfb is None:
                    bar_slo_ttfb = strict.get("ttfb_ms")
                if bar_slo_tpob is None:
                    bar_slo_tpob = strict.get("tpob_ms")
        plot_bar_p99(
            summary=summary,
            output=args.bar_output or default_bar_output(args.output),
            task=args.bar_task,
            rate=args.bar_rate,
            schedulers=schedulers,
            slo_ttfb_ms=bar_slo_ttfb,
            slo_tpob_ms=bar_slo_tpob,
            title=args.title,
            dpi=args.dpi,
        )
    if not args.no_breakdown:
        bd_slo_ttfb = args.bar_slo_ttfb_ms
        bd_slo_tpob = args.bar_slo_tpob_ms
        if (bd_slo_ttfb is None or bd_slo_tpob is None) and args.bar_task:
            cfg_path = args.slo_config or (args.summary.parent / "slo_config.json")
            if cfg_path.exists():
                import json as _json
                task_slo = _json.loads(cfg_path.read_text()).get(args.bar_task, {})
                strict = task_slo.get("strict", {})
                if bd_slo_ttfb is None:
                    bd_slo_ttfb = strict.get("ttfb_ms")
                if bd_slo_tpob is None:
                    bd_slo_tpob = strict.get("tpob_ms")
        plot_breakdown_bar(
            summary=summary,
            output=args.breakdown_output or default_breakdown_bar_output(args.output),
            task=args.bar_task,
            rate=args.bar_rate,
            schedulers=schedulers,
            slo_ttfb_ms=bd_slo_ttfb,
            slo_tpob_ms=bd_slo_tpob,
            title=args.title,
            dpi=args.dpi,
        )
    if not args.no_user_experiment:
        ue_task = args.user_experiment_task or args.bar_task
        ue_slo_ttfb = args.ue_slo_ttfb_ms
        ue_slo_tpob = args.ue_slo_tpob_ms
        if (ue_slo_ttfb is None or ue_slo_tpob is None) and ue_task:
            cfg_path = args.slo_config or (args.summary.parent / "slo_config.json")
            if cfg_path.exists():
                import json as _json
                task_slo = _json.loads(cfg_path.read_text()).get(ue_task, {})
                strict = task_slo.get("strict", {})
                if ue_slo_ttfb is None:
                    ue_slo_ttfb = strict.get("ttfb_ms")
                if ue_slo_tpob is None:
                    ue_slo_tpob = strict.get("tpob_ms")
        plot_user_experiment(
            summary=summary,
            output=args.user_experiment_output or default_user_experiment_output(args.output),
            task=ue_task,
            schedulers=schedulers,
            slo_ttfb_ms=ue_slo_ttfb,
            slo_tpob_ms=ue_slo_tpob,
            title=args.title,
            dpi=args.dpi,
        )
    if not args.no_block_unmask and (args.block_unmask_summary or args.block_unmask_bellman_log):
        plot_block_unmask_steps(
            summary_path=args.block_unmask_summary,
            bellman_log_path=args.block_unmask_bellman_log,
            output=args.block_unmask_output or default_block_unmask_output(args.output),
            task=args.block_unmask_task or args.bar_task,
            model_label=args.block_unmask_model_label,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
