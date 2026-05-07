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
        "relaxed_all": 0.0
      }
    }
  }
}

Example:
    python ~/sglang/test/plot_dlm_slo_summary.py \
        --summary /tmp/dlm_sched_comparison/slo_summary.json

    python ~/sglang/test/plot_dlm_slo_summary.py \
        --metrics all ttfb tpob
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt


DEFAULT_SCHEDULER_ORDER = ["TTFB", "DECODE", "LST", "SOLA", "FCFS", "PREFILL"]

STYLE = {
    "TTFB": {"label": "TTFB", "color": "#4C72B0", "marker": "o", "linestyle": "-"},
    "DECODE": {"label": "DECODE", "color": "#DD8452", "marker": "s", "linestyle": "-"},
    "LST": {"label": "LST", "color": "#8172B2", "marker": "^", "linestyle": "-"},
    "SOLA": {"label": "SOLA", "color": "#C44E52", "marker": "s", "linestyle": "-"},
    "FCFS": {"label": "FCFS", "color": "#55A868", "marker": "D", "linestyle": "-"},
    "PREFILL": {"label": "PREFILL", "color": "#64B5CD", "marker": "v", "linestyle": "-"},
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
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def sorted_tasks(summary: Summary, requested: Optional[List[str]]) -> List[str]:
    tasks = set()
    for scheduler_data in summary.values():
        for rate_data in scheduler_data.values():
            tasks.update(rate_data.keys())

    if requested:
        return [task for task in requested if task in tasks]
    return sorted(tasks)


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
        if not metrics or field not in metrics:
            continue
        xs.append(rate_key(rate))
        ys.append(float(metrics[field]))

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

    fig_height = max(2.15 * len(rows), 5.0)
    fig_width = max(4.2 * len(tasks), 7.5)
    fig, axes = plt.subplots(
        len(rows),
        len(tasks),
        figsize=(fig_width, fig_height),
        sharey=True,
        squeeze=False,
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
                    linewidth=1.8,
                    markersize=4.0,
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
                ax.set_title(task.upper(), fontsize=12, fontweight="bold")
            if row == len(rows) - 1:
                ax.set_xlabel("Requests/s", fontsize=10)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=10, fontweight="bold")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 6),
            frameon=False,
            bbox_to_anchor=(0.5, 0.99),
            fontsize=10,
        )

    metric_label = " / ".join(METRIC_LABEL[metric] for metric in metrics)
    fig.suptitle(title or f"{metric_label} SLO attainment comparison", y=1.02, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
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
        "--output",
        type=Path,
        default=Path("/tmp/dlm_sched_comparison/slo_attainment_comparison.png"),
        help="Output image path.",
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
    args = parser.parse_args()

    summary = load_summary(args.summary)
    tasks = sorted_tasks(summary, args.tasks)
    schedulers = sorted_schedulers(summary, args.schedulers)
    metrics = [args.metric] if args.metric else args.metrics
    plot_summary(
        summary=summary,
        output=args.output,
        metrics=metrics,
        tasks=tasks,
        schedulers=schedulers,
        title=args.title,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
