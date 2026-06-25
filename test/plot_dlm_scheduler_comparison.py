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
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt


DEFAULT_SCHEDULER_ORDER = ["TTFB", "DECODE", "LST", "SOLA", "FCFS", "PREFILL"]

STYLE = {
    "DECODE": {"label": "DECODE", "color": "#DD8452", "marker": "s", "linestyle": "-"},
    "LST": {"label": "LST", "color": "#8172B2", "marker": "^", "linestyle": "-"},
    "SOLA": {"label": "SOLA", "color": "#C44E52", "marker": "s", "linestyle": "-"},
    "FCFS": {"label": "FCFS", "color": "#55A868", "marker": "D", "linestyle": "-"},
    "PREFILL": {"label": "PREFILL", "color": "#64B5CD", "marker": "v", "linestyle": "-"},
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

    fig_width = max(4.2 * len(tasks), 7.5)
    fig, axes = plt.subplots(
        len(rows),
        len(tasks),
        figsize=(fig_width, 7.2),
        sharey=False,
        squeeze=False,
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
                    linewidth=1.8,
                    markersize=4.0,
                )

            if log_scale:
                ax.set_yscale("log", base=log_base)
            if normalize_baseline:
                ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.1, alpha=0.75)
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

    if log_scale:
        scale_label = f"log2 scale" if log_base == 2 else f"log base {log_base} scale"
    else:
        scale_label = "linear scale"
    norm_label = f", normalized to {normalize_baseline}" if normalize_baseline else ""
    fig.suptitle(title or f"p99 latency comparison ({scale_label}{norm_label})", y=1.03, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
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


def _draw_slo_lines(ax, has_strict: bool, has_release: bool, has_other: bool, slo_strict, slo_relaxed) -> None:
    # slo_type 없는 레코드(other)는 단독일 때 반대 tier가 없으면 해당 SLO 선을 그림
    draw_strict = has_strict or (has_other and not has_release)
    draw_release = has_release or (has_other and not has_strict)
    if draw_strict and slo_strict:
        ax.axvline(slo_strict["ttfb_ms"], color="#C44E52", linestyle="--", linewidth=1.2, alpha=0.85)
        ax.axhline(slo_strict["tpob_ms"], color="#C44E52", linestyle="--", linewidth=1.2, alpha=0.85)
    if draw_release and slo_relaxed:
        ax.axvline(slo_relaxed["ttfb_ms"], color="#4472C4", linestyle="--", linewidth=1.2, alpha=0.85)
        ax.axhline(slo_relaxed["tpob_ms"], color="#4472C4", linestyle="--", linewidth=1.2, alpha=0.85)


def plot_scatter_ttfb_tpob(
    latency_root: Path,
    output: Path,
    task: str,
    rate: str,
    schedulers: List[str],
    slo_config_path: Optional[Path],
    title: Optional[str],
    dpi: int,
) -> None:
    slo_strict: Optional[Dict] = None
    slo_relaxed: Optional[Dict] = None
    cfg_path = slo_config_path or (latency_root / "slo_config.json")
    if cfg_path.exists():
        with cfg_path.open() as f:
            task_slo = json.load(f).get(task, {})
        slo_strict = task_slo.get("strict")
        slo_relaxed = task_slo.get("relaxed")

    # Load and split records per scheduler
    sched_split: List[Tuple[str, str, List, List, List, List, List, List]] = []
    for scheduler in schedulers:
        path = latency_root / f"scheduler_{scheduler}" / f"request_rate_{rate}" / task / f"request_latency_{task}.jsonl"
        records = _read_request_jsonl(path)
        if not records:
            continue
        label = STYLE.get(scheduler, {}).get("label", scheduler)
        color = STYLE.get(scheduler, {}).get("color", "#888888")
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
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)

    for col, (label, sched_color, sx, sy, rx, ry, ox, oy) in enumerate(sched_split):
        ax = axes[0][col]
        dot_s = _STRICT_COLOR  if color_by_tier else sched_color
        dot_r = _RELEASE_COLOR if color_by_tier else sched_color
        dot_o = sched_color
        if sx:
            s_met_x, s_met_y, s_viol_x, s_viol_y = _split_by_slo(sx, sy, slo_strict)
            if s_met_x:
                ax.scatter(s_met_x, s_met_y, s=8, alpha=0.6, color=dot_s, label="strict (met)", rasterized=True)
            if s_viol_x:
                ax.scatter(s_viol_x, s_viol_y, s=8, alpha=0.6, color=_lighten_color(dot_s), label="strict (violated)", rasterized=True)
        if rx:
            r_met_x, r_met_y, r_viol_x, r_viol_y = _split_by_slo(rx, ry, slo_relaxed)
            if r_met_x:
                ax.scatter(r_met_x, r_met_y, s=8, alpha=0.6, color=dot_r, label="release (met)", rasterized=True)
            if r_viol_x:
                ax.scatter(r_viol_x, r_viol_y, s=8, alpha=0.6, color=_lighten_color(dot_r), label="release (violated)", rasterized=True)
        if ox:
            ax.scatter(ox, oy, s=8, alpha=0.6, color=dot_o, label="other", rasterized=True)
        _draw_slo_lines(ax, bool(sx), bool(rx), bool(ox), slo_strict, slo_relaxed)
        ax.set_xlabel("TTFB (ms)", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel("TPOB (ms)", fontsize=10)
            ax.legend(fontsize=8, frameon=False, markerscale=2)
        ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=8)

    fig.suptitle(
        title or f"TTFB vs TPOB per request — {task.upper()} @ {rate} req/s",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {output}")

    # ── Combined figure ────────────────────────────────────────────────────────
    # color_by_tier: dots colored by SLO tier (strict=red, release=blue),
    #                marker shape distinguishes scheduler.
    # single_type:   dots colored by scheduler (existing behaviour).
    if not single_type and not color_by_tier:
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
                    ax2.scatter(s_met_x, s_met_y, s=10, alpha=0.6, color=_STRICT_COLOR,
                                marker=mkr, label=f"{label} strict (met)", rasterized=True)
                if s_viol_x:
                    ax2.scatter(s_viol_x, s_viol_y, s=10, alpha=0.6,
                                color=_lighten_color(_STRICT_COLOR), marker=mkr,
                                label=f"{label} strict (viol)", rasterized=True)
            if rx:
                r_met_x, r_met_y, r_viol_x, r_viol_y = _split_by_slo(rx, ry, slo_relaxed)
                if r_met_x:
                    ax2.scatter(r_met_x, r_met_y, s=10, alpha=0.6, color=_RELEASE_COLOR,
                                marker=mkr, label=f"{label} release (met)", rasterized=True)
                if r_viol_x:
                    ax2.scatter(r_viol_x, r_viol_y, s=10, alpha=0.6,
                                color=_lighten_color(_RELEASE_COLOR), marker=mkr,
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
                ax2.scatter(met_x, met_y, s=5, alpha=0.6, color=sched_color, label=label, rasterized=True)
            if viol_x:
                ax2.scatter(viol_x, viol_y, s=5, alpha=0.6, color=_lighten_color(sched_color),
                            label=f"{label} (violated)", rasterized=True)
        slo_label = "strict" if has_any_strict else "relaxed"

    _draw_slo_lines(ax2, has_any_strict, has_any_release, has_any_other, slo_strict, slo_relaxed)
    ax2.set_xlabel("TTFB (ms)", fontsize=11)
    ax2.set_ylabel("TPOB (ms)", fontsize=11)
    ax2.legend(fontsize=9, frameon=False, markerscale=2)
    ax2.grid(True, color="#dddddd", linewidth=0.8, alpha=0.85)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="both", labelsize=9)
    ax2.set_title(
        title or f"TTFB vs TPOB all schedulers ({slo_label}) — {task.upper()} @ {rate} req/s",
        fontsize=11, fontweight="bold",
    )
    fig2.tight_layout()
    combined_output = output.with_name(f"{output.stem}_combined{output.suffix}")
    fig2.savefig(combined_output, dpi=dpi, bbox_inches="tight")
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
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
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
    ax.set_title(title or f"p99 TTFB & TPOB — {task.upper()} @ {rate} req/s",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {output}")


def default_breakdown_bar_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}_breakdown_bar{output.suffix}")


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

    labels: List[str] = []
    p95_ttfb_list: List[float] = []
    p95_tpob_list: List[float] = []
    ttfb_parts: List[List[float]] = []
    tpob_parts: List[List[float]] = []
    ttfb_known_sums: List[float] = []
    tpob_known_sums: List[float] = []

    for scheduler in schedulers:
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

        ttfb_forward = _float_metric(metrics, "p95_ttfb_selected_forward_ms")
        if ttfb_forward is None:
            ttfb_forward = _float_metric(metrics, "p95_ideal_ttfb_ms") or 0.0
        ttfb_sched_wait = _float_metric(metrics, "p95_ttfb_selected_sched_wait_ms")
        if ttfb_sched_wait is None:
            ttfb_sched_wait = _float_metric(metrics, "p95_sched_wait_ms") or 0.0
        ttfb_first_unmask_gap = _float_metric(metrics, "p95_ttfb_selected_first_unmask_gap_ms")
        if ttfb_first_unmask_gap is None:
            ttfb_first_unmask_gap = _float_metric(metrics, "p95_first_unmask_gap_ms") or 0.0
        ttfb_first_block_dw = _float_metric(metrics, "p95_ttfb_selected_first_block_decode_wait_ms") or 0.0
        ttfb_other = _float_metric(metrics, "p95_ttfb_selected_other_ms")

        tpob_forward = _float_metric(metrics, "p95_tpob_selected_forward_ms")
        if tpob_forward is None:
            tpob_forward = _float_metric(metrics, "p95_ideal_tpob_ms") or 0.0
        tpob_decode_wait = _float_metric(metrics, "p95_tpob_selected_decode_wait_ms")
        if tpob_decode_wait is None:
            tpob_decode_wait = _float_metric(metrics, "p95_decode_wait_ms") or 0.0
        tpob_other = _float_metric(metrics, "p95_tpob_selected_other_ms")

        ttfb_known = ttfb_forward + ttfb_sched_wait + ttfb_first_unmask_gap + ttfb_first_block_dw
        if ttfb_other is None:
            ttfb_other = max(0.0, p95_ttfb - ttfb_known)
        tpob_known = tpob_forward + tpob_decode_wait
        if tpob_other is None:
            tpob_other = max(0.0, p95_tpob - tpob_known)

        labels.append(STYLE.get(scheduler, {}).get("label", scheduler))
        p95_ttfb_list.append(p95_ttfb)
        p95_tpob_list.append(p95_tpob)
        ttfb_parts.append([ttfb_forward, ttfb_sched_wait, ttfb_first_unmask_gap, ttfb_first_block_dw, ttfb_other])
        tpob_parts.append([tpob_forward, tpob_decode_wait, tpob_other])
        ttfb_known_sums.append(ttfb_known + ttfb_other)
        tpob_known_sums.append(tpob_known + tpob_other)

    if not labels:
        print(f"[plot] no breakdown data for task={task!r} rate={rate!r}; skipped breakdown bar")
        return

    # ── colours / hatches / labels ────────────────────────────────────────────
    # TTFB bottom -> top: [forward, sched_wait, first_unmask_gap, first_block_dw, others]
    TTFB_COLORS  = ["#E8A838", "#FAD7A0", "#FADEA0", "#C97A00", "#B8B8B8"]
    TTFB_HATCHES = [None,      None,      "\\",      "////",    ".."]
    TTFB_LABELS  = ["Forward", "Prefill delay", "P-D delay", "Decode delay", "Others"]

    # TPOB bottom -> top: [forward, decode_wait, others]
    TPOB_COLORS  = ["#AED6F1", "#154360", "#B8B8B8"]
    TPOB_HATCHES = [None,      "//",      ".."]
    TPOB_LABELS  = ["Forward", "Decode delay", "Others"]

    n = len(labels)
    width = 0.35
    xs = list(range(n))
    ttfb_xs = [x - width / 2 for x in xs]
    tpob_xs = [x + width / 2 for x in xs]

    fig, ax = plt.subplots(figsize=(max(7.0, 2.0 * n), 7.5))

    max_val = max(
        max(p95_ttfb_list),
        max(p95_tpob_list),
        max(ttfb_known_sums),
        max(tpob_known_sums),
    )
    offset = max_val * 0.008

    # ── TTFB stacked bars ─────────────────────────────────────────────────────
    ttfb_bottoms = [0.0] * n
    for i, (color, hatch, lbl) in enumerate(zip(TTFB_COLORS, TTFB_HATCHES, TTFB_LABELS)):
        heights = [ttfb_parts[j][i] if i < len(ttfb_parts[j]) else 0.0 for j in range(n)]
        ax.bar(ttfb_xs, heights, width, bottom=ttfb_bottoms,
               color=color, hatch=hatch, edgecolor="white", linewidth=0.5)
        if i == 0:  # annotate forward compute only
            for j, (bx, h, bot) in enumerate(zip(ttfb_xs, heights, ttfb_bottoms)):
                if h >= p95_ttfb_list[j] * 0.04:
                    ax.text(bx, bot + h / 2, f"{h:.0f}", ha="center", va="center",
                            fontsize=8, color="black", fontweight="bold")
        ttfb_bottoms = [b + h for b, h in zip(ttfb_bottoms, heights)]

    for j, (bx, total, known) in enumerate(zip(ttfb_xs, p95_ttfb_list, ttfb_known_sums)):
        label_y = max(total, known)
        ax.text(bx, label_y + offset, f"{total:.0f} ms", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#7D5C00")

    # ── TPOB stacked bars ─────────────────────────────────────────────────────
    tpob_bottoms = [0.0] * n
    for i, (color, hatch, lbl) in enumerate(zip(TPOB_COLORS, TPOB_HATCHES, TPOB_LABELS)):
        heights = [tpob_parts[j][i] if i < len(tpob_parts[j]) else 0.0 for j in range(n)]
        ax.bar(tpob_xs, heights, width, bottom=tpob_bottoms,
               color=color, hatch=hatch, edgecolor="white", linewidth=0.5)
        if i == 0:  # annotate forward compute only
            for j, (bx, h, bot) in enumerate(zip(tpob_xs, heights, tpob_bottoms)):
                if h >= p95_tpob_list[j] * 0.04:
                    ax.text(bx, bot + h / 2, f"{h:.0f}", ha="center", va="center",
                            fontsize=8, color="black", fontweight="bold")
        tpob_bottoms = [b + h for b, h in zip(tpob_bottoms, heights)]

    for j, (bx, total, known) in enumerate(zip(tpob_xs, p95_tpob_list, tpob_known_sums)):
        label_y = max(total, known)
        ax.text(bx, label_y + offset, f"{total:.0f} ms", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#154360")

    # ── SLO lines with right-side value labels ────────────────────────────────
    if slo_ttfb_ms is not None:
        ax.axhline(slo_ttfb_ms, color="#E8A838", linestyle="--", linewidth=1.8, zorder=3)
        ax.text(1.01, slo_ttfb_ms, f"TTFB SLO ({slo_ttfb_ms:.0f} ms)",
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=8.5, color="#B8820A", fontweight="bold", clip_on=False)
    if slo_tpob_ms is not None:
        ax.axhline(slo_tpob_ms, color="#2980B9", linestyle="--", linewidth=1.8, zorder=3)
        ax.text(1.01, slo_tpob_ms, f"TPOB SLO ({slo_tpob_ms:.0f} ms)",
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=8.5, color="#2980B9", fontweight="bold", clip_on=False)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.grid(True, axis="y", color="#dddddd", linestyle="--", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        title or f"p95 TTFB & TPOB Delay Breakdown — {task.upper()} @ {rate} req/s",
        fontsize=13, fontweight="bold",
    )

    # ── Bottom legends: 3 separate boxes ─────────────────────────────────────
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    _lkw = dict(fontsize=8.5, frameon=True, framealpha=1.0, edgecolor="#aaaaaa",
                fancybox=False, borderpad=0.6)

    slo_handles = []
    if slo_ttfb_ms is not None:
        slo_handles.append(Line2D([0], [0], color="#E8A838", linestyle="--", linewidth=1.5,
                                   label=f"TTFB SLO ({slo_ttfb_ms:.0f} ms)"))
    if slo_tpob_ms is not None:
        slo_handles.append(Line2D([0], [0], color="#2980B9", linestyle="--", linewidth=1.5,
                                   label=f"TPOB SLO ({slo_tpob_ms:.0f} ms)"))

    group_handles = [
        Patch(facecolor="#E8A838", edgecolor="#cccccc", label="TTFB p95 (total)"),
        Patch(facecolor="#AED6F1", edgecolor="#cccccc", label="TPOB p95 (total)"),
    ]

    ttfb_comp_handles = [
        Patch(facecolor=c, hatch=h or "", edgecolor="white" if h else "#cccccc", label=lbl)
        for i, (c, h, lbl) in enumerate(zip(TTFB_COLORS, TTFB_HATCHES, TTFB_LABELS))
    ]
    tpob_comp_handles = [
        Patch(facecolor=c, hatch=h or "", edgecolor="white" if h else "#cccccc", label=lbl)
        for i, (c, h, lbl) in enumerate(zip(TPOB_COLORS, TPOB_HATCHES, TPOB_LABELS))
    ]

    x_groups = 0.20 if slo_handles else 0.0
    x_comps  = x_groups + 0.20

    extra_artists = []
    if slo_handles:
        leg1 = ax.legend(handles=slo_handles, loc="upper left",
                         bbox_to_anchor=(0.0, -0.12), bbox_transform=ax.transAxes,
                         title="SLO Lines", title_fontsize=8.5, **_lkw)
        leg1.get_frame().set_linewidth(0.8)
        ax.add_artist(leg1)
        extra_artists.append(leg1)

    leg2 = ax.legend(handles=group_handles, loc="upper left",
                     bbox_to_anchor=(x_groups, -0.12), bbox_transform=ax.transAxes,
                     title="Bar Groups (Totals)", title_fontsize=8.5, **_lkw)
    leg2.get_frame().set_linewidth(0.8)
    ax.add_artist(leg2)
    extra_artists.append(leg2)

    leg3 = ax.legend(handles=ttfb_comp_handles + tpob_comp_handles, loc="upper left",
                     bbox_to_anchor=(x_comps, -0.06), bbox_transform=ax.transAxes,
                     title="Delay Components (Stacked)", title_fontsize=8.5,
                     ncol=2, **_lkw)
    leg3.get_frame().set_linewidth(0.8)
    extra_artists.append(leg3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", bbox_extra_artists=extra_artists)
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
    series: List[Tuple[str, List[float], List[float], List[float]]] = []
    all_rates: List[float] = []
    all_xs: List[float] = []
    all_ys: List[float] = []

    for scheduler in schedulers:
        rates, xs, ys = collect_p95_pair_series(summary, scheduler, task)
        if not xs:
            continue
        series.append((scheduler, rates, xs, ys))
        all_rates.extend(rates)
        all_xs.extend(xs)
        all_ys.extend(ys)

    if not series:
        raise ValueError(f"No p95 TTFB/TPOB data found for task={task!r}.")

    max_x = max(all_xs + ([slo_ttfb_ms] if slo_ttfb_ms is not None else []))
    max_y = max(all_ys + ([slo_tpob_ms] if slo_tpob_ms is not None else []))
    x_max = max_x * 1.12
    y_max = max_y * 1.12
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    if slo_ttfb_ms is not None and slo_tpob_ms is not None:
        y_frac = min(max(slo_tpob_ms / y_max, 0.0), 1.0)
        ax.axvspan(0, slo_ttfb_ms, ymin=0, ymax=y_frac, facecolor="#EAF4E9", alpha=0.95, zorder=0)
        ax.axvspan(0, slo_ttfb_ms, ymin=y_frac, ymax=1, facecolor="#FFF7E8", alpha=0.85, zorder=0)
        ax.axvspan(slo_ttfb_ms, x_max, ymin=0, ymax=y_frac, facecolor="#FFF7E8", alpha=0.85, zorder=0)
        ax.axvspan(slo_ttfb_ms, x_max, ymin=y_frac, ymax=1, facecolor="#FCEEEE", alpha=0.85, zorder=0)
        ax.axvline(slo_ttfb_ms, color="#444444", linestyle="--", linewidth=1.8, alpha=0.9, zorder=2)
        ax.axhline(slo_tpob_ms, color="#444444", linestyle="--", linewidth=1.8, alpha=0.9, zorder=2)

        ax.text(
            slo_ttfb_ms + x_max * 0.012,
            y_max * 0.965,
            f"First-block SLO\n{slo_ttfb_ms:.0f} ms",
            ha="left",
            va="top",
            fontsize=11,
            color="#222222",
        )
        ax.text(
            x_max * 0.67,
            slo_tpob_ms + y_max * 0.012,
            f"Inter-block delay SLO = {slo_tpob_ms:.0f} ms",
            ha="left",
            va="bottom",
            fontsize=11,
            color="#222222",
        )
        ax.text(x_max * 0.04, y_max * 0.085, "satisfy\nboth", fontsize=11, color="#222222")
        ax.text(x_max * 0.04, y_max * 0.52, "inter-block\nviolation", fontsize=11, color="#222222")
        ax.text(x_max * 0.62, y_max * 0.11, "first-block\nviolation", fontsize=11, color="#222222")
        ax.text(x_max * 0.64, y_max * 0.52, "both\nviolate", fontsize=11, color="#222222")

    if all_rates:
        sorted_rates = sorted(set(all_rates))
        start_label, end_label = nice_rate_labels([sorted_rates[0], sorted_rates[-1]])
        order_label = f"{start_label} -> {end_label}"
        ax.text(
            0.015,
            0.96,
            f"marker order: {order_label} req/s",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            color="#222222",
        )

    label_offsets = {
        "TTFB": (10, -18),
        "DECODE": (-88, 6),
        "LST": (10, 0),
        "SOLA": (10, 2),
        "FCFS": (10, -2),
        "PREFILL": (-88, 10),
    }

    handles = []
    labels: List[str] = []
    for scheduler, _rates, xs, ys in series:
        style = STYLE.get(scheduler, {})
        color = style.get("color")
        label = style.get("label", scheduler)
        (line,) = ax.plot(
            xs,
            ys,
            label=label,
            color=color,
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            linewidth=2.5,
            markersize=7.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=4,
        )
        handles.append(line)
        labels.append(label)

        for x0, y0, x1, y1 in zip(xs, ys, xs[1:], ys[1:]):
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "lw": 1.8,
                    "alpha": 0.82,
                    "shrinkA": 8,
                    "shrinkB": 8,
                },
                zorder=5,
            )

        dx, dy = label_offsets.get(scheduler, (8, 8))
        ax.annotate(
            label,
            xy=(xs[-1], ys[-1]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="center",
            fontsize=10.5,
            color="#111111",
        )

    ax.set_xlabel("p95 Time until first block (ms)", fontsize=13)
    ax.set_ylabel("p95 Max delay between consecutive blocks (ms)", fontsize=13)
    ax.grid(True, axis="both", color="#dddddd", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title(
        "New SLO violation trajectory under increasing request rate",
        fontsize=17,
        pad=14,
    )
    ax.legend(
        handles,
        labels,
        loc="upper right",
        ncol=2,
        frameon=False,
        fontsize=10.5,
        handlelength=2.6,
    )
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
    """Combined user-experiment figure.

    Top: p95 TTFB/TPOB trajectory as request rate increases.
    Bottom: the previous two request-rate panels for p95 TTFB and p95 TPOB.
    """
    fig = plt.figure(figsize=(13.5, 12.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.28, 1.0], hspace=0.22, wspace=0.18)
    ax_trajectory = fig.add_subplot(grid[0, :])
    ax_ttfb = fig.add_subplot(grid[1, 0])
    ax_tpob = fig.add_subplot(grid[1, 1])

    _draw_user_trajectory(
        ax=ax_trajectory,
        summary=summary,
        task=task,
        schedulers=schedulers,
        slo_ttfb_ms=slo_ttfb_ms,
        slo_tpob_ms=slo_tpob_ms,
    )
    _draw_user_rate_panels(
        axes=(ax_ttfb, ax_tpob),
        summary=summary,
        task=task,
        schedulers=schedulers,
        slo_ttfb_ms=slo_ttfb_ms,
        slo_tpob_ms=slo_tpob_ms,
    )

    if title:
        fig.suptitle(title, fontsize=15, y=0.995)

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
    parser.add_argument("--scatter-output", type=Path, default=None,
                        help="Output path for scatter chart. Defaults to '<output_stem>_scatter.<ext>'.")
    parser.add_argument("--slo-config", type=Path, default=None,
                        help="slo_config.json path for scatter SLO lines. Defaults to <output_dir>/slo_config.json.")
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
    if not args.no_p99:
        normalize_baseline = args.p99_normalize_baseline or None
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
        plot_scatter_ttfb_tpob(
            latency_root=args.output.parent,
            output=args.scatter_output or default_scatter_output(args.output),
            task=args.bar_task,
            rate=args.bar_rate,
            schedulers=schedulers,
            slo_config_path=args.slo_config,
            title=args.title,
            dpi=args.dpi,
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
            cfg_path = args.slo_config or (args.output.parent / "slo_config.json")
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
            cfg_path = args.slo_config or (args.output.parent / "slo_config.json")
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
            cfg_path = args.slo_config or (args.output.parent / "slo_config.json")
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


if __name__ == "__main__":
    main()
