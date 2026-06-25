#!/usr/bin/env python3
"""
Plot remaining_compute_mode comparison from run_dlm_bellman_test.sh output.

Generates three figures:
  1. Scatter: per-request TTFB vs TPOB, one subplot per mode
  2. SLO attainment bar: strict/relaxed attainment per mode
  3. p95 latency bar: TTFB and TPOB p95 per mode

Usage:
    python test/plot_dlm_bellman_test.py \
        --output-root /mnt/nvme0/kdg6245/dlm_bellman_test \
        --tasks gsm8k \
        --rates 9.5 \
        --model-path inclusionAI/LLaDA2.0-mini
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

MODE_STYLE = {
    "bellman": {"label": "Bellman",  "color": "#8172B2", "marker": "o"},
    "zero":    {"label": "Zero (EDF)","color": "#55A868", "marker": "s"},
    "worst":   {"label": "Worst",    "color": "#C44E52", "marker": "^"},
    "oracle":  {"label": "Oracle",   "color": "#4472C4", "marker": "D"},
}

_STRICT_COLOR  = "#C44E52"
_RELEASE_COLOR = "#4472C4"


def _lighten(hex_color: str, amount: float = 0.55) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02X}{g:02X}{b:02X}"


def _read_jsonl(path: Path) -> List[Dict]:
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


def _split_records(records: List[Dict]):
    strict_x, strict_y, release_x, release_y, other_x, other_y = [], [], [], [], [], []
    for r in records:
        if r.get("ttfb_ms") is None or r.get("tpob_ms") is None:
            continue
        slo_type = r.get("slo_type")
        if slo_type == "strict":
            strict_x.append(r["ttfb_ms"]); strict_y.append(r["tpob_ms"])
        elif slo_type == "release":
            release_x.append(r["ttfb_ms"]); release_y.append(r["tpob_ms"])
        else:
            other_x.append(r["ttfb_ms"]); other_y.append(r["tpob_ms"])
    return strict_x, strict_y, release_x, release_y, other_x, other_y


def _split_by_slo(xs, ys, slo):
    if not slo:
        return xs, ys, [], []
    ttfb_lim = slo.get("ttfb_ms")
    tpob_lim = slo.get("tpob_ms")
    met_x, met_y, viol_x, viol_y = [], [], [], []
    for x, y in zip(xs, ys):
        ok = ((ttfb_lim is None or x <= ttfb_lim) and
              (tpob_lim is None or y <= tpob_lim))
        if ok:
            met_x.append(x); met_y.append(y)
        else:
            viol_x.append(x); viol_y.append(y)
    return met_x, met_y, viol_x, viol_y


def plot_scatter(
    output_root: Path,
    modes: List[str],
    task: str,
    rate: str,
    slo_config_path: Path,
    output: Path,
    dpi: int,
) -> None:
    slo_strict = slo_relaxed = None
    if slo_config_path.exists():
        cfg = json.loads(slo_config_path.read_text()).get(task, {})
        slo_strict  = cfg.get("strict")
        slo_relaxed = cfg.get("relaxed")

    mode_data = []
    for mode in modes:
        path = (output_root / f"rc_mode_{mode}" / f"request_rate_{rate}"
                / task / f"dlm_request_latency_{task}.jsonl")
        records = _read_jsonl(path)
        if records:
            mode_data.append((mode, records))

    if not mode_data:
        print(f"[plot] no scatter records for task={task} rate={rate}; skipped")
        return

    n = len(mode_data)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)

    for col, (mode, records) in enumerate(mode_data):
        ax = axes[0][col]
        style = MODE_STYLE.get(mode, {"label": mode, "color": "#888888"})
        color = style["color"]

        sx, sy, rx, ry, ox, oy = _split_records(records)
        has_strict  = bool(sx)
        has_release = bool(rx)

        if sx:
            mx, my, vx, vy = _split_by_slo(sx, sy, slo_strict)
            if mx: ax.scatter(mx, my, s=8, alpha=0.6, color=_STRICT_COLOR,
                              label="strict (met)", rasterized=True)
            if vx: ax.scatter(vx, vy, s=8, alpha=0.6, color=_lighten(_STRICT_COLOR),
                              label="strict (viol)", rasterized=True)
        if rx:
            mx, my, vx, vy = _split_by_slo(rx, ry, slo_relaxed)
            if mx: ax.scatter(mx, my, s=8, alpha=0.6, color=_RELEASE_COLOR,
                              label="release (met)", rasterized=True)
            if vx: ax.scatter(vx, vy, s=8, alpha=0.6, color=_lighten(_RELEASE_COLOR),
                              label="release (viol)", rasterized=True)
        if ox:
            mx, my, vx, vy = _split_by_slo(ox, oy, slo_strict or slo_relaxed)
            if mx: ax.scatter(mx, my, s=8, alpha=0.6, color=color,
                              label="met", rasterized=True)
            if vx: ax.scatter(vx, vy, s=8, alpha=0.6, color=_lighten(color),
                              label="violated", rasterized=True)

        draw_strict  = has_strict  or (bool(ox) and not has_release)
        draw_release = has_release or (bool(ox) and not has_strict)
        if draw_strict and slo_strict:
            ax.axvline(slo_strict["ttfb_ms"],  color=_STRICT_COLOR,  linestyle="--", linewidth=1.2, alpha=0.85)
            ax.axhline(slo_strict["tpob_ms"],  color=_STRICT_COLOR,  linestyle="--", linewidth=1.2, alpha=0.85)
        if draw_release and slo_relaxed:
            ax.axvline(slo_relaxed["ttfb_ms"], color=_RELEASE_COLOR, linestyle="--", linewidth=1.2, alpha=0.85)
            ax.axhline(slo_relaxed["tpob_ms"], color=_RELEASE_COLOR, linestyle="--", linewidth=1.2, alpha=0.85)

        ax.set_title(style["label"], fontsize=11, fontweight="bold")
        ax.set_xlabel("TTFB (ms)", fontsize=10)
        if col == 0:
            ax.set_ylabel("TPOB (ms)", fontsize=10)
            ax.legend(fontsize=8, frameon=False, markerscale=2)
        ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)

    fig.suptitle(f"TTFB vs TPOB — {task.upper()} @ {rate} req/s", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {output}")


def _read_slo_rates(path: Path, task: str) -> Optional[Dict]:
    """Read the rates dict from a dlm_slorate.py output JSON. Returns None if missing."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    task_data = data.get(task.lower()) or data.get(task.upper()) or next(iter(data.values()), {})
    return task_data.get("rates", task_data)


def plot_slo_attainment(
    output_root: Path,
    modes: List[str],
    task: str,
    output: Path,
    dpi: int,
) -> None:
    labels, strict_all, relaxed_all = [], [], []
    strict_ttfb_list, strict_tpob_list = [], []
    relaxed_ttfb_list, relaxed_tpob_list = [], []

    for mode in modes:
        rates = _read_slo_rates(output_root / f"rc_mode_{mode}" / f"slo_rates_{task}.json", task)
        if rates is None or rates.get("strict_all") is None:
            continue
        style = MODE_STYLE.get(mode, {"label": mode})
        labels.append(style["label"])
        strict_all.append(float(rates.get("strict_all", 0)))
        relaxed_all.append(float(rates.get("relaxed_all") or 0))
        strict_ttfb_list.append(float(rates.get("strict_ttfb") or 0))
        strict_tpob_list.append(float(rates.get("strict_tpob") or 0))
        relaxed_ttfb_list.append(float(rates.get("relaxed_ttfb") or 0))
        relaxed_tpob_list.append(float(rates.get("relaxed_tpob") or 0))

    if not labels:
        print(f"[plot] no SLO attainment data for task={task}; skipped")
        return

    n = len(labels)
    xs = list(range(n))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(max(6.0, 1.8 * n), 5.2))

    groups = [
        (offsets[0], strict_ttfb_list,  "#C44E52", "Strict TTFB"),
        (offsets[1], strict_tpob_list,  "#E8713C", "Strict TPOB"),
        (offsets[2], relaxed_ttfb_list, "#4472C4", "Relaxed TTFB"),
        (offsets[3], relaxed_tpob_list, "#7FA8D5", "Relaxed TPOB"),
    ]
    for offset, vals, color, lbl in groups:
        ax.bar([x + offset * width for x in xs], vals, width * 0.92,
               color=color, label=lbl, edgecolor="white", linewidth=0.5)

    for j, (x, sa) in enumerate(zip(xs, strict_all)):
        ax.text(x + offsets[1] * width, strict_tpob_list[j] + 0.01,
                f"{sa:.1%}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("SLO Attainment Rate", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False, ncol=4, loc="upper right")
    ax.set_title(f"SLO Attainment — {task.upper()} (all rates)", fontsize=13, fontweight="bold")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {output}")


def plot_slo_by_rate(
    output_root: Path,
    modes: List[str],
    task: str,
    rates: List[str],
    output: Path,
    dpi: int,
) -> None:
    """Line chart: X = request rate, Y = strict_all attainment, one line per mode."""
    rate_floats = []
    for r in rates:
        try:
            rate_floats.append(float(r))
        except ValueError:
            rate_floats.append(r)

    has_data = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_strict, ax_relaxed = axes

    for mode in modes:
        style = MODE_STYLE.get(mode, {"label": mode, "color": "#888888", "marker": "o"})
        strict_vals, relaxed_vals = [], []
        for rate in rates:
            path = output_root / f"rc_mode_{mode}" / f"slo_rates_{task}_rate{rate}.json"
            r = _read_slo_rates(path, task)
            strict_vals.append(float(r["strict_all"]) if r and r.get("strict_all") is not None else None)
            relaxed_vals.append(float(r["relaxed_all"]) if r and r.get("relaxed_all") is not None else None)

        xs = [rf for rf, sv in zip(rate_floats, strict_vals) if sv is not None]
        ys_strict  = [sv for sv in strict_vals  if sv is not None]
        ys_relaxed = [rv for rv in relaxed_vals if rv is not None]

        if not xs:
            continue
        has_data = True

        ax_strict.plot(xs, ys_strict, marker=style["marker"], color=style["color"],
                       label=style["label"], linewidth=1.8, markersize=6)
        ax_relaxed.plot(xs, ys_relaxed, marker=style["marker"], color=style["color"],
                        label=style["label"], linewidth=1.8, markersize=6)

    if not has_data:
        print(f"[plot] no per-rate SLO data for task={task}; skipped slo_by_rate")
        plt.close(fig)
        return

    for ax, title in ((ax_strict, "Strict"), (ax_relaxed, "Relaxed")):
        ax.set_xlabel("Request Rate (req/s)", fontsize=10)
        ax.set_ylabel("SLO Attainment", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, frameon=False)
        ax.set_title(f"{title} SLO — {task.upper()}", fontsize=11, fontweight="bold")
        if isinstance(rate_floats[0], float):
            ax.set_xticks(rate_floats)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {output}")


def plot_p95_bar(
    output_root: Path,
    modes: List[str],
    task: str,
    rate: str,
    model_tag: str,
    slo_config_path: Path,
    output: Path,
    dpi: int,
) -> None:
    slo_ttfb = slo_tpob = None
    if slo_config_path.exists():
        cfg = json.loads(slo_config_path.read_text()).get(task, {})
        strict = cfg.get("strict", {})
        slo_ttfb = strict.get("ttfb_ms")
        slo_tpob = strict.get("tpob_ms")

    labels, ttfb_vals, tpob_vals = [], [], []
    for mode in modes:
        path = (output_root / f"rc_mode_{mode}" / f"request_rate_{rate}"
                / task / f"{task}_{model_tag}.json")
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        ls = d.get("latency_stats", {})
        ttfb = ls.get("p95_ttfb_ms")
        tpob = ls.get("p95_tpob_ms")
        if ttfb is None or tpob is None:
            continue
        style = MODE_STYLE.get(mode, {"label": mode})
        labels.append(style["label"])
        ttfb_vals.append(float(ttfb))
        tpob_vals.append(float(tpob))

    if not labels:
        print(f"[plot] no p95 data for task={task} rate={rate}; skipped p95 bar")
        return

    n = len(labels)
    xs = list(range(n))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(5.0, 1.8 * n), 5.0))
    ax.bar([x - width / 2 for x in xs], ttfb_vals, width,
           label="TTFB p95", color="#E8A838", edgecolor="white")
    ax.bar([x + width / 2 for x in xs], tpob_vals, width,
           label="TPOB p95", color="#4472C4", edgecolor="white")

    for x, tv, pv in zip(xs, ttfb_vals, tpob_vals):
        ax.text(x - width / 2, tv + 20, f"{tv:.0f}", ha="center", va="bottom", fontsize=9)
        ax.text(x + width / 2, pv + 20, f"{pv:.0f}", ha="center", va="bottom", fontsize=9)

    if slo_ttfb is not None:
        ax.axhline(slo_ttfb, color="#E8A838", linestyle="--", linewidth=1.5,
                   label=f"TTFB SLO ({slo_ttfb:.0f} ms)")
    if slo_tpob is not None:
        ax.axhline(slo_tpob, color="#4472C4", linestyle="--", linewidth=1.5,
                   label=f"TPOB SLO ({slo_tpob:.0f} ms)")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=False)
    ax.set_title(f"p95 TTFB & TPOB — {task.upper()} @ {rate} req/s",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot remaining_compute_mode comparison from run_dlm_bellman_test.sh output."
    )
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Root output directory of run_dlm_bellman_test.sh")
    parser.add_argument("--modes", nargs="+", default=list(MODE_STYLE.keys()),
                        help="Modes to include (default: bellman zero worst oracle)")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k"])
    parser.add_argument("--rates", nargs="+", default=["9.5"])
    parser.add_argument("--model-path", type=str, default="inclusionAI/LLaDA2.0-mini")
    parser.add_argument("--slo-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write plots. Defaults to --output-root/plots/")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    out_dir = args.output_dir or (args.output_root / "plots")
    slo_config = args.slo_config or (args.output_root / "slo_config.json")
    model_tag = args.model_path.replace("/", "_")

    # Filter modes to those that actually have data
    available_modes = []
    for mode in args.modes:
        if any((args.output_root / f"rc_mode_{mode}").exists() for _ in [None]):
            available_modes.append(mode)
    if not available_modes:
        available_modes = args.modes

    for task in args.tasks:
        for rate in args.rates:
            plot_scatter(
                output_root=args.output_root,
                modes=available_modes,
                task=task,
                rate=rate,
                slo_config_path=slo_config,
                output=out_dir / f"scatter_{task}_rate{rate}.png",
                dpi=args.dpi,
            )
            plot_p95_bar(
                output_root=args.output_root,
                modes=available_modes,
                task=task,
                rate=rate,
                model_tag=model_tag,
                slo_config_path=slo_config,
                output=out_dir / f"p95_bar_{task}_rate{rate}.png",
                dpi=args.dpi,
            )

        plot_slo_attainment(
            output_root=args.output_root,
            modes=available_modes,
            task=task,
            output=out_dir / f"slo_attainment_{task}.png",
            dpi=args.dpi,
        )
        plot_slo_by_rate(
            output_root=args.output_root,
            modes=available_modes,
            task=task,
            rates=args.rates,
            output=out_dir / f"slo_by_rate_{task}.png",
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
