#!/usr/bin/env python3
"""
Parse dlm_benchmark.py summary JSON outputs and generate throughput results JSON
compatible with plot_throughput_figures.py.

Usage:
    python parse_throughput_results.py \
        --baseline  /path/to/summary_baseline.json \
        --ours      /path/to/summary_ours.json \
        --model     LLaDA2.0-mini \
        --output    /mnt/nvme0/kdg6245/throughput_llada2_results.json

    # throughput_scaling (gsm8k effectiveness) 포함 시:
    python parse_throughput_results.py \
        --baseline  /path/to/summary_baseline.json \
        --ours      /path/to/summary_ours.json \
        --model     LLaDA2.0-mini \
        --scaling-configs "2048/32/64" "1024/32/32" "512/32/16" \
        --output    /mnt/nvme0/kdg6245/throughput_llada2_results.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

TASK_DOMAIN = {
    "humaneval": "Code",
    "math":      "Reasoning",
    "gsm8k":     "Math",
    "sharegpt":  "ChatBot",
    "mmlu":      "Knowledge",
    "gpqa":      "Reasoning",
    "ruler_1k":  "Long context",
    "ruler_2k":  "Long context",
    "ruler_3k":  "Long context",
    "ruler_4k":  "Long context",
    "ruler_1~4k mix": "Long context",
    "ruler_1_4k": "Long context",
}

LONG_CONTEXT_TASKS = {"ruler_1k", "ruler_2k", "ruler_3k", "ruler_4k", "ruler_1~4k mix", "ruler_1_4k"}


def _flatten_steps(values: Any) -> list[int]:
    flat: list[int] = []
    if values is None:
        return flat
    if not isinstance(values, list):
        return [int(values)]
    for v in values:
        if isinstance(v, list):
            flat.extend(_flatten_steps(v))
        else:
            flat.append(int(v))
    return flat


def _step_stats(block_steps: list[int]) -> dict:
    if not block_steps:
        return {}
    arr = np.array(block_steps, dtype=float)
    mean = float(np.mean(arr))
    std  = float(np.std(arr))
    cov  = std / mean if mean > 0 else 0.0
    cov_over_mean = cov / mean if mean > 0 else 0.0
    return {
        "steps_per_block_mean":          round(mean, 3),
        "steps_per_block_std":           round(std,  3),
        "steps_per_block_cov":           round(cov,  3),
        "steps_per_block_cov_over_mean": round(cov_over_mean, 3),
    }


def _get_block_steps(step_data: dict, task: str) -> list[int]:
    td = step_data.get(task, {})
    return _flatten_steps(td.get("block_steps", []))


def _get_throughput(results: dict, task: str) -> float | None:
    t = results.get(task, {}).get("output_throughput_tok_s")
    return float(t) if t is not None else None


def build_variability_entry(
    task: str,
    step_data_ours: dict,
    results_baseline: dict,
    results_ours: dict,
) -> dict | None:
    block_steps = _get_block_steps(step_data_ours, task)
    stats = _step_stats(block_steps)
    if not stats:
        print(f"  [warn] no block_steps for task={task!r}, skipping")
        return None

    thr_base = _get_throughput(results_baseline, task)
    thr_ours = _get_throughput(results_ours, task)
    if thr_base is None or thr_ours is None or thr_base == 0:
        print(f"  [warn] missing throughput for task={task!r} "
              f"(baseline={thr_base}, ours={thr_ours}), speedup=None")
        speedup = None
    else:
        speedup = round(thr_ours / thr_base, 3)

    return {
        "domain": TASK_DOMAIN.get(task, "Unknown"),
        "task": task,
        **stats,
        "throughput_speedup": speedup,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, type=Path,
                        help="summary JSON from dlm_benchmark.py (baseline scheduler)")
    parser.add_argument("--ours", required=True, type=Path,
                        help="summary JSON from dlm_benchmark.py (our scheduler)")
    parser.add_argument("--model", default="LLaDA2.0-mini",
                        help="Model name written into output JSON")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output JSON path (e.g. /mnt/nvme0/kdg6245/throughput_llada2_results.json)")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Tasks to include (default: all tasks found in --ours)")
    args = parser.parse_args()

    print(f"[parse] loading baseline: {args.baseline}")
    with args.baseline.open() as f:
        baseline = json.load(f)
    print(f"[parse] loading ours:     {args.ours}")
    with args.ours.open() as f:
        ours = json.load(f)

    results_baseline = baseline.get("results", {})
    results_ours     = ours.get("results", {})
    step_data_ours   = ours.get("step_data", {})

    all_tasks = args.tasks or list(step_data_ours.keys())
    normal_tasks = [t for t in all_tasks if t not in LONG_CONTEXT_TASKS]
    long_tasks   = [t for t in all_tasks if t in LONG_CONTEXT_TASKS]

    task_step_variability = []
    for task in normal_tasks:
        entry = build_variability_entry(task, step_data_ours, results_baseline, results_ours)
        if entry:
            task_step_variability.append(entry)

    long_context_step_variability = []
    for task in long_tasks:
        entry = build_variability_entry(task, step_data_ours, results_baseline, results_ours)
        if entry:
            long_context_step_variability.append(entry)

    out = {
        "model": args.model,
        "source": "parsed_from_dlm_benchmark",
        "task_step_variability": task_step_variability,
    }
    if long_context_step_variability:
        out["long_context_step_variability"] = long_context_step_variability

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[parse] wrote {args.output}")
    print(f"  tasks:        {[e['task'] for e in task_step_variability]}")
    if long_context_step_variability:
        print(f"  long context: {[e['task'] for e in long_context_step_variability]}")


if __name__ == "__main__":
    main()
