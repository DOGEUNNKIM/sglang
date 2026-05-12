#!/usr/bin/env python3
"""Plot per-task distribution of forward steps required per output block."""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_steps(log_dir: str, task: str) -> list:
    path = os.path.join(log_dir, f"step_stats_{task}.jsonl")
    if not os.path.exists(path):
        print(f"[warn] no step log for {task}: {path}", file=sys.stderr)
        return []
    steps = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                steps.extend(rec.get("final_block_steps", []))
            except json.JSONDecodeError:
                continue
    return steps


def main():
    parser = argparse.ArgumentParser(description="Plot step distribution per task")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    task_steps = {}
    for task in args.tasks:
        steps = load_steps(args.log_dir, task)
        if steps:
            task_steps[task] = steps
        else:
            print(f"[warn] no step data for {task}", file=sys.stderr)

    if not task_steps:
        print("No step data found, skipping plot.", file=sys.stderr)
        return

    n = len(task_steps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

    for i, (task, steps) in enumerate(task_steps.items()):
        ax = axes[0][i]
        max_step = max(steps)
        bins = list(range(1, max_step + 2))
        ax.hist(steps, bins=bins, align="left", rwidth=0.8, color="steelblue", edgecolor="white")
        mean_steps = float(np.mean(steps))
        ax.axvline(mean_steps, color="red", linestyle="--", linewidth=1.2, label=f"mean={mean_steps:.1f}")
        ax.set_title(task)
        ax.set_xlabel("Steps per block")
        ax.set_ylabel("Count")
        ax.set_xticks(range(1, max_step + 1))
        ax.legend(fontsize=8)

    fig.suptitle("Steps per output block")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
