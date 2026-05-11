#!/usr/bin/env python3
"""Plot Bellman table convergence vs ground-truth (GT) over time.

For each task, reads bellman_log_*.jsonl files and plots:
  - MAE between TB (table) and GT (empirical mean steps per remaining-mask count)
  - Optionally: TB vs GT per r value at a specific block index

Usage:
  python test/plot_bellman_convergence.py \
      --log-dir /tmp/dlm_results/request_rate_200/threads_200 \
      --tasks humaneval math gsm8k \
      --output /tmp/convergence.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_bellman_log(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_gt_p50(records: list[dict], block_size: int) -> np.ndarray:
    """Empirical p50 (median) forwards-to-finish per remaining-mask count."""
    buckets: list[list[float]] = [[] for _ in range(block_size + 1)]
    for rec in records:
        traj = rec["traj"]
        n = len(traj)
        for i, r in enumerate(traj):
            if 0 <= r <= block_size:
                buckets[r].append(n - i)
    p50 = np.full(block_size + 1, np.nan)
    for r, vals in enumerate(buckets):
        if vals:
            p50[r] = float(np.median(vals))
    return p50


def compute_gt(records: list[dict], block_size: int) -> np.ndarray:
    """Empirical mean forwards-to-finish from all trajectory data.

    For each (r_before, r_after) transition, GT[r_before] += 1 per step and
    we count how many observations there are.  GT[r] = mean total steps
    observed starting from r.

    Simpler approach: from each trajectory, for position i the remaining steps
    to finish the block = len(traj) - i.  Average that per r value.
    """
    sums = np.zeros(block_size + 1)
    counts = np.zeros(block_size + 1)
    for rec in records:
        traj = rec["traj"]
        n = len(traj)
        for i, r in enumerate(traj):
            remaining_steps = n - i  # steps from position i to end of block
            if 0 <= r <= block_size:
                sums[r] += remaining_steps
                counts[r] += 1
    gt = np.full(block_size + 1, np.nan)
    mask = counts > 0
    gt[mask] = sums[mask] / counts[mask]
    return gt


def compute_mae_series(records: list[dict], gt: np.ndarray, block_size: int) -> tuple[list[float], list[float]]:
    """MAE between table snapshot and GT at each logged block.

    X-axis is relative time in seconds since the first record.
    Falls back to block index if 'time' is not present.
    """
    t0 = records[0].get("time") if records else None
    xs = []
    maes = []
    valid = ~np.isnan(gt)
    if valid.sum() == 0:
        return xs, maes
    for rec in records:
        table = np.array(rec["table"][: block_size + 1], dtype=float)
        mae = float(np.mean(np.abs(table[valid] - gt[valid])))
        x = (rec["time"] - t0) if (t0 is not None and "time" in rec) else rec["block"]
        xs.append(x)
        maes.append(mae)
    return xs, maes


def plot_convergence(args):
    log_dir = Path(args.log_dir)
    tasks = args.tasks
    output = args.output

    # 2 rows: Row 0 = MAE over time, Row 1 = TB vs GT at final block
    fig, axes = plt.subplots(2, len(tasks), figsize=(5 * len(tasks), 8), squeeze=False)

    for col, task in enumerate(tasks):
        log_path = log_dir / f"bellman_log_{task}.jsonl"
        ax_mae = axes[0][col]
        ax_tbl = axes[1][col]

        if not log_path.exists():
            for ax in (ax_mae, ax_tbl):
                ax.set_title(f"{task}\n(no log)")
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        records = load_bellman_log(log_path)
        if not records:
            for ax in (ax_mae, ax_tbl):
                ax.set_title(f"{task}\n(empty)")
            continue

        block_size = len(records[0]["table"]) - 1
        # block=0 is the initial state logged before any updates.
        init_record = records[0] if records[0].get("block") == 0 else None
        update_records = records[1:] if init_record is not None else records

        gt = compute_gt(update_records, block_size)
        gt_p50 = compute_gt_p50(update_records, block_size)
        use_time = "time" in records[0]

        # Row 0: MAE over time
        block_ids, maes = compute_mae_series(update_records, gt, block_size)
        ax_mae.plot(block_ids, maes, linewidth=1.2, color="steelblue")
        ax_mae.set_xlabel("Time (s)" if use_time else "Completed blocks")
        ax_mae.set_ylabel("MAE (forwards)")
        ax_mae.set_title(f"{task} — MAE vs GT")
        ax_mae.grid(True, alpha=0.3)

        # Row 1: TB vs GT (mean & p50) at final block
        last_table = np.array(records[-1]["table"][: block_size + 1])
        init_table = np.array(init_record["table"][: block_size + 1]) if init_record is not None else None
        xs = np.arange(block_size + 1)
        if init_table is not None:
            ax_tbl.plot(xs, init_table, label="TB (init)", color="gray", linestyle=":", linewidth=1.0)
        ax_tbl.plot(xs, last_table, label="TB (final)", color="steelblue")
        valid_mean = ~np.isnan(gt)
        if valid_mean.any():
            ax_tbl.plot(xs[valid_mean], gt[valid_mean], label="GT mean", color="crimson", linestyle="--")
        valid_p50 = ~np.isnan(gt_p50)
        if valid_p50.any():
            ax_tbl.plot(xs[valid_p50], gt_p50[valid_p50], label="GT p50", color="darkorange", linestyle=":")
        ax_tbl.set_xlabel("Remaining masks (r)")
        ax_tbl.set_ylabel("Estimated forwards")
        ax_tbl.set_title(f"{task} — TB vs GT (final)")
        ax_tbl.legend(fontsize=8)
        ax_tbl.grid(True, alpha=0.3)

    fig.suptitle("Bellman table convergence", fontsize=13)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True, help="Directory containing bellman_log_*.jsonl files")
    parser.add_argument("--tasks", nargs="+", default=["humaneval", "math", "gsm8k"])
    parser.add_argument("--output", default="/tmp/bellman_convergence.png", help="Output path (MAE + TB vs GT)")
    args = parser.parse_args()
    plot_convergence(args)


if __name__ == "__main__":
    main()
