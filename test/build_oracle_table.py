"""Build a Bellman table from bellman_log JSONL files.

Two modes (--mode):
  mean  (default): Aggregate all block trajectories to compute empirical mean
                   E[forwards | r] for each remaining-mask count r.
  last:            Extract the final table state from the last valid log entry
                   (the stochastic-median converged table). Used for warm_start.

Usage:
    python test/build_oracle_table.py \
        --bellman-log path/to/bellman_log.jsonl [additional_logs ...] \
        --block-size 32 \
        --output oracle_table.json [--mode {mean,last}]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def build_table(log_paths: list[Path], block_size: int) -> tuple[list[float], dict[int, int]]:
    observations: defaultdict[int, list[float]] = defaultdict(list)

    for path in log_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip corrupted lines (e.g. partial writes from SIGKILL)
                traj = entry.get("traj", [])
                if not traj:
                    continue
                n = len(traj)
                for i, r in enumerate(traj):
                    r = int(r)
                    if 0 < r <= block_size:
                        observations[r].append(float(n - i))

    table = []
    counts = {}
    for r in range(block_size + 1):
        obs = observations[r]
        counts[r] = len(obs)
        if obs:
            table.append(sum(obs) / len(obs))
        else:
            table.append(float(r) / 2.0 + 1.0)  # fallback: same as DllmConfig default

    return table, counts


def extract_last_table(log_paths: list[Path], block_size: int) -> list[float]:
    """Return the last valid table[] entry across all log files."""
    last_table = None
    for path in log_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "table" in entry:
                    last_table = entry["table"]
    if last_table is None or len(last_table) != block_size + 1:
        print("[warn] no valid table entry found; using default initialization")
        return [float(r) / 2.0 + 1.0 for r in range(block_size + 1)]
    return [float(v) for v in last_table]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Bellman table from trajectory logs.")
    parser.add_argument("--bellman-log", nargs="+", required=True, type=Path,
                        help="bellman_log JSONL file(s) produced by a bellman-mode run")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--output", required=True, type=Path,
                        help="Output JSON file for the table")
    parser.add_argument("--mode", choices=("mean", "last"), default="mean",
                        help="mean: empirical mean (oracle); last: final log entry (warm_start)")
    parser.add_argument("--min-obs", type=int, default=10,
                        help="[mean mode] Minimum observations per r to trust the mean (else fallback)")
    args = parser.parse_args()

    missing = [p for p in args.bellman_log if not p.exists()]
    if missing:
        parser.error(f"bellman-log file(s) not found: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "last":
        table = extract_last_table(args.bellman_log, args.block_size)
        with open(args.output, "w") as f:
            json.dump(table, f)
        print(f"[warm_start] block_size  : {args.block_size}")
        print(f"[warm_start] table written to: {args.output}")
        print()
        print(f"  {'r':>4}  {'V[r]':>10}")
        print(f"  {'-'*4}  {'-'*10}")
        for r in range(1, args.block_size + 1):
            print(f"  {r:>4}  {table[r]:>10.3f}")
        return

    # mode == "mean"
    table, counts = build_table(args.bellman_log, args.block_size)

    # Apply min-obs fallback: r values with too few observations keep the default
    total_obs = sum(counts.values())
    fallback_rs = []
    for r in range(1, args.block_size + 1):
        if counts[r] < args.min_obs:
            table[r] = float(r) / 2.0 + 1.0
            fallback_rs.append(r)

    with open(args.output, "w") as f:
        json.dump(table, f)

    # Diagnostic output
    print(f"[oracle] total blocks observed : {total_obs}")
    print(f"[oracle] block_size            : {args.block_size}")
    print(f"[oracle] min_obs threshold     : {args.min_obs}")
    print(f"[oracle] fallback r values     : {fallback_rs if fallback_rs else 'none'}")
    print(f"[oracle] table written to      : {args.output}")
    print()
    print(f"  {'r':>4}  {'mean_fwd':>10}  {'n_obs':>8}  {'note'}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*10}")
    for r in range(1, args.block_size + 1):
        note = "fallback" if r in fallback_rs else ("ok" if counts[r] >= args.min_obs else "fallback")
        print(f"  {r:>4}  {table[r]:>10.3f}  {counts[r]:>8}  {note}")


if __name__ == "__main__":
    main()
