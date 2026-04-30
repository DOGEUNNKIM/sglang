#!/usr/bin/env python3
"""
Compute per-task DLM SLO satisfaction rates from dlm_benchmark.py latency logs.

Examples:
    # From a dlm_benchmark.py summary JSON.
    python test/dlm_slorate.py --summary outputs/summary_inclusionAI_LLaDA2.0-mini.json

    # From per-task JSONL files in an output directory.
    python test/dlm_slorate.py --latency-dir outputs --tasks gsm8k humaneval math

    # Override SLOs.
    python test/dlm_slorate.py --summary outputs/summary.json --slo-config slo.json

SLO config format:
{
  "gsm8k": {
    "strict": {"ttfb_ms": 2500, "tpob_ms": 2500},
    "relaxed": {"ttfb_ms": 12000, "tpob_ms": 12000}
  }
}
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


TASKS = ("gsm8k", "humaneval", "math")


@dataclass(frozen=True)
class SloTarget:
    ttfb_ms: float
    tpob_ms: float


@dataclass(frozen=True)
class TaskSlo:
    strict: SloTarget
    relaxed: SloTarget


DEFAULT_BASE_MS: Dict[str, Dict[str, float]] = {
    "gsm8k": {"ttfb_ms": 417.2, "tpob_ms": 354.5},
    "humaneval": {"ttfb_ms": 312.9, "tpob_ms": 172.7},
    "math": {"ttfb_ms": 404.5, "tpob_ms": 442.3},
}


def build_default_slos(strict_factor: float, relaxed_factor: float) -> Dict[str, TaskSlo]:
    return {
        task: TaskSlo(
            strict=SloTarget(
                ttfb_ms=base["ttfb_ms"] * strict_factor,
                tpob_ms=base["tpob_ms"] * strict_factor,
            ),
            relaxed=SloTarget(
                ttfb_ms=base["ttfb_ms"] * relaxed_factor,
                tpob_ms=base["tpob_ms"] * relaxed_factor,
            ),
        )
        for task, base in DEFAULT_BASE_MS.items()
    }


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_summary_latency(summary_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    with summary_path.open() as f:
        summary = json.load(f)
    latency_data = summary.get("latency_data") or {}
    return {
        task.lower(): list((payload or {}).get("request") or [])
        for task, payload in latency_data.items()
    }


def _load_dir_latency(latency_dir: Path, tasks: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
    return {
        task: _read_jsonl(latency_dir / f"request_latency_{task}.jsonl")
        for task in tasks
    }


def load_request_latency(
    summary_path: Optional[Path],
    latency_dir: Optional[Path],
    tasks: Iterable[str],
) -> Dict[str, List[Dict[str, Any]]]:
    if summary_path is not None:
        loaded = _load_summary_latency(summary_path)
        if loaded:
            return {task: loaded.get(task, []) for task in tasks}
    if latency_dir is None:
        raise ValueError("Either --summary or --latency-dir is required")
    return _load_dir_latency(latency_dir, tasks)


def _load_slo_config(path: Path, default_slos: Mapping[str, TaskSlo]) -> Dict[str, TaskSlo]:
    with path.open() as f:
        raw = json.load(f)

    slos = dict(default_slos)
    for task, payload in raw.items():
        task_key = task.lower()
        strict = payload.get("strict") or {}
        relaxed = payload.get("relaxed") or {}
        base = slos.get(task_key)
        slos[task_key] = TaskSlo(
            strict=SloTarget(
                ttfb_ms=float(strict.get("ttfb_ms", base.strict.ttfb_ms if base else 0.0)),
                tpob_ms=float(strict.get("tpob_ms", base.strict.tpob_ms if base else 0.0)),
            ),
            relaxed=SloTarget(
                ttfb_ms=float(relaxed.get("ttfb_ms", base.relaxed.ttfb_ms if base else 0.0)),
                tpob_ms=float(relaxed.get("tpob_ms", base.relaxed.tpob_ms if base else 0.0)),
            ),
        )
    return slos


def _metric_value(record: Mapping[str, Any], key: str) -> Optional[float]:
    value = record.get(key)
    if value is None:
        return None
    return float(value)


def _rate(pass_count: int, total_count: int) -> Optional[float]:
    if total_count == 0:
        return None
    return pass_count / total_count


def _format_rate(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _count_metric(
    records: List[Dict[str, Any]],
    key: str,
    threshold_ms: float,
    missing: str,
) -> Tuple[int, int]:
    passed = 0
    total = 0
    for record in records:
        value = _metric_value(record, key)
        if value is None:
            if missing == "fail":
                total += 1
            continue
        total += 1
        if value <= threshold_ms:
            passed += 1
    return passed, total


def _count_all(
    records: List[Dict[str, Any]],
    target: SloTarget,
    missing: str,
) -> Tuple[int, int]:
    passed = 0
    total = 0
    for record in records:
        ttfb = _metric_value(record, "ttfb_ms")
        tpob = _metric_value(record, "tpob_ms")
        if ttfb is None or tpob is None:
            if missing == "fail":
                total += 1
            continue
        total += 1
        if ttfb <= target.ttfb_ms and tpob <= target.tpob_ms:
            passed += 1
    return passed, total


def compute_slo_rates(
    latency_by_task: Mapping[str, List[Dict[str, Any]]],
    slos: Mapping[str, TaskSlo],
    missing: str = "exclude",
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    overall_counts = {
        "strict_ttfb": [0, 0],
        "strict_tpob": [0, 0],
        "strict_all": [0, 0],
        "relaxed_ttfb": [0, 0],
        "relaxed_tpob": [0, 0],
        "relaxed_all": [0, 0],
    }

    for task, records in latency_by_task.items():
        if task not in slos:
            continue
        task_slo = slos[task]
        counts = {
            "strict_ttfb": _count_metric(records, "ttfb_ms", task_slo.strict.ttfb_ms, missing),
            "strict_tpob": _count_metric(records, "tpob_ms", task_slo.strict.tpob_ms, missing),
            "strict_all": _count_all(records, task_slo.strict, missing),
            "relaxed_ttfb": _count_metric(records, "ttfb_ms", task_slo.relaxed.ttfb_ms, missing),
            "relaxed_tpob": _count_metric(records, "tpob_ms", task_slo.relaxed.tpob_ms, missing),
            "relaxed_all": _count_all(records, task_slo.relaxed, missing),
        }
        for key, (passed, total) in counts.items():
            overall_counts[key][0] += passed
            overall_counts[key][1] += total

        results[task] = {
            "num_records": len(records),
            "slo_ms": {
                "strict": {
                    "ttfb_ms": task_slo.strict.ttfb_ms,
                    "tpob_ms": task_slo.strict.tpob_ms,
                },
                "relaxed": {
                    "ttfb_ms": task_slo.relaxed.ttfb_ms,
                    "tpob_ms": task_slo.relaxed.tpob_ms,
                },
            },
            "counts": {
                key: {"pass": passed, "total": total}
                for key, (passed, total) in counts.items()
            },
            "rates": {
                key: _rate(passed, total)
                for key, (passed, total) in counts.items()
            },
        }

    results["overall"] = {
        "num_records": sum(len(records) for records in latency_by_task.values()),
        "counts": {
            key: {"pass": value[0], "total": value[1]}
            for key, value in overall_counts.items()
        },
        "rates": {
            key: _rate(value[0], value[1])
            for key, value in overall_counts.items()
        },
    }
    return results


def print_table(results: Mapping[str, Dict[str, Any]], tasks: Iterable[str]) -> None:
    headers = [
        "Task",
        "N",
        "Strict TTFB",
        "Strict TPOB",
        "Strict All",
        "Relaxed TTFB",
        "Relaxed TPOB",
        "Relaxed All",
    ]
    rows: List[List[str]] = []
    for task in list(tasks) + ["overall"]:
        result = results.get(task)
        if result is None:
            continue
        rates = result.get("rates", {})
        rows.append(
            [
                task.upper(),
                str(result.get("num_records", 0)),
                _format_rate(rates.get("strict_ttfb")),
                _format_rate(rates.get("strict_tpob")),
                _format_rate(rates.get("strict_all")),
                _format_rate(rates.get("relaxed_ttfb")),
                _format_rate(rates.get("relaxed_tpob")),
                _format_rate(rates.get("relaxed_all")),
            ]
        )

    widths = [
        max(len(str(row[idx])) for row in [headers] + rows)
        for idx in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute strict/relaxed DLM SLO satisfaction rates."
    )
    parser.add_argument("--summary", type=Path, help="summary_*.json from dlm_benchmark.py")
    parser.add_argument(
        "--latency-dir",
        type=Path,
        help="Directory containing request_latency_<task>.jsonl files",
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), help="Tasks to evaluate")
    parser.add_argument(
        "--strict-factor",
        type=float,
        default=5.0,
        help="Default strict multiplier for built-in base TTFB/TPOB values",
    )
    parser.add_argument(
        "--relaxed-factor",
        type=float,
        default=25.0,
        help="Default relaxed multiplier for built-in base TTFB/TPOB values",
    )
    parser.add_argument(
        "--slo-config",
        type=Path,
        help="Optional JSON file overriding per-task strict/relaxed SLOs in ms",
    )
    parser.add_argument(
        "--missing",
        choices=("exclude", "fail"),
        default="exclude",
        help="How to handle records with missing TTFB/TPOB values",
    )
    parser.add_argument("--output-json", type=Path, help="Write detailed results to JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [task.lower() for task in args.tasks]
    slos = build_default_slos(args.strict_factor, args.relaxed_factor)
    if args.slo_config is not None:
        slos = _load_slo_config(args.slo_config, slos)

    latency_by_task = load_request_latency(args.summary, args.latency_dir, tasks)
    results = compute_slo_rates(latency_by_task, slos, missing=args.missing)
    print_table(results, tasks)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSLO results saved -> {args.output_json}")


if __name__ == "__main__":
    main()
