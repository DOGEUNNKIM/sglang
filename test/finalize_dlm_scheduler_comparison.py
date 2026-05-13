#!/usr/bin/env python3
"""Finalize plots/summaries from completed DLM scheduler comparison outputs.

This script does not launch servers or run benchmarks. It scans an existing
OUTPUT_ROOT and regenerates artifacts from whatever task/rate/scheduler runs
already finished.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _num_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def _read_metric(path: Path, field: str) -> float | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    value = data.get(field)
    if value is None:
        value = (data.get("latency_stats") or {}).get(field)
    if value is None:
        return None
    return float(value)


def _read_bench_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    latency = data.get("latency_stats") or {}
    return {
        "score": data.get("score"),
        "score_std": data.get("score:std"),
        "throughput_tok_s": data.get("output_throughput_tok_s"),
        "p99_ttfb_ms": latency.get("p99_ttfb_ms"),
        "p99_tpob_ms": latency.get("p99_tpob_ms"),
    }


def discover_runs(output_root: Path, model_tag: str) -> dict[tuple[str, str, str], Path]:
    runs: dict[tuple[str, str, str], Path] = {}
    suffix = f"_{model_tag}.json"
    for path in output_root.glob(f"scheduler_*/request_rate_*/*{suffix}"):
        if path.name.startswith("summary_"):
            continue
        try:
            scheduler = path.parents[1].name.removeprefix("scheduler_")
            rate = path.parent.name.removeprefix("request_rate_")
        except IndexError:
            continue
        task = path.name[: -len(suffix)]
        runs[(scheduler, rate, task)] = path.parent
    return runs


def build_slo_config(
    output_root: Path,
    runs: dict[tuple[str, str, str], Path],
    model_tag: str,
    strict_multiplier: float,
    release_multiplier: float,
) -> dict[str, Any]:
    tasks = sorted({task for _, _, task in runs})
    slos: dict[str, Any] = {}
    for task in tasks:
        ttfb_candidates: list[tuple[tuple[int, float | str], float]] = []
        tpob_candidates: list[tuple[tuple[int, float | str], float]] = []
        for scheduler, rate, run_task in runs:
            if run_task != task:
                continue
            json_path = (
                output_root
                / f"scheduler_{scheduler}"
                / f"request_rate_{rate}"
                / f"{task}_{model_tag}.json"
            )
            if scheduler == "TTFB":
                value = _read_metric(json_path, "p50_ideal_ttfb_ms")
                if value is not None:
                    ttfb_candidates.append((_num_key(rate), value))
            if scheduler == "DECODE":
                value = _read_metric(json_path, "p50_ideal_tpob_ms")
                if value is not None:
                    tpob_candidates.append((_num_key(rate), value))
        if not ttfb_candidates or not tpob_candidates:
            continue
        ideal_ttfb = sorted(ttfb_candidates, key=lambda x: x[0])[-1][1]
        ideal_tpob = sorted(tpob_candidates, key=lambda x: x[0])[-1][1]
        slos[task] = {
            "strict": {
                "ttfb_ms": ideal_ttfb * strict_multiplier,
                "tpob_ms": ideal_tpob * strict_multiplier,
            },
            "relaxed": {
                "ttfb_ms": ideal_ttfb * release_multiplier,
                "tpob_ms": ideal_tpob * release_multiplier,
            },
        }
    return slos


def run_cmd(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/nvme0/kdg6245/dlm_sched_comparison_LLADA2"),
    )
    parser.add_argument("--model-path", default="inclusionAI/LLaDA2.0-mini")
    parser.add_argument("--strict-multiplier", type=float, default=10.0)
    parser.add_argument("--release-multiplier", type=float, default=20.0)
    parser.add_argument(
        "--step-plots",
        action="store_true",
        help="Generate per-run step_dist_*.png files under scheduler_*/request_rate_*.",
    )
    parser.add_argument("--skip-summary-plots", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_tag = args.model_path.replace("/", "_")
    output_root = args.output_root
    runs = discover_runs(output_root, model_tag)
    if not runs:
        raise SystemExit(f"No completed runs found under {output_root}")

    print(f"[discover] found {len(runs)} completed task runs")

    slo_config = build_slo_config(
        output_root,
        runs,
        model_tag,
        args.strict_multiplier,
        args.release_multiplier,
    )
    slo_config_path = output_root / "slo_config.json"
    slo_config_path.write_text(json.dumps(slo_config, indent=2))
    print(f"[slo_config] written to {slo_config_path} ({len(slo_config)} tasks)")

    tasks_by_out_dir: dict[Path, list[str]] = {}
    for (_, _, task), out_dir in sorted(runs.items()):
        if task not in slo_config:
            continue
        if not (out_dir / f"request_latency_{task}.jsonl").exists():
            continue
        tasks_by_out_dir.setdefault(out_dir, []).append(task)

    for out_dir, tasks in sorted(tasks_by_out_dir.items(), key=lambda item: str(item[0])):
        run_cmd(
            [
                sys.executable,
                str(repo_root / "test" / "dlm_slorate.py"),
                "--latency-dir",
                str(out_dir),
                "--tasks",
                *sorted(tasks),
                "--slo-config",
                str(slo_config_path),
                "--output-json",
                str(out_dir / "slo_rates.json"),
            ]
        )

    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for (scheduler, rate, task), out_dir in sorted(runs.items()):
        slo_path = out_dir / "slo_rates.json"
        rates_d: dict[str, Any] = {}
        if slo_path.exists():
            try:
                rates_d = (
                    json.loads(slo_path.read_text()).get(task, {}).get("rates", {})
                )
            except Exception:
                rates_d = {}
        bench_path = out_dir / f"{task}_{model_tag}.json"
        summary.setdefault(scheduler, {}).setdefault(rate, {})[task] = {
            **_read_bench_metrics(bench_path),
            "strict_ttfb": rates_d.get("strict_ttfb"),
            "strict_tpob": rates_d.get("strict_tpob"),
            "strict_all": rates_d.get("strict_all"),
            "relaxed_ttfb": rates_d.get("relaxed_ttfb"),
            "relaxed_tpob": rates_d.get("relaxed_tpob"),
            "relaxed_all": rates_d.get("relaxed_all"),
        }

    summary_path = output_root / "slo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[summary] written to {summary_path}")

    if args.step_plots:
        for (_, _, task), out_dir in sorted(runs.items()):
            step_path = out_dir / f"step_stats_{task}.jsonl"
            if not step_path.exists():
                continue
            run_cmd(
                [
                    sys.executable,
                    str(repo_root / "test" / "plot_step_dist.py"),
                    "--log-dir",
                    str(out_dir),
                    "--tasks",
                    task,
                    "--output",
                    str(out_dir / f"step_dist_{model_tag}_{task}.png"),
                ]
            )

    if not args.skip_summary_plots:
        run_cmd(
            [
                sys.executable,
                str(repo_root / "test" / "plot_dlm_slo_summary.py"),
                "--summary",
                str(summary_path),
                "--output",
                str(output_root / "slo_attainment_comparison.png"),
                "--slo-config",
                str(slo_config_path),
                "--p99-normalize-baseline",
                "",
                "--no-scatter",
                "--no-bar",
            ]
        )

    print(f"[done] finalized existing outputs under {output_root}")


if __name__ == "__main__":
    main()
