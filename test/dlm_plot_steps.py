#!/usr/bin/env python3
"""
저장된 step 데이터로 분포 그래프를 그립니다.

Usage:
    # summary JSON으로부터 (dlm_benchmark.py가 정상 완료된 경우)
    python test/dlm_plot_steps.py \
        --summary /tmp/dlm_results/summary_inclusionAI_LLaDA2.0-mini.json \
        --block-size 32

    # 개별 task step 파일로부터 (부분 실행된 경우)
    python test/dlm_plot_steps.py \
        --steps-dir /tmp/dlm_results \
        --tasks gsm8k humaneval math \
        --block-size 32
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_from_summary(summary_path: str) -> Dict[str, List[int]]:
    with open(summary_path) as f:
        data = json.load(f)
    return {task: steps for task, steps in data.get("step_data", {}).items() if steps}


def load_from_steps_dir(steps_dir: str, tasks: List[str]) -> Dict[str, List[int]]:
    step_data = {}
    for task in tasks:
        p = Path(steps_dir) / f"steps_{task}.jsonl"
        if p.exists():
            with open(p) as f:
                all_steps = []
                for line in f:
                    line = line.strip()
                    if line:
                        val = json.loads(line)
                        # flat list 또는 list of lists 모두 처리
                        if val and isinstance(val[0], list):
                            for row in val:
                                all_steps.extend(row)
                        else:
                            all_steps.extend(val)
            step_data[task] = all_steps
            print(f"[{task}] loaded {len(all_steps)} block steps from {p}")
        else:
            print(f"[{task}] {p} 없음 — 스킵")
    return step_data


def plot(step_data: Dict[str, List[int]], block_size: int,
         model_name: str, output_dir: str):

    tasks = list(step_data.keys())
    if not tasks:
        print("플롯할 데이터가 없습니다.")
        return

    colors = {"gsm8k": "#4C72B0", "humaneval": "#DD8452", "math": "#55A868"}

    # ── 히스토그램 ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), sharey=False)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        steps = step_data[task]
        bins = range(1, block_size + 2)
        counts, edges = np.histogram(steps, bins=bins)
        pct = counts / counts.sum() * 100

        ax.bar(edges[:-1], pct, width=0.8, align="center",
               color=colors.get(task, "#888"), edgecolor="white", linewidth=0.5)

        mean_val = np.mean(steps)
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.2,
                   label=f"mean={mean_val:.2f}")

        ax.set_title(task.upper(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Steps per block", fontsize=11)
        ax.set_ylabel("Blocks (%)", fontsize=11)
        ax.set_xticks(range(1, block_size + 1, max(1, block_size // 8)))
        ax.set_xlim(0.4, block_size + 0.6)
        ax.legend(fontsize=9)

        stats_txt = (f"n={len(steps)}\n"
                     f"mean={mean_val:.2f}\n"
                     f"median={np.median(steps):.1f}\n"
                     f"max={max(steps)}")
        ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
                fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    model_tag = model_name.replace("/", "_")
    fig.suptitle(f"Block-level unmask steps — {model_name}", fontsize=13)
    fig.tight_layout()
    out1 = Path(output_dir) / f"step_dist_{model_tag}.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"[plot] histogram → {out1}")
    plt.close(fig)

    # ── Box plot 비교 ─────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(max(5, 2.5 * len(tasks)), 4))
    data_list = [step_data[t] for t in tasks]
    bp = ax2.boxplot(data_list, labels=[t.upper() for t in tasks],
                     patch_artist=True, notch=False)
    for patch, task in zip(bp["boxes"], tasks):
        patch.set_facecolor(colors.get(task, "#888"))
        patch.set_alpha(0.7)

    ax2.set_ylabel("Steps per block", fontsize=11)
    ax2.set_title(f"Step distribution by task — {model_name}", fontsize=12)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax2.set_yticks(range(1, block_size + 1, max(1, block_size // 8)))

    out2 = Path(output_dir) / f"step_boxplot_{model_tag}.png"
    fig2.tight_layout()
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"[plot] boxplot    → {out2}")
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(
        description="DLM step 분포 시각화 (저장된 데이터 사용)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--summary", type=str, default=None,
                        help="dlm_benchmark.py가 생성한 summary JSON 경로")
    parser.add_argument("--steps-dir", type=str, default="/tmp/dlm_results",
                        help="steps_<task>.jsonl 파일들이 있는 디렉토리")
    parser.add_argument("--tasks", nargs="+",
                        default=["gsm8k", "humaneval", "math"],
                        help="--steps-dir 사용 시 로드할 task 목록")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--model-name", type=str, default="LLaDA2.0-mini")
    parser.add_argument("--output-dir", type=str, default="/tmp/dlm_results")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.summary:
        step_data = load_from_summary(args.summary)
        print(f"summary에서 로드: {list(step_data.keys())}")
    else:
        step_data = load_from_steps_dir(args.steps_dir, args.tasks)

    plot(step_data, block_size=args.block_size,
         model_name=args.model_name, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
