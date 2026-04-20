#!/usr/bin/env python3
"""
DLM inference benchmark: GSM8K, HumanEval, MATH
각 task별로 block당 실제 unmask step 수를 수집해 matplotlib으로 시각화.

Usage (기존 서버에 연결):
    python test/dlm_benchmark.py \
        --base-url http://localhost:30000 \
        --model inclusionAI/LLaDA2.0-mini \
        --tasks gsm8k humaneval math

Usage (서버 자동 실행):
    python test/dlm_benchmark.py \
        --model-path inclusionAI/LLaDA2.0-mini \
        --tasks gsm8k humaneval math \
        --tp-size 2 --num-examples 200

step_log_file 설정 예시 (config.yaml):
    threshold: 0.95
    step_log_file: /tmp/dlm_steps.jsonl
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

# ──────────────────────────────────────────────────────────────────────────────
# Step stats 수집
# ──────────────────────────────────────────────────────────────────────────────

STEP_LOG_FILE = "/tmp/dlm_step_stats.jsonl"


def clear_step_log():
    try:
        os.remove(STEP_LOG_FILE)
    except FileNotFoundError:
        pass


def read_step_log() -> List[int]:
    """JSONL에서 모든 block_steps를 flat list로 읽어 반환."""
    all_steps = []
    try:
        with open(STEP_LOG_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_steps.extend(json.loads(line))
    except FileNotFoundError:
        pass
    return all_steps


# ──────────────────────────────────────────────────────────────────────────────
# MATH local eval (GPT-4 없이 sympy 기반 채점)
# ──────────────────────────────────────────────────────────────────────────────

MATH_QUERY_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.""".strip()

_ANSWER_RE = re.compile(r"Answer\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _extract_answer(text: str) -> Optional[str]:
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def _math_equal(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    norm = lambda s: re.sub(r"\\left|\\right|\\!", "", s).replace(" ", "").replace(",", "").lower()
    if norm(pred) == norm(gold):
        return True
    try:
        import sympy
        return sympy.simplify(sympy.sympify(pred) - sympy.sympify(gold)) == 0
    except Exception:
        return False


class LocalMathEval:
    MATH_URL = "https://openaipublic.blob.core.windows.net/simple-evals/math_test.csv"

    def __init__(self, num_examples: Optional[int], num_threads: int, data_path: Optional[str] = None):
        import pandas
        filename = data_path or self.MATH_URL
        df = pandas.read_csv(filename, storage_options={"timeout": 60} if "://" in filename else {})
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, min(num_examples, len(examples)))
        self.examples = examples
        self.num_threads = num_threads

    def __call__(self, sampler) -> object:
        from sglang.test import simple_eval_common as common
        from sglang.test.simple_eval_common import HTML_JINJA, SingleEvalResult

        def fn(row: dict) -> SingleEvalResult:
            prompt_messages = [sampler._pack_message(
                content=MATH_QUERY_TEMPLATE.format(**row), role="user"
            )]
            try:
                response_text = sampler(prompt_messages) or ""
            except Exception:
                response_text = ""
            extracted = _extract_answer(response_text)
            gold = _extract_answer(str(row["Answer"])) or str(row["Answer"])
            score = float(_math_equal(extracted, gold))
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=gold,
                extracted_answer=extracted,
            )
            return SingleEvalResult(
                html=html, score=score,
                convo=prompt_messages + [dict(content=response_text, role="assistant")],
            )

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results, default_stats=("mean", "std"))


# ──────────────────────────────────────────────────────────────────────────────
# 서버 관리
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_PORT = 31900
DEFAULT_URL = f"http://localhost:{DEFAULT_PORT}"


def _wait_server_ready(base_url: str, timeout: int = 600) -> bool:
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def launch_server(args) -> subprocess.Popen:
    # step_log_file을 algorithm config에 넣기 위해 임시 yaml 생성
    config_path = "/tmp/dlm_algo_config.yaml"
    with open(config_path, "w") as f:
        f.write(f"threshold: {args.threshold}\n")
        f.write(f"step_log_file: {STEP_LOG_FILE}\n")

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model_path,
        "--port", str(DEFAULT_PORT),
        "--trust-remote-code",
        "--dllm-algorithm", "LowConfidence",
        "--dllm-algorithm-config", config_path,
        "--tp-size", str(args.tp_size),
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--max-running-requests", str(args.max_running_requests),
        "--attention-backend", "flashinfer",
    ]
    if args.disable_cuda_graph:
        cmd += ["--disable-cuda-graph"]
    print(f"[server] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    if not _wait_server_ready(DEFAULT_URL):
        proc.kill()
        raise RuntimeError("Server failed to start")
    print("[server] ready")
    return proc


# ──────────────────────────────────────────────────────────────────────────────
# 태스크 실행
# ──────────────────────────────────────────────────────────────────────────────

TASK_API = {"gsm8k": "completion", "humaneval": "chat", "math": "chat"}
TASK_MAX_TOKENS = {"gsm8k": 512, "humaneval": 512, "math": 1024}


def _make_sampler(task: str, base_url: str, model: str):
    from sglang.test.simple_eval_common import ChatCompletionSampler, CompletionSampler
    kw = dict(model=model, max_tokens=TASK_MAX_TOKENS[task],
              base_url=f"{base_url}/v1", temperature=0.0)
    if TASK_API[task] == "completion":
        return CompletionSampler(**kw, stop=["Question", "Assistant:", "<|separator|>"])
    # humaneval: chat API, stop token 없음
    return ChatCompletionSampler(**kw)


def run_task(task: str, base_url: str, model: str,
             num_examples: Optional[int], num_threads: int,
             gsm8k_data_path: Optional[str] = None,
             math_data_path: Optional[str] = None) -> Dict:
    sampler = _make_sampler(task, base_url, model)

    if task == "gsm8k":
        from sglang.test.simple_eval_gsm8k import GSM8KEval
        eval_obj = GSM8KEval(num_examples=num_examples, num_threads=num_threads,
                             num_shots=5, data_path=gsm8k_data_path)
    elif task == "humaneval":
        try:
            from sglang.test.simple_eval_humaneval import HumanEval
        except ImportError:
            print("[humaneval] human-eval 패키지 없음: pip install human-eval")
            return {"score": None, "error": "human-eval not installed"}
        he_total = 164  # HumanEval has exactly 164 problems
        he_examples = min(num_examples, he_total) if num_examples else None
        eval_obj = HumanEval(num_examples=he_examples, num_threads=num_threads,
                             num_samples_per_task=1, ks_passes=[1])
    elif task == "math":
        eval_obj = LocalMathEval(num_examples=num_examples, num_threads=num_threads,
                                 data_path=math_data_path)
    else:
        raise ValueError(f"Unknown task: {task}")

    tic = time.perf_counter()
    result = eval_obj(sampler)
    latency = time.perf_counter() - tic

    total_tokens = sum(sampler._completion_tokens)
    return {
        "score": result.score,
        "latency_s": round(latency, 2),
        "output_throughput_tok_s": round(total_tokens / latency if latency > 0 else 0, 2),
        "total_completion_tokens": total_tokens,
        **(result.metrics or {}),
    }


# ──────────────────────────────────────────────────────────────────────────────
# matplotlib 시각화
# ──────────────────────────────────────────────────────────────────────────────

def plot_step_distributions(
    step_data: Dict[str, List[int]],
    block_size: int,
    model_name: str,
    output_dir: str,
):
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = [t for t, steps in step_data.items() if steps]
    if not tasks:
        print("[plot] step 데이터 없음 (step_log_file 설정 확인)")
        return

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), sharey=False)
    if len(tasks) == 1:
        axes = [axes]

    colors = {"gsm8k": "#4C72B0", "humaneval": "#DD8452", "math": "#55A868"}

    for ax, task in zip(axes, tasks):
        steps = step_data[task]
        bins = range(1, block_size + 2)  # 1 ~ block_size

        counts, edges = np.histogram(steps, bins=bins)
        pct = counts / counts.sum() * 100

        ax.bar(edges[:-1], pct, width=0.8, align="center",
               color=colors.get(task, "#777"), edgecolor="white", linewidth=0.5)

        mean_val = np.mean(steps)
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.2,
                   label=f"mean={mean_val:.2f}")

        ax.set_title(task.upper(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Steps per block", fontsize=11)
        ax.set_ylabel("Blocks (%)", fontsize=11)
        ax.set_xticks(range(1, block_size + 1))
        ax.set_xlim(0.4, block_size + 0.6)
        ax.legend(fontsize=9)

        # 통계 텍스트
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

    out_path = Path(output_dir) / f"step_dist_{model_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path}")
    plt.close(fig)

    # task 간 비교 (box plot)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    data_list = [step_data[t] for t in tasks]
    bp = ax2.boxplot(data_list, labels=[t.upper() for t in tasks],
                     patch_artist=True, notch=False)
    for patch, task in zip(bp["boxes"], tasks):
        patch.set_facecolor(colors.get(task, "#777"))
        patch.set_alpha(0.7)

    ax2.set_ylabel("Steps per block", fontsize=11)
    ax2.set_title(f"Step distribution by task — {model_name}", fontsize=12)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax2.set_yticks(range(1, block_size + 1))

    out_path2 = Path(output_dir) / f"step_boxplot_{model_tag}.png"
    fig2.tight_layout()
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path2}")
    plt.close(fig2)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DLM benchmark (GSM8K / HumanEval / MATH) + step 분포 시각화",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 서버 연결
    parser.add_argument("--base-url", type=str, default=None,
                        help="기존 서버 URL (예: http://localhost:30000)")
    parser.add_argument("--model", type=str, default=None,
                        help="--base-url 사용 시 모델 이름")

    # 서버 자동 실행
    parser.add_argument("--model-path", type=str, default=None,
                        help="HuggingFace 모델 경로 → 서버 자동 실행")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.88)
    parser.add_argument("--max-running-requests", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="LowConfidence unmasking threshold")
    parser.add_argument("--disable-cuda-graph", action="store_true",
                        help="서버 실행 시 --disable-cuda-graph 전달 (sm_80 등 미지원 GPU용)")

    # 벤치마크
    parser.add_argument("--tasks", nargs="+",
                        choices=["gsm8k", "humaneval", "math"],
                        default=["gsm8k", "humaneval", "math"])
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=64)
    parser.add_argument("--gsm8k-data-path", type=str, default=None)
    parser.add_argument("--math-data-path", type=str, default=None)

    # 출력
    parser.add_argument("--output-dir", type=str, default="/tmp/dlm_results")
    parser.add_argument("--block-size", type=int, default=None,
                        help="모델의 block_size (LLaDA2=32, SDAR=4). "
                             "미지정 시 step 축 범위를 자동 추정")

    args = parser.parse_args()

    if args.base_url is None and args.model_path is None:
        parser.error("--base-url 또는 --model-path 중 하나 필요")
    if args.base_url and args.model is None:
        parser.error("--base-url 사용 시 --model 필요")

    # step_log_file이 서버에 기록되려면,
    # 기존 서버는 반드시 step_log_file 설정된 config로 실행된 상태여야 함.
    # 자동 실행 시 launch_server()가 config yaml을 자동 생성.
    if args.base_url:
        print(f"[warn] 기존 서버 사용 시 서버가 step_log_file={STEP_LOG_FILE} 로 실행됐는지 확인하세요.")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    server_proc = None
    base_url = args.base_url
    model = args.model

    try:
        if args.model_path:
            server_proc = launch_server(args)
            base_url = DEFAULT_URL
            model = args.model_path

        all_results: Dict[str, Dict] = {}
        step_data: Dict[str, List[int]] = {}

        for task in args.tasks:
            print(f"\n{'='*60}")
            print(f"Task: {task.upper()}  |  model: {model}")
            print(f"{'='*60}")

            clear_step_log()
            # 이번 task의 raw step 데이터를 별도 파일에도 저장
            task_step_file = Path(args.output_dir) / f"steps_{task}.jsonl"
            if task_step_file.exists():
                task_step_file.unlink()

            metrics = run_task(
                task=task, base_url=base_url, model=model,
                num_examples=args.num_examples, num_threads=args.num_threads,
                gsm8k_data_path=args.gsm8k_data_path,
                math_data_path=args.math_data_path,
            )
            all_results[task] = metrics

            steps = read_step_log()
            step_data[task] = steps
            # task별 raw step 파일에 복사 (plot 스크립트가 나중에 읽을 수 있도록)
            if steps:
                task_step_file = Path(args.output_dir) / f"steps_{task}.jsonl"
                with open(task_step_file, "w") as f:
                    f.write(json.dumps(steps) + "\n")

            score = metrics.get("score")
            score_str = f"{score:.4f}" if score is not None else "N/A"
            print(f"\n[{task}] score={score_str}  "
                  f"throughput={metrics.get('output_throughput_tok_s', 0):.1f} tok/s")
            if steps:
                import numpy as np
                print(f"[{task}] step stats: n={len(steps)}  "
                      f"mean={np.mean(steps):.2f}  "
                      f"median={np.median(steps):.1f}  "
                      f"max={max(steps)}")

            # task별 결과 저장
            model_tag = model.replace("/", "_")
            out_path = Path(args.output_dir) / f"{task}_{model_tag}.json"
            with open(out_path, "w") as f:
                json.dump({"task": task, "model": model,
                           "step_stats": {
                               "n_blocks": len(steps),
                               "mean": float(sum(steps) / len(steps)) if steps else None,
                               "max": max(steps) if steps else None,
                           },
                           **metrics}, f, indent=2)

        # ── 요약 출력 ────────────────────────────────────────────
        print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
        for task, m in all_results.items():
            s = m.get("score")
            print(f"  {task:<12} score={f'{s:.4f}' if s is not None else 'N/A'}"
                  f"  throughput={m.get('output_throughput_tok_s', 0):.1f} tok/s")

        # ── 전체 결과 저장 (plot 전에 먼저) ──────────────────────────
        summary_path = Path(args.output_dir) / f"summary_{model.replace('/', '_')}.json"
        with open(summary_path, "w") as f:
            json.dump({"model": model, "results": all_results,
                       "step_data": {t: v for t, v in step_data.items()}}, f, indent=2)
        print(f"Summary saved → {summary_path}")

        # ── 플롯 ─────────────────────────────────────────────────
        block_size = args.block_size
        if block_size is None:
            all_steps_flat = [s for v in step_data.values() for s in v]
            block_size = max(all_steps_flat) if all_steps_flat else 32

        plot_step_distributions(step_data, block_size=block_size,
                                model_name=model, output_dir=args.output_dir)

    finally:
        if server_proc is not None:
            print("\n[server] shutting down...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
