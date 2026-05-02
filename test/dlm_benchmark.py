#!/usr/bin/env python3
"""
DLM inference benchmark: GSM8K, HumanEval, MATH
각 task별로 block당 실제 unmask step 수를 수집해 matplotlib으로 시각화.

Usage (서버 실행):
    python -m sglang.launch_server \
    --model-path inclusionAI/LLaDA2.0-mini \
    --port 30001 \
    --trust-remote-code \
    --dllm-algorithm LowConfidence \
    --dllm-algorithm-config /tmp/dlm_algo_config.yaml \
    --attention-backend flashinfer \
    --max-running-requests 16 \
    --cuda-graph-max-bs 16 \
    --disable-cuda-graph-padding

Usage (서버 이용한 test):
    SLO_MULTIPLIER=5.0 \
    FORWARD_TIME_S=0.03 \
    python test/dlm_benchmark.py \
            --base-url http://localhost:30001 \
            --model inclusionAI/LLaDA2.0-mini \
            --tasks gsm8k humaneval math \
            --num-examples 200 \
            --num-threads 200 \
            --warmup 16 \
            --num-output-blocks 0\
            --block-size 32\
            --request-rate 8\
            --log

Usage (서버 자동 실행):
    python test/dlm_benchmark.py \
        --model-path inclusionAI/LLaDA2.0-mini \
        --tasks gsm8k humaneval math \
        --tp-size 2 --num-examples 200 \
        --num-threads 200

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
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

# ──────────────────────────────────────────────────────────────────────────────
# Step stats 수집
# ──────────────────────────────────────────────────────────────────────────────

STEP_LOG_FILE = "/tmp/dlm_step_stats.jsonl"
REQUEST_LATENCY_LOG_FILE = "/tmp/dlm_request_latency.jsonl"
BATCH_LATENCY_LOG_FILE = "/tmp/dlm_batch_latency.jsonl"


def clear_step_log():
    try:
        os.remove(STEP_LOG_FILE)
    except FileNotFoundError:
        pass


def clear_latency_logs():
    for path in (REQUEST_LATENCY_LOG_FILE, BATCH_LATENCY_LOG_FILE):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        pass
    return records


def read_latency_logs() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "request": _read_jsonl(REQUEST_LATENCY_LOG_FILE),
        "batch": _read_jsonl(BATCH_LATENCY_LOG_FILE),
    }


def _flatten_steps(values) -> List[int]:
    flat_steps: List[int] = []
    if values is None:
        return flat_steps
    if not isinstance(values, list):
        return [int(values)]
    for value in values:
        if isinstance(value, list):
            flat_steps.extend(_flatten_steps(value))
        else:
            flat_steps.append(int(value))
    return flat_steps


def _normalize_step_record(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        raw_forward_calls = raw.get("raw_forward_calls")
        if "final_block_steps" in raw:
            block_steps = _flatten_steps(raw.get("final_block_steps", []))
        elif "kv_saved" in raw and "block_steps" in raw:
            steps = _flatten_steps(raw.get("block_steps", []))
            saved = raw.get("kv_saved", [])
            block_steps = [
                int(step)
                for step, is_saved in zip(steps, saved)
                if bool(is_saved)
            ]
        else:
            block_steps = _flatten_steps(raw.get("block_steps", []))
    else:
        raw_forward_calls = None
        block_steps = _flatten_steps(raw)
    return {
        "raw_forward_calls": (
            int(raw_forward_calls) if raw_forward_calls is not None else None
        ),
        "block_steps": block_steps,
    }


def read_step_log() -> Dict[str, Any]:
    """JSONL에서 raw forward calls, block steps, forward duration을 읽어 반환."""
    step_records = []
    raw_forward_calls = []
    block_steps = []
    forward_durations_ms = []
    try:
        with open(STEP_LOG_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    raw = json.loads(line)
                    record = _normalize_step_record(raw)
                    step_records.append(record)
                    if record["raw_forward_calls"] is not None:
                        raw_forward_calls.append(record["raw_forward_calls"])
                    block_steps.extend(record["block_steps"])
                    if isinstance(raw, dict) and raw.get("forward_duration_ms") is not None:
                        forward_durations_ms.append(float(raw["forward_duration_ms"]))
    except FileNotFoundError:
        pass
    return {
        "records": step_records,
        "raw_forward_calls": raw_forward_calls,
        "block_steps": block_steps,
        "forward_durations_ms": forward_durations_ms,
    }


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
    config_path = "/tmp/dlm_algo_config.yaml"
    with open(config_path, "w") as f:
        f.write(f"threshold: {args.threshold}\n")
        if args.log:
            f.write(f"step_log_file: {STEP_LOG_FILE}\n")
            f.write(f"request_latency_log_file: {REQUEST_LATENCY_LOG_FILE}\n")
            f.write(f"batch_latency_log_file: {BATCH_LATENCY_LOG_FILE}\n")

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
# Cache flush & Warmup
# ──────────────────────────────────────────────────────────────────────────────

def flush_server_cache(base_url: str, timeout: float = 30.0) -> bool:
    """Call /flush_cache to release fragmented GPU memory between tasks."""
    import requests as req_lib
    try:
        resp = req_lib.post(f"{base_url}/flush_cache", timeout=timeout)
        if resp.status_code == 200:
            print(f"[flush_cache] {resp.text.strip()}")
            return True
        print(f"[flush_cache] failed ({resp.status_code}): {resp.text.strip()}")
        return False
    except Exception as e:
        print(f"[flush_cache] error: {e}")
        return False

_WARMUP_MAX_TOKENS = 2048
# Intentionally minimal prompt so seq_len starts near 0 and sweeps through
# all FlashInfer compilation buckets (64, 128, 256, 512, 1024, 2048) during
# generation, rather than skipping low buckets due to a long prompt.
_WARMUP_PROMPT = "Hi"


def run_warmup(base_url: str, model: str, num_requests: int = 4) -> None:
    """Send dummy requests with the largest expected seq_len to compile all
    Triton kernel shapes before benchmarking starts."""
    if num_requests <= 0:
        return
    import concurrent.futures
    from sglang.test.simple_eval_common import ChatCompletionSampler
    sampler = ChatCompletionSampler(
        model=model,
        max_tokens=_WARMUP_MAX_TOKENS,
        base_url=f"{base_url}/v1",
        temperature=0.0,
    )
    prompt = [{"role": "user", "content": _WARMUP_PROMPT}]
    print(f"[warmup] sending {num_requests} dummy requests (max_tokens={_WARMUP_MAX_TOKENS})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as ex:
        futs = [ex.submit(sampler, prompt) for _ in range(num_requests)]
        concurrent.futures.wait(futs)
    print("[warmup] done")


# ──────────────────────────────────────────────────────────────────────────────
# 태스크 실행
# ──────────────────────────────────────────────────────────────────────────────

TASK_API = {"gsm8k": "completion", "humaneval": "chat", "math": "chat"}
TASK_MAX_TOKENS = {"gsm8k": 512, "humaneval": 512, "math": 1024}


def _make_sampler(task: str, base_url: str, model: str,
                  num_output_blocks: int = 0, block_size: int = 32):
    from sglang.test.simple_eval_common import ChatCompletionSampler, CompletionSampler
    if num_output_blocks > 0:
        max_tokens = num_output_blocks * block_size
        extra_body = {"ignore_eos": True}
    else:
        max_tokens = TASK_MAX_TOKENS[task]
        extra_body = None
    kw = dict(model=model, max_tokens=max_tokens,
              base_url=f"{base_url}/v1", temperature=0.0)
    if TASK_API[task] == "completion":
        return CompletionSampler(**kw, stop=["Question", "Assistant:", "<|separator|>"],
                                 extra_body=extra_body)
    return ChatCompletionSampler(**kw, extra_body=extra_body)


class RateLimitedSampler:
    """Throttle request start times across all worker threads."""

    def __init__(self, sampler, request_rate: float):
        self.sampler = sampler
        self.interval_s = 1.0 / request_rate
        self.lock = threading.Lock()
        self.next_start_time = time.perf_counter()

    def __getattr__(self, name):
        return getattr(self.sampler, name)

    def __call__(self, *args, **kwargs):
        with self.lock:
            now = time.perf_counter()
            start_time = max(now, self.next_start_time)
            self.next_start_time = start_time + self.interval_s

        delay = start_time - now
        if delay > 0:
            time.sleep(delay)
        return self.sampler(*args, **kwargs)


def run_task(task: str, base_url: str, model: str,
             num_examples: Optional[int], num_threads: int,
             gsm8k_data_path: Optional[str] = None,
             math_data_path: Optional[str] = None,
             request_rate: Optional[float] = None,
             num_output_blocks: int = 0,
             block_size: int = 32) -> Dict:
    sampler = _make_sampler(task, base_url, model,
                            num_output_blocks=num_output_blocks,
                            block_size=block_size)
    if request_rate is not None:
        sampler = RateLimitedSampler(sampler, request_rate)

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
        "request_rate": request_rate,
        "output_throughput_tok_s": round(total_tokens / latency if latency > 0 else 0, 2),
        "total_completion_tokens": total_tokens,
        **(result.metrics or {}),
    }


# ──────────────────────────────────────────────────────────────────────────────
# matplotlib 시각화
# ──────────────────────────────────────────────────────────────────────────────

def plot_step_distributions(
    step_data: Dict[str, Dict[str, List[int]]],
    block_size: int,
    model_name: str,
    output_dir: str,
):
    import matplotlib.pyplot as plt
    import numpy as np

    def _hist_pct(values: List[int], upper: int):
        bins = range(1, upper + 2)
        counts, edges = np.histogram(values, bins=bins)
        pct = counts / counts.sum() * 100 if counts.sum() else counts
        return pct, edges

    def _stats_text(values: List[int]) -> str:
        return (
            f"n={len(values)}\n"
            f"mean={np.mean(values):.2f}\n"
            f"median={np.median(values):.1f}\n"
            f"max={max(values)}"
        )

    def _tick_step(upper: int) -> int:
        return max(1, upper // 8)

    tasks = [
        task
        for task, metrics in step_data.items()
        if metrics.get("raw_forward_calls") or metrics.get("block_steps")
    ]
    if not tasks:
        print("[plot] step 데이터 없음 (step_log_file 설정 확인)")
        return

    forward_upper = max(
        [
            max(step_data[t]["raw_forward_calls"])
            for t in tasks
            if step_data[t]["raw_forward_calls"]
        ],
        default=block_size,
    )
    block_upper = max(
        block_size,
        max(
            [max(step_data[t]["block_steps"]) for t in tasks if step_data[t]["block_steps"]],
            default=block_size,
        ),
    )

    fig, axes = plt.subplots(2, len(tasks), figsize=(5 * len(tasks), 8), sharey=False)
    if len(tasks) == 1:
        axes = [[axes[0]], [axes[1]]]

    colors = {"gsm8k": "#4C72B0", "humaneval": "#DD8452", "math": "#55A868"}

    for idx, task in enumerate(tasks):
        raw_forward_calls = step_data[task].get("raw_forward_calls", [])
        block_steps = step_data[task].get("block_steps", [])
        color = colors.get(task, "#777")

        forward_ax = axes[0][idx]
        block_ax = axes[1][idx]

        if raw_forward_calls:
            pct, edges = _hist_pct(raw_forward_calls, forward_upper)
            forward_ax.bar(
                edges[:-1],
                pct,
                width=0.8,
                align="center",
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
            mean_val = np.mean(raw_forward_calls)
            forward_ax.axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=1.2,
                label=f"mean={mean_val:.2f}",
            )
            forward_ax.text(
                0.97,
                0.97,
                _stats_text(raw_forward_calls),
                transform=forward_ax.transAxes,
                fontsize=8,
                va="top",
                ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )
            forward_ax.legend(fontsize=9)
        else:
            forward_ax.text(0.5, 0.5, "no forward-call data", ha="center", va="center")

        forward_ax.set_title(
            f"{task.upper()} — forward calls", fontsize=13, fontweight="bold"
        )
        forward_ax.set_xlabel("Raw forward calls per run", fontsize=11)
        forward_ax.set_ylabel("Runs (%)", fontsize=11)
        forward_ax.set_xticks(range(1, forward_upper + 1, _tick_step(forward_upper)))
        forward_ax.set_xlim(0.4, forward_upper + 0.6)

        if block_steps:
            pct, edges = _hist_pct(block_steps, block_upper)
            block_ax.bar(
                edges[:-1],
                pct,
                width=0.8,
                align="center",
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
            mean_val = np.mean(block_steps)
            block_ax.axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=1.2,
                label=f"mean={mean_val:.2f}",
            )
            block_ax.text(
                0.97,
                0.97,
                _stats_text(block_steps),
                transform=block_ax.transAxes,
                fontsize=8,
                va="top",
                ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )
            block_ax.legend(fontsize=9)
        else:
            block_ax.text(0.5, 0.5, "no block-step data", ha="center", va="center")

        block_ax.set_title(
            f"{task.upper()} — block steps", fontsize=13, fontweight="bold"
        )
        block_ax.set_xlabel("Steps per block", fontsize=11)
        block_ax.set_ylabel("Blocks (%)", fontsize=11)
        block_ax.set_xticks(range(1, block_upper + 1, _tick_step(block_upper)))
        block_ax.set_xlim(0.4, block_upper + 0.6)

    model_tag = model_name.replace("/", "_")
    fig.suptitle(
        f"Raw forward calls per run and unmask steps per block — {model_name}",
        fontsize=13,
    )
    fig.tight_layout()

    out_path = Path(output_dir) / f"step_dist_{model_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path}")
    plt.close(fig)

    # task 간 비교 (box plot)
    fig2, axes2 = plt.subplots(1, 2, figsize=(max(8, 2.5 * len(tasks) + 3), 4))
    forward_tasks = [task for task in tasks if step_data[task].get("raw_forward_calls")]
    block_tasks = [task for task in tasks if step_data[task].get("block_steps")]

    if forward_tasks:
        forward_bp = axes2[0].boxplot(
            [step_data[t]["raw_forward_calls"] for t in forward_tasks],
            tick_labels=[t.upper() for t in forward_tasks],
            patch_artist=True,
            notch=False,
        )
        for patch, task in zip(forward_bp["boxes"], forward_tasks):
            patch.set_facecolor(colors.get(task, "#777"))
            patch.set_alpha(0.7)
        axes2[0].set_ylabel("Raw forward calls per run", fontsize=11)
        axes2[0].set_title("Forward-call distribution by task", fontsize=12)
        axes2[0].yaxis.grid(True, linestyle="--", alpha=0.6)
    else:
        axes2[0].text(0.5, 0.5, "no forward-call data", ha="center", va="center")
        axes2[0].set_axis_off()

    if block_tasks:
        block_bp = axes2[1].boxplot(
            [step_data[t]["block_steps"] for t in block_tasks],
            tick_labels=[t.upper() for t in block_tasks],
            patch_artist=True,
            notch=False,
        )
        for patch, task in zip(block_bp["boxes"], block_tasks):
            patch.set_facecolor(colors.get(task, "#777"))
            patch.set_alpha(0.7)
        axes2[1].set_ylabel("Steps per block", fontsize=11)
        axes2[1].set_title("Block-step distribution by task", fontsize=12)
        axes2[1].yaxis.grid(True, linestyle="--", alpha=0.6)
        axes2[1].set_yticks(range(1, block_upper + 1, _tick_step(block_upper)))
    else:
        axes2[1].text(0.5, 0.5, "no block-step data", ha="center", va="center")
        axes2[1].set_axis_off()

    out_path2 = Path(output_dir) / f"step_boxplot_{model_tag}.png"
    fig2.tight_layout()
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path2}")
    plt.close(fig2)


def _values(records: List[Dict[str, Any]], key: str) -> List[float]:
    return [
        float(record[key])
        for record in records
        if record.get(key) is not None
    ]


def _read_dllm_slo_config(config_path: Optional[str] = None) -> Dict[str, Optional[float]]:
    """Read active DLM SLOs from the YAML-like config written by the sweep script."""
    config_path = config_path or os.environ.get("CONFIG_PATH", "/tmp/dlm_algo_config.yaml")
    slo = {"ttfb_slo_ms": None, "tpob_slo_ms": None}
    try:
        with open(config_path) as f:
            for line in f:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key == "ttfb_slo":
                    slo["ttfb_slo_ms"] = float(value) * 1000.0
                elif key == "tpob_slo":
                    slo["tpob_slo_ms"] = float(value) * 1000.0
    except FileNotFoundError:
        pass
    except ValueError as e:
        print(f"[warn] failed to parse DLM SLO config {config_path}: {e}")
    return slo


def _unique_request_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop duplicate request latency records emitted by multiple TP ranks."""
    unique = []
    seen = set()
    for record in records:
        rid = record.get("rid")
        key = rid if rid is not None else id(record)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    import numpy as np

    return float(np.percentile(values, q))


def _ideal_ttfb_values(records: List[Dict[str, Any]]) -> List[float]:
    values = []
    for record in records:
        ttfb = record.get("ttfb_ms")
        sched_wait = record.get("sched_wait_ms")
        first_unmask_gap = record.get("first_unmask_gap_ms")
        if ttfb is None or sched_wait is None or first_unmask_gap is None:
            continue
        values.append(float(ttfb) - float(sched_wait) - float(first_unmask_gap))
    return values


def _ideal_tpob_values(records: List[Dict[str, Any]]) -> List[float]:
    values = []
    for record in records:
        tpob = record.get("tpob_ms")
        decode_delay = record.get("decode_delay_ms")
        if tpob is None or decode_delay is None:
            continue
        values.append(float(tpob) - float(decode_delay))
    return values


def _prefill_block_counts_by_request(
    records: List[Dict[str, Any]],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        rids = record.get("rids") or []
        modes = record.get("per_req_mode") or []
        for rid, mode in zip(rids, modes):
            if rid not in counts:
                counts[rid] = 0
            if mode == "prefill":
                counts[rid] += 1
    return counts


def _weighted_batch_mean(
    records: List[Dict[str, Any]],
    phase: str,
    value_key: str,
    weight_key: str,
) -> Optional[float]:
    total = 0.0
    weight_sum = 0
    for record in records:
        if record.get("phase") != phase:
            continue
        value = record.get(value_key)
        weight = int(record.get(weight_key) or 0)
        if value is None or weight <= 0:
            continue
        total += float(value) * weight
        weight_sum += weight
    return total / weight_sum if weight_sum > 0 else None


REQ_PHASE_LABELS = {
    "queuing_prefill": "queuing prefill",
    "staging_prefill": "staging prefill",
    "queuing_decode": "queuing decode",
    "staging_decode": "staging decode",
}


def _weighted_mean_by_count(
    records: List[Dict[str, Any]],
    value_key: str,
    count_key: str,
) -> Optional[float]:
    total = 0.0
    count_sum = 0
    for record in records:
        value = record.get(value_key)
        count = int(record.get(count_key) or 0)
        if value is None or count <= 0:
            continue
        total += float(value) * count
        count_sum += count
    return total / count_sum if count_sum > 0 else None


def _count_per_req_phases(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {phase: 0 for phase in REQ_PHASE_LABELS}
    for record in records:
        for phase in record.get("per_req_phase") or []:
            if phase in counts:
                counts[phase] += 1
            elif phase is not None:
                counts[phase] = counts.get(phase, 0) + 1
    return counts


def _batch_visual_phase(record: Dict[str, Any]) -> str:
    if record.get("phase") == "mixed_prefill_decode":
        return "mixed_prefill_decode"
    phases = set(record.get("per_req_phase") or [])
    has_mask = record.get("per_req_has_mask") or []
    if has_mask and any(has_mask) and any(not value for value in has_mask):
        return "mixed"
    if has_mask and any(has_mask):
        return "decode"
    if "queuing_prefill" in phases:
        return "initial_prefill"
    if "staging_prefill" in phases:
        return "staging_prefill"
    return record.get("phase", "unknown")


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def print_latency_summary(task: str, summary: Dict[str, Any]) -> None:
    rows = [
        (
            "Wait/Queue",
            [
                ("sched_wait mean", "mean_sched_wait_ms"),
                ("sched_wait p95", "p95_sched_wait_ms"),
                ("decode_delay mean", "mean_decode_delay_ms"),
                ("decode_delay p95", "p95_decode_delay_ms"),
                ("first_unmask_gap mean", "mean_first_unmask_gap_ms"),
                ("first_unmask_gap p95", "p95_first_unmask_gap_ms"),
            ],
        ),
        (
            "Observed",
            [
                ("TTFB mean", "mean_ttfb_ms"),
                ("TTFB p95", "p95_ttfb_ms"),
                ("TTFB p99", "p99_ttfb_ms"),
                ("TPOB mean", "mean_tpob_ms"),
                ("TPOB p95", "p95_tpob_ms"),
                ("TPOB p99", "p99_tpob_ms"),
            ],
        ),
        (
            "Ideal",
            [
                ("ideal TTFB mean", "mean_ideal_ttfb_ms"),
                ("ideal TTFB p95", "p95_ideal_ttfb_ms"),
                ("ideal TPOB mean", "mean_ideal_tpob_ms"),
                ("ideal TPOB p95", "p95_ideal_tpob_ms"),
            ],
        ),
        (
            "Blocks/Batches",
            [
                ("generated blocks mean", "mean_generated_blocks"),
                ("prefill blocks mean", "mean_prefill_blocks"),
                ("prefill req ms", "avg_prefill_req_ms"),
                ("decode block ms", "avg_decode_block_ms"),
                ("initial prefill ms", "avg_initial_prefill_req_ms"),
                ("staging prefill ms", "avg_staging_prefill_req_ms"),
            ],
        ),
        (
            "Admission",
            [
                ("admission window", "dllm_admission_window"),
                ("admitted mean", "mean_dllm_admitted_reqs"),
                ("admitted max", "max_dllm_admitted_reqs"),
                ("waiting q mean", "mean_dllm_waiting_queue_size"),
                ("waiting q max", "max_dllm_waiting_queue_size"),
                (
                    "pending next mean",
                    "mean_dllm_pending_next_round_reqs_size",
                ),
                ("pending next max", "max_dllm_pending_next_round_reqs_size"),
            ],
        ),
        (
            "Phase",
            [
                ("mixed batches", "mixed_mask_batches"),
                ("mixed prefill/decode", "mixed_prefill_decode_batches"),
                ("phase switches", "phase_switches"),
            ],
        ),
    ]

    print(f"[{task}] latency stats")
    for title, metrics in rows:
        print(f"  {title}:")
        for label, key in metrics:
            print(f"    {label:<24} {_fmt_metric(summary.get(key))}")

    phase_counts = summary.get("request_phase_counts")
    if phase_counts:
        print("  Request phases:")
        for phase, count in phase_counts.items():
            print(f"    {phase:<24} {count}")


def summarize_latency_metrics(
    request_records: List[Dict[str, Any]],
    batch_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    import numpy as np

    ttfb_ms = _values(request_records, "ttfb_ms")
    tpob_ms = _values(request_records, "tpob_ms")
    sched_wait_ms = _values(request_records, "sched_wait_ms")
    decode_delay_ms = _values(request_records, "decode_delay_ms")
    first_unmask_gap_ms = _values(request_records, "first_unmask_gap_ms")
    ideal_ttfb_ms = _ideal_ttfb_values(request_records)
    ideal_tpob_ms = _ideal_tpob_values(request_records)
    generated_block_counts = _values(request_records, "num_output_blocks")
    prefill_block_counts_by_req = _prefill_block_counts_by_request(batch_records)
    prefill_block_counts = list(prefill_block_counts_by_req.values())
    avg_prefill_req_ms = _weighted_batch_mean(
        batch_records, "prefill", "duration_ms", "num_reqs"
    )
    avg_decode_block_ms = _weighted_batch_mean(
        batch_records, "decode", "duration_ms", "num_output_blocks"
    )
    avg_initial_prefill_req_ms = _weighted_mean_by_count(
        batch_records, "duration_ms", "num_initial_prefill_reqs"
    )
    avg_staging_prefill_req_ms = _weighted_mean_by_count(
        batch_records, "duration_ms", "num_staging_prefill_reqs"
    )
    avg_actual_decode_block_ms = _weighted_mean_by_count(
        batch_records, "duration_ms", "num_output_blocks"
    )
    dllm_admitted_reqs = _values(batch_records, "dllm_admitted_reqs")
    dllm_waiting_queue_size = _values(batch_records, "dllm_waiting_queue_size")
    dllm_pending_next_round_reqs_size = _values(
        batch_records, "dllm_pending_next_round_reqs_size"
    )
    if not dllm_pending_next_round_reqs_size:
        dllm_pending_next_round_reqs_size = _values(
            batch_records, "dllm_staging_queue_size"
        )
    dllm_admission_window = _values(batch_records, "dllm_admission_window")
    phase_sequence = [record.get("phase") for record in batch_records]
    req_phase_counts = _count_per_req_phases(batch_records)

    return {
        "n_request_records": len(request_records),
        "mean_sched_wait_ms": float(np.mean(sched_wait_ms)) if sched_wait_ms else None,
        "p50_sched_wait_ms": _percentile(sched_wait_ms, 50),
        "p95_sched_wait_ms": _percentile(sched_wait_ms, 95),
        "p99_sched_wait_ms": _percentile(sched_wait_ms, 99),
        "mean_decode_delay_ms": float(np.mean(decode_delay_ms)) if decode_delay_ms else None,
        "p50_decode_delay_ms": _percentile(decode_delay_ms, 50),
        "p95_decode_delay_ms": _percentile(decode_delay_ms, 95),
        "p99_decode_delay_ms": _percentile(decode_delay_ms, 99),
        "mean_first_unmask_gap_ms": float(np.mean(first_unmask_gap_ms)) if first_unmask_gap_ms else None,
        "p50_first_unmask_gap_ms": _percentile(first_unmask_gap_ms, 50),
        "p95_first_unmask_gap_ms": _percentile(first_unmask_gap_ms, 95),
        "p99_first_unmask_gap_ms": _percentile(first_unmask_gap_ms, 99),
        "mean_ttfb_ms": float(np.mean(ttfb_ms)) if ttfb_ms else None,
        "p50_ttfb_ms": _percentile(ttfb_ms, 50),
        "p95_ttfb_ms": _percentile(ttfb_ms, 95),
        "p99_ttfb_ms": _percentile(ttfb_ms, 99),
        "mean_tpob_ms": float(np.mean(tpob_ms)) if tpob_ms else None,
        "p50_tpob_ms": _percentile(tpob_ms, 50),
        "p95_tpob_ms": _percentile(tpob_ms, 95),
        "p99_tpob_ms": _percentile(tpob_ms, 99),
        "n_ideal_ttfb_records": len(ideal_ttfb_ms),
        "mean_ideal_ttfb_ms": (
            float(np.mean(ideal_ttfb_ms)) if ideal_ttfb_ms else None
        ),
        "p50_ideal_ttfb_ms": _percentile(ideal_ttfb_ms, 50),
        "p95_ideal_ttfb_ms": _percentile(ideal_ttfb_ms, 95),
        "p99_ideal_ttfb_ms": _percentile(ideal_ttfb_ms, 99),
        "n_ideal_tpob_records": len(ideal_tpob_ms),
        "mean_ideal_tpob_ms": (
            float(np.mean(ideal_tpob_ms)) if ideal_tpob_ms else None
        ),
        "p50_ideal_tpob_ms": _percentile(ideal_tpob_ms, 50),
        "p95_ideal_tpob_ms": _percentile(ideal_tpob_ms, 95),
        "p99_ideal_tpob_ms": _percentile(ideal_tpob_ms, 99),
        "n_generated_block_records": len(generated_block_counts),
        "mean_generated_blocks": (
            float(np.mean(generated_block_counts)) if generated_block_counts else None
        ),
        "n_prefill_block_records": len(prefill_block_counts),
        "mean_prefill_blocks": (
            float(np.mean(prefill_block_counts)) if prefill_block_counts else None
        ),
        "n_batch_records": len(batch_records),
        "avg_prefill_req_ms": avg_prefill_req_ms,
        "avg_decode_block_ms": avg_decode_block_ms,
        "avg_initial_prefill_req_ms": avg_initial_prefill_req_ms,
        "avg_staging_prefill_req_ms": avg_staging_prefill_req_ms,
        "avg_actual_decode_block_ms": avg_actual_decode_block_ms,
        "dllm_admission_window": (
            int(max(dllm_admission_window)) if dllm_admission_window else None
        ),
        "mean_dllm_admitted_reqs": (
            float(np.mean(dllm_admitted_reqs)) if dllm_admitted_reqs else None
        ),
        "max_dllm_admitted_reqs": (
            int(max(dllm_admitted_reqs)) if dllm_admitted_reqs else None
        ),
        "mean_dllm_waiting_queue_size": (
            float(np.mean(dllm_waiting_queue_size))
            if dllm_waiting_queue_size
            else None
        ),
        "max_dllm_waiting_queue_size": (
            int(max(dllm_waiting_queue_size)) if dllm_waiting_queue_size else None
        ),
        "mean_dllm_pending_next_round_reqs_size": (
            float(np.mean(dllm_pending_next_round_reqs_size))
            if dllm_pending_next_round_reqs_size
            else None
        ),
        "max_dllm_pending_next_round_reqs_size": (
            int(max(dllm_pending_next_round_reqs_size))
            if dllm_pending_next_round_reqs_size
            else None
        ),
        "request_phase_counts": req_phase_counts,
        "mixed_mask_batches": sum(
            1 for record in batch_records if record.get("is_mixed_mask_batch")
        ),
        "prefill_batches": phase_sequence.count("prefill"),
        "decode_batches": phase_sequence.count("decode"),
        "mixed_prefill_decode_batches": phase_sequence.count("mixed_prefill_decode"),
        "phase_switches": sum(
            1
            for prev, cur in zip(phase_sequence, phase_sequence[1:])
            if prev != cur
        ),
    }


def plot_forward_latency(
    step_data: Dict[str, Dict],
    model_name: str,
    output_dir: str,
):
    """Visualize forward call latency distribution and timeline."""
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = [
        task
        for task, metrics in step_data.items()
        if metrics.get("forward_durations_ms")
    ]
    if not tasks:
        print("[plot] no forward_duration_ms data (check step_log_file config)")
        return

    colors = {"gsm8k": "#4C72B0", "humaneval": "#DD8452", "math": "#55A868"}
    model_tag = model_name.replace("/", "_")

    # ── 히스토그램 + 타임라인 ──────────────────────────────────────────
    fig, axes = plt.subplots(2, len(tasks), figsize=(5 * len(tasks), 8), sharey=False)
    if len(tasks) == 1:
        axes = [[axes[0]], [axes[1]]]

    for idx, task in enumerate(tasks):
        durations = step_data[task]["forward_durations_ms"]
        color = colors.get(task, "#777")
        hist_ax = axes[0][idx]
        seq_ax = axes[1][idx]

        # histogram
        hist_ax.hist(durations, bins=40, color=color, edgecolor="white", linewidth=0.4)
        mean_val = float(np.mean(durations))
        p95_val = float(np.percentile(durations, 95))
        hist_ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.2,
                        label=f"mean={mean_val:.1f} ms")
        hist_ax.axvline(p95_val, color="orange", linestyle=":", linewidth=1.2,
                        label=f"p95={p95_val:.1f} ms")
        hist_ax.set_title(f"{task.upper()} — Forward latency distribution", fontsize=12, fontweight="bold")
        hist_ax.set_xlabel("Forward latency (ms)", fontsize=10)
        hist_ax.set_ylabel("Count", fontsize=10)
        hist_ax.legend(fontsize=8)
        hist_ax.text(
            0.97, 0.97,
            f"n={len(durations)}\nmean={mean_val:.1f}\n"
            f"p50={np.percentile(durations, 50):.1f}\n"
            f"p95={p95_val:.1f}\n"
            f"p99={np.percentile(durations, 99):.1f}\n"
            f"max={max(durations):.1f}",
            transform=hist_ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

        # timeline (latency per call sequence)
        seq_ax.plot(durations, color=color, linewidth=0.6, alpha=0.7)
        seq_ax.axhline(mean_val, color="red", linestyle="--", linewidth=1.0,
                       label=f"mean={mean_val:.1f} ms")
        seq_ax.axhline(p95_val, color="orange", linestyle=":", linewidth=1.0,
                       label=f"p95={p95_val:.1f} ms")
        seq_ax.set_title(f"{task.upper()} — Forward latency timeline", fontsize=12, fontweight="bold")
        seq_ax.set_xlabel("Forward call sequence", fontsize=10)
        seq_ax.set_ylabel("Latency (ms)", fontsize=10)
        seq_ax.legend(fontsize=8)

    fig.suptitle(f"Forward call latency — {model_name}", fontsize=13)
    fig.tight_layout()
    out_path = Path(output_dir) / f"forward_latency_{model_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path}")
    plt.close(fig)

    # ── Task 간 박스 플롯 비교 ─────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(max(6, 2.5 * len(tasks) + 2), 4))
    bp = ax2.boxplot(
        [step_data[t]["forward_durations_ms"] for t in tasks],
        tick_labels=[t.upper() for t in tasks],
        patch_artist=True,
        notch=False,
    )
    for patch, task in zip(bp["boxes"], tasks):
        patch.set_facecolor(colors.get(task, "#777"))
        patch.set_alpha(0.7)
    ax2.set_ylabel("Forward latency (ms)", fontsize=11)
    ax2.set_title("Forward latency distribution by task", fontsize=12)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5)

    fig2.tight_layout()
    out_path2 = Path(output_dir) / f"forward_latency_boxplot_{model_tag}.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path2}")
    plt.close(fig2)


def plot_context_length_distribution(
    latency_data: Dict[str, Dict[str, List[Dict[str, Any]]]],
    model_name: str,
    output_dir: str,
):
    """Plot histogram of input context lengths per request for each task."""
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = [
        task
        for task, data in latency_data.items()
        if any("input_len" in r for r in data.get("request", []))
    ]
    if not tasks:
        print("[plot] no input_len data in request latency log")
        return

    colors = {"gsm8k": "#4C72B0", "humaneval": "#DD8452", "math": "#55A868"}
    model_tag = model_name.replace("/", "_")

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4))
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        input_lens = [
            r["input_len"]
            for r in latency_data[task]["request"]
            if "input_len" in r
        ]
        color = colors.get(task, "#777")
        mean_val = float(np.mean(input_lens))
        p50 = float(np.percentile(input_lens, 50))

        ax.hist(input_lens, bins=30, color=color, edgecolor="white", linewidth=0.4)
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.2,
                   label=f"mean={mean_val:.0f}")
        ax.axvline(p50, color="orange", linestyle=":", linewidth=1.2,
                   label=f"p50={p50:.0f}")
        ax.set_title(f"{task.upper()} — Context length distribution",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Input token count", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=8)
        ax.text(
            0.97, 0.97,
            f"n={len(input_lens)}\nmean={mean_val:.0f}\n"
            f"p50={p50:.0f}\n"
            f"min={min(input_lens)}\nmax={max(input_lens)}",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    fig.suptitle(f"Request context length distribution — {model_name}", fontsize=13)
    fig.tight_layout()
    out_path = Path(output_dir) / f"context_length_{model_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path}")
    plt.close(fig)


def plot_scheduling_delays(
    latency_data: Dict[str, Dict[str, List[Dict[str, Any]]]],
    model_name: str,
    output_dir: str,
):
    """Histogram of sched_wait_ms and decode_delay_ms per task."""
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = list(latency_data.keys())
    colors = {"gsm8k": "#4C72B0", "humaneval": "#DD8452", "math": "#55A868"}
    model_tag = model_name.replace("/", "_")

    metrics = [
        ("sched_wait_ms",      "Scheduling wait (ms)",      "Scheduling wait — queue entry → first forward"),
        ("first_unmask_gap_ms", "Prefill→unmask gap (ms)",  "Prefill→first unmask gap — last prefill forward → first unmask batch"),
        ("decode_delay_ms",    "Inter-block delay (ms)",    "Decode delay — block ready → next batch"),
    ]

    for field, xlabel, suptitle in metrics:
        task_data = {
            task: _values(latency_data[task]["request"], field)
            for task in tasks
        }
        task_data = {t: v for t, v in task_data.items() if v}
        if not task_data:
            print(f"[plot] no {field} data — skipping")
            continue

        n = len(task_data)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, (task, vals) in zip(axes, task_data.items()):
            arr = np.array(vals)
            color = colors.get(task, "#777")
            mean_v = float(np.mean(arr))
            p50   = float(np.percentile(arr, 50))
            p95   = float(np.percentile(arr, 95))
            p99   = float(np.percentile(arr, 99))

            ax.hist(arr, bins=40, color=color, edgecolor="white", linewidth=0.4)
            ax.axvline(mean_v, color="red",    linestyle="--", linewidth=1.2, label=f"mean={mean_v:.1f}")
            ax.axvline(p50,   color="orange",  linestyle=":",  linewidth=1.2, label=f"p50={p50:.1f}")
            ax.axvline(p95,   color="purple",  linestyle="-.", linewidth=1.2, label=f"p95={p95:.1f}")

            ax.set_title(f"{task.upper()}", fontsize=12, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.legend(fontsize=8)
            ax.text(
                0.97, 0.97,
                f"n={len(arr)}\nmean={mean_v:.1f}\np50={p50:.1f}\np95={p95:.1f}\np99={p99:.1f}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

        fig.suptitle(f"{suptitle} — {model_name}", fontsize=13)
        fig.tight_layout()
        out_path = Path(output_dir) / f"{field}_{model_tag}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[plot] saved → {out_path}")
        plt.close(fig)

    # Batch-level DLLM admission queue occupancy.
    field = "dllm_waiting_queue_size"
    task_data = {
        task: _values(latency_data[task]["batch"], field)
        for task in tasks
    }
    task_data = {t: v for t, v in task_data.items() if v}
    if not task_data:
        print(f"[plot] no {field} data — skipping")
        return

    n = len(task_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (task, vals) in zip(axes, task_data.items()):
        arr = np.array(vals)
        color = colors.get(task, "#777")
        mean_v = float(np.mean(arr))
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        max_v = int(max(arr))
        bins = range(0, max_v + 2)

        ax.hist(arr, bins=bins, color=color, edgecolor="white", linewidth=0.4)
        ax.axvline(mean_v, color="red", linestyle="--", linewidth=1.2,
                   label=f"mean={mean_v:.1f}")
        ax.axvline(p50, color="orange", linestyle=":", linewidth=1.2,
                   label=f"p50={p50:.1f}")
        ax.axvline(p95, color="purple", linestyle="-.", linewidth=1.2,
                   label=f"p95={p95:.1f}")
        ax.set_title(f"{task.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("DLLM waiting queue size per forward", fontsize=10)
        ax.set_ylabel("Forward batches", fontsize=10)
        ax.legend(fontsize=8)
        ax.text(
            0.97,
            0.97,
            f"n={len(arr)}\nmean={mean_v:.1f}\np50={p50:.1f}\np95={p95:.1f}\nmax={max_v}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    fig.suptitle(f"DLLM waiting queue occupancy per forward — {model_name}", fontsize=13)
    fig.tight_layout()
    out_path = Path(output_dir) / f"{field}_{model_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path}")
    plt.close(fig)


def plot_ttfb_per_request(
    latency_data: Dict[str, Dict[str, List[Dict[str, Any]]]],
    model_name: str,
    output_dir: str,
):
    """Scatter + running-mean of TTFB ordered by request completion."""
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = [
        task
        for task, data in latency_data.items()
        if any("ttfb_ms" in r for r in data.get("request", []))
    ]
    if not tasks:
        print("[plot] no ttfb_ms data in request latency log — skipping")
        return

    colors = {"gsm8k": "#4C72B0", "humaneval": "#DD8452", "math": "#55A868"}
    model_tag = model_name.replace("/", "_")

    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 4), sharey=False)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        records = latency_data[task].get("request", [])
        ttfb_vals = [r["ttfb_ms"] for r in records if "ttfb_ms" in r]
        if not ttfb_vals:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        color = colors.get(task, "#777")
        x = np.arange(len(ttfb_vals))
        arr = np.array(ttfb_vals)

        ax.scatter(x, arr, color=color, s=8, alpha=0.5, label="TTFB")

        # running mean with window=max(1, n//20)
        window = max(1, len(arr) // 20)
        kernel = np.ones(window) / window
        running_mean = np.convolve(arr, kernel, mode="valid")
        offset = window // 2
        ax.plot(x[offset: offset + len(running_mean)], running_mean,
                color="black", linewidth=1.5, label=f"running mean (w={window})")

        mean_v = float(np.mean(arr))
        p95_v  = float(np.percentile(arr, 95))
        ax.axhline(mean_v, color="red",    linestyle="--", linewidth=1.0,
                   label=f"mean={mean_v:.1f} ms")
        ax.axhline(p95_v,  color="orange", linestyle=":",  linewidth=1.0,
                   label=f"p95={p95_v:.1f} ms")

        ax.set_title(f"{task.upper()} — TTFB per request", fontsize=12, fontweight="bold")
        ax.set_xlabel("Request index (completion order)", fontsize=10)
        ax.set_ylabel("TTFB (ms)", fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
        ax.text(
            0.97, 0.97,
            f"n={len(arr)}\nmean={mean_v:.1f}\n"
            f"p50={np.percentile(arr, 50):.1f}\n"
            f"p95={p95_v:.1f}\n"
            f"p99={np.percentile(arr, 99):.1f}",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    fig.suptitle(f"TTFB per request (completion order) — {model_name}", fontsize=13)
    fig.tight_layout()
    out_path = Path(output_dir) / f"ttfb_per_request_{model_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path}")
    plt.close(fig)


def plot_tpob_per_request(
    latency_data: Dict[str, Dict[str, List[Dict[str, Any]]]],
    model_name: str,
    output_dir: str,
):
    """Scatter + running-mean of mean TPOB per request, ordered by completion."""
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = [
        task
        for task, data in latency_data.items()
        if any("tpob_ms" in r for r in data.get("request", []))
    ]
    if not tasks:
        print("[plot] no tpob_ms data in request latency log — skipping")
        return

    colors = {"gsm8k": "#4C72B0", "humaneval": "#DD8452", "math": "#55A868"}
    model_tag = model_name.replace("/", "_")

    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 4), sharey=False)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        records = latency_data[task].get("request", [])
        tpob_vals = [r["tpob_ms"] for r in records if "tpob_ms" in r]
        if not tpob_vals:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        color = colors.get(task, "#777")
        x = np.arange(len(tpob_vals))
        arr = np.array(tpob_vals)

        ax.scatter(x, arr, color=color, s=8, alpha=0.5, label="mean TPOB")

        window = max(1, len(arr) // 20)
        kernel = np.ones(window) / window
        running_mean = np.convolve(arr, kernel, mode="valid")
        offset = window // 2
        ax.plot(x[offset: offset + len(running_mean)], running_mean,
                color="black", linewidth=1.5, label=f"running mean (w={window})")

        mean_v = float(np.mean(arr))
        p95_v  = float(np.percentile(arr, 95))
        ax.axhline(mean_v, color="red",    linestyle="--", linewidth=1.0,
                   label=f"mean={mean_v:.1f} ms")
        ax.axhline(p95_v,  color="orange", linestyle=":",  linewidth=1.0,
                   label=f"p95={p95_v:.1f} ms")

        ax.set_title(f"{task.upper()} — mean TPOB per request",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Request index (completion order)", fontsize=10)
        ax.set_ylabel("Mean TPOB (ms)", fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
        ax.text(
            0.97, 0.97,
            f"n={len(arr)}\nmean={mean_v:.1f}\n"
            f"p50={np.percentile(arr, 50):.1f}\n"
            f"p95={p95_v:.1f}\n"
            f"p99={np.percentile(arr, 99):.1f}",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    fig.suptitle(f"Mean TPOB per request (completion order) — {model_name}", fontsize=13)
    fig.tight_layout()
    out_path = Path(output_dir) / f"tpob_per_request_{model_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path}")
    plt.close(fig)


def plot_request_slo_scatter(
    latency_data: Dict[str, Dict[str, List[Dict[str, Any]]]],
    slo_data: Dict[str, Dict[str, Optional[float]]],
    model_name: str,
    output_dir: str,
):
    """Scatter requests by normalized TTFB/TPOB against their SLOs."""
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = []
    for task, data in latency_data.items():
        slo = slo_data.get(task, {})
        if slo.get("ttfb_slo_ms") is None or slo.get("tpob_slo_ms") is None:
            continue
        if any(
            r.get("ttfb_ms") is not None and r.get("tpob_ms") is not None
            for r in data.get("request", [])
        ):
            tasks.append(task)
    if not tasks:
        print("[plot] no request records with both TTFB/TPOB SLOs — skipping")
        return

    colors = {"gsm8k": "#4C72B0", "humaneval": "#DD8452", "math": "#55A868"}
    model_tag = model_name.replace("/", "_")

    fig, axes = plt.subplots(1, len(tasks), figsize=(5.8 * len(tasks), 5), sharex=False, sharey=False)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        records = _unique_request_records(latency_data[task].get("request", []))
        ttfb_slo_ms = float(slo_data[task]["ttfb_slo_ms"])
        tpob_slo_ms = float(slo_data[task]["tpob_slo_ms"])

        points = [
            (float(r["ttfb_ms"]) / ttfb_slo_ms, float(r["tpob_ms"]) / tpob_slo_ms)
            for r in records
            if r.get("ttfb_ms") is not None and r.get("tpob_ms") is not None
        ]
        if not points:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        arr = np.array(points)
        x = arr[:, 0]
        y = arr[:, 1]
        both_ok = (x <= 1.0) & (y <= 1.0)
        ttfb_only_bad = (x > 1.0) & (y <= 1.0)
        tpob_only_bad = (x <= 1.0) & (y > 1.0)
        both_bad = (x > 1.0) & (y > 1.0)

        color = colors.get(task, "#777")
        ax.scatter(x[both_ok], y[both_ok], s=14, alpha=0.55, color=color, label="both ok")
        ax.scatter(x[ttfb_only_bad], y[ttfb_only_bad], s=18, alpha=0.75,
                   color="#C44E52", marker=">", label="TTFB miss")
        ax.scatter(x[tpob_only_bad], y[tpob_only_bad], s=18, alpha=0.75,
                   color="#8172B2", marker="^", label="TPOB miss")
        ax.scatter(x[both_bad], y[both_bad], s=22, alpha=0.8,
                   color="#222222", marker="x", label="both miss")

        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"{task.upper()} — request SLO map",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("TTFB / TTFB SLO", fontsize=10)
        ax.set_ylabel("Mean TPOB / TPOB SLO", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=7, loc="upper right")

        max_x = max(1.05, float(np.percentile(x, 99)) * 1.08)
        max_y = max(1.05, float(np.percentile(y, 99)) * 1.08)
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y)

        total = len(x)
        ok_count = int(np.sum(both_ok))
        ax.text(
            0.03, 0.97,
            f"n={total}\n"
            f"SLO ok={ok_count / total * 100:.1f}%\n"
            f"TTFB SLO={ttfb_slo_ms:.1f} ms\n"
            f"TPOB SLO={tpob_slo_ms:.1f} ms\n"
            f"p95 x={np.percentile(x, 95):.2f}\n"
            f"p95 y={np.percentile(y, 95):.2f}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
        )

    fig.suptitle(f"Request distribution against TTFB/TPOB SLO — {model_name}", fontsize=13)
    fig.tight_layout()
    out_path = Path(output_dir) / f"request_slo_scatter_{model_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path}")
    plt.close(fig)


def plot_avg_block_steps_per_request(
    latency_data: Dict[str, Dict[str, List[Dict[str, Any]]]],
    model_name: str,
    output_dir: str,
):
    """Scatter + running-mean of avg decode forward steps per block, per request."""
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = [
        task
        for task, data in latency_data.items()
        if any("avg_block_steps" in r and r["avg_block_steps"] is not None
               for r in data.get("request", []))
    ]
    if not tasks:
        print("[plot] no avg_block_steps data in request latency log — skipping")
        return

    colors = {"gsm8k": "#4C72B0", "humaneval": "#DD8452", "math": "#55A868"}
    model_tag = model_name.replace("/", "_")

    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 4), sharey=False)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        records = latency_data[task].get("request", [])
        vals = [
            r["avg_block_steps"]
            for r in records
            if "avg_block_steps" in r and r["avg_block_steps"] is not None
        ]
        if not vals:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        color = colors.get(task, "#777")
        x = np.arange(len(vals))
        arr = np.array(vals)

        ax.scatter(x, arr, color=color, s=8, alpha=0.5, label="avg steps/block")

        window = max(1, len(arr) // 20)
        kernel = np.ones(window) / window
        running_mean = np.convolve(arr, kernel, mode="valid")
        offset = window // 2
        ax.plot(x[offset: offset + len(running_mean)], running_mean,
                color="black", linewidth=1.5, label=f"running mean (w={window})")

        mean_v = float(np.mean(arr))
        p95_v  = float(np.percentile(arr, 95))
        ax.axhline(mean_v, color="red",    linestyle="--", linewidth=1.0,
                   label=f"mean={mean_v:.2f}")
        ax.axhline(p95_v,  color="orange", linestyle=":",  linewidth=1.0,
                   label=f"p95={p95_v:.2f}")

        ax.set_title(f"{task.upper()} — avg decode steps per block",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Request index (completion order)", fontsize=10)
        ax.set_ylabel("Avg decode forward steps / block", fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
        ax.text(
            0.97, 0.97,
            f"n={len(arr)}\nmean={mean_v:.2f}\n"
            f"p50={np.percentile(arr, 50):.2f}\n"
            f"p95={p95_v:.2f}\n"
            f"p99={np.percentile(arr, 99):.2f}",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    fig.suptitle(
        f"Avg decode forward steps per block (completion order) — {model_name}",
        fontsize=13,
    )
    fig.tight_layout()
    out_path = Path(output_dir) / f"avg_block_steps_per_request_{model_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path}")
    plt.close(fig)


def plot_tpob_by_block_index(
    latency_data: Dict[str, Dict[str, List[Dict[str, Any]]]],
    model_name: str,
    output_dir: str,
):
    """Box plot of TPOB (inter-block time) grouped by block transition index.

    tpob_list_ms[i] is the gap from block i to block i+1.
    Sample count decreases at higher indices since fewer requests produce many blocks.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = [
        task
        for task, data in latency_data.items()
        if any("tpob_list_ms" in r for r in data.get("request", []))
    ]
    if not tasks:
        print("[plot] no tpob_list_ms data in request latency log — skipping")
        return

    colors = {"gsm8k": "#4C72B0", "humaneval": "#DD8452", "math": "#55A868"}
    model_tag = model_name.replace("/", "_")

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5), sharey=False)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        request_records = latency_data[task].get("request", [])
        by_index: Dict[int, List[float]] = {}
        for rec in request_records:
            tpob_list = rec.get("tpob_list_ms")
            if not tpob_list:
                continue
            for idx, val in enumerate(tpob_list):
                by_index.setdefault(idx, []).append(float(val))

        if not by_index:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{task.upper()}", fontsize=12, fontweight="bold")
            continue

        sorted_indices = sorted(by_index.keys())
        data_groups = [by_index[i] for i in sorted_indices]
        tick_labels = [f"blk {i}→{i+1}\n(n={len(by_index[i])})" for i in sorted_indices]
        color = colors.get(task, "#777")

        bp = ax.boxplot(
            data_groups,
            tick_labels=tick_labels,
            patch_artist=True,
            notch=False,
            showfliers=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        means = [float(np.mean(g)) for g in data_groups]
        ax.plot(range(1, len(sorted_indices) + 1), means, "r--o",
                linewidth=1.2, markersize=4, label="mean")

        overall_mean = float(np.mean([v for g in data_groups for v in g]))
        ax.axhline(overall_mean, color="gray", linestyle=":", linewidth=1.0,
                   label=f"overall mean={overall_mean:.1f}")

        ax.set_title(f"{task.upper()} — TPOB by block transition",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Block transition", fontsize=10)
        ax.set_ylabel("TPOB (ms)", fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    fig.suptitle(f"TPOB distribution by block index — {model_name}", fontsize=13)
    fig.tight_layout()
    out_path = Path(output_dir) / f"tpob_by_block_index_{model_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path}")
    plt.close(fig)


def plot_block_steps_per_request(
    latency_data: Dict[str, Dict[str, List[Dict[str, Any]]]],
    model_name: str,
    output_dir: str,
):
    """Line plot of decode forward steps per block index, one line per request.

    Each line connects block_steps_list[0], [1], ... for a single request.
    Line alpha varies from light (first request) to dark (last request) so
    ordering effects are visible without overplotting.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = [
        task
        for task, data in latency_data.items()
        if any(
            r.get("block_steps_list") for r in data.get("request", [])
        )
    ]
    if not tasks:
        print("[plot] no block_steps_list data in request latency log — skipping")
        return

    colors = {"gsm8k": "#4C72B0", "humaneval": "#DD8452", "math": "#55A868"}
    model_tag = model_name.replace("/", "_")

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5), sharey=False)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        records = [
            r for r in latency_data[task].get("request", [])
            if r.get("block_steps_list")
        ]
        if not records:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{task.upper()}", fontsize=12, fontweight="bold")
            continue

        color = colors.get(task, "#777")
        n = len(records)

        for i, rec in enumerate(records):
            steps = rec["block_steps_list"]
            alpha = 0.15 + 0.70 * (i / max(n - 1, 1))
            ax.plot(
                range(len(steps)),
                steps,
                color=color,
                alpha=alpha,
                linewidth=0.8,
            )

        # overlay per-block mean
        max_blocks = max(len(r["block_steps_list"]) for r in records)
        means = []
        for b in range(max_blocks):
            vals = [r["block_steps_list"][b] for r in records if b < len(r["block_steps_list"])]
            means.append(float(np.mean(vals)))
        ax.plot(range(max_blocks), means, "r--o",
                linewidth=1.5, markersize=4, zorder=5, label="mean")

        ax.set_title(f"{task.upper()} — decode steps per block (per request)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Block index", fontsize=10)
        ax.set_ylabel("Decode forward steps", fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Decode steps per block index — {model_name}\n"
        f"(line alpha: light=early request → dark=late request)",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = Path(output_dir) / f"block_steps_per_request_{model_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path}")
    plt.close(fig)


def plot_latency_metrics(
    latency_data: Dict[str, Dict[str, List[Dict[str, Any]]]],
    model_name: str,
    output_dir: str,
):
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = [
        task
        for task, records in latency_data.items()
        if records.get("request") or records.get("batch")
    ]
    if not tasks:
        print("[plot] latency 데이터 없음 (request/batch latency log 설정 확인)")
        return

    colors = {"gsm8k": "#4C72B0", "humaneval": "#DD8452", "math": "#55A868"}
    model_tag = model_name.replace("/", "_")

    # ── Request-level TTFB/TPOB box plots ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 2.8 * len(tasks) + 3), 4))
    for ax, key, title in [
        (axes[0], "ttfb_ms", "TTFB per request"),
        (axes[1], "tpob_ms", "Mean TPOB per request"),
    ]:
        plot_tasks = [
            task for task in tasks if _values(latency_data[task]["request"], key)
        ]
        if plot_tasks:
            data = [
                _values(latency_data[task]["request"], key)
                for task in plot_tasks
            ]
            bp = ax.boxplot(
                data,
                tick_labels=[task.upper() for task in plot_tasks],
                patch_artist=True,
                notch=False,
            )
            for patch, task in zip(bp["boxes"], plot_tasks):
                patch.set_facecolor(colors.get(task, "#777"))
                patch.set_alpha(0.7)
            ax.set_ylabel("Latency (ms)")
            ax.set_title(title)
            ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        else:
            ax.text(0.5, 0.5, f"no {key} data", ha="center", va="center")
            ax.set_axis_off()

    out_path = Path(output_dir) / f"request_latency_{model_tag}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path}")
    plt.close(fig)

    # ── Average initial/staging prefill request / decode block time ───
    initial_prefill_vals = [
        _weighted_mean_by_count(
            latency_data[task]["batch"], "duration_ms", "num_initial_prefill_reqs"
        )
        for task in tasks
    ]
    staging_prefill_vals = [
        _weighted_mean_by_count(
            latency_data[task]["batch"], "duration_ms", "num_staging_prefill_reqs"
        )
        for task in tasks
    ]
    decode_vals = [
        _weighted_mean_by_count(
            latency_data[task]["batch"], "duration_ms", "num_output_blocks"
        )
        for task in tasks
    ]

    fig2, ax2 = plt.subplots(figsize=(max(7, 2.5 * len(tasks)), 4))
    x = np.arange(len(tasks))
    width = 0.26
    initial_prefill_plot = [0.0 if v is None else v for v in initial_prefill_vals]
    staging_prefill_plot = [0.0 if v is None else v for v in staging_prefill_vals]
    decode_plot = [0.0 if v is None else v for v in decode_vals]
    ax2.bar(
        x - width,
        initial_prefill_plot,
        width,
        label="Initial prefill req avg",
        color="#74A9CF",
    )
    ax2.bar(
        x,
        staging_prefill_plot,
        width,
        label="Staging prefill req avg",
        color="#6BA292",
    )
    ax2.bar(
        x + width,
        decode_plot,
        width,
        label="Decode block avg",
        color="#C44E52",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([task.upper() for task in tasks])
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Average processing time")
    ax2.legend()
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
    for idx, value in enumerate(initial_prefill_vals):
        if value is not None:
            ax2.text(idx - width, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    for idx, value in enumerate(staging_prefill_vals):
        if value is not None:
            ax2.text(idx, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    for idx, value in enumerate(decode_vals):
        if value is not None:
            ax2.text(idx + width, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    out_path2 = Path(output_dir) / f"batch_latency_{model_tag}.png"
    fig2.tight_layout()
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path2}")
    plt.close(fig2)

    # ── Prefill/decode execution order timeline ───────────────────────
    fig3, ax3 = plt.subplots(figsize=(max(9, 0.18 * max(
        [len(latency_data[t]["batch"]) for t in tasks] or [1]
    )), max(3, 0.7 * len(tasks) + 1.5)))
    phase_color = {
        "initial_prefill": "#74A9CF",
        "staging_prefill": "#2C7BB6",
        "decode": "#D7191C",
        "mixed": "#Fdae61",
        "mixed_prefill_decode": "#7B3294",
        "prefill": "#2C7BB6",
        "unknown": "#777777",
    }
    phase_marker = {
        "initial_prefill": "s",
        "staging_prefill": "D",
        "decode": "o",
        "mixed": "P",
        "mixed_prefill_decode": "X",
        "prefill": "s",
        "unknown": "x",
    }

    for y, task in enumerate(tasks):
        batch_records = latency_data[task]["batch"]
        for fallback_seq, record in enumerate(batch_records):
            phase = _batch_visual_phase(record)
            seq = int(record.get("seq") if record.get("seq") is not None else fallback_seq)
            duration_ms = float(record.get("duration_ms") or 0.0)
            size = 35 + min(duration_ms, 500.0) * 0.25
            ax3.scatter(
                seq,
                y,
                s=size,
                c=phase_color.get(phase, phase_color["unknown"]),
                marker=phase_marker.get(phase, "x"),
                alpha=0.8,
                edgecolors="white",
                linewidths=0.4,
            )

    ax3.set_yticks(range(len(tasks)))
    ax3.set_yticklabels([task.upper() for task in tasks])
    ax3.set_xlabel("DLM batch sequence")
    ax3.set_title("DLM execution order by request composition")
    ax3.grid(True, axis="x", linestyle="--", alpha=0.35)
    handles = [
        plt.Line2D([0], [0], marker=phase_marker[p], color="w", label=p.replace("_", " "),
                   markerfacecolor=phase_color[p], markersize=8)
        for p in (
            "initial_prefill",
            "staging_prefill",
            "decode",
            "mixed",
            "mixed_prefill_decode",
        )
    ]
    ax3.legend(handles=handles, loc="upper right")

    out_path3 = Path(output_dir) / f"phase_sequence_{model_tag}.png"
    fig3.tight_layout()
    fig3.savefig(out_path3, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path3}")
    plt.close(fig3)

    # ── Per-batch request phase composition ───────────────────────────
    phase_keys = [
        "queuing_prefill",
        "staging_prefill",
        "queuing_decode",
        "staging_decode",
    ]
    phase_colors = {
        "queuing_prefill": "#74A9CF",
        "staging_prefill": "#2C7BB6",
        "queuing_decode": "#Fdae61",
        "staging_decode": "#D7191C",
    }
    fig4, axes4 = plt.subplots(
        len(tasks),
        1,
        figsize=(max(9, 0.18 * max([len(latency_data[t]["batch"]) for t in tasks] or [1])), max(3, 1.8 * len(tasks))),
        squeeze=False,
    )
    for row, task in enumerate(tasks):
        ax = axes4[row][0]
        batch_records = latency_data[task]["batch"]
        seqs = [
            int(record.get("seq") if record.get("seq") is not None else idx)
            for idx, record in enumerate(batch_records)
        ]
        bottom = np.zeros(len(batch_records))
        for phase_key in phase_keys:
            counts = [
                (record.get("per_req_phase") or []).count(phase_key)
                for record in batch_records
            ]
            ax.bar(
                seqs,
                counts,
                bottom=bottom,
                width=0.9,
                color=phase_colors[phase_key],
                label=REQ_PHASE_LABELS[phase_key],
                linewidth=0,
            )
            bottom += np.array(counts)
        ax.set_ylabel(task.upper())
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        if row == 0:
            ax.legend(loc="upper right", ncol=2, fontsize=8)
    axes4[-1][0].set_xlabel("DLM batch sequence")
    fig4.suptitle("Per-batch request phase composition")
    out_path4 = Path(output_dir) / f"phase_composition_{model_tag}.png"
    fig4.tight_layout()
    fig4.savefig(out_path4, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path4}")
    plt.close(fig4)


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
    parser.add_argument("--num-threads", type=int, default=200)
    parser.add_argument("--request-rate", type=float, default=None,
                        help="전체 request 시작 rate(req/s). 미지정 시 기존 closed-loop 방식")
    parser.add_argument("--gsm8k-data-path", type=str, default=None)
    parser.add_argument("--math-data-path", type=str, default=None)

    # 출력
    parser.add_argument("--output-dir", type=str, default="/tmp/dlm_results")
    parser.add_argument("--block-size", type=int, default=None,
                        help="모델의 block_size (LLaDA2=32, SDAR=4). "
                             "미지정 시 step 축 범위를 자동 추정")
    parser.add_argument("--log", action="store_true",
                        help="DLM step/request/batch 로그를 수집하고 그래프를 생성")
    parser.add_argument("--warmup", type=int, default=4,
                        help="벤치마크 전 Triton 커널 컴파일을 위해 보낼 더미 요청 수 (0 = 비활성)")
    parser.add_argument("--num-output-blocks", type=int, default=0,
                        help="생성할 block 수 고정 (0=비활성, EOS 무시하고 정확히 N블록 생성)")

    args = parser.parse_args()

    if args.base_url is None and args.model_path is None:
        parser.error("--base-url 또는 --model-path 중 하나 필요")
    if args.base_url and args.model is None:
        parser.error("--base-url 사용 시 --model 필요")
    if args.request_rate is not None and args.request_rate <= 0:
        parser.error("--request-rate는 양수여야 합니다")

    if args.base_url and args.log:
        print(
            "[warn] 기존 서버 사용 시 서버가 "
            f"step_log_file={STEP_LOG_FILE}, "
            f"request_latency_log_file={REQUEST_LATENCY_LOG_FILE}, "
            f"batch_latency_log_file={BATCH_LATENCY_LOG_FILE} "
            "로 실행됐는지 확인하세요."
        )

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
        step_data: Dict[str, Dict[str, List[int]]] = {}
        latency_data: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        slo_data: Dict[str, Dict[str, Optional[float]]] = {}
        effective_block_size = args.block_size or 32

        for task in args.tasks:
            print(f"\n{'='*60}")
            print(f"Task: {task.upper()}  |  model: {model}")
            print(f"{'='*60}")
            flush_server_cache(base_url)
            run_warmup(base_url, model, num_requests=args.warmup)

            if args.log:
                clear_step_log()
                clear_latency_logs()
                task_step_file = Path(args.output_dir) / f"steps_{task}.jsonl"
                if task_step_file.exists():
                    task_step_file.unlink()
                task_request_latency_file = (
                    Path(args.output_dir) / f"request_latency_{task}.jsonl"
                )
                task_batch_latency_file = (
                    Path(args.output_dir) / f"batch_latency_{task}.jsonl"
                )
                for path in (task_request_latency_file, task_batch_latency_file):
                    if path.exists():
                        path.unlink()

            metrics = run_task(
                task=task, base_url=base_url, model=model,
                num_examples=args.num_examples, num_threads=args.num_threads,
                gsm8k_data_path=args.gsm8k_data_path,
                math_data_path=args.math_data_path,
                request_rate=args.request_rate,
                num_output_blocks=args.num_output_blocks,
                block_size=effective_block_size,
            )
            all_results[task] = metrics

            step_metrics = read_step_log() if args.log else {
                "records": [],
                "raw_forward_calls": [],
                "block_steps": [],
                "forward_durations_ms": [],
            }
            raw_forward_calls = step_metrics["raw_forward_calls"]
            block_steps = step_metrics["block_steps"]
            forward_durations_ms = step_metrics.get("forward_durations_ms", [])
            step_data[task] = {
                "raw_forward_calls": raw_forward_calls,
                "block_steps": block_steps,
                "forward_durations_ms": forward_durations_ms,
            }
            latency_records = (
                read_latency_logs() if args.log else {"request": [], "batch": []}
            )
            latency_data[task] = latency_records
            slo_data[task] = _read_dllm_slo_config()
            if args.log and step_metrics["records"]:
                task_step_file = Path(args.output_dir) / f"steps_{task}.jsonl"
                with open(task_step_file, "w") as f:
                    for record in step_metrics["records"]:
                        f.write(json.dumps(record) + "\n")
            if args.log and latency_records["request"]:
                with open(task_request_latency_file, "w") as f:
                    for record in latency_records["request"]:
                        f.write(json.dumps(record) + "\n")
            if args.log and latency_records["batch"]:
                with open(task_batch_latency_file, "w") as f:
                    for record in latency_records["batch"]:
                        f.write(json.dumps(record) + "\n")

            latency_summary = summarize_latency_metrics(
                latency_records["request"],
                latency_records["batch"],
            )
            metrics["latency_stats"] = latency_summary

            score = metrics.get("score")
            score_str = f"{score:.4f}" if score is not None else "N/A"
            print(f"\n[{task}] score={score_str}  "
                  f"throughput={metrics.get('output_throughput_tok_s', 0):.1f} tok/s")
            if raw_forward_calls:
                import numpy as np
                print(f"[{task}] forward call stats: n={len(raw_forward_calls)}  "
                      f"mean={np.mean(raw_forward_calls):.2f}  "
                      f"median={np.median(raw_forward_calls):.1f}  "
                      f"max={max(raw_forward_calls)}")
            if block_steps:
                import numpy as np
                print(f"[{task}] block step stats: n={len(block_steps)}  "
                      f"mean={np.mean(block_steps):.2f}  "
                      f"median={np.median(block_steps):.1f}  "
                      f"max={max(block_steps)}")
            if forward_durations_ms:
                import numpy as np
                print(f"[{task}] forward latency (ms): n={len(forward_durations_ms)}  "
                      f"mean={np.mean(forward_durations_ms):.1f}  "
                      f"p50={np.percentile(forward_durations_ms, 50):.1f}  "
                      f"p95={np.percentile(forward_durations_ms, 95):.1f}  "
                      f"p99={np.percentile(forward_durations_ms, 99):.1f}  "
                      f"max={max(forward_durations_ms):.1f}")
            if args.log:
                print_latency_summary(task, latency_summary)
            else:
                print(f"[{task}] DLM logging disabled; pass --log to collect latency/step stats")

            # task별 결과 저장
            model_tag = model.replace("/", "_")
            out_path = Path(args.output_dir) / f"{task}_{model_tag}.json"
            with open(out_path, "w") as f:
                json.dump({"task": task, "model": model,
                           "step_stats": {
                               "n_runs": len(raw_forward_calls),
                               "mean_raw_forward_calls": (
                                   float(sum(raw_forward_calls) / len(raw_forward_calls))
                                   if raw_forward_calls else None
                               ),
                               "max_raw_forward_calls": (
                                   max(raw_forward_calls) if raw_forward_calls else None
                               ),
                               "n_blocks": len(block_steps),
                               "mean_block_steps": (
                                   float(sum(block_steps) / len(block_steps))
                                   if block_steps else None
                               ),
                               "max_block_steps": (
                                   max(block_steps) if block_steps else None
                               ),
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
                       "slo_data": slo_data,
                       "step_data": {t: v for t, v in step_data.items()},
                       "latency_data": {
                           t: {
                               "request": v.get("request", []),
                               "batch": v.get("batch", []),
                           }
                           for t, v in latency_data.items()
                       }}, f, indent=2)
        print(f"Summary saved → {summary_path}")

        # ── 플롯 ─────────────────────────────────────────────────
        block_size = args.block_size
        if block_size is None:
            all_steps_flat = [
                step
                for metrics in step_data.values()
                for step in metrics.get("block_steps", [])
            ]
            block_size = max(all_steps_flat) if all_steps_flat else 32

        if args.log:
            plot_step_distributions(step_data, block_size=block_size,
                                    model_name=model, output_dir=args.output_dir)
            plot_forward_latency(step_data, model_name=model, output_dir=args.output_dir)
            plot_context_length_distribution(latency_data, model_name=model, output_dir=args.output_dir)
            plot_scheduling_delays(latency_data, model_name=model, output_dir=args.output_dir)
            plot_ttfb_per_request(latency_data, model_name=model, output_dir=args.output_dir)
            plot_tpob_per_request(latency_data, model_name=model, output_dir=args.output_dir)
            plot_request_slo_scatter(
                latency_data,
                slo_data=slo_data,
                model_name=model,
                output_dir=args.output_dir,
            )
            plot_avg_block_steps_per_request(latency_data, model_name=model, output_dir=args.output_dir)
            plot_tpob_by_block_index(latency_data, model_name=model, output_dir=args.output_dir)
            plot_block_steps_per_request(latency_data, model_name=model, output_dir=args.output_dir)
            # takes too long time to make figure
            #plot_latency_metrics(latency_data, model_name=model, output_dir=args.output_dir)

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
