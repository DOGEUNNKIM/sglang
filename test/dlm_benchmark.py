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
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

# ──────────────────────────────────────────────────────────────────────────────
# Step stats 수집
# ──────────────────────────────────────────────────────────────────────────────

STEP_LOG_FILE = os.environ.get("STEP_LOG_FILE", "/tmp/dlm_step_stats.jsonl")
REQUEST_LATENCY_LOG_FILE = os.environ.get("REQUEST_LATENCY_LOG_FILE", "/tmp/dlm_request_latency.jsonl")
BATCH_LATENCY_LOG_FILE = os.environ.get("BATCH_LATENCY_LOG_FILE", "/tmp/dlm_batch_latency.jsonl")


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
# RULER long-context eval (simonjegou/ruler)
# ──────────────────────────────────────────────────────────────────────────────

RULER_DATASET_ID = "simonjegou/ruler"


class RulerEval:
    """Evaluate on RULER (NIAH + related subtasks) at a fixed context length."""

    def __init__(self, context_length: int, num_examples: Optional[int], num_threads: int):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets  # required for ruler tasks")

        ds = load_dataset(RULER_DATASET_ID, str(context_length), split="test")
        examples = [dict(row) for row in ds]
        if num_examples:
            examples = random.Random(0).sample(examples, min(num_examples, len(examples)))
        self.examples = examples
        self.num_threads = num_threads

    def __call__(self, sampler) -> object:
        from sglang.test import simple_eval_common as common
        from sglang.test.simple_eval_common import SingleEvalResult

        def fn(row: dict) -> SingleEvalResult:
            prompt_messages = [sampler._pack_message(
                content=f"{row['context']}\n\n{row['question']}",
                role="user",
            )]
            try:
                response_text = sampler(prompt_messages) or ""
            except Exception:
                response_text = ""
            gold_answers = row["answer"]
            score = float(any(ans.lower() in response_text.lower() for ans in gold_answers))
            return SingleEvalResult(score=score)

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results, default_stats=("mean", "std"))


# ──────────────────────────────────────────────────────────────────────────────
# ShareGPT latency benchmark (no accuracy)
# ──────────────────────────────────────────────────────────────────────────────

SHAREGPT_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"


class ShareGPTEval:
    """Send ShareGPT first-turn prompts; measures latency only (score=None)."""

    def __init__(self, num_examples: int, num_threads: int, data_path: Optional[str] = None):
        import json
        from sglang.benchmark.utils import download_and_cache_hf_file, is_file_valid_json

        filename = data_path
        if not filename or not is_file_valid_json(filename):
            filename = download_and_cache_hf_file(
                repo_id=SHAREGPT_REPO_ID,
                filename=SHAREGPT_FILENAME,
            )

        with open(filename) as f:
            dataset = json.load(f)

        dataset = [
            data for data in dataset
            if len(data.get("conversations", data.get("conversation", []))) >= 1
        ]
        random.shuffle(dataset)

        self.examples = []
        for data in dataset:
            if len(self.examples) >= num_examples:
                break
            convs = data.get("conversations", data.get("conversation", []))
            if convs:
                self.examples.append({"prompt": convs[0]["value"]})

        self.num_threads = num_threads

    def __call__(self, sampler) -> object:
        from sglang.test import simple_eval_common as common
        from sglang.test.simple_eval_common import SingleEvalResult

        def fn(row: dict) -> SingleEvalResult:
            prompt_messages = [sampler._pack_message(content=row["prompt"], role="user")]
            try:
                sampler(prompt_messages)
            except Exception:
                pass
            return SingleEvalResult(score=None)

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)


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
_WARMUP_PROMPT = "Hi"


def run_warmup(base_url: str, model: str, num_requests: int = 4) -> None:
    """Send dummy requests to compile all Triton kernel shapes before benchmarking."""
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

TASK_API = {
    "gsm8k": "completion",
    "humaneval": "chat",
    "math": "chat",
    "gpqa": "chat",
    "mmlu": "chat",
    "ruler_4k": "chat",
    "ruler_8k": "chat",
    "ruler_16k": "chat",
    "sharegpt": "chat",
}
TASK_MAX_TOKENS = {
    "gsm8k": 512,
    "humaneval": 512,
    "math": 1024,
    "gpqa": 1024,
    "mmlu": 512,
    "sharegpt": 512,
    "ruler_4k": 128,
    "ruler_8k": 128,
    "ruler_16k": 128,
}

GPQA_DEFAULT_URL = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
MMLU_DEFAULT_URL = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"


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
             gpqa_data_path: Optional[str] = None,
             mmlu_data_path: Optional[str] = None,
             sharegpt_data_path: Optional[str] = None,
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
    elif task == "gpqa":
        from sglang.test.simple_eval_gpqa import GPQAEval
        filename = gpqa_data_path or GPQA_DEFAULT_URL
        GPQA_TOTAL = 198  # gpqa_diamond.csv has 198 examples
        gpqa_examples = min(num_examples, GPQA_TOTAL) if num_examples else None
        eval_obj = GPQAEval(filename=filename, num_examples=gpqa_examples,
                            num_threads=num_threads)
    elif task == "mmlu":
        from sglang.test.simple_eval_mmlu import MMLUEval
        filename = mmlu_data_path or MMLU_DEFAULT_URL
        eval_obj = MMLUEval(filename=filename, num_examples=num_examples,
                            num_threads=num_threads)
    elif task in ("ruler_4k", "ruler_8k", "ruler_16k"):
        context_length = int(task.split("_")[1].rstrip("k")) * 1024
        ruler_examples = num_examples or 500
        eval_obj = RulerEval(context_length=context_length, num_examples=ruler_examples,
                             num_threads=num_threads)
    elif task == "sharegpt":
        sharegpt_examples = num_examples or 1000
        eval_obj = ShareGPTEval(num_examples=sharegpt_examples, num_threads=num_threads,
                                data_path=sharegpt_data_path)
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
            labels=[t.upper() for t in forward_tasks],
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
            labels=[t.upper() for t in block_tasks],
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


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    import numpy as np

    return float(np.percentile(values, q))


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
    "incoming_prefill": "initial prefill",
    "staging_prefill": "staging prefill",
    "incoming_decode": "initial decode",
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
    phases = set(record.get("per_req_phase") or [])
    has_mask = record.get("per_req_has_mask") or []
    if has_mask and any(has_mask) and any(not value for value in has_mask):
        return "mixed"
    if has_mask and any(has_mask):
        return "decode"
    if "incoming_prefill" in phases:
        return "initial_prefill"
    if "staging_prefill" in phases:
        return "staging_prefill"
    return record.get("phase", "unknown")


def summarize_latency_metrics(
    request_records: List[Dict[str, Any]],
    batch_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    import numpy as np

    ttfb_ms = _values(request_records, "ttfb_ms")
    tpob_ms = _values(request_records, "tpob_ms")
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
    phase_sequence = [record.get("phase") for record in batch_records]
    req_phase_counts = _count_per_req_phases(batch_records)

    return {
        "n_request_records": len(request_records),
        "mean_ttfb_ms": float(np.mean(ttfb_ms)) if ttfb_ms else None,
        "p50_ttfb_ms": _percentile(ttfb_ms, 50),
        "p95_ttfb_ms": _percentile(ttfb_ms, 95),
        "p99_ttfb_ms": _percentile(ttfb_ms, 99),
        "mean_tpob_ms": float(np.mean(tpob_ms)) if tpob_ms else None,
        "p50_tpob_ms": _percentile(tpob_ms, 50),
        "p95_tpob_ms": _percentile(tpob_ms, 95),
        "p99_tpob_ms": _percentile(tpob_ms, 99),
        "n_batch_records": len(batch_records),
        "avg_prefill_req_ms": avg_prefill_req_ms,
        "avg_decode_block_ms": avg_decode_block_ms,
        "avg_initial_prefill_req_ms": avg_initial_prefill_req_ms,
        "avg_staging_prefill_req_ms": avg_staging_prefill_req_ms,
        "avg_actual_decode_block_ms": avg_actual_decode_block_ms,
        "request_phase_counts": req_phase_counts,
        "mixed_mask_batches": sum(
            1 for record in batch_records if record.get("is_mixed_mask_batch")
        ),
        "prefill_batches": phase_sequence.count("prefill"),
        "decode_batches": phase_sequence.count("decode"),
        "phase_switches": sum(
            1
            for prev, cur in zip(phase_sequence, phase_sequence[1:])
            if prev != cur
        ),
    }


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
                labels=[task.upper() for task in plot_tasks],
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
        "prefill": "#2C7BB6",
        "unknown": "#777777",
    }
    phase_marker = {
        "initial_prefill": "s",
        "staging_prefill": "D",
        "decode": "o",
        "mixed": "P",
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
        for p in ("initial_prefill", "staging_prefill", "decode", "mixed")
    ]
    ax3.legend(handles=handles, loc="upper right")

    out_path3 = Path(output_dir) / f"phase_sequence_{model_tag}.png"
    fig3.tight_layout()
    fig3.savefig(out_path3, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path3}")
    plt.close(fig3)

    # ── Per-batch request phase composition ───────────────────────────
    phase_keys = [
        "incoming_prefill",
        "staging_prefill",
        "incoming_decode",
        "staging_decode",
    ]
    phase_colors = {
        "incoming_prefill": "#74A9CF",
        "staging_prefill": "#2C7BB6",
        "incoming_decode": "#Fdae61",
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
                        choices=["gsm8k", "humaneval", "math", "gpqa", "mmlu",
                                 "ruler_4k", "ruler_8k", "ruler_16k", "sharegpt"],
                        default=["gsm8k", "humaneval", "math"])
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=64)
    parser.add_argument("--request-rate", type=float, default=None,
                        help="전체 request 시작 rate(req/s). 미지정 시 기존 closed-loop 방식")
    parser.add_argument("--warmup", type=int, default=0,
                        help="벤치마크 전 warmup 요청 수 (0=비활성화)")
    parser.add_argument("--num-output-blocks", type=int, default=0,
                        help="고정 출력 블록 수 (0=task별 기본값 사용)")
    parser.add_argument("--gsm8k-data-path", type=str, default=None)
    parser.add_argument("--math-data-path", type=str, default=None)
    parser.add_argument("--gpqa-data-path", type=str, default=None)
    parser.add_argument("--mmlu-data-path", type=str, default=None)
    parser.add_argument("--sharegpt-data-path", type=str, default=None)

    # 출력
    parser.add_argument("--output-dir", type=str, default="/tmp/dlm_results")
    parser.add_argument("--block-size", type=int, default=None,
                        help="모델의 block_size (LLaDA2=32, SDAR=4). "
                             "미지정 시 step 축 범위를 자동 추정")
    parser.add_argument("--log", action="store_true",
                        help="DLM step/request/batch 로그를 수집하고 그래프를 생성")

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

        if getattr(args, "warmup", 0) > 0:
            run_warmup(base_url, model, num_requests=args.warmup)

        all_results: Dict[str, Dict] = {}
        step_data: Dict[str, Dict[str, List[int]]] = {}
        latency_data: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        for task in args.tasks:
            print(f"\n{'='*60}")
            print(f"Task: {task.upper()}  |  model: {model}")
            print(f"{'='*60}")

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
                gpqa_data_path=getattr(args, "gpqa_data_path", None),
                mmlu_data_path=getattr(args, "mmlu_data_path", None),
                sharegpt_data_path=getattr(args, "sharegpt_data_path", None),
                request_rate=args.request_rate,
                num_output_blocks=getattr(args, "num_output_blocks", 0),
                block_size=args.block_size or 32,
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
            step_data[task] = {
                "raw_forward_calls": raw_forward_calls,
                "block_steps": block_steps,
            }
            latency_records = (
                read_latency_logs() if args.log else {"request": [], "batch": []}
            )
            latency_data[task] = latency_records
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
            if args.log:
                print(
                    f"[{task}] latency stats: "
                    f"mean_ttfb_ms={latency_summary.get('mean_ttfb_ms')}  "
                    f"p95_ttfb_ms={latency_summary.get('p95_ttfb_ms')}  "
                    f"p99_ttfb_ms={latency_summary.get('p99_ttfb_ms')}  "
                    f"mean_tpob_ms={latency_summary.get('mean_tpob_ms')}  "
                    f"p95_tpob_ms={latency_summary.get('p95_tpob_ms')}  "
                    f"p99_tpob_ms={latency_summary.get('p99_tpob_ms')}  "
                    f"avg_prefill_req_ms={latency_summary.get('avg_prefill_req_ms')}  "
                    f"avg_decode_block_ms={latency_summary.get('avg_decode_block_ms')}  "
                    f"initial_prefill_req_ms={latency_summary.get('avg_initial_prefill_req_ms')}  "
                    f"staging_prefill_req_ms={latency_summary.get('avg_staging_prefill_req_ms')}  "
                    f"mixed_batches={latency_summary.get('mixed_mask_batches')}  "
                    f"phase_switches={latency_summary.get('phase_switches')}"
                )
            else:
                print(f"[{task}] DLM logging disabled; pass --log to collect latency/step stats")
            if latency_summary.get("request_phase_counts"):
                print(
                    f"[{task}] request phase counts: "
                    f"{latency_summary['request_phase_counts']}"
                )

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

        #if args.log:
        #    plot_step_distributions(step_data, block_size=block_size,
        #                            model_name=model, output_dir=args.output_dir)
        #    plot_latency_metrics(latency_data, model_name=model, output_dir=args.output_dir)

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
