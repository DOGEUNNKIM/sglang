"""
Standard LLM benchmark: HumanEval accuracy + throughput via sglang server.

Usage (external server already running):
    python test/LLM_comparison.py \
        --base-url http://localhost:31000 \
        --model inclusionAI/Ling-mini-2.0 \
        --output-dir /tmp/llm_comparison

Usage (launch server automatically):
    python test/LLM_comparison.py \
        --model inclusionAI/Ling-mini-2.0 \
        --launch-server \
        --chunked_prefill_size 2048\
        --tp-size 1 \
        --output-dir /tmp/llm_comparison
"""

import argparse
import json
import os
import subprocess
import sys
import time
import threading
import urllib.request

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
from pathlib import Path
from typing import Optional

DEFAULT_PORT = 31000
DEFAULT_URL = f"http://localhost:{DEFAULT_PORT}"


# ── Server helpers ─────────────────────────────────────────────────────────────

def _wait_server_ready(base_url: str, timeout: int = 600) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base_url}/health", timeout=5)
            return True
        except Exception:
            pass
        time.sleep(3)
    return False


def launch_server(model: str, port: int, tp_size: int,
                  max_running_requests: int,
                  chunked_prefill_size: Optional[int] = None) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model,
        "--port", str(port),
        "--trust-remote-code",
        "--tp-size", str(tp_size),
        "--max-running-requests", str(max_running_requests),
        "--attention-backend", "flashinfer",
    ]
    if chunked_prefill_size is not None:
        cmd += ["--chunked-prefill-size", str(chunked_prefill_size)]
    print(f"[server] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    if not _wait_server_ready(f"http://localhost:{port}"):
        proc.kill()
        raise RuntimeError("Server failed to start within 600s")
    print("[server] ready")
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


# ── Warmup ────────────────────────────────────────────────────────────────────

def run_warmup(base_url: str, model: str, num_requests: int = 4) -> None:
    if num_requests <= 0:
        return
    import concurrent.futures
    from sglang.test.simple_eval_common import ChatCompletionSampler
    sampler = ChatCompletionSampler(
        model=model, max_tokens=512,
        base_url=f"{base_url}/v1", temperature=0.0,
    )
    prompt = [{"role": "user", "content": "Hi"}]
    print(f"[warmup] sending {num_requests} requests...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as ex:
        futs = [ex.submit(sampler, prompt) for _ in range(num_requests)]
        concurrent.futures.wait(futs)
    print("[warmup] done")


# ── Rate-limited sampler ───────────────────────────────────────────────────────

class RateLimitedSampler:
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


# ── HumanEval run ─────────────────────────────────────────────────────────────

def run_humaneval(base_url: str, model: str, num_examples: Optional[int],
                  num_threads: int, request_rate: Optional[float],
                  num_samples_per_task: int = 1,
                  max_tokens: int = 512) -> dict:
    try:
        from sglang.test.simple_eval_humaneval import HumanEval
    except ImportError:
        print("[humaneval] human-eval package missing: pip install human-eval")
        sys.exit(1)
    from sglang.test.simple_eval_common import ChatCompletionSampler

    sampler = ChatCompletionSampler(
        model=model, max_tokens=max_tokens,
        base_url=f"{base_url}/v1", temperature=0.0,
    )
    if request_rate is not None:
        sampler = RateLimitedSampler(sampler, request_rate)

    he_total = 164
    he_examples = min(num_examples, he_total) if num_examples else None
    eval_obj = HumanEval(
        num_examples=he_examples,
        num_threads=num_threads,
        num_samples_per_task=num_samples_per_task,
        ks_passes=[1],
    )

    tic = time.perf_counter()
    result = eval_obj(sampler)
    latency = time.perf_counter() - tic

    total_tokens = sum(sampler._completion_tokens)
    return {
        "score": result.score,
        "pass@1": (result.metrics or {}).get("pass@1", result.score),
        "latency_s": round(latency, 2),
        "request_rate": request_rate,
        "output_throughput_tok_s": round(total_tokens / latency if latency > 0 else 0, 2),
        "total_completion_tokens": total_tokens,
        "num_examples": he_examples or he_total,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Standard LLM HumanEval benchmark")
    parser.add_argument("--model", default="inclusionAI/Ling-mini-2.0")
    parser.add_argument("--base-url", default=None,
                        help="URL of already-running sglang server. "
                             "If omitted, use --launch-server.")
    parser.add_argument("--launch-server", action="store_true",
                        help="Launch sglang server automatically.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-running-requests", type=int, default=164)
    parser.add_argument("--chunked-prefill-size", type=int, default=2048,
                        help="Max tokens per forward pass chunk (activation memory budget). "
                             "Default: auto (2048–4096 based on GPU). "
                             "Activation memory ∝ chunked_prefill_size × 1.5 GB.")
    parser.add_argument("--request-rate", type=float, default=164,
                        help="Requests per second. None = unlimited.")
    parser.add_argument("--num-examples", type=int, default=None,
                        help="Subset of HumanEval problems (max 164).")
    parser.add_argument("--num-threads", type=int, default=164)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--num-samples-per-task", type=int, default=1,
                        help="Samples per problem for pass@k estimation.")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max output tokens per request.")
    parser.add_argument("--output-dir", default="/tmp/llm_comparison")
    args = parser.parse_args()

    base_url = args.base_url or f"http://localhost:{args.port}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    server_proc = None
    try:
        if args.launch_server:
            server_proc = launch_server(
                args.model, args.port, args.tp_size, args.max_running_requests,
                chunked_prefill_size=args.chunked_prefill_size,
            )
        else:
            print(f"[info] using external server at {base_url}")

        run_warmup(base_url, args.model, args.warmup)

        print("\n[run] HumanEval ...")
        metrics = run_humaneval(
            base_url=base_url,
            model=args.model,
            num_examples=args.num_examples,
            num_threads=args.num_threads,
            request_rate=args.request_rate,
            num_samples_per_task=args.num_samples_per_task,
            max_tokens=args.max_tokens,
        )

        print(f"\n{'='*50}")
        print(f"  model      : {args.model}")
        print(f"  pass@1     : {metrics['pass@1']:.4f}")
        print(f"  throughput : {metrics['output_throughput_tok_s']:.1f} tok/s")
        print(f"  latency    : {metrics['latency_s']:.1f} s")
        print(f"  examples   : {metrics['num_examples']}")
        if args.request_rate:
            print(f"  rate       : {args.request_rate} req/s")
        print(f"{'='*50}\n")

        model_tag = args.model.replace("/", "_")
        out_path = output_dir / f"humaneval_{model_tag}.json"
        with open(out_path, "w") as f:
            json.dump({"model": args.model, "task": "humaneval", **metrics}, f, indent=2)
        print(f"[saved] {out_path}")

    finally:
        if server_proc is not None:
            stop_server(server_proc)


if __name__ == "__main__":
    main()
