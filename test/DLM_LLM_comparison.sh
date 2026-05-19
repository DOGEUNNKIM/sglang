#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DLM_MODEL="${DLM_MODEL:-inclusionAI/LLaDA2.0-mini}"
LLM_MODEL="${LLM_MODEL:-inclusionAI/Ling-mini-2.0}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
TP_SIZE="${TP_SIZE:-1}"
DLM_MAX_RUNNING_REQUESTS="${DLM_MAX_RUNNING_REQUESTS:-64}"
LLM_MAX_RUNNING_REQUESTS="${LLM_MAX_RUNNING_REQUESTS:-164}"
WARMUP="${WARMUP:-32}"

DLM_RATES="${DLM_RATES:-164}"
LLM_RATES="${LLM_RATES:-164}"

NUM_EXAMPLES="${NUM_EXAMPLES:-164}"
NUM_OUTPUT_BLOCKS="${NUM_OUTPUT_BLOCKS:-0}"

DLM_PORT="${DLM_PORT:-31000}"
LLM_PORT="${LLM_PORT:-32000}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/nvme0/kdg6245}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/dlm_llm_comparison}"

FORWARD_TIME_S="${FORWARD_TIME_S:-0.04}"
THRESHOLD="${THRESHOLD:-0.95}"
GPU_FREE_MEMORY_MIN_MB="${GPU_FREE_MEMORY_MIN_MB:-70000}"

SERVER_PID=""

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
wait_server_ready() {
    local _url="${1}"
    local deadline=$((SECONDS + 600))
    until python -c "import urllib.request; urllib.request.urlopen('${_url}/health', timeout=5)" >/dev/null 2>&1; do
        if (( SECONDS >= deadline )); then
            echo "Server failed to become ready: ${_url}" >&2
            return 1
        fi
        sleep 3
    done
}

wait_gpu_memory_released() {
    local _prefix="${1}"
    local _gpu_deadline=$(( SECONDS + 60 ))
    until python3 -c "
import subprocess, sys
visible = '${CUDA_VISIBLE_DEVICES:-}'.strip()
tp_size = int('${TP_SIZE}')
min_free = int('${GPU_FREE_MEMORY_MIN_MB}')
out = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode()
free_by_idx = {}
for line in out.strip().splitlines():
    idx, free = [x.strip() for x in line.split(',', 1)]
    free_by_idx[idx] = int(free)
if visible:
    selected = [x.strip() for x in visible.split(',') if x.strip()]
    selected = [x for x in selected if x in free_by_idx][:tp_size]
else:
    selected = sorted(free_by_idx, key=lambda x: int(x))[:tp_size]
if len(selected) < tp_size:
    selected = sorted(free_by_idx, key=lambda x: int(x))[:tp_size]
sys.exit(0 if selected and all(free_by_idx[i] > min_free for i in selected) else 1)" 2>/dev/null; do
        if (( SECONDS >= _gpu_deadline )); then
            echo "[${_prefix}] WARNING: GPU memory not fully released after 60s, proceeding anyway" >&2
            break
        fi
        echo "[${_prefix}] waiting for GPU memory release..."
        sleep 3
    done
}

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        echo "[server] shutting down pid=${SERVER_PID}"
        local child_pids
        child_pids=$(pgrep -P "${SERVER_PID}" 2>/dev/null || true)
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        local deadline=$(( SECONDS + 30 ))
        while kill -0 "${SERVER_PID}" >/dev/null 2>&1; do
            if (( SECONDS >= deadline )); then
                kill -9 "${SERVER_PID}" >/dev/null 2>&1 || true
                break
            fi
            sleep 1
        done
        wait "${SERVER_PID}" >/dev/null 2>&1 || true
        if [[ -n "${child_pids}" ]]; then
            # shellcheck disable=SC2086
            kill -9 ${child_pids} >/dev/null 2>&1 || true
        fi
        wait_gpu_memory_released "server"
    fi
    SERVER_PID=""
}

trap stop_server EXIT

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
mkdir -p "${OUTPUT_ROOT}"

# Kill stale servers on both ports
for _port in "${DLM_PORT}" "${LLM_PORT}"; do
    _stale=$(lsof -ti tcp:"${_port}" 2>/dev/null || true)
    if [[ -n "${_stale}" ]]; then
        echo "[startup] killing stale process on port ${_port}: ${_stale}"
        # shellcheck disable=SC2086
        kill -9 ${_stale} 2>/dev/null || true
    fi
done
wait_gpu_memory_released "startup"

# ──────────────────────────────────────────────────────────────────────────────
# DLM benchmark (LLaDA2.0-mini, FCFS scheduler)
# ──────────────────────────────────────────────────────────────────────────────
echo
echo "============================================================"
echo "DLM Benchmark: ${DLM_MODEL}  (FCFS, humaneval)"
echo "Rates: ${DLM_RATES}"
echo "============================================================"

_dlm_rates=(${DLM_RATES})

for RATE in "${_dlm_rates[@]}"; do
    OUT_DIR="${OUTPUT_ROOT}/dlm/rate_${RATE}"
    mkdir -p "${OUT_DIR}"

    CONFIG_PATH="${OUT_DIR}/dlm_config.yaml"
    STEP_LOG_FILE="${OUT_DIR}/step_stats.jsonl"
    REQUEST_LATENCY_LOG_FILE="${OUT_DIR}/request_latency.jsonl"
    BATCH_LATENCY_LOG_FILE="${OUT_DIR}/batch_latency.jsonl"
    export CONFIG_PATH STEP_LOG_FILE REQUEST_LATENCY_LOG_FILE BATCH_LATENCY_LOG_FILE

    cat > "${CONFIG_PATH}" <<EOF
threshold: ${THRESHOLD}
dllm_admission_window: ${DLM_MAX_RUNNING_REQUESTS}
forward_time_s: ${FORWARD_TIME_S}
strict_prob: 1
scheduler_mode: fcfs
step_log_file: ${STEP_LOG_FILE}
request_latency_log_file: ${REQUEST_LATENCY_LOG_FILE}
batch_latency_log_file: ${BATCH_LATENCY_LOG_FILE}
EOF

    echo
    echo "---- DLM  rate=${RATE} ----"

    PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6 \
    python -m sglang.launch_server \
        --model-path "${DLM_MODEL}" \
        --port "${DLM_PORT}" \
        --trust-remote-code \
        --dllm-algorithm LowConfidence \
        --dllm-algorithm-config "${CONFIG_PATH}" \
        --attention-backend flashinfer \
        --max-running-requests "${DLM_MAX_RUNNING_REQUESTS}" \
        --cuda-graph-max-bs "${DLM_MAX_RUNNING_REQUESTS}" \
        --disable-cuda-graph-padding \
        --tp-size "${TP_SIZE}" \
        --mem-fraction-static 0.95 \
        >> "${OUT_DIR}/server.log" 2>&1 &
    SERVER_PID=$!

    echo "[server] pid=${SERVER_PID}, waiting for http://localhost:${DLM_PORT}/health"
    wait_server_ready "http://localhost:${DLM_PORT}"
    echo "[server] ready"

    PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6 \
    python test/dlm_benchmark.py \
        --base-url "http://localhost:${DLM_PORT}" \
        --model "${DLM_MODEL}" \
        --tasks humaneval \
        --block-size "${BLOCK_SIZE}" \
        --request-rate "${RATE}" \
        --num-threads "${NUM_EXAMPLES}" \
        --num-examples "${NUM_EXAMPLES}" \
        --warmup "${WARMUP}" \
        --num-output-blocks "${NUM_OUTPUT_BLOCKS}" \
        --output-dir "${OUT_DIR}" \
        --tp-size "${TP_SIZE}"

    stop_server
done

# ──────────────────────────────────────────────────────────────────────────────
# LLM benchmark (Ling-mini-2.0)
# ──────────────────────────────────────────────────────────────────────────────
echo
echo "============================================================"
echo "LLM Benchmark: ${LLM_MODEL}  (humaneval)"
echo "Rates: ${LLM_RATES}"
echo "============================================================"

_llm_rates=(${LLM_RATES})

for RATE in "${_llm_rates[@]}"; do
    OUT_DIR="${OUTPUT_ROOT}/llm/rate_${RATE}"
    mkdir -p "${OUT_DIR}"

    echo
    echo "---- LLM  rate=${RATE} ----"

    PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6 \
    python -m sglang.launch_server \
        --model-path "${LLM_MODEL}" \
        --port "${LLM_PORT}" \
        --trust-remote-code \
        --attention-backend flashinfer \
        --max-running-requests "${LLM_MAX_RUNNING_REQUESTS}" \
        --tp-size "${TP_SIZE}" \
        --mem-fraction-static 0.95 \
        >> "${OUT_DIR}/server.log" 2>&1 &
    SERVER_PID=$!

    echo "[server] pid=${SERVER_PID}, waiting for http://localhost:${LLM_PORT}/health"
    wait_server_ready "http://localhost:${LLM_PORT}"
    echo "[server] ready"

    python test/LLM_benchmark.py \
        --base-url "http://localhost:${LLM_PORT}" \
        --model "${LLM_MODEL}" \
        --request-rate "${RATE}" \
        --num-examples "${NUM_EXAMPLES}" \
        --num-threads "${NUM_EXAMPLES}" \
        --warmup "${WARMUP}" \
        --output-dir "${OUT_DIR}"

    stop_server
done

# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────
python3 -c "
import json
from pathlib import Path

dlm_model  = '${DLM_MODEL}'
llm_model  = '${LLM_MODEL}'
dlm_rates  = '${DLM_RATES}'.split()
llm_rates  = '${LLM_RATES}'.split()
output_root = Path('${OUTPUT_ROOT}')

dlm_tag = dlm_model.replace('/', '_')
llm_tag = llm_model.replace('/', '_')

W = 80
print()
print('=' * W)
print('DLM vs LLM — HumanEval Throughput Comparison')
print('=' * W)
hdr = f\"{'Model':<32} {'Rate':>6}  {'Throughput':>14}  {'pass@1':>8}  {'Latency':>10}\"
print(hdr)
print('-' * W)

def fmt_row(model, rate, d, score_key):
    thr = d.get('output_throughput_tok_s', 0) or 0
    sc  = d.get(score_key) or d.get('score') or 0
    lat = d.get('latency_s', 0) or 0
    return f'{model:<32} {rate:>6}  {thr:>13.1f}  {sc:>8.4f}  {lat:>9.1f}s'

for rate in dlm_rates:
    p = output_root / 'dlm' / f'rate_{rate}' / f'humaneval_{dlm_tag}.json'
    if not p.exists():
        print(f'{dlm_model:<32} {rate:>6}  {\"(no result)\":>14}')
        continue
    d = json.loads(p.read_text())
    print(fmt_row(dlm_model, rate, d, 'score'))

print()

for rate in llm_rates:
    p = output_root / 'llm' / f'rate_{rate}' / f'humaneval_{llm_tag}.json'
    if not p.exists():
        print(f'{llm_model:<32} {rate:>6}  {\"(no result)\":>14}')
        continue
    d = json.loads(p.read_text())
    print(fmt_row(llm_model, rate, d, 'pass@1'))

print('=' * W)
print(f'Results saved under: ${OUTPUT_ROOT}')
"

echo
echo "Done."
