#!/usr/bin/env bash
# Run all schedulers for a single task/rate, then generate scatter plot.
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-JetLM/SDAR-8B-Chat}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
TASK="${TASK:-gsm8k}"
RATE="${RATE:-4}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1000}"
NUM_OUTPUT_BLOCKS="${NUM_OUTPUT_BLOCKS:-0}"
SCHEDULERS=(${SCHEDULERS:-TTFB DECODE LST SOLA FCFS PREFILL})

MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
WARMUP="${WARMUP:-32}"
TP_SIZE="${TP_SIZE:-1}"
FORWARD_TIME_S="${FORWARD_TIME_S:-0.08}"
THRESHOLD="${THRESHOLD:-0.95}"
STRICT_MULTIPLIER="${STRICT_MULTIPLIER:-10.0}"
RELEASE_MULTIPLIER="${RELEASE_MULTIPLIER:-20.0}"
STRICT_PROB="${STRICT_PROB:-1}"
PREFILL_FORWARD_TIME_S="${PREFILL_FORWARD_TIME_S:-}"
DECODE_FORWARD_TIME_S="${DECODE_FORWARD_TIME_S:-}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/nvme0/kdg6245}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/dlm_scatter_comparison_SDAR}"
PORT="${PORT:-30005}"
BASE_URL="http://localhost:${PORT}"
CONFIG_PATH="${OUTPUT_ROOT}/dlm_algo_config.yaml"
STEP_LOG_FILE="${OUTPUT_ROOT}/dlm_step_stats.jsonl"
REQUEST_LATENCY_LOG_FILE="${OUTPUT_ROOT}/dlm_request_latency.jsonl"
BATCH_LATENCY_LOG_FILE="${OUTPUT_ROOT}/dlm_batch_latency.jsonl"
export CONFIG_PATH STEP_LOG_FILE REQUEST_LATENCY_LOG_FILE BATCH_LATENCY_LOG_FILE
GPU_FREE_MEMORY_MIN_MB="${GPU_FREE_MEMORY_MIN_MB:-70000}"

SERVER_PID=""

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
_compute_task_slo() {
    local _ideal_ttfb_ms="${1}" _ideal_tpob_ms="${2}" _multiplier="${3}"
    python3 -c "
m = float('${_multiplier}')
print(f'{float(\"${_ideal_ttfb_ms}\")*m/1000:.4f} {float(\"${_ideal_tpob_ms}\")*m/1000:.4f}')
"
}

write_dllm_config() {
    local _strict_ttfb="${1:-}" _strict_tpob="${2:-}"
    local _release_ttfb="${3:-}" _release_tpob="${4:-}"
    local _scheduler_mode="${5:-prefill}"
    local _admission_window="${6:-${MAX_RUNNING_REQUESTS}}"

    cat > "${CONFIG_PATH}" <<EOF
threshold: ${THRESHOLD}
dllm_admission_window: ${_admission_window}
forward_time_s: ${FORWARD_TIME_S}
strict_prob: ${STRICT_PROB}
scheduler_mode: ${_scheduler_mode}
step_log_file: ${STEP_LOG_FILE}
request_latency_log_file: ${REQUEST_LATENCY_LOG_FILE}
batch_latency_log_file: ${BATCH_LATENCY_LOG_FILE}
EOF
    [[ -n "${_strict_ttfb}" ]]           && echo "strict_ttfb_slo: ${_strict_ttfb}"                 >> "${CONFIG_PATH}"
    [[ -n "${_strict_tpob}" ]]           && echo "strict_tpob_slo: ${_strict_tpob}"                 >> "${CONFIG_PATH}"
    [[ -n "${_release_ttfb}" ]]          && echo "release_ttfb_slo: ${_release_ttfb}"               >> "${CONFIG_PATH}"
    [[ -n "${_release_tpob}" ]]          && echo "release_tpob_slo: ${_release_tpob}"               >> "${CONFIG_PATH}"
    [[ -n "${PREFILL_FORWARD_TIME_S}" ]] && echo "prefill_forward_time_s: ${PREFILL_FORWARD_TIME_S}" >> "${CONFIG_PATH}"
    [[ -n "${DECODE_FORWARD_TIME_S}" ]]  && echo "decode_forward_time_s: ${DECODE_FORWARD_TIME_S}"   >> "${CONFIG_PATH}"
    return 0
}

_parse_calib_metric() {
    local _out="${1}" _task="${2}" _field="${3}"
    local _model_tag="${MODEL_PATH//\//_}"
    local _json="${_out}/${_task}_${_model_tag}.json"
    python3 -c "
import json, sys
try:
    d = json.load(open('${_json}'))
    v = d.get('${_field}') or d.get('latency_stats', {}).get('${_field}')
    print('' if v is None else f'{v:.4f}')
except Exception:
    print('')
"
}

wait_server_ready() {
    local deadline=$((SECONDS + 600))
    until python -c "import urllib.request; urllib.request.urlopen('${BASE_URL}/health', timeout=5)" >/dev/null 2>&1; do
        if (( SECONDS >= deadline )); then
            echo "Server failed to become ready: ${BASE_URL}" >&2
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
SERVER_LOG="${OUTPUT_ROOT}/server_log.txt"

_stale_pid=$(lsof -ti tcp:"${PORT}" 2>/dev/null || true)
if [[ -n "${_stale_pid}" ]]; then
    echo "[startup] killing stale process on port ${PORT}: ${_stale_pid}"
    # shellcheck disable=SC2086
    kill -9 ${_stale_pid} 2>/dev/null || true
    wait_gpu_memory_released "startup"
fi

declare -A IDEAL_TTFB_MS IDEAL_TPOB_MS
IDEAL_TTFB_MS["${TASK}"]=""
IDEAL_TPOB_MS["${TASK}"]=""

# ──────────────────────────────────────────────────────────────────────────────
# Run schedulers
# ──────────────────────────────────────────────────────────────────────────────
for SCHEDULER in "${SCHEDULERS[@]}"; do
    OUT_DIR="${OUTPUT_ROOT}/scheduler_${SCHEDULER}/request_rate_${RATE}/${TASK}"
    mkdir -p "${OUT_DIR}"

    CONFIG_PATH="${OUT_DIR}/dllm_algo_config.yaml"
    STEP_LOG_FILE="${OUT_DIR}/dlm_step_stats_${TASK}.jsonl"
    REQUEST_LATENCY_LOG_FILE="${OUT_DIR}/dlm_request_latency_${TASK}.jsonl"
    BATCH_LATENCY_LOG_FILE="${OUT_DIR}/dlm_batch_latency_${TASK}.jsonl"
    export CONFIG_PATH STEP_LOG_FILE REQUEST_LATENCY_LOG_FILE BATCH_LATENCY_LOG_FILE
    RUN_SERVER_LOG="${OUT_DIR}/server.log"

    _ideal_ttfb="${IDEAL_TTFB_MS[${TASK}]:-}"
    _ideal_tpob="${IDEAL_TPOB_MS[${TASK}]:-}"

    _strict_ttfb="" _strict_tpob="" _release_ttfb="" _release_tpob=""
    if [[ -n "${_ideal_ttfb}" && -n "${_ideal_tpob}" ]]; then
        read -r _strict_ttfb  _strict_tpob  <<< "$(_compute_task_slo "${_ideal_ttfb}"  "${_ideal_tpob}"  "${STRICT_MULTIPLIER}")"
        read -r _release_ttfb _release_tpob <<< "$(_compute_task_slo "${_ideal_ttfb}"  "${_ideal_tpob}"  "${RELEASE_MULTIPLIER}")"
        echo "[slo] task=${TASK}  ideal=${_ideal_ttfb}ms/${_ideal_tpob}ms  strict=${_strict_ttfb}s/${_strict_tpob}s  release=${_release_ttfb}s/${_release_tpob}s"
    else
        echo "[slo] task=${TASK}  no calibration data — SLOs omitted"
    fi

    if [[ "${SCHEDULER}" == "DECODE" ]]; then
        _admission_window="${MAX_RUNNING_REQUESTS}"
    else
        _admission_window="${NUM_EXAMPLES}"
    fi

    case "${SCHEDULER}" in
        LST)    _scheduler_mode="lst"    ;;
        FCFS)   _scheduler_mode="fcfs"   ;;
        SOLA)   _scheduler_mode="sola"   ;;
        TTFB)   _scheduler_mode="ttfb"   ;;
        *)      _scheduler_mode="prefill" ;;
    esac

    write_dllm_config \
        "${_strict_ttfb}" "${_strict_tpob}" \
        "${_release_ttfb}" "${_release_tpob}" \
        "${_scheduler_mode}" \
        "${_admission_window}"

    echo
    echo "============================================================"
    echo "scheduler=${SCHEDULER}  rate=${RATE}  task=${TASK}"
    echo "============================================================"

    PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6 \
    python -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --port "${PORT}" \
        --trust-remote-code \
        --dllm-algorithm LowConfidence \
        --dllm-algorithm-config "${CONFIG_PATH}" \
        --attention-backend flashinfer \
        --max-running-requests "${MAX_RUNNING_REQUESTS}" \
        --cuda-graph-max-bs "${MAX_RUNNING_REQUESTS}" \
        --disable-cuda-graph-padding \
        --tp-size "${TP_SIZE}" \
        --mem-fraction-static 0.90 \
        >> "${RUN_SERVER_LOG}" 2>&1 &
    SERVER_PID=$!

    echo "[server] pid=${SERVER_PID}, waiting for ${BASE_URL}/health"
    wait_server_ready
    echo "[server] ready"

    PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6 \
    python test/dlm_benchmark.py \
        --base-url "${BASE_URL}" \
        --model "${MODEL_PATH}" \
        --tasks "${TASK}" \
        --block-size "${BLOCK_SIZE}" \
        --log \
        --request-rate "${RATE}" \
        --num-threads "${NUM_EXAMPLES}" \
        --num-examples "${NUM_EXAMPLES}" \
        --warmup "${WARMUP}" \
        --num-output-blocks "${NUM_OUTPUT_BLOCKS}" \
        --output-dir "${OUT_DIR}" \
        --tp-size "${TP_SIZE}"

    stop_server

    # Extract calibration values inline
    if [[ "${SCHEDULER}" == "TTFB" ]]; then
        _val_ttfb=$(_parse_calib_metric "${OUT_DIR}" "${TASK}" "p50_ideal_ttfb_ms")
        _val_tpob=$(_parse_calib_metric "${OUT_DIR}" "${TASK}" "p50_ideal_tpob_ms")
        IDEAL_TTFB_MS["${TASK}"]="${_val_ttfb:-}"
        IDEAL_TPOB_MS["${TASK}"]="${_val_tpob:-}"
        echo "[ideal] task=${TASK}  ideal_ttfb=${_val_ttfb:-N/A}ms  ideal_tpob=${_val_tpob:-N/A}ms"
    fi

done  # SCHEDULER

# ──────────────────────────────────────────────────────────────────────────────
# Write slo_config.json
# ──────────────────────────────────────────────────────────────────────────────
SLO_CONFIG_PATH="${OUTPUT_ROOT}/slo_config.json"
python3 -c "
import json
strict_m  = float('${STRICT_MULTIPLIER}')
release_m = float('${RELEASE_MULTIPLIER}')
ittfb = float('${IDEAL_TTFB_MS[${TASK}]:-0}')
itpob = float('${IDEAL_TPOB_MS[${TASK}]:-0}')
slos = {}
if ittfb > 0 and itpob > 0:
    slos['${TASK}'] = {
        'strict':  {'ttfb_ms': ittfb * strict_m,  'tpob_ms': itpob * strict_m},
        'relaxed': {'ttfb_ms': ittfb * release_m, 'tpob_ms': itpob * release_m},
    }
with open('${SLO_CONFIG_PATH}', 'w') as f:
    json.dump(slos, f, indent=2)
print('[slo_config] written to ${SLO_CONFIG_PATH}')
print(json.dumps(slos, indent=2))
"

# ──────────────────────────────────────────────────────────────────────────────
# Build slo_summary.json (needed by plot script)
# ──────────────────────────────────────────────────────────────────────────────
SUMMARY_PATH="${OUTPUT_ROOT}/slo_summary.json"
python3 -c "
import json
from pathlib import Path

output_root = Path('${OUTPUT_ROOT}')
schedulers  = '${SCHEDULERS[*]}'.split()
task        = '${TASK}'
rate        = '${RATE}'
model_tag   = '${MODEL_PATH}'.replace('/', '_')

def _read_bench_metrics(out_dir):
    p = Path(out_dir) / f'{task}_{model_tag}.json'
    if not p.exists():
        return None, None, None, None
    try:
        d = json.loads(p.read_text())
        ls = d.get('latency_stats', d)
        return d.get('score'), d.get('score:std'), ls.get('p99_ttfb_ms'), ls.get('p99_tpob_ms')
    except Exception:
        return None, None, None, None

summary = {}
for sched in schedulers:
    out_dir  = output_root / f'scheduler_{sched}' / f'request_rate_{rate}' / task
    slo_path = out_dir / 'slo_rates.json'
    score, score_std, p99_ttfb, p99_tpob = _read_bench_metrics(out_dir)
    if rate not in summary.get(sched, {}):
        summary.setdefault(sched, {})[rate] = {}
    summary[sched][rate][task] = {
        'score': score, 'score_std': score_std,
        'p99_ttfb_ms': p99_ttfb, 'p99_tpob_ms': p99_tpob,
        'strict_ttfb': None, 'strict_tpob': None, 'strict_all': None,
        'relaxed_ttfb': None, 'relaxed_tpob': None, 'relaxed_all': None,
    }
    if slo_path.exists():
        data = json.loads(slo_path.read_text())
        if task in data:
            rates_d = data[task].get('rates', {})
            summary[sched][rate][task].update({
                'strict_ttfb':  rates_d.get('strict_ttfb'),
                'strict_tpob':  rates_d.get('strict_tpob'),
                'strict_all':   rates_d.get('strict_all'),
                'relaxed_ttfb': rates_d.get('relaxed_ttfb'),
                'relaxed_tpob': rates_d.get('relaxed_tpob'),
                'relaxed_all':  rates_d.get('relaxed_all'),
            })

Path('${SUMMARY_PATH}').write_text(json.dumps(summary, indent=2))
print(f'[summary] saved → ${SUMMARY_PATH}')
"

# Run dlm_slorate.py per scheduler for slo_rates.json
for SCHEDULER in "${SCHEDULERS[@]}"; do
    OUT_DIR="${OUTPUT_ROOT}/scheduler_${SCHEDULER}/request_rate_${RATE}/${TASK}"
    python test/dlm_slorate.py \
        --latency-dir "${OUT_DIR}" \
        --tasks "${TASK}" \
        --slo-config "${SLO_CONFIG_PATH}" \
        --output-json "${OUT_DIR}/slo_rates.json" 2>/dev/null || true
done

# ──────────────────────────────────────────────────────────────────────────────
# Scatter plot
# ──────────────────────────────────────────────────────────────────────────────
echo
echo "============================================================"
echo "Generating scatter plot"
echo "============================================================"

python test/plot_dlm_scheduler_comparison.py \
    --summary "${SUMMARY_PATH}" \
    --output "${OUTPUT_ROOT}/scatter.png" \
    --slo-config "${SLO_CONFIG_PATH}" \
    --bar-task "${TASK}" \
    --bar-rate "${RATE}" \
    --no-p99 \
    --no-sr \
    --no-bar

echo
echo "Done. Scatter: ${OUTPUT_ROOT}/scatter_scatter.png"
echo "      Combined: ${OUTPUT_ROOT}/scatter_scatter_combined.png"
