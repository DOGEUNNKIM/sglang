#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-JetLM/SDAR-4B-Chat}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/dlm_sched_comparison}"
BLOCK_SIZE="${BLOCK_SIZE:-4}"
WARMUP="${WARMUP:-16}"
NUM_OUTPUT_BLOCKS="${NUM_OUTPUT_BLOCKS:-0}"
REQUEST_RATES=(${REQUEST_RATES:-2 4 6 8 10 12 14})
TASKS=(${TASKS:-humaneval gsm8k}) # humaneval gsm8k math
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
DLLM_ADMISSION_WINDOW="${DLLM_ADMISSION_WINDOW:-200}"
PORT="${PORT:-30000}"
BASE_URL="${BASE_URL:-http://localhost:${PORT}}"
THRESHOLD="${THRESHOLD:-0.95}"
NUM_THREADS="${NUM_THREADS:-200}"
SCHEDULERS=(${SCHEDULERS:-LST FCFS PREFILL TTFB DECODE})
STRICT_MULTIPLIER="${STRICT_MULTIPLIER:-5.0}"
RELEASE_MULTIPLIER="${RELEASE_MULTIPLIER:-25.0}"
STRICT_PROB="${STRICT_PROB:-0.5}"
FORWARD_TIME_S="${FORWARD_TIME_S:-0.030}"
PREFILL_FORWARD_TIME_S="${PREFILL_FORWARD_TIME_S:-}"
DECODE_FORWARD_TIME_S="${DECODE_FORWARD_TIME_S:-}"
EXPECTED_UNMASK_PER_FORWARD="${EXPECTED_UNMASK_PER_FORWARD:-}"
CONFIG_PATH="${CONFIG_PATH:-/tmp/dlm_algo_config_sched_cmp.yaml}"
STEP_LOG_FILE="${STEP_LOG_FILE:-/tmp/dlm_step_stats_cmp.jsonl}"
REQUEST_LATENCY_LOG_FILE="${REQUEST_LATENCY_LOG_FILE:-/tmp/dlm_request_latency_cmp.jsonl}"
BATCH_LATENCY_LOG_FILE="${BATCH_LATENCY_LOG_FILE:-/tmp/dlm_batch_latency_cmp.jsonl}"
TP_SIZE="${TP_SIZE:-1}"

SERVER_PID=""

_ideal_ttfb_ms() {
    case "${1}" in
        gsm8k)     echo "417.2" ;;
        humaneval) echo "312.9" ;;
        math)      echo "404.5" ;;
        *)         echo "417.2" ;;
    esac
}

_default_expected_unmask_per_forward() {
    case "${1}" in
        gsm8k)     echo "2.666666667" ;;
        humaneval) echo "5.333333334" ;;
        math)      echo "2.133333334" ;;
        *)         echo "2.666666667" ;;
    esac
}

_compute_task_slo() {
    local task="${1}" multiplier="${2}"
    if [[ -z "${multiplier}" ]]; then
        echo ""
        return
    fi
    local _decode_fwd_s="${DECODE_FORWARD_TIME_S:-${FORWARD_TIME_S}}"
    local _unmask_per_fwd="${EXPECTED_UNMASK_PER_FORWARD:-$(_default_expected_unmask_per_forward "${task}")}"
    python3 -c "
import math
ttfb_ms     = float('$(_ideal_ttfb_ms "${task}")')
block_size  = int('${BLOCK_SIZE}')
decode_fwd_s = float('${_decode_fwd_s}')
unmask_per_fwd = float('${_unmask_per_fwd}')
m = float('${multiplier}')

decode_steps = math.ceil(block_size / unmask_per_fwd)
ideal_tpob_s = decode_steps * decode_fwd_s

print(f'{ttfb_ms*m/1000:.4f} {ideal_tpob_s*m:.4f} {unmask_per_fwd:.4f}')
"
}

write_dllm_config() {
    local _strict_ttfb="${1:-}"
    local _strict_tpob="${2:-}"
    local _release_ttfb="${3:-}"
    local _release_tpob="${4:-}"
    local _unmask_per_fwd="${EXPECTED_UNMASK_PER_FORWARD:-${5:-}}"
    local _scheduler_mode="${6:-prefill}"
    local _admission_window="${7:-${DLLM_ADMISSION_WINDOW}}"

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
    [[ -n "${_strict_ttfb}" ]]           && echo "strict_ttfb_slo: ${_strict_ttfb}"                   >> "${CONFIG_PATH}"
    [[ -n "${_strict_tpob}" ]]           && echo "strict_tpob_slo: ${_strict_tpob}"                   >> "${CONFIG_PATH}"
    [[ -n "${_release_ttfb}" ]]          && echo "release_ttfb_slo: ${_release_ttfb}"                 >> "${CONFIG_PATH}"
    [[ -n "${_release_tpob}" ]]          && echo "release_tpob_slo: ${_release_tpob}"                 >> "${CONFIG_PATH}"
    [[ -n "${_unmask_per_fwd}" ]]        && echo "expected_unmask_per_forward: ${_unmask_per_fwd}"     >> "${CONFIG_PATH}"
    [[ -n "${PREFILL_FORWARD_TIME_S}" ]] && echo "prefill_forward_time_s: ${PREFILL_FORWARD_TIME_S}"   >> "${CONFIG_PATH}"
    [[ -n "${DECODE_FORWARD_TIME_S}" ]]  && echo "decode_forward_time_s: ${DECODE_FORWARD_TIME_S}"     >> "${CONFIG_PATH}"
    return 0
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

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        echo "[server] shutting down pid=${SERVER_PID}"
        local child_pids
        child_pids=$(pgrep -P "${SERVER_PID}" 2>/dev/null || true)

        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        local deadline=$(( SECONDS + 30 ))
        while kill -0 "${SERVER_PID}" >/dev/null 2>&1; do
            if (( SECONDS >= deadline )); then
                echo "[server] grace period expired, sending SIGKILL to ${SERVER_PID}"
                kill -9 "${SERVER_PID}" >/dev/null 2>&1 || true
                break
            fi
            sleep 1
        done
        wait "${SERVER_PID}" >/dev/null 2>&1 || true
        if [[ -n "${child_pids}" ]]; then
            echo "[server] killing surviving worker pids: ${child_pids}"
            # shellcheck disable=SC2086
            kill -9 ${child_pids} >/dev/null 2>&1 || true
        fi
        sleep 2
    fi
    SERVER_PID=""
}

trap stop_server EXIT

SERVER_LOG="${OUTPUT_ROOT}/server_log.txt"
mkdir -p "${OUTPUT_ROOT}"

for SCHEDULER in "${SCHEDULERS[@]}"; do
    for RATE in "${REQUEST_RATES[@]}"; do
        _admission_window="${DLLM_ADMISSION_WINDOW}"
        if [[ "${SCHEDULER}" == "DECODE" ]]; then
            _admission_window="${MAX_RUNNING_REQUESTS}"
        fi

        OUT_DIR="${OUTPUT_ROOT}/scheduler_${SCHEDULER}/request_rate_${RATE}"
        mkdir -p "${OUT_DIR}"

        for TASK in "${TASKS[@]}"; do
            echo
            echo "============================================================"
            echo "Scheduler comparison: scheduler=${SCHEDULER}, rate=${RATE}, task=${TASK}"
            echo "output_dir=${OUT_DIR}"
            echo "A fresh server will be started for this run."
            echo "============================================================"

            _strict_vals=$(_compute_task_slo "${TASK}" "${STRICT_MULTIPLIER}")
            _release_vals=$(_compute_task_slo "${TASK}" "${RELEASE_MULTIPLIER}")
            read -r _strict_ttfb _strict_tpob _task_unmask_per_fwd <<< "${_strict_vals}"
            read -r _release_ttfb _release_tpob _                  <<< "${_release_vals}"
            if [[ -n "${_strict_vals}" ]]; then
                echo "[slo] task=${TASK}  strict=${_strict_ttfb}s/${_strict_tpob}s  release=${_release_ttfb}s/${_release_tpob}s  unmask/fwd=${_task_unmask_per_fwd}  (${STRICT_MULTIPLIER}×/${RELEASE_MULTIPLIER}× ideal)"
            fi

            case "${SCHEDULER}" in
                LST)    _scheduler_mode="lst"    ;;
                FCFS)   _scheduler_mode="fcfs"   ;;
                SOLA)   _scheduler_mode="sola"   ;;
                TTFB)   _scheduler_mode="ttfb"   ;;
                *)      _scheduler_mode="prefill" ;;
            esac

            write_dllm_config \
                "${_strict_ttfb:-}" "${_strict_tpob:-}" \
                "${_release_ttfb:-}" "${_release_tpob:-}" \
                "${_task_unmask_per_fwd:-}" \
                "${_scheduler_mode}" \
                "${_admission_window}"

            echo "===== scheduler=${SCHEDULER} rate=${RATE} task=${TASK} =====" >> "${SERVER_LOG}"

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
                >> "${SERVER_LOG}" 2>&1 &
            SERVER_PID=$!

            echo "[server] pid=${SERVER_PID}, waiting for ${BASE_URL}/health"
            wait_server_ready
            echo "[server] ready"

            BENCH_ARGS=(
                test/dlm_benchmark.py
                --base-url "${BASE_URL}"
                --model "${MODEL_PATH}"
                --tasks "${TASK}"
                --block-size "${BLOCK_SIZE}"
                --log
                --request-rate "${RATE}"
                --num-threads "${NUM_THREADS}"
                --warmup "${WARMUP}"
                --num-output-blocks "${NUM_OUTPUT_BLOCKS}"
                --output-dir "${OUT_DIR}"
                --tp-size "${TP_SIZE}"
            )

            PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6 \
            python "${BENCH_ARGS[@]}"

            stop_server
        done  # TASK
    done  # RATE
done  # SCHEDULER

echo
echo "============================================================"
echo "DLM Scheduler Comparison — SLO Summary"
echo "============================================================"

for SCHEDULER in "${SCHEDULERS[@]}"; do
    for RATE in "${REQUEST_RATES[@]}"; do
        OUT_DIR="${OUTPUT_ROOT}/scheduler_${SCHEDULER}/request_rate_${RATE}"
        SLO_PATH="${OUT_DIR}/slo_rates.json"

        echo
        echo "scheduler=${SCHEDULER}  request_rate=${RATE}"
        echo "------------------------------------------------------------"

        python test/dlm_slorate.py \
            --latency-dir "${OUT_DIR}" \
            --tasks "${TASKS[@]}" \
            --output-json "${SLO_PATH}"
    done
done

echo
echo "Done. Results are under ${OUTPUT_ROOT}/scheduler_*/request_rate_*/"
