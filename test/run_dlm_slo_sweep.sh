#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-inclusionAI/LLaDA2.0-mini}" #JetLM/SDAR-4B-Chat
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/dlm_results}"
BLOCK_SIZE="${BLOCK_SIZE:-32}" #4
WARMUP="${WARMUP:-16}"
NUM_OUTPUT_BLOCKS="${NUM_OUTPUT_BLOCKS:-0}"
REQUEST_RATES=(${REQUEST_RATES:-8}) #0.5 1 1.5
TASKS=(${TASKS:-math}) ##### TASK humaneval math gsm8k
NUM_EXAMPLES="${NUM_EXAMPLES:-1000}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
PORT="${PORT:-30000}"
#DLLM_ADMISSION_WINDOW="${DLLM_ADMISSION_WINDOW:-100}"
BASE_URL="${BASE_URL:-http://localhost:${PORT}}"
THRESHOLD="${THRESHOLD:-0.95}"
NUM_THREADS_SWEEP=(${NUM_THREADS_SWEEP:-1000})  # sweep values 50 100 150 200
SCHEDULER="${SCHEDULER:-LST}"               # LST | PREFILL | DECODE | FCFS | SOLA | TTFB
STRICT_MULTIPLIER="${STRICT_MULTIPLIER:-5.0}"    # strict SLO = multiplier × ideal latency
RELEASE_MULTIPLIER="${RELEASE_MULTIPLIER:-25.0}" # release SLO = multiplier × ideal latency
STRICT_PROB="${STRICT_PROB:-1}"               # fraction of requests assigned strict SLO
FORWARD_TIME_S="${FORWARD_TIME_S:-0.030}"          # shared fallback (s)
PREFILL_FORWARD_TIME_S="${PREFILL_FORWARD_TIME_S:-}"  # override prefill fwd time (s)
DECODE_FORWARD_TIME_S="${DECODE_FORWARD_TIME_S:-}"    # override decode fwd time (s)
CONFIG_PATH="${CONFIG_PATH:-/tmp/dlm_algo_config.yaml}"
STEP_LOG_FILE="${STEP_LOG_FILE:-/tmp/dlm_step_stats.jsonl}"
REQUEST_LATENCY_LOG_FILE="${REQUEST_LATENCY_LOG_FILE:-/tmp/dlm_request_latency.jsonl}"
BATCH_LATENCY_LOG_FILE="${BATCH_LATENCY_LOG_FILE:-/tmp/dlm_batch_latency.jsonl}"
TP_SIZE="${TP_SIZE:-1}"

SERVER_PID=""

# Ideal baseline latencies (ms) per task — used with SLO_MULTIPLIER.
# Source: empirical measurements with DECODE scheduler (TPOB) and TTFB scheduler (TTFB).
# TODO: update with measurements for the current model.
_ideal_ttfb_ms() {
    case "${1}" in
        gsm8k)     echo "417.2" ;;
        humaneval) echo "312.9" ;;
        math)      echo "404.5" ;;
        *)         echo "417.2" ;;
    esac
}

_ideal_tpob_ms() {
    case "${1}" in
        gsm8k)     echo "348.5" ;;
        humaneval) echo "158.8" ;;
        math)      echo "436.5" ;;
        *)         echo "348.5" ;;
    esac
}

# Compute absolute SLO values.
# Returns "ttfb_s tpob_s" on stdout, or "" if multiplier is empty.
_compute_task_slo() {
    local task="${1}" multiplier="${2}"
    if [[ -z "${multiplier}" ]]; then
        echo ""
        return
    fi
    python3 -c "
m = float('${multiplier}')
print(f'{float(\"$(_ideal_ttfb_ms "${task}")\") * m / 1000:.4f} {float(\"$(_ideal_tpob_ms "${task}")\") * m / 1000:.4f}')
"
}

# write_dllm_config STRICT_TTFB STRICT_TPOB RELEASE_TTFB RELEASE_TPOB SCHEDULER_MODE
# All SLO args are in seconds; empty = omit from YAML.
# SCHEDULER_MODE: "lst" | "fcfs" | "prefill" — written as scheduler_mode to YAML.
write_dllm_config() {
    local _strict_ttfb="${1:-}"
    local _strict_tpob="${2:-}"
    local _release_ttfb="${3:-}"
    local _release_tpob="${4:-}"
    local _scheduler_mode="${5:-prefill}"

    cat > "${CONFIG_PATH}" <<EOF
threshold: ${THRESHOLD}
dllm_admission_window: ${DLLM_ADMISSION_WINDOW}
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
        # Capture TP worker PIDs before killing master (they reparent to init afterward).
        local child_pids
        child_pids=$(pgrep -P "${SERVER_PID}" 2>/dev/null || true)

        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        # Wait up to 30 s for graceful shutdown; force-kill if it hangs (TP>1).
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
        # Kill surviving TP workers / detokenizer so GPU memory is released.
        if [[ -n "${child_pids}" ]]; then
            echo "[server] killing surviving worker pids: ${child_pids}"
            # shellcheck disable=SC2086
            kill -9 ${child_pids} >/dev/null 2>&1 || true
        fi
        sleep 2  # let CUDA release memory before next server starts
    fi
    SERVER_PID=""
}

trap stop_server EXIT

for RATE in "${REQUEST_RATES[@]}"; do
    for THREADS in "${NUM_THREADS_SWEEP[@]}"; do
        DLLM_ADMISSION_WINDOW="${THREADS}"
        OUT_DIR="${OUTPUT_ROOT}/request_rate_${RATE}/threads_${THREADS}"

        mkdir -p "${OUT_DIR}"

    for TASK in "${TASKS[@]}"; do
        echo
        echo "============================================================"
        echo "DLM benchmark: rate=${RATE}, threads=${THREADS}, task=${TASK}, output_dir=${OUT_DIR}"
        echo "A fresh server will be started for this task."
        echo "============================================================"

        echo "[scheduler] ${SCHEDULER}"
        if [[ "${SCHEDULER}" == "DECODE" ]]; then
            DLLM_ADMISSION_WINDOW="${MAX_RUNNING_REQUESTS}"
            echo "[scheduler] DLLM_ADMISSION_WINDOW overridden to ${DLLM_ADMISSION_WINDOW}"
        fi

        # Compute strict and release SLO values for all schedulers (scatter plot + LST deadlines).
        _strict_vals=$(_compute_task_slo "${TASK}" "${STRICT_MULTIPLIER}")
        _release_vals=$(_compute_task_slo "${TASK}" "${RELEASE_MULTIPLIER}")
        read -r _strict_ttfb _strict_tpob   <<< "${_strict_vals}"
        read -r _release_ttfb _release_tpob <<< "${_release_vals}"
        if [[ -n "${_strict_vals}" ]]; then
            echo "[slo] task=${TASK}  ideal=$(_ideal_ttfb_ms "${TASK}")ms(TTFB)/$(_ideal_tpob_ms "${TASK}")ms(TPOB)  strict=${_strict_ttfb}s/${_strict_tpob}s  release=${_release_ttfb}s/${_release_tpob}s  (${STRICT_MULTIPLIER}×/${RELEASE_MULTIPLIER}× ideal)"
        fi

        # Map SCHEDULER env var to scheduler_mode YAML value.
        case "${SCHEDULER}" in
            LST)    _scheduler_mode="lst"    ;;
            FCFS)   _scheduler_mode="fcfs"   ;;
            SOLA)   _scheduler_mode="sola"   ;;
            TTFB)   _scheduler_mode="ttfb"   ;;
            *)      _scheduler_mode="prefill" ;;  # PREFILL, DECODE, or unrecognised
        esac

        write_dllm_config "${_strict_ttfb:-}" "${_strict_tpob:-}" "${_release_ttfb:-}" "${_release_tpob:-}" "${_scheduler_mode}"
        echo "===== request_rate=${RATE} task=${TASK} =====" >> /tmp/dlm_results/run_dlm_slo_server_log.txt

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
            >> /tmp/dlm_results/run_dlm_slo_server_log.txt 2>&1 &
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
            --num-threads "${THREADS}"
            --warmup "${WARMUP}"
            --num-output-blocks "${NUM_OUTPUT_BLOCKS}"
            --output-dir "${OUT_DIR}"
            --tp-size "${TP_SIZE}"
        )

        if [[ -n "${NUM_EXAMPLES}" ]]; then
            BENCH_ARGS+=(--num-examples "${NUM_EXAMPLES}")
        fi

        PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6 \
        python "${BENCH_ARGS[@]}"

        stop_server
    done
    done  # THREADS
done  # RATE

echo
echo "============================================================"
echo "DLM SLO rates"
echo "============================================================"

for RATE in "${REQUEST_RATES[@]}"; do
    for THREADS in "${NUM_THREADS_SWEEP[@]}"; do
        OUT_DIR="${OUTPUT_ROOT}/request_rate_${RATE}/threads_${THREADS}"
        SLO_PATH="${OUT_DIR}/slo_rates.json"

        echo
        echo "DLM SLO rate: request_rate=${RATE}, threads=${THREADS}"
        echo "------------------------------------------------------------"

        python test/dlm_slorate.py \
            --latency-dir "${OUT_DIR}" \
            --tasks "${TASKS[@]}" \
            --output-json "${SLO_PATH}"
    done
done

echo
echo "Done. Results are under ${OUTPUT_ROOT}/request_rate_*/threads_*/"
