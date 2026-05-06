#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-inclusionAI/LLaDA2.0-mini}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/dlm_sched_comparison}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
WARMUP="${WARMUP:-16}"
NUM_OUTPUT_BLOCKS="${NUM_OUTPUT_BLOCKS:-0}"
REQUEST_RATES=(${REQUEST_RATES:-2 4 6 8 10 12 14})
TASKS=(${TASKS:-humaneval gsm8k}) # humaneval gsm8k math
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
# NUM_THREADS / DLLM_ADMISSION_WINDOW: when unset, auto-detected from full dataset size per task.
NUM_THREADS="${NUM_THREADS:-}"
DLLM_ADMISSION_WINDOW="${DLLM_ADMISSION_WINDOW:-}"
PORT="${PORT:-30000}"
BASE_URL="${BASE_URL:-http://localhost:${PORT}}"
THRESHOLD="${THRESHOLD:-0.95}"
SCHEDULERS=(${SCHEDULERS:-LST FCFS PREFILL TTFB DECODE})
STRICT_MULTIPLIER="${STRICT_MULTIPLIER:-5.0}"
RELEASE_MULTIPLIER="${RELEASE_MULTIPLIER:-25.0}"
STRICT_PROB="${STRICT_PROB:-0.5}"
FORWARD_TIME_S="${FORWARD_TIME_S:-0.030}"
PREFILL_FORWARD_TIME_S="${PREFILL_FORWARD_TIME_S:-}"
DECODE_FORWARD_TIME_S="${DECODE_FORWARD_TIME_S:-}"
CONFIG_PATH="${CONFIG_PATH:-/tmp/dlm_algo_config_sched_cmp.yaml}"
STEP_LOG_FILE="${STEP_LOG_FILE:-/tmp/dlm_step_stats_cmp.jsonl}"
REQUEST_LATENCY_LOG_FILE="${REQUEST_LATENCY_LOG_FILE:-/tmp/dlm_request_latency_cmp.jsonl}"
BATCH_LATENCY_LOG_FILE="${BATCH_LATENCY_LOG_FILE:-/tmp/dlm_batch_latency_cmp.jsonl}"
TP_SIZE="${TP_SIZE:-1}"
# Calibration: rate and example count for ideal TTFB/TPOB measurement.
# CALIB_RATE should be high enough that the GPU batch is typically full.
CALIB_RATE="${CALIB_RATE:-4}"
CALIB_NUM_EXAMPLES="${CALIB_NUM_EXAMPLES:-100}"

SERVER_PID=""

# Full test-set sizes (fixed benchmarks — not sampled).
_task_dataset_size() {
    case "${1}" in
        gsm8k)     echo "1319" ;;
        humaneval) echo "164"  ;;
        math)      echo "5000" ;;
        *)         echo "500"  ;;
    esac
}

# _compute_task_slo IDEAL_TTFB_MS IDEAL_TPOB_MS MULTIPLIER
# Returns "ttfb_s tpob_s" on stdout.
_compute_task_slo() {
    local _ideal_ttfb_ms="${1}" _ideal_tpob_ms="${2}" _multiplier="${3}"
    python3 -c "
m = float('${_multiplier}')
print(f'{float(\"${_ideal_ttfb_ms}\")*m/1000:.4f} {float(\"${_ideal_tpob_ms}\")*m/1000:.4f}')
"
}

# write_dllm_config STRICT_TTFB STRICT_TPOB RELEASE_TTFB RELEASE_TPOB SCHEDULER_MODE ADMISSION_WINDOW
write_dllm_config() {
    local _strict_ttfb="${1:-}"
    local _strict_tpob="${2:-}"
    local _release_ttfb="${3:-}"
    local _release_tpob="${4:-}"
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

# _run_calib_bench SCHEDULER_MODE ADMISSION_WINDOW OUT_DIR TASK [--num-examples N]
# Starts server, runs benchmark, stops server.
_run_calib_bench() {
    local _mode="${1}" _window="${2}" _out="${3}" _task="${4}"
    shift 4
    mkdir -p "${_out}"

    write_dllm_config "" "" "" "" "${_mode}" "${_window}"
    echo "===== calib mode=${_mode} task=${_task} =====" >> "${SERVER_LOG}"

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
    echo "[calib server] pid=${SERVER_PID}, waiting for ${BASE_URL}/health"
    wait_server_ready
    echo "[calib server] ready"

    PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6 \
    python test/dlm_benchmark.py \
        --base-url "${BASE_URL}" \
        --model "${MODEL_PATH}" \
        --tasks "${_task}" \
        --block-size "${BLOCK_SIZE}" \
        --log \
        --request-rate "${CALIB_RATE}" \
        --num-threads "${CALIB_NUM_EXAMPLES}" \
        --num-examples "${CALIB_NUM_EXAMPLES}" \
        --warmup "${WARMUP}" \
        --num-output-blocks "${NUM_OUTPUT_BLOCKS}" \
        --output-dir "${_out}" \
        --tp-size "${TP_SIZE}" \
        "$@"

    stop_server
}

# _parse_calib_metric OUT_DIR TASK FIELD
# Prints the value of FIELD from {task}_{model_tag}.json, or "" if missing.
_parse_calib_metric() {
    local _out="${1}" _task="${2}" _field="${3}"
    local _model_tag="${MODEL_PATH//\//_}"
    local _json="${_out}/${_task}_${_model_tag}.json"
    python3 -c "
import json, sys
try:
    d = json.load(open('${_json}'))
    v = d.get('${_field}')
    print('' if v is None else f'{v:.4f}')
except Exception:
    print('')
"
}

trap stop_server EXIT

SERVER_LOG="${OUTPUT_ROOT}/server_log.txt"
mkdir -p "${OUTPUT_ROOT}"

# ── Calibration phase ────────────────────────────────────────────────────────
# ideal TTFB: TTFB scheduler (minimises time-to-first-block)
# ideal TPOB: DECODE scheduler (batch always full, decode-only throughput)
echo
echo "============================================================"
echo "Calibration: measuring ideal TTFB (TTFB sched) and TPOB (DECODE sched)"
echo "  rate=${CALIB_RATE} req/s, ${CALIB_NUM_EXAMPLES} examples per task"
echo "============================================================"

declare -A IDEAL_TTFB_MS IDEAL_TPOB_MS

for TASK in "${TASKS[@]}"; do
    CALIB_TTFB_DIR="${OUTPUT_ROOT}/calibration/ttfb/${TASK}"
    CALIB_TPOB_DIR="${OUTPUT_ROOT}/calibration/decode/${TASK}"

    echo "[calib] task=${TASK}: running TTFB scheduler..."
    _run_calib_bench "ttfb" "$(_task_dataset_size "${TASK}")" "${CALIB_TTFB_DIR}" "${TASK}"
    _ttfb=$(_parse_calib_metric "${CALIB_TTFB_DIR}" "${TASK}" "p50_ttfb_ms")

    echo "[calib] task=${TASK}: running DECODE scheduler..."
    _run_calib_bench "prefill" "${MAX_RUNNING_REQUESTS}" "${CALIB_TPOB_DIR}" "${TASK}"
    _tpob=$(_parse_calib_metric "${CALIB_TPOB_DIR}" "${TASK}" "p50_tpob_ms")

    if [[ -z "${_ttfb}" || "${_ttfb}" == "None" ]]; then
        echo "[calib] WARNING: could not parse ideal TTFB for ${TASK}; SLOs will be empty" >&2
        _ttfb=""
    fi
    if [[ -z "${_tpob}" || "${_tpob}" == "None" ]]; then
        echo "[calib] WARNING: could not parse ideal TPOB for ${TASK}; SLOs will be empty" >&2
        _tpob=""
    fi

    IDEAL_TTFB_MS["${TASK}"]="${_ttfb}"
    IDEAL_TPOB_MS["${TASK}"]="${_tpob}"
    echo "[calib] task=${TASK}  ideal_ttfb=${_ttfb}ms  ideal_tpob=${_tpob}ms"
done

# ── Main comparison ──────────────────────────────────────────────────────────
for SCHEDULER in "${SCHEDULERS[@]}"; do
    for RATE in "${REQUEST_RATES[@]}"; do
        OUT_DIR="${OUTPUT_ROOT}/scheduler_${SCHEDULER}/request_rate_${RATE}"
        mkdir -p "${OUT_DIR}"

        for TASK in "${TASKS[@]}"; do
            _task_size=$(_task_dataset_size "${TASK}")
            _task_threads="${NUM_THREADS:-${_task_size}}"

            if [[ "${SCHEDULER}" == "DECODE" ]]; then
                _admission_window="${MAX_RUNNING_REQUESTS}"
            else
                _admission_window="${DLLM_ADMISSION_WINDOW:-${_task_size}}"
            fi

            echo
            echo "============================================================"
            echo "Scheduler comparison: scheduler=${SCHEDULER}, rate=${RATE}, task=${TASK}"
            echo "threads=${_task_threads}, admission_window=${_admission_window}, output_dir=${OUT_DIR}"
            echo "A fresh server will be started for this run."
            echo "============================================================"

            _ideal_ttfb="${IDEAL_TTFB_MS[${TASK}]:-}"
            _ideal_tpob="${IDEAL_TPOB_MS[${TASK}]:-}"

            _strict_ttfb="" _strict_tpob=""
            _release_ttfb="" _release_tpob=""
            if [[ -n "${_ideal_ttfb}" && -n "${_ideal_tpob}" ]]; then
                read -r _strict_ttfb  _strict_tpob  <<< "$(_compute_task_slo "${_ideal_ttfb}"  "${_ideal_tpob}"  "${STRICT_MULTIPLIER}")"
                read -r _release_ttfb _release_tpob <<< "$(_compute_task_slo "${_ideal_ttfb}"  "${_ideal_tpob}"  "${RELEASE_MULTIPLIER}")"
                echo "[slo] task=${TASK}  ideal=${_ideal_ttfb}ms/${_ideal_tpob}ms  strict=${_strict_ttfb}s/${_strict_tpob}s  release=${_release_ttfb}s/${_release_tpob}s"
            else
                echo "[slo] task=${TASK}  no calibration data — SLOs omitted"
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
                --num-threads "${_task_threads}"
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
