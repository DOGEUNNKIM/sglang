#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-inclusionAI/LLaDA2.0-mini}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"

################################
TASKS=(${TASKS:-math mmlu gsm8k sharegpt ruler_4k gpqa humaneval})  ##### TASK humaneval math gsm8k gpqa mmlu ruler_4k ruler_8k ruler_16k sharegpt
RETRY_TASKS=(${RETRY_TASKS:-})  # unset = default retry; RETRY_TASKS="" = plot-only
SUMMARY_TASKS=(${SUMMARY_TASKS:-})  # default: RETRY_TASKS in retry mode, otherwise TASKS
# TP = 1일때 batch 32
# 전부 재수행하는데 24시간 정도
RATES_GSM8K="${RATES_GSM8K:-8 8.5 9 9.5}" #1000 # 200 : 10 15 20 25
RATES_MMLU="${RATES_MMLU:-2 2.2 2.4 2.6}" #1000 # 200 : 2 2.5 3 3.5
RATES_MATH="${RATES_MATH:-3 3.15 3.3 3.45}" #1000 #200 : 3.5 4 4.5 5
RATES_SHAREGPT="${RATES_SHAREGPT:-1.3 1.5 1.7 1.9}" #1000 #200 1.5 2 2.5 3
RATES_RULER_4K="${RATES_RULER_4K:-3 5 7 9}" #400 2.5 3 3.5 4
RATES_HUMANEVAL="${RATES_HUMANEVAL:-10 20 30 40}" #200 20 25 30 35
RATES_GPQA="${RATES_GPQA:-1 2 3 4}" #200
RATES_RULER_8K="${RATES_RULER_8K:-0.5 0.75 1 1.25}"
RATES_RULER_16K="${RATES_RULER_16K:-0.5 0.75 1 1.25}"
RATES_LONGBENCH_V2="${RATES_LONGBENCH_V2:-1 1.5 2 2.5}"
# TP = 2일때
# FORWARD_TIME_S를 변경해줘야함
#RATES_GSM8K="${RATES_GSM8K:-6 7 8 9}"
#RATES_MMLU="${RATES_MMLU:-1 1.5 2 2.5}"
#RATES_HUMANEVAL="${RATES_HUMANEVAL:-12 16 20 24}"
#RATES_MATH="${RATES_MATH:-0.5 1 1.5 2}"
#RATES_GPQA="${RATES_GPQA:-0.5 1 1.5 2}"
#RATES_SHAREGPT="${RATES_SHAREGPT:-2 2.5 3 3.5}"
#RATES_RULER_4K="${RATES_RULER_4K:-3.5 4 4.5 5}"
#RATES_RULER_8K="${RATES_RULER_8K:-0.5 0.75 1 1.25}"
#RATES_RULER_16K="${RATES_RULER_16K:-0.5 0.75 1 1.25}"
#RATES_LONGBENCH_V2="${RATES_LONGBENCH_V2:-1 1.5 2 2.5}"
# Per-task example cap (empty = full dataset). Override via env, e.g. NUM_EXAMPLES_MATH=100.
NUM_EXAMPLES_GSM8K="${NUM_EXAMPLES_GSM8K:-1000}" #1000
NUM_EXAMPLES_MATH="${NUM_EXAMPLES_MATH:-1000}" #1000
NUM_EXAMPLES_MMLU="${NUM_EXAMPLES_MMLU:-1000}" #1000
NUM_EXAMPLES_SHAREGPT="${NUM_EXAMPLES_SHAREGPT:-1000}" #1000
NUM_EXAMPLES_HUMANEVAL="${NUM_EXAMPLES_HUMANEVAL:-200}"
NUM_EXAMPLES_GPQA="${NUM_EXAMPLES_GPQA:-200}"
NUM_EXAMPLES_RULER_4K="${NUM_EXAMPLES_RULER_4K:-200}" #150 for OoM issue -> need to retry
NUM_EXAMPLES_LONGBENCH_V2="${NUM_EXAMPLES_LONGBENCH_V2:-}"
NUM_EXAMPLES_RULER_8K="${NUM_EXAMPLES_RULER_8K:-200}"
NUM_EXAMPLES_RULER_16K="${NUM_EXAMPLES_RULER_16K:-200}"
SCHEDULERS=(${SCHEDULERS:-TTFB DECODE FCFS PREFILL SOLA LST}) # TTFB DECODE LST SOLA FCFS PREFILL
STRICT_MULTIPLIER="${STRICT_MULTIPLIER:-10.0}"
RELEASE_MULTIPLIER="${RELEASE_MULTIPLIER:-20.0}"
STRICT_PROB="${STRICT_PROB:-1}"
#TP가 2, batch 32이면 Forward 0.03
#TP가 1, batch 32이면 Forward 0.04
#TP가 1, batch 16이면 Forward 0.03
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
WARMUP="${WARMUP:-32}"
TP_SIZE="${TP_SIZE:-1}" 
FORWARD_TIME_S="${FORWARD_TIME_S:-0.04}"
################################

SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/nvme0/kdg6245}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/dlm_sched_comparison_LLADA2}"

if [[ "${#SUMMARY_TASKS[@]}" -eq 0 ]]; then
    if [[ "${#RETRY_TASKS[@]}" -gt 0 ]]; then
        SUMMARY_TASKS=("${RETRY_TASKS[@]}")
    else
        SUMMARY_TASKS=("${TASKS[@]}")
    fi
fi

if [[ "${#RETRY_TASKS[@]}" -eq 0 ]]; then
    echo "[plot-only] RETRY_TASKS is empty — skipping all benchmark runs, plot only"
else
    echo "[retry] benchmark tasks: ${RETRY_TASKS[*]}"
fi
echo "[summary] summary tasks: ${SUMMARY_TASKS[*]}"
REQUEST_RATES=(${REQUEST_RATES:-})  # fallback when task has no per-task rates
NUM_OUTPUT_BLOCKS="${NUM_OUTPUT_BLOCKS:-0}"
# NUM_THREADS / DLLM_ADMISSION_WINDOW: when unset, auto-detected from full dataset size per task.
NUM_THREADS="${NUM_THREADS:-}"
DLLM_ADMISSION_WINDOW="${DLLM_ADMISSION_WINDOW:-}"
PORT="${PORT:-30002}"
BASE_URL="${BASE_URL:-http://localhost:${PORT}}"
THRESHOLD="${THRESHOLD:-0.95}"
PREFILL_FORWARD_TIME_S="${PREFILL_FORWARD_TIME_S:-}"
DECODE_FORWARD_TIME_S="${DECODE_FORWARD_TIME_S:-}"
CONFIG_PATH="${CONFIG_PATH:-${OUTPUT_ROOT}/dlm_algo_config_sched_cmp_LLADA2.yaml}"
STEP_LOG_FILE="${STEP_LOG_FILE:-${OUTPUT_ROOT}/dlm_step_stats_LLADA2.jsonl}"
REQUEST_LATENCY_LOG_FILE="${REQUEST_LATENCY_LOG_FILE:-${OUTPUT_ROOT}/dlm_request_latency_LLADA2.jsonl}"
BATCH_LATENCY_LOG_FILE="${BATCH_LATENCY_LOG_FILE:-${OUTPUT_ROOT}/dlm_batch_latency_LLADA2.jsonl}"
export STEP_LOG_FILE REQUEST_LATENCY_LOG_FILE BATCH_LATENCY_LOG_FILE
GPU_FREE_MEMORY_MIN_MB="${GPU_FREE_MEMORY_MIN_MB:-70000}"

SERVER_PID=""

# Full test-set sizes (fixed benchmarks — not sampled).
_task_dataset_size() {
    case "${1}" in
        gsm8k)         echo "1314"  ;;
        humaneval)     echo "164"   ;;
        math)          echo "5000"  ;;
        gpqa)          echo "198"   ;;
        mmlu)          echo "14042" ;;
        longbench_v2)  echo "503"   ;;
        ruler_4k)      echo "6500"  ;;
        ruler_8k)      echo "6500"  ;;
        ruler_16k)     echo "6500"  ;;
        sharegpt)      echo "70000" ;;
        *)             echo "4000"  ;;
    esac
}

# Per-task request cap. Returns empty string when no cap is set (use full dataset).
_task_max_examples() {
    case "${1}" in
        gsm8k)         echo "${NUM_EXAMPLES_GSM8K}" ;;
        humaneval)     echo "${NUM_EXAMPLES_HUMANEVAL}" ;;
        math)          echo "${NUM_EXAMPLES_MATH}" ;;
        gpqa)          echo "${NUM_EXAMPLES_GPQA}" ;;
        mmlu)          echo "${NUM_EXAMPLES_MMLU}" ;;
        longbench_v2)  echo "${NUM_EXAMPLES_LONGBENCH_V2}" ;;
        ruler_4k)      echo "${NUM_EXAMPLES_RULER_4K}" ;;
        ruler_8k)      echo "${NUM_EXAMPLES_RULER_8K}" ;;
        ruler_16k)     echo "${NUM_EXAMPLES_RULER_16K}" ;;
        sharegpt)      echo "${NUM_EXAMPLES_SHAREGPT}" ;;
        *)             echo "" ;;
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
        # Wait for GPU memory to be released (CUDA holds memory until process fully exits)
        wait_gpu_memory_released "server"
    fi
    SERVER_PID=""
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
    v = d.get('${_field}') or d.get('latency_stats', {}).get('${_field}')
    print('' if v is None else f'{v:.4f}')
except Exception:
    print('')
"
}

trap stop_server EXIT

SERVER_LOG="${OUTPUT_ROOT}/server_log.txt"
RUN_MARKER="${OUTPUT_ROOT}/.retry_run_started"
mkdir -p "${OUTPUT_ROOT}"

# Kill any stale server already occupying the port (leftover from a prior run).
_stale_pid=$(lsof -ti tcp:"${PORT}" 2>/dev/null || true)
if [[ -n "${_stale_pid}" ]]; then
    echo "[startup] killing stale process(es) on port ${PORT}: ${_stale_pid}"
    kill -9 ${_stale_pid} 2>/dev/null || true
    wait_gpu_memory_released "startup"
fi

declare -A IDEAL_TTFB_MS IDEAL_TPOB_MS

# Returns space-separated rates for a given task.
_task_rates() {
    case "${1}" in
        gsm8k)         echo "${RATES_GSM8K}" ;;
        humaneval)     echo "${RATES_HUMANEVAL}" ;;
        math)          echo "${RATES_MATH}" ;;
        gpqa)          echo "${RATES_GPQA}" ;;
        mmlu)          echo "${RATES_MMLU}" ;;
        longbench_v2)  echo "${RATES_LONGBENCH_V2}" ;;
        ruler_4k)      echo "${RATES_RULER_4K}" ;;
        ruler_8k)      echo "${RATES_RULER_8K}" ;;
        ruler_16k)     echo "${RATES_RULER_16K}" ;;
        sharegpt)      echo "${RATES_SHAREGPT}" ;;
        *)             echo "${REQUEST_RATES[*]}" ;;
    esac
}

_in_retry_tasks() {
    local _t="${1}"
    for _r in "${RETRY_TASKS[@]}"; do [[ "${_r}" == "${_t}" ]] && return 0; done
    return 1
}

_same_task_set() {
    [[ "${#TASKS[@]}" -eq "${#RETRY_TASKS[@]}" ]] || return 1
    local _tasks_key _retry_key
    _tasks_key=$(printf '%s\n' "${TASKS[@]}" | sort | tr '\n' ' ')
    _retry_key=$(printf '%s\n' "${RETRY_TASKS[@]}" | sort | tr '\n' ' ')
    [[ "${_tasks_key}" == "${_retry_key}" ]]
}

# Selective cleanup for retry mode
if [[ "${#RETRY_TASKS[@]}" -gt 0 && -d "${OUTPUT_ROOT}" ]]; then
    echo "[clean] retry mode — removing results for tasks: ${RETRY_TASKS[*]}"
    touch "${RUN_MARKER}"
    rm -f "${OUTPUT_ROOT}/slo_summary.json" \
          "${OUTPUT_ROOT}/slo_config.json" \
          "${OUTPUT_ROOT}/server_log.txt"
    if _same_task_set; then
        echo "[clean] RETRY_TASKS matches TASKS — removing all scheduler results"
        for _dir in "${OUTPUT_ROOT}"/scheduler_*; do
            [[ -d "${_dir}" ]] || continue
            echo "[clean]   rm -rf ${_dir}"
            rm -rf "${_dir}"
        done
    else
        for _task in "${RETRY_TASKS[@]}"; do
            for _dir in "${OUTPUT_ROOT}"/scheduler_*/request_rate_*/"${_task}"; do
                if [[ -d "${_dir}" ]]; then
                    echo "[clean]   rm -rf ${_dir}"
                    rm -rf "${_dir}"
                fi
            done
        done
    fi
fi

# ── Main comparison ──────────────────────────────────────────────────────────
for SCHEDULER in "${SCHEDULERS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        if [[ "${#RETRY_TASKS[@]}" -eq 0 ]]; then
            continue
        fi
        if ! _in_retry_tasks "${TASK}"; then
            echo "[retry] skipping task=${TASK} (not in RETRY_TASKS)"
            continue
        fi
        _rates=($(_task_rates "${TASK}"))
        for RATE in "${_rates[@]}"; do
        OUT_DIR="${OUTPUT_ROOT}/scheduler_${SCHEDULER}/request_rate_${RATE}/${TASK}"
        mkdir -p "${OUT_DIR}"

        {
            CONFIG_PATH="${OUT_DIR}/dllm_algo_config.yaml"
            STEP_LOG_FILE="${OUT_DIR}/dlm_step_stats_${TASK}.jsonl"
            REQUEST_LATENCY_LOG_FILE="${OUT_DIR}/dlm_request_latency_${TASK}.jsonl"
            BATCH_LATENCY_LOG_FILE="${OUT_DIR}/dlm_batch_latency_${TASK}.jsonl"
            export STEP_LOG_FILE REQUEST_LATENCY_LOG_FILE BATCH_LATENCY_LOG_FILE
            RUN_SERVER_LOG="${OUT_DIR}/server.log"

            _task_size=$(_task_dataset_size "${TASK}")
            _task_examples=$(_task_max_examples "${TASK}")
            _effective_size="${_task_examples:-${_task_size}}"
            _task_threads="${NUM_THREADS:-${_effective_size}}"

            if [[ "${SCHEDULER}" == "DECODE" ]]; then
                _admission_window="${MAX_RUNNING_REQUESTS}"
            else
                _admission_window="${DLLM_ADMISSION_WINDOW:-${_effective_size}}"
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

            {
                echo "===== scheduler=${SCHEDULER} rate=${RATE} task=${TASK} ====="
                echo "model=${MODEL_PATH}"
                echo "output_dir=${OUT_DIR}"
            } | tee -a "${SERVER_LOG}" > "${RUN_SERVER_LOG}"

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
                --mem-fraction-static 0.95 \
                >> "${RUN_SERVER_LOG}" 2>&1 &
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
            [[ -n "${_task_examples}" ]] && BENCH_ARGS+=(--num-examples "${_task_examples}")

            PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6 \
            python "${BENCH_ARGS[@]}"

            stop_server
        }  # TASK block
        done  # RATE
    done  # TASK

    # After TTFB scheduler: extract both ideal_ttfb and ideal_tpob from highest rate
    if [[ "${SCHEDULER}" == "TTFB" ]]; then
        echo
        echo "[ideal] TTFB sched done — extracting p50_ideal_ttfb_ms and p50_ideal_tpob_ms at highest rate per task"
        for TASK in "${SUMMARY_TASKS[@]}"; do
            _rates=($(_task_rates "${TASK}"))
            _high_rate="${_rates[-1]}"
            _out="${OUTPUT_ROOT}/scheduler_TTFB/request_rate_${_high_rate}/${TASK}"
            _val_ttfb=$(_parse_calib_metric "${_out}" "${TASK}" "p50_ideal_ttfb_ms")
            _val_tpob=$(_parse_calib_metric "${_out}" "${TASK}" "p50_ideal_tpob_ms")
            if [[ -z "${_val_ttfb}" ]]; then
                echo "[ideal] WARNING: p50_ideal_ttfb_ms not found for task=${TASK} rate=${_high_rate}" >&2
            fi
            if [[ -z "${_val_tpob}" ]]; then
                echo "[ideal] WARNING: p50_ideal_tpob_ms not found for task=${TASK} rate=${_high_rate}" >&2
            fi
            IDEAL_TTFB_MS["${TASK}"]="${_val_ttfb:-}"
            IDEAL_TPOB_MS["${TASK}"]="${_val_tpob:-}"
            echo "  task=${TASK}  rate=${_high_rate}  ideal_ttfb=${_val_ttfb:-N/A}ms  ideal_tpob=${_val_tpob:-N/A}ms"
        done
    fi

done  # SCHEDULER

if [[ "${#RETRY_TASKS[@]}" -gt 0 ]]; then
    _missing_logs=0
    echo
    echo "[validate] checking freshly generated latency logs for summary tasks"
    for SCHEDULER in "${SCHEDULERS[@]}"; do
        for TASK in "${SUMMARY_TASKS[@]}"; do
            _rates=($(_task_rates "${TASK}"))
            for RATE in "${_rates[@]}"; do
                _latency_log="${OUTPUT_ROOT}/scheduler_${SCHEDULER}/request_rate_${RATE}/${TASK}/dlm_request_latency_${TASK}.jsonl"
                if [[ ! -s "${_latency_log}" || ! "${_latency_log}" -nt "${RUN_MARKER}" ]]; then
                    echo "[validate] ERROR: missing or stale latency log: ${_latency_log}" >&2
                    _missing_logs=1
                fi
            done
        done
    done
    if [[ "${_missing_logs}" -ne 0 ]]; then
        echo "[validate] retry run did not produce all latency logs; refusing to build a mixed summary" >&2
        exit 1
    fi
fi

_load_ideal_metrics_from_ttfb_summaries() {
    local _missing=0
    echo "[ideal] extracting p50_ideal_ttfb_ms and p50_ideal_tpob_ms from existing TTFB summaries"
    for TASK in "${SUMMARY_TASKS[@]}"; do
        _rates=($(_task_rates "${TASK}"))
        _high_rate="${_rates[-1]}"
        _out="${OUTPUT_ROOT}/scheduler_TTFB/request_rate_${_high_rate}/${TASK}"
        _val_ttfb=$(_parse_calib_metric "${_out}" "${TASK}" "p50_ideal_ttfb_ms")
        _val_tpob=$(_parse_calib_metric "${_out}" "${TASK}" "p50_ideal_tpob_ms")
        if [[ -z "${_val_ttfb}" || -z "${_val_tpob}" ]]; then
            echo "[ideal] ERROR: missing ideal metrics for task=${TASK} rate=${_high_rate} under ${_out}" >&2
            _missing=1
        fi
        IDEAL_TTFB_MS["${TASK}"]="${_val_ttfb:-}"
        IDEAL_TPOB_MS["${TASK}"]="${_val_tpob:-}"
        echo "  task=${TASK}  rate=${_high_rate}  ideal_ttfb=${_val_ttfb:-N/A}ms  ideal_tpob=${_val_tpob:-N/A}ms"
    done
    return "${_missing}"
}

_write_slo_config() {
    python3 -c "
import json, sys
tasks = '${SUMMARY_TASKS[*]}'.split()
strict_m  = float('${STRICT_MULTIPLIER}')
release_m = float('${RELEASE_MULTIPLIER}')
ideal_ttfb = {$(for T in "${SUMMARY_TASKS[@]}"; do echo "'${T}': float('${IDEAL_TTFB_MS[${T}]:-0}'),"; done)}
ideal_tpob = {$(for T in "${SUMMARY_TASKS[@]}"; do echo "'${T}': float('${IDEAL_TPOB_MS[${T}]:-0}'),"; done)}
slos = {}
for t in tasks:
    ittfb, itpob = ideal_ttfb.get(t, 0), ideal_tpob.get(t, 0)
    if ittfb > 0 and itpob > 0:
        slos[t] = {
            'strict':  {'ttfb_ms': ittfb * strict_m,  'tpob_ms': itpob * strict_m},
            'relaxed': {'ttfb_ms': ittfb * release_m, 'tpob_ms': itpob * release_m},
        }
with open('${SLO_CONFIG_PATH}', 'w') as f:
    json.dump(slos, f, indent=2)
print('[slo_config] written to ${SLO_CONFIG_PATH}')
print(json.dumps(slos, indent=2))
"
}

# ── Write calibrated SLO config for dlm_slorate.py ───────────────────────────
SLO_CONFIG_PATH="${OUTPUT_ROOT}/slo_config.json"
_write_slo_config

echo
echo "============================================================"
echo "DLM Scheduler Comparison — SLO Summary"
echo "============================================================"

if [[ "${#RETRY_TASKS[@]}" -eq 0 ]]; then
    _missing_slo_inputs=0
    echo "[plot-only] rebuilding missing slo_rates.json files from existing latency logs"
    for SCHEDULER in "${SCHEDULERS[@]}"; do
        for TASK in "${SUMMARY_TASKS[@]}"; do
            _rates=($(_task_rates "${TASK}"))
            for RATE in "${_rates[@]}"; do
                OUT_DIR="${OUTPUT_ROOT}/scheduler_${SCHEDULER}/request_rate_${RATE}/${TASK}"
                SLO_PATH="${OUT_DIR}/slo_rates.json"
                if [[ -s "${SLO_PATH}" ]]; then
                    continue
                fi
                _latency_log="${OUT_DIR}/dlm_request_latency_${TASK}.jsonl"
                if [[ ! -s "${_latency_log}" ]]; then
                    echo "[plot-only] ERROR: missing latency log for SLO rates: ${_latency_log}" >&2
                    _missing_slo_inputs=1
                    continue
                fi

                echo
                echo "[plot-only] scheduler=${SCHEDULER}  task=${TASK}  request_rate=${RATE}"
                echo "------------------------------------------------------------"
                python test/dlm_slorate.py \
                    --latency-dir "${OUT_DIR}" \
                    --tasks "${TASK}" \
                    --slo-config "${SLO_CONFIG_PATH}" \
                    --output-json "${SLO_PATH}"
            done
        done
    done
    if [[ "${_missing_slo_inputs}" -ne 0 ]]; then
        echo "[plot-only] refusing to continue with incomplete latency inputs" >&2
        exit 1
    fi

    _missing_slo_rates=0
    echo "[plot-only] validating slo_rates.json files"
    for SCHEDULER in "${SCHEDULERS[@]}"; do
        for TASK in "${SUMMARY_TASKS[@]}"; do
            _rates=($(_task_rates "${TASK}"))
            for RATE in "${_rates[@]}"; do
                SLO_PATH="${OUTPUT_ROOT}/scheduler_${SCHEDULER}/request_rate_${RATE}/${TASK}/slo_rates.json"
                if [[ ! -s "${SLO_PATH}" ]]; then
                    echo "[plot-only] ERROR: missing existing SLO rates: ${SLO_PATH}" >&2
                    _missing_slo_rates=1
                fi
            done
        done
    done
    if [[ "${_missing_slo_rates}" -ne 0 ]]; then
        echo "[plot-only] refusing to continue with incomplete SLO rate inputs" >&2
        exit 1
    fi
else
    for SCHEDULER in "${SCHEDULERS[@]}"; do
        for TASK in "${SUMMARY_TASKS[@]}"; do
            _rates=($(_task_rates "${TASK}"))
            for RATE in "${_rates[@]}"; do
                OUT_DIR="${OUTPUT_ROOT}/scheduler_${SCHEDULER}/request_rate_${RATE}/${TASK}"
                SLO_PATH="${OUT_DIR}/slo_rates.json"

                echo
                echo "scheduler=${SCHEDULER}  task=${TASK}  request_rate=${RATE}"
                echo "------------------------------------------------------------"

                python test/dlm_slorate.py \
                    --latency-dir "${OUT_DIR}" \
                    --tasks "${TASK}" \
                    --slo-config "${SLO_CONFIG_PATH}" \
                    --output-json "${SLO_PATH}"
            done
        done
    done
fi

# ── Consolidated SLO summary ──────────────────────────────────────────────────
SUMMARY_PATH="${OUTPUT_ROOT}/slo_summary.json"
python3 -c "
import json, os
from pathlib import Path

output_root = '${OUTPUT_ROOT}'
schedulers  = '${SCHEDULERS[*]}'.split()
tasks       = '${SUMMARY_TASKS[*]}'.split()
task_rate_map = {
    'gsm8k':        '${RATES_GSM8K}'.split(),
    'humaneval':    '${RATES_HUMANEVAL}'.split(),
    'math':         '${RATES_MATH}'.split(),
    'gpqa':         '${RATES_GPQA}'.split(),
    'mmlu':         '${RATES_MMLU}'.split(),
    'longbench_v2': '${RATES_LONGBENCH_V2}'.split(),
    'ruler_4k':     '${RATES_RULER_4K}'.split(),
    'ruler_8k':     '${RATES_RULER_8K}'.split(),
    'ruler_16k':    '${RATES_RULER_16K}'.split(),
    'sharegpt':     '${RATES_SHAREGPT}'.split(),
}

model_tag = '${MODEL_PATH}'.replace('/', '_')

def _read_bench_metrics(out_dir, task):
    p = Path(out_dir) / f'{task}_{model_tag}.json'
    if not p.exists():
        return (None,) * 28
    try:
        d = json.loads(p.read_text())
        ls = d.get('latency_stats', d)
        return (
            d.get('score'),
            d.get('score:std'),
            ls.get('p95_ttfb_ms'),
            ls.get('p95_tpob_ms'),
            ls.get('p99_ttfb_ms'),
            ls.get('p99_tpob_ms'),
            ls.get('mean_ttfb_ms'),
            ls.get('mean_tpob_ms'),
            ls.get('mean_sched_wait_ms'),
            ls.get('mean_first_unmask_gap_ms'),
            ls.get('mean_decode_wait_ms'),
            ls.get('p95_ideal_ttfb_ms'),
            ls.get('p95_ideal_tpob_ms'),
            ls.get('p95_sched_wait_ms'),
            ls.get('p95_first_unmask_gap_ms'),
            ls.get('p95_decode_wait_ms'),
            ls.get('p95_ttfb_selected_target_ms'),
            ls.get('p95_ttfb_selected_total_ms'),
            ls.get('p95_ttfb_selected_forward_ms'),
            ls.get('p95_ttfb_selected_sched_wait_ms'),
            ls.get('p95_ttfb_selected_first_unmask_gap_ms'),
            ls.get('p95_ttfb_selected_first_block_decode_wait_ms'),
            ls.get('p95_ttfb_selected_other_ms'),
            ls.get('p95_tpob_selected_target_ms'),
            ls.get('p95_tpob_selected_total_ms'),
            ls.get('p95_tpob_selected_forward_ms'),
            ls.get('p95_tpob_selected_decode_wait_ms'),
            ls.get('p95_tpob_selected_other_ms'),
        )
    except Exception:
        return (None,) * 28

summary = {}
for sched in schedulers:
    summary[sched] = {}
    for task in tasks:
        for rate in task_rate_map.get(task, []):
            out_dir  = Path(output_root) / f'scheduler_{sched}' / f'request_rate_{rate}' / task
            slo_path = out_dir / 'slo_rates.json'
            if not slo_path.exists():
                continue
            data = json.loads(slo_path.read_text())
            if rate not in summary[sched]:
                summary[sched][rate] = {}
            if task not in data:
                continue
            rates_d = data[task].get('rates', {})
            score, score_std, p95_ttfb, p95_tpob, p99_ttfb, p99_tpob, \
                mean_ttfb, mean_tpob, mean_sched_wait, mean_first_unmask_gap, mean_decode_wait, \
                p95_ideal_ttfb, p95_ideal_tpob, p95_sched_wait, p95_first_unmask_gap, p95_decode_wait, \
                p95_ttfb_selected_target, p95_ttfb_selected_total, p95_ttfb_selected_forward, \
                p95_ttfb_selected_sched_wait, p95_ttfb_selected_first_unmask_gap, p95_ttfb_selected_first_block_dw, p95_ttfb_selected_other, \
                p95_tpob_selected_target, p95_tpob_selected_total, p95_tpob_selected_forward, \
                p95_tpob_selected_decode_wait, p95_tpob_selected_other = \
                _read_bench_metrics(out_dir, task)
            summary[sched][rate][task] = {
                'score':                  score,
                'score_std':              score_std,
                'strict_ttfb':            rates_d.get('strict_ttfb'),
                'strict_tpob':            rates_d.get('strict_tpob'),
                'strict_all':             rates_d.get('strict_all'),
                'relaxed_ttfb':           rates_d.get('relaxed_ttfb'),
                'relaxed_tpob':           rates_d.get('relaxed_tpob'),
                'relaxed_all':            rates_d.get('relaxed_all'),
                'p95_ttfb_ms':            p95_ttfb,
                'p95_tpob_ms':            p95_tpob,
                'p99_ttfb_ms':            p99_ttfb,
                'p99_tpob_ms':            p99_tpob,
                'mean_ttfb_ms':           mean_ttfb,
                'mean_tpob_ms':           mean_tpob,
                'mean_sched_wait_ms':     mean_sched_wait,
                'mean_first_unmask_gap_ms': mean_first_unmask_gap,
                'mean_decode_wait_ms':    mean_decode_wait,
                'p95_ideal_ttfb_ms':      p95_ideal_ttfb,
                'p95_ideal_tpob_ms':      p95_ideal_tpob,
                'p95_sched_wait_ms':      p95_sched_wait,
                'p95_first_unmask_gap_ms': p95_first_unmask_gap,
                'p95_decode_wait_ms':     p95_decode_wait,
                'p95_ttfb_selected_target_ms': p95_ttfb_selected_target,
                'p95_ttfb_selected_total_ms': p95_ttfb_selected_total,
                'p95_ttfb_selected_forward_ms': p95_ttfb_selected_forward,
                'p95_ttfb_selected_sched_wait_ms': p95_ttfb_selected_sched_wait,
                'p95_ttfb_selected_first_unmask_gap_ms': p95_ttfb_selected_first_unmask_gap,
                'p95_ttfb_selected_first_block_decode_wait_ms': p95_ttfb_selected_first_block_dw,
                'p95_ttfb_selected_other_ms': p95_ttfb_selected_other,
                'p95_tpob_selected_target_ms': p95_tpob_selected_target,
                'p95_tpob_selected_total_ms': p95_tpob_selected_total,
                'p95_tpob_selected_forward_ms': p95_tpob_selected_forward,
                'p95_tpob_selected_decode_wait_ms': p95_tpob_selected_decode_wait,
                'p95_tpob_selected_other_ms': p95_tpob_selected_other,
            }

Path('${SUMMARY_PATH}').write_text(json.dumps(summary, indent=2))
print(f'[summary] saved → ${SUMMARY_PATH}')

# Print table
header = f\"{'Scheduler':<10} {'Task':<12} {'Rate':>6} {'Acc':>8} {'Str-TTFB':>10} {'Str-TPOB':>10} {'Str-All':>9} {'Rel-TTFB':>10} {'Rel-TPOB':>10} {'Rel-All':>9} {'P99-TTFB':>10} {'P99-TPOB':>10}\"
print()
print(header)
print('-' * len(header))
def fmt(v): return f'{v:.3f}' if v is not None else 'N/A'
def fms(v): return f'{v:.1f}' if v is not None else 'N/A'
for sched in schedulers:
    for task in tasks:
        for rate in task_rate_map.get(task, []):
            r = summary.get(sched, {}).get(rate, {}).get(task)
            if r is None:
                continue
            print(f'{sched:<10} {task:<12} {rate:>6} {fmt(r[\"score\"]):>8} {fmt(r[\"strict_ttfb\"]):>10} {fmt(r[\"strict_tpob\"]):>10} {fmt(r[\"strict_all\"]):>9} {fmt(r[\"relaxed_ttfb\"]):>10} {fmt(r[\"relaxed_tpob\"]):>10} {fmt(r[\"relaxed_all\"]):>9} {fms(r[\"p99_ttfb_ms\"]):>10} {fms(r[\"p99_tpob_ms\"]):>10}')
"

echo
echo "Done. Results are under ${OUTPUT_ROOT}/scheduler_*/request_rate_*/"
echo "Consolidated SLO summary: ${SUMMARY_PATH}"
