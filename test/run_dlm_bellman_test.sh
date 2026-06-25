#!/usr/bin/env bash
set -euo pipefail

# Compare remaining_compute_mode: bellman vs zero (EDF) vs worst vs oracle.
# Only meaningful with LST scheduler (slack-based scheduling).

###### hyper parameters ######
BLOCK_SIZE="${BLOCK_SIZE:-32}"
MODEL_PATH="${MODEL_PATH:-inclusionAI/LLaDA2.0-mini}"
WARMUP="${WARMUP:-32}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
NUM_OUTPUT_BLOCKS="${NUM_OUTPUT_BLOCKS:-0}"
TASKS=(${TASKS:-math mmlu gsm8k sharegpt ruler_4k gpqa humaneval}) #math mmlu gsm8k sharegpt ruler_4k gpqa humaneval
SCHEDULER="${SCHEDULER:-LST}"   # LST is required for remaining_compute_mode to have effect
DLLM_ADMISSION_WINDOW="${DLLM_ADMISSION_WINDOW:-1000}"
REMAINING_COMPUTE_MODES=(${REMAINING_COMPUTE_MODES:-bellman zero worst oracle})

# Per-task request rates
RATES_GSM8K=(${RATES_GSM8K:-9 9.5 10})
RATES_MMLU=(${RATES_MMLU:-2.4 2.6 2.8})
RATES_MATH=(${RATES_MATH:-3.3 3.45 3.6})
RATES_SHAREGPT=(${RATES_SHAREGPT:-1.7 1.9 2.1})
RATES_RULER_4K=(${RATES_RULER_4K:-5 7 9})
RATES_HUMANEVAL=(${RATES_HUMANEVAL:-40 50 60})
RATES_GPQA=(${RATES_GPQA:-3 4 5})

# Per-task number of examples
NUM_EXAMPLES_GSM8K="${NUM_EXAMPLES_GSM8K:-1000}"
NUM_EXAMPLES_MMLU="${NUM_EXAMPLES_MMLU:-1000}"
NUM_EXAMPLES_MATH="${NUM_EXAMPLES_MATH:-1000}"
NUM_EXAMPLES_SHAREGPT="${NUM_EXAMPLES_SHAREGPT:-1000}"
NUM_EXAMPLES_RULER_4K="${NUM_EXAMPLES_RULER_4K:-400}"
NUM_EXAMPLES_HUMANEVAL="${NUM_EXAMPLES_HUMANEVAL:-200}"
NUM_EXAMPLES_GPQA="${NUM_EXAMPLES_GPQA:-200}"
###### hyper parameters ######

SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/nvme0/kdg6245}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/dlm_bellman_test}"
PORT="${PORT:-30009}"
BASE_URL="${BASE_URL:-http://localhost:${PORT}}"
THRESHOLD="${THRESHOLD:-0.95}"
STRICT_PROB="${STRICT_PROB:-1}"
FORWARD_TIME_S="${FORWARD_TIME_S:-0.040}"
PREFILL_FORWARD_TIME_S="${PREFILL_FORWARD_TIME_S:-}"
DECODE_FORWARD_TIME_S="${DECODE_FORWARD_TIME_S:-}"
TP_SIZE="${TP_SIZE:-1}"
SLO_CONFIG_PATH="${SLO_CONFIG_PATH:-/mnt/nvme0/kdg6245/dlm_sched_comparison_LLADA2/slo_config.json}"

CONFIG_PATH="${OUTPUT_ROOT}/dlm_algo_config.yaml"
STEP_LOG_FILE="${OUTPUT_ROOT}/dlm_step_stats.jsonl"
REQUEST_LATENCY_LOG_FILE="${OUTPUT_ROOT}/dlm_request_latency.jsonl"
BATCH_LATENCY_LOG_FILE="${OUTPUT_ROOT}/dlm_batch_latency.jsonl"
export STEP_LOG_FILE REQUEST_LATENCY_LOG_FILE BATCH_LATENCY_LOG_FILE

_task_rates() {
    case "${1}" in
        gsm8k)     echo "${RATES_GSM8K[@]}" ;;
        mmlu)      echo "${RATES_MMLU[@]}" ;;
        math)      echo "${RATES_MATH[@]}" ;;
        sharegpt)  echo "${RATES_SHAREGPT[@]}" ;;
        ruler_4k)  echo "${RATES_RULER_4K[@]}" ;;
        humaneval) echo "${RATES_HUMANEVAL[@]}" ;;
        gpqa)      echo "${RATES_GPQA[@]}" ;;
        *)         echo "10" ;;
    esac
}

_task_num_examples() {
    case "${1}" in
        gsm8k)     echo "${NUM_EXAMPLES_GSM8K}" ;;
        mmlu)      echo "${NUM_EXAMPLES_MMLU}" ;;
        math)      echo "${NUM_EXAMPLES_MATH}" ;;
        sharegpt)  echo "${NUM_EXAMPLES_SHAREGPT}" ;;
        ruler_4k)  echo "${NUM_EXAMPLES_RULER_4K}" ;;
        humaneval) echo "${NUM_EXAMPLES_HUMANEVAL}" ;;
        gpqa)      echo "${NUM_EXAMPLES_GPQA}" ;;
        *)         echo "1000" ;;
    esac
}

# Precompute per-task SLO values in seconds from the reference JSON (called once).
declare -A _SLO_STRICT_TTFB_S _SLO_STRICT_TPOB_S _SLO_RELEASE_TTFB_S _SLO_RELEASE_TPOB_S
_precompute_slo_tables() {
    while IFS=' ' read -r _t _st _sp _rt _rp; do
        _SLO_STRICT_TTFB_S["$_t"]="$_st"
        _SLO_STRICT_TPOB_S["$_t"]="$_sp"
        _SLO_RELEASE_TTFB_S["$_t"]="$_rt"
        _SLO_RELEASE_TPOB_S["$_t"]="$_rp"
    done < <(python3 -c "
import json
d = json.load(open('${SLO_CONFIG_PATH}'))
for task, v in d.items():
    s = v['strict']; r = v['relaxed']
    print(f\"{task} {s['ttfb_ms']/1000:.6f} {s['tpob_ms']/1000:.6f} {r['ttfb_ms']/1000:.6f} {r['tpob_ms']/1000:.6f}\")
")
}

# write_dllm_config STRICT_TTFB STRICT_TPOB RELEASE_TTFB RELEASE_TPOB SCHEDULER_MODE REMAINING_COMPUTE_MODE [BELLMAN_LOG_FILE] [ORACLE_TABLE_PATH]
write_dllm_config() {
    local _strict_ttfb="${1:-}"
    local _strict_tpob="${2:-}"
    local _release_ttfb="${3:-}"
    local _release_tpob="${4:-}"
    local _scheduler_mode="${5:-lst}"
    local _rc_mode="${6:-bellman}"
    local _bellman_log_file="${7:-}"
    local _oracle_table_path="${8:-}"

    cat > "${CONFIG_PATH}" <<EOF
threshold: ${THRESHOLD}
dllm_admission_window: ${DLLM_ADMISSION_WINDOW}
forward_time_s: ${FORWARD_TIME_S}
strict_prob: ${STRICT_PROB}
scheduler_mode: ${_scheduler_mode}
remaining_compute_mode: ${_rc_mode}
step_log_file: ${STEP_LOG_FILE}
request_latency_log_file: ${REQUEST_LATENCY_LOG_FILE}
batch_latency_log_file: ${BATCH_LATENCY_LOG_FILE}
EOF
    [[ -n "${_strict_ttfb}" ]]           && echo "strict_ttfb_slo: ${_strict_ttfb}"                  >> "${CONFIG_PATH}"
    [[ -n "${_strict_tpob}" ]]           && echo "strict_tpob_slo: ${_strict_tpob}"                  >> "${CONFIG_PATH}"
    [[ -n "${_release_ttfb}" ]]          && echo "release_ttfb_slo: ${_release_ttfb}"                >> "${CONFIG_PATH}"
    [[ -n "${_release_tpob}" ]]          && echo "release_tpob_slo: ${_release_tpob}"                >> "${CONFIG_PATH}"
    [[ -n "${PREFILL_FORWARD_TIME_S}" ]] && echo "prefill_forward_time_s: ${PREFILL_FORWARD_TIME_S}" >> "${CONFIG_PATH}"
    [[ -n "${DECODE_FORWARD_TIME_S}" ]]  && echo "decode_forward_time_s: ${DECODE_FORWARD_TIME_S}"   >> "${CONFIG_PATH}"
    [[ -n "${_bellman_log_file}" ]]      && echo "bellman_log_file: ${_bellman_log_file}"             >> "${CONFIG_PATH}"
    [[ -n "${_oracle_table_path}" ]]     && echo "oracle_table_path: ${_oracle_table_path}"           >> "${CONFIG_PATH}"
    return 0
}

# Extract oracle table (empirical mean) from bellman_log.
_extract_oracle_table() {
    local _log_file="${1}" _out_file="${2}"
    python3 test/build_oracle_table.py \
        --bellman-log "${_log_file}" \
        --block-size "${BLOCK_SIZE}" \
        --mode mean \
        --output "${_out_file}"
}

# Extract warm_start table (last log entry) from bellman_log.
_extract_warmstart_table() {
    local _log_file="${1}" _out_file="${2}"
    python3 test/build_oracle_table.py \
        --bellman-log "${_log_file}" \
        --block-size "${BLOCK_SIZE}" \
        --mode last \
        --output "${_out_file}"
}

SERVER_PID=""

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

echo "[init] removing previous output: ${OUTPUT_ROOT}"
rm -rf "${OUTPUT_ROOT}"

SERVER_LOG="${OUTPUT_ROOT}/server_log.txt"
mkdir -p "${OUTPUT_ROOT}"

# Validate SLO config and precompute tables
if [[ ! -f "${SLO_CONFIG_PATH}" ]]; then
    echo "[error] SLO_CONFIG_PATH not found: ${SLO_CONFIG_PATH}" >&2
    exit 1
fi
echo "[slo] loading from ${SLO_CONFIG_PATH}"
_precompute_slo_tables

case "${SCHEDULER}" in
    LST)    _scheduler_mode="lst"    ;;
    FCFS)   _scheduler_mode="fcfs"   ;;
    SOLA)   _scheduler_mode="sola"   ;;
    TTFB)   _scheduler_mode="ttfb"   ;;
    *)      _scheduler_mode="prefill" ;;
esac

# Build execution order:
#   bellman always first (all warm/oracle modes depend on its log)
#   oracle always last (convention)
#   warm_start / warm_start_continue come before oracle
_has_oracle=0; _has_bellman=0; _has_warm=0
for _m in "${REMAINING_COMPUTE_MODES[@]}"; do
    [[ "$_m" == "oracle"                ]] && _has_oracle=1
    [[ "$_m" == "bellman"               ]] && _has_bellman=1
    [[ "$_m" == "warm_start" || "$_m" == "warm_start_continue" ]] && _has_warm=1
done
_ordered_modes=()
if [[ $_has_bellman -eq 1 || $_has_oracle -eq 1 || $_has_warm -eq 1 ]]; then
    _ordered_modes+=("bellman")
fi
for _m in "${REMAINING_COMPUTE_MODES[@]}"; do
    [[ "$_m" == "bellman" || "$_m" == "oracle" || "$_m" == "warm_start" || "$_m" == "warm_start_continue" ]] && continue
    _ordered_modes+=("$_m")
done
for _m in "${REMAINING_COMPUTE_MODES[@]}"; do
    [[ "$_m" == "warm_start" || "$_m" == "warm_start_continue" ]] && _ordered_modes+=("$_m")
done
[[ $_has_oracle -eq 1 ]] && _ordered_modes+=("oracle")

# Tables per "task:rate" — populated after bellman runs
declare -A _ORACLE_TABLES      # empirical mean (for oracle mode)
declare -A _WARMSTART_TABLES   # last log entry  (for warm_start / warm_start_continue)
_ORACLE_TABLE_DIR="${OUTPUT_ROOT}/oracle_tables"
[[ $_has_oracle -eq 1 || $_has_warm -eq 1 ]] && mkdir -p "${_ORACLE_TABLE_DIR}"

for RC_MODE in "${_ordered_modes[@]}"; do
    for TASK in "${TASKS[@]}"; do
        read -ra _task_rate_arr <<< "$(_task_rates "${TASK}")"
        _num_examples="$(_task_num_examples "${TASK}")"
        _strict_ttfb="${_SLO_STRICT_TTFB_S["${TASK}"]:-}"
        _strict_tpob="${_SLO_STRICT_TPOB_S["${TASK}"]:-}"
        _release_ttfb="${_SLO_RELEASE_TTFB_S["${TASK}"]:-}"
        _release_tpob="${_SLO_RELEASE_TPOB_S["${TASK}"]:-}"

        for RATE in "${_task_rate_arr[@]}"; do
            OUT_DIR="${OUTPUT_ROOT}/rc_mode_${RC_MODE}/request_rate_${RATE}/${TASK}"
            mkdir -p "${OUT_DIR}"

            echo
            echo "============================================================"
            echo "remaining_compute_mode=${RC_MODE}  scheduler=${SCHEDULER}  rate=${RATE}  task=${TASK}"
            echo "output_dir=${OUT_DIR}"
            echo "============================================================"
            echo "[slo] strict=${_strict_ttfb}s/${_strict_tpob}s  release=${_release_ttfb}s/${_release_tpob}s"

            STEP_LOG_FILE="${OUT_DIR}/dlm_step_stats_${TASK}.jsonl"
            REQUEST_LATENCY_LOG_FILE="${OUT_DIR}/dlm_request_latency_${TASK}.jsonl"
            BATCH_LATENCY_LOG_FILE="${OUT_DIR}/dlm_batch_latency_${TASK}.jsonl"
            export STEP_LOG_FILE REQUEST_LATENCY_LOG_FILE BATCH_LATENCY_LOG_FILE

            # All modes write a log file so I/O overhead is equal across modes.
            # bellman's log is also used to build the oracle table afterward.
            _bellman_log_arg="${OUT_DIR}/bellman_log_${TASK}.jsonl"
            _oracle_table_arg=""
            if [[ "${RC_MODE}" == "oracle" ]]; then
                _oracle_table_arg="${_ORACLE_TABLES["${TASK}:${RATE}"]:-}"
                if [[ -z "${_oracle_table_arg}" ]]; then
                    echo "[oracle] ERROR: no oracle table for task=${TASK} rate=${RATE} — did bellman run first?" >&2
                    exit 1
                fi
                echo "[oracle] using table: ${_oracle_table_arg}"
            elif [[ "${RC_MODE}" == "warm_start" || "${RC_MODE}" == "warm_start_continue" ]]; then
                _oracle_table_arg="${_WARMSTART_TABLES["${TASK}:${RATE}"]:-}"
                if [[ -z "${_oracle_table_arg}" ]]; then
                    echo "[${RC_MODE}] ERROR: no warm_start table for task=${TASK} rate=${RATE} — did bellman run first?" >&2
                    exit 1
                fi
                echo "[${RC_MODE}] using table: ${_oracle_table_arg}"
            fi

            write_dllm_config \
                "${_strict_ttfb}" "${_strict_tpob}" \
                "${_release_ttfb}" "${_release_tpob}" \
                "${_scheduler_mode}" "${RC_MODE}" \
                "${_bellman_log_arg}" "${_oracle_table_arg}"

            echo "===== rc_mode=${RC_MODE} rate=${RATE} task=${TASK} =====" >> "${SERVER_LOG}"

            PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6 \
            python -m sglang.launch_server \
                --model-path "${MODEL_PATH}" \
                --port "${PORT}" \
                --trust-remote-code \
                --dllm-algorithm LowConfidence \
                --dllm-algorithm-config "${CONFIG_PATH}" \
                --attention-backend flashinfer \
                --max-running-requests "${MAX_RUNNING_REQUESTS}" \
                --tp-size "${TP_SIZE}" \
                --mem-fraction-static 0.95 \
                --cuda-graph-max-bs "${MAX_RUNNING_REQUESTS}" \
                --disable-cuda-graph-padding \
                >> "${SERVER_LOG}" 2>&1 &
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
                --num-threads "${DLLM_ADMISSION_WINDOW}" \
                --warmup "${WARMUP}" \
                --num-output-blocks "${NUM_OUTPUT_BLOCKS}" \
                --output-dir "${OUT_DIR}" \
                --tp-size "${TP_SIZE}" \
                --num-examples "${_num_examples}"

            stop_server

            # After bellman run: extract oracle (mean) and warm_start (last) tables
            if [[ "${RC_MODE}" == "bellman" ]]; then
                if [[ $_has_oracle -eq 1 ]]; then
                    _oracle_out="${_ORACLE_TABLE_DIR}/oracle_table_${TASK}_rate${RATE}.json"
                    _extract_oracle_table "${_bellman_log_arg}" "${_oracle_out}"
                    _ORACLE_TABLES["${TASK}:${RATE}"]="${_oracle_out}"
                fi
                if [[ $_has_warm -eq 1 ]]; then
                    _warmstart_out="${_ORACLE_TABLE_DIR}/warmstart_table_${TASK}_rate${RATE}.json"
                    _extract_warmstart_table "${_bellman_log_arg}" "${_warmstart_out}"
                    _WARMSTART_TABLES["${TASK}:${RATE}"]="${_warmstart_out}"
                fi
            fi
        done  # RATE
    done  # TASK
done  # RC_MODE

echo
echo "============================================================"
echo "DLM remaining_compute_mode Comparison — SLO rates"
echo "============================================================"

for RC_MODE in "${_ordered_modes[@]}"; do
    for TASK in "${TASKS[@]}"; do
        read -ra _task_rate_arr <<< "$(_task_rates "${TASK}")"
        echo
        echo "rc_mode=${RC_MODE}  task=${TASK}"
        echo "============================================================"

        # Per-rate SLO (one call per rate, N=num_examples each)
        for RATE in "${_task_rate_arr[@]}"; do
            echo "  rate=${RATE}"
            echo "  ----------------------------------------------------------"
            python test/dlm_slorate.py \
                --latency-dir "${OUTPUT_ROOT}/rc_mode_${RC_MODE}/request_rate_${RATE}/${TASK}" \
                --tasks "${TASK}" \
                --slo-config "${SLO_CONFIG_PATH}" \
                --output-json "${OUTPUT_ROOT}/rc_mode_${RC_MODE}/slo_rates_${TASK}_rate${RATE}.json"
        done

        # Aggregate across all rates for this task
        _latency_dirs=()
        for RATE in "${_task_rate_arr[@]}"; do
            _latency_dirs+=("${OUTPUT_ROOT}/rc_mode_${RC_MODE}/request_rate_${RATE}/${TASK}")
        done
        echo "  [aggregate rates=${_task_rate_arr[*]}]"
        echo "  ----------------------------------------------------------"
        python test/dlm_slorate.py \
            --latency-dir "${_latency_dirs[@]}" \
            --tasks "${TASK}" \
            --slo-config "${SLO_CONFIG_PATH}" \
            --output-json "${OUTPUT_ROOT}/rc_mode_${RC_MODE}/slo_rates_${TASK}.json"
    done
done

echo
echo "============================================================"
echo "Throughput Summary"
echo "============================================================"

# Write per-task rates to JSON so the python summary can iterate correctly.
_TASK_RATES_FILE="${OUTPUT_ROOT}/task_rates.json"
python3 - << PYEOF > "${_TASK_RATES_FILE}"
import json
d = {}
$(for _T in "${TASKS[@]}"; do
    read -ra _r <<< "$(_task_rates "${_T}")"
    echo "d['${_T}'] = [$(printf '"%s", ' "${_r[@]}" | sed 's/, $//')]"
done)
print(json.dumps(d, indent=2))
PYEOF

python3 - << PYEOF
import json
from pathlib import Path

output_root = '${OUTPUT_ROOT}'
tasks = '${TASKS[*]}'.split()
modes = '${_ordered_modes[*]}'.split()
model_tag = '${MODEL_PATH}'.replace('/', '_')
task_rates = json.loads(Path('${_TASK_RATES_FILE}').read_text())

print(f"{'Mode':<10} {'Task':<14} {'Rate':>6} {'p50TTFB':>10} {'p95TTFB':>10} {'p50TPOB':>10} {'p95TPOB':>10}")
print('-' * 70)
for mode in modes:
    for task in tasks:
        for rate in task_rates.get(task, []):
            p = Path(output_root) / f'rc_mode_{mode}' / f'request_rate_{rate}' / task / f'{task}_{model_tag}.json'
            if not p.exists():
                print(f'{mode:<10} {task:<14} {rate:>6} {"N/A":>10} {"N/A":>10} {"N/A":>10} {"N/A":>10}')
                continue
            try:
                d = json.loads(p.read_text())
                ls = d.get('latency_stats', {})
                fmt = lambda v: f'{v:.1f}' if v is not None else 'N/A'
                print(f'{mode:<10} {task:<14} {rate:>6}'
                      f' {fmt(ls.get("p50_ttfb_ms")):>10}'
                      f' {fmt(ls.get("p95_ttfb_ms")):>10}'
                      f' {fmt(ls.get("p50_tpob_ms")):>10}'
                      f' {fmt(ls.get("p95_tpob_ms")):>10}')
            except Exception as e:
                print(f'{mode:<10} {task:<14} {rate:>6} {"ERR":>10} {str(e)[:20]}')
PYEOF

echo
echo "============================================================"
echo "Scatter & SLO attainment plots"
echo "============================================================"

for TASK in "${TASKS[@]}"; do
    read -ra _task_rate_arr <<< "$(_task_rates "${TASK}")"
    python3 test/plot_dlm_bellman_test.py \
        --output-root "${OUTPUT_ROOT}" \
        --modes "${_ordered_modes[@]}" \
        --tasks "${TASK}" \
        --rates "${_task_rate_arr[@]}" \
        --model-path "${MODEL_PATH}" \
        --slo-config "${SLO_CONFIG_PATH}" \
        --output-dir "${OUTPUT_ROOT}/plots" \
        --dpi 200
done

echo
echo "Done. Results: ${OUTPUT_ROOT}/"
echo "Plots:         ${OUTPUT_ROOT}/plots/"
