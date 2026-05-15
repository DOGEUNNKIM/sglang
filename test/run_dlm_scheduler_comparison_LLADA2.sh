#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-inclusionAI/LLaDA2.0-mini}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"

################################
TASKS=(${TASKS:-ruler_1_4k})  ##### TASK humaneval math gsm8k gpqa mmlu ruler_1k ruler_2k ruler_3k ruler_4k ruler_8k ruler_16k sharegpt
RATES_GSM8K="${RATES_GSM8K:-200}"
RATES_MMLU="${RATES_MMLU:-200}"
RATES_HUMANEVAL="${RATES_HUMANEVAL:-200}"
RATES_MATH="${RATES_MATH:-200}"
RATES_GPQA="${RATES_GPQA:-200}"
RATES_SHAREGPT="${RATES_SHAREGPT:-200}"
RATES_RULER_1_4K="${RATES_RULER_1_4K:-200}"
RATES_RULER_1K="${RATES_RULER_1K:-200}"
RATES_RULER_2K="${RATES_RULER_2K:-200}"
RATES_RULER_3K="${RATES_RULER_3K:-200}"
RATES_RULER_4K="${RATES_RULER_4K:-200}"
RATES_RULER_8K="${RATES_RULER_8K:-200}"
RATES_RULER_16K="${RATES_RULER_16K:-200}"
RATES_LONGBENCH_V2="${RATES_LONGBENCH_V2:-200}"
NUM_EXAMPLES_GSM8K="${NUM_EXAMPLES_GSM8K:-200}"
NUM_EXAMPLES_HUMANEVAL="${NUM_EXAMPLES_HUMANEVAL:-200}"
NUM_EXAMPLES_MATH="${NUM_EXAMPLES_MATH:-200}"
NUM_EXAMPLES_GPQA="${NUM_EXAMPLES_GPQA:-200}"
NUM_EXAMPLES_MMLU="${NUM_EXAMPLES_MMLU:-200}"
NUM_EXAMPLES_SHAREGPT="${NUM_EXAMPLES_SHAREGPT:-200}"
NUM_EXAMPLES_RULER_1_4K="${NUM_EXAMPLES_RULER_1_4K:-200}"
NUM_EXAMPLES_RULER_1K="${NUM_EXAMPLES_RULER_1K:-200}"
NUM_EXAMPLES_RULER_2K="${NUM_EXAMPLES_RULER_2K:-200}"
NUM_EXAMPLES_RULER_3K="${NUM_EXAMPLES_RULER_3K:-200}"
NUM_EXAMPLES_RULER_4K="${NUM_EXAMPLES_RULER_4K:-200}"
NUM_EXAMPLES_LONGBENCH_V2="${NUM_EXAMPLES_LONGBENCH_V2:-}"
NUM_EXAMPLES_RULER_8K="${NUM_EXAMPLES_RULER_8K:-200}"
NUM_EXAMPLES_RULER_16K="${NUM_EXAMPLES_RULER_16K:-200}"
SCHEDULERS=(${SCHEDULERS:-FCFS})
STRICT_MULTIPLIER="${STRICT_MULTIPLIER:-10.0}"
RELEASE_MULTIPLIER="${RELEASE_MULTIPLIER:-20.0}"
STRICT_PROB="${STRICT_PROB:-1}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
WARMUP="${WARMUP:-32}"
TP_SIZE="${TP_SIZE:-1}"
FORWARD_TIME_S="${FORWARD_TIME_S:-0.030}"
################################

SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/nvme0/kdg6245}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/dlm_sched_comparison_LLADA2_baseline}"
REQUEST_RATES=(${REQUEST_RATES:-})
NUM_OUTPUT_BLOCKS="${NUM_OUTPUT_BLOCKS:-0}"
NUM_THREADS="${NUM_THREADS:-}"
DLLM_ADMISSION_WINDOW="${DLLM_ADMISSION_WINDOW:-}"
PORT="${PORT:-30010}"
BASE_URL="${BASE_URL:-http://localhost:${PORT}}"
THRESHOLD="${THRESHOLD:-0.95}"
PREFILL_FORWARD_TIME_S="${PREFILL_FORWARD_TIME_S:-}"
DECODE_FORWARD_TIME_S="${DECODE_FORWARD_TIME_S:-}"
CONFIG_PATH="${CONFIG_PATH:-${OUTPUT_ROOT}/dlm_algo_config_sched_cmp_LLADA2_baseline.yaml}"
STEP_LOG_FILE="${STEP_LOG_FILE:-${OUTPUT_ROOT}/dlm_step_stats_LLADA2_baseline.jsonl}"
REQUEST_LATENCY_LOG_FILE="${REQUEST_LATENCY_LOG_FILE:-${OUTPUT_ROOT}/dlm_request_latency_LLADA2_baseline.jsonl}"
BATCH_LATENCY_LOG_FILE="${BATCH_LATENCY_LOG_FILE:-${OUTPUT_ROOT}/dlm_batch_latency_LLADA2_baseline.jsonl}"
export STEP_LOG_FILE REQUEST_LATENCY_LOG_FILE BATCH_LATENCY_LOG_FILE
GPU_FREE_MEMORY_MIN_MB="${GPU_FREE_MEMORY_MIN_MB:-70000}"

SERVER_PID=""

_task_dataset_size() {
    case "${1}" in
        gsm8k)         echo "1314"  ;;
        humaneval)     echo "164"   ;;
        math)          echo "5000"  ;;
        gpqa)          echo "198"   ;;
        mmlu)          echo "14042" ;;
        longbench_v2)  echo "503"   ;;
        ruler_1_4k)    echo "6500"  ;;
        ruler_1k)      echo "6500"  ;;
        ruler_2k)      echo "6500"  ;;
        ruler_3k)      echo "6500"  ;;
        ruler_4k)      echo "6500"  ;;
        ruler_8k)      echo "6500"  ;;
        ruler_16k)     echo "6500"  ;;
        sharegpt)      echo "70000" ;;
        *)             echo "4000"  ;;
    esac
}

_task_max_examples() {
    case "${1}" in
        gsm8k)         echo "${NUM_EXAMPLES_GSM8K}" ;;
        humaneval)     echo "${NUM_EXAMPLES_HUMANEVAL}" ;;
        math)          echo "${NUM_EXAMPLES_MATH}" ;;
        gpqa)          echo "${NUM_EXAMPLES_GPQA}" ;;
        mmlu)          echo "${NUM_EXAMPLES_MMLU}" ;;
        longbench_v2)  echo "${NUM_EXAMPLES_LONGBENCH_V2}" ;;
        ruler_1_4k)    echo "${NUM_EXAMPLES_RULER_1_4K}" ;;
        ruler_1k)      echo "${NUM_EXAMPLES_RULER_1K}" ;;
        ruler_2k)      echo "${NUM_EXAMPLES_RULER_2K}" ;;
        ruler_3k)      echo "${NUM_EXAMPLES_RULER_3K}" ;;
        ruler_4k)      echo "${NUM_EXAMPLES_RULER_4K}" ;;
        ruler_8k)      echo "${NUM_EXAMPLES_RULER_8K}" ;;
        ruler_16k)     echo "${NUM_EXAMPLES_RULER_16K}" ;;
        sharegpt)      echo "${NUM_EXAMPLES_SHAREGPT}" ;;
        *)             echo "" ;;
    esac
}

_compute_task_slo() {
    local _ideal_ttfb_ms="${1}" _ideal_tpob_ms="${2}" _multiplier="${3}"
    python3 -c "
m = float('${_multiplier}')
print(f'{float(\"${_ideal_ttfb_ms}\")*m/1000:.4f} {float(\"${_ideal_tpob_ms}\")*m/1000:.4f}')
"
}

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
        wait_gpu_memory_released "server"
    fi
    SERVER_PID=""
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

# _cleanup_benchmark_outputs OUT_DIR TASK
# Keep the logs needed by dlm_slorate.py and plot_step_dist.py, and drop
# duplicated/raw benchmark artifacts that make large sweeps fill scratch quickly.
_cleanup_benchmark_outputs() {
    local _out="${1}" _task="${2}"
    local _model_tag="${MODEL_PATH//\//_}"
    local _summary="${_out}/summary_${_model_tag}.json"

    rm -f "${_out}/steps_${_task}.jsonl"
    rm -f "${_out}/step_dist_${_model_tag}.png" \
          "${_out}/step_boxplot_${_model_tag}.png" \
          "${_out}/request_latency_${_model_tag}.png" \
          "${_out}/batch_latency_${_model_tag}.png" \
          "${_out}/phase_sequence_${_model_tag}.png" \
          "${_out}/phase_composition_${_model_tag}.png"

    if [[ -f "${_summary}" ]]; then
        python3 - "${_summary}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
tmp = path.with_suffix(path.suffix + ".tmp")
with path.open() as f:
    data = json.load(f)
data.pop("latency_data", None)
data.pop("step_data", None)
with tmp.open("w") as f:
    json.dump(data, f, indent=2)
os.replace(tmp, path)
PY
    fi
}

trap stop_server EXIT

SERVER_LOG="${OUTPUT_ROOT}/server_log.txt"
mkdir -p "${OUTPUT_ROOT}"

_stale_pid=$(lsof -ti tcp:"${PORT}" 2>/dev/null || true)
if [[ -n "${_stale_pid}" ]]; then
    echo "[startup] killing stale process(es) on port ${PORT}: ${_stale_pid}"
    kill -9 ${_stale_pid} 2>/dev/null || true
    wait_gpu_memory_released "startup"
fi

declare -A IDEAL_TTFB_MS IDEAL_TPOB_MS

_task_rates() {
    case "${1}" in
        gsm8k)         echo "${RATES_GSM8K}" ;;
        humaneval)     echo "${RATES_HUMANEVAL}" ;;
        math)          echo "${RATES_MATH}" ;;
        gpqa)          echo "${RATES_GPQA}" ;;
        mmlu)          echo "${RATES_MMLU}" ;;
        longbench_v2)  echo "${RATES_LONGBENCH_V2}" ;;
        ruler_1_4k)    echo "${RATES_RULER_1_4K}" ;;
        ruler_1k)      echo "${RATES_RULER_1K}" ;;
        ruler_2k)      echo "${RATES_RULER_2K}" ;;
        ruler_3k)      echo "${RATES_RULER_3K}" ;;
        ruler_4k)      echo "${RATES_RULER_4K}" ;;
        ruler_8k)      echo "${RATES_RULER_8K}" ;;
        ruler_16k)     echo "${RATES_RULER_16K}" ;;
        sharegpt)      echo "${RATES_SHAREGPT}" ;;
        *)             echo "${REQUEST_RATES[*]}" ;;
    esac
}

# ── Main comparison ──────────────────────────────────────────────────────────
for SCHEDULER in "${SCHEDULERS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        _rates=($(_task_rates "${TASK}"))
        for RATE in "${_rates[@]}"; do
        OUT_DIR="${OUTPUT_ROOT}/scheduler_${SCHEDULER}/request_rate_${RATE}"
        mkdir -p "${OUT_DIR}"

        {
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
                --mem-fraction-static 0.95 \
                >> "${SERVER_LOG}" 2>&1 &
            SERVER_PID=$!

            echo "[server] pid=${SERVER_PID}, waiting for ${BASE_URL}/health"
            wait_server_ready
            echo "[server] ready"

            rm -f "${STEP_LOG_FILE}" "${REQUEST_LATENCY_LOG_FILE}" "${BATCH_LATENCY_LOG_FILE}"

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
            )
            [[ -n "${_task_examples}" ]] && BENCH_ARGS+=(--num-examples "${_task_examples}")

            PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6 \
            python "${BENCH_ARGS[@]}"

            [[ -f "${STEP_LOG_FILE}" ]] && cp "${STEP_LOG_FILE}" "${OUT_DIR}/step_stats_${TASK}.jsonl"
            _cleanup_benchmark_outputs "${OUT_DIR}" "${TASK}"

            stop_server
            rm -f "${STEP_LOG_FILE}" "${REQUEST_LATENCY_LOG_FILE}" "${BATCH_LATENCY_LOG_FILE}"
        }  # TASK block
        done  # RATE
    done  # TASK

    if [[ "${SCHEDULER}" == "TTFB" ]]; then
        echo
        echo "[ideal] TTFB sched done — extracting p50_ideal_ttfb_ms at highest rate per task"
        for TASK in "${TASKS[@]}"; do
            _rates=($(_task_rates "${TASK}"))
            _high_rate="${_rates[-1]}"
            _out="${OUTPUT_ROOT}/scheduler_TTFB/request_rate_${_high_rate}"
            _val=$(_parse_calib_metric "${_out}" "${TASK}" "p50_ideal_ttfb_ms")
            if [[ -z "${_val}" ]]; then
                echo "[ideal] WARNING: p50_ideal_ttfb_ms not found for task=${TASK} rate=${_high_rate}" >&2
            fi
            IDEAL_TTFB_MS["${TASK}"]="${_val:-}"
            echo "  task=${TASK}  rate=${_high_rate}  ideal_ttfb=${_val:-N/A}ms"
        done
    fi

    if [[ "${SCHEDULER}" == "DECODE" ]]; then
        echo
        echo "[ideal] DECODE sched done — extracting p50_ideal_tpob_ms at highest rate per task"
        for TASK in "${TASKS[@]}"; do
            _rates=($(_task_rates "${TASK}"))
            _high_rate="${_rates[-1]}"
            _out="${OUTPUT_ROOT}/scheduler_DECODE/request_rate_${_high_rate}"
            _val=$(_parse_calib_metric "${_out}" "${TASK}" "p50_ideal_tpob_ms")
            if [[ -z "${_val}" ]]; then
                echo "[ideal] WARNING: p50_ideal_tpob_ms not found for task=${TASK} rate=${_high_rate}" >&2
            fi
            IDEAL_TPOB_MS["${TASK}"]="${_val:-}"
            echo "  task=${TASK}  rate=${_high_rate}  ideal_tpob=${_val:-N/A}ms"
        done
    fi

done  # SCHEDULER

# ── Write calibrated SLO config for dlm_slorate.py ───────────────────────────
SLO_CONFIG_PATH="${OUTPUT_ROOT}/slo_config.json"
python3 -c "
import json, sys
tasks = '${TASKS[*]}'.split()
strict_m  = float('${STRICT_MULTIPLIER}')
release_m = float('${RELEASE_MULTIPLIER}')
ideal_ttfb = {$(for T in "${TASKS[@]}"; do echo "'${T}': float('${IDEAL_TTFB_MS[${T}]:-0}'),"; done)}
ideal_tpob = {$(for T in "${TASKS[@]}"; do echo "'${T}': float('${IDEAL_TPOB_MS[${T}]:-0}'),"; done)}
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

echo
echo "============================================================"
echo "DLM Scheduler Comparison — SLO Summary"
echo "============================================================"

for SCHEDULER in "${SCHEDULERS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        _rates=($(_task_rates "${TASK}"))
        for RATE in "${_rates[@]}"; do
            OUT_DIR="${OUTPUT_ROOT}/scheduler_${SCHEDULER}/request_rate_${RATE}"
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

# ── Consolidated SLO summary ──────────────────────────────────────────────────
SUMMARY_PATH="${OUTPUT_ROOT}/slo_summary.json"
python3 -c "
import json, os
from pathlib import Path

output_root = '${OUTPUT_ROOT}'
schedulers  = '${SCHEDULERS[*]}'.split()
tasks       = '${TASKS[*]}'.split()
task_rate_map = {
    'gsm8k':        '${RATES_GSM8K}'.split(),
    'humaneval':    '${RATES_HUMANEVAL}'.split(),
    'math':         '${RATES_MATH}'.split(),
    'gpqa':         '${RATES_GPQA}'.split(),
    'mmlu':         '${RATES_MMLU}'.split(),
    'longbench_v2': '${RATES_LONGBENCH_V2}'.split(),
    'ruler_1_4k':   '${RATES_RULER_1_4K}'.split(),
    'ruler_1k':     '${RATES_RULER_1K}'.split(),
    'ruler_2k':     '${RATES_RULER_2K}'.split(),
    'ruler_3k':     '${RATES_RULER_3K}'.split(),
    'ruler_4k':     '${RATES_RULER_4K}'.split(),
    'ruler_8k':     '${RATES_RULER_8K}'.split(),
    'ruler_16k':    '${RATES_RULER_16K}'.split(),
    'sharegpt':     '${RATES_SHAREGPT}'.split(),
}

model_tag = '${MODEL_PATH}'.replace('/', '_')

def _read_bench_metrics(out_dir, task):
    p = Path(out_dir) / f'{task}_{model_tag}.json'
    if not p.exists():
        return None, None, None, None, None
    try:
        d = json.loads(p.read_text())
        ls = d.get('latency_stats', d)
        return (
            d.get('score'),
            d.get('score:std'),
            d.get('output_throughput_tok_s'),
            ls.get('p99_ttfb_ms'),
            ls.get('p99_tpob_ms'),
        )
    except Exception:
        return None, None, None, None, None

summary = {}
for sched in schedulers:
    summary[sched] = {}
    for task in tasks:
        for rate in task_rate_map.get(task, []):
            out_dir  = Path(output_root) / f'scheduler_{sched}' / f'request_rate_{rate}'
            slo_path = out_dir / 'slo_rates.json'
            if not slo_path.exists():
                continue
            data = json.loads(slo_path.read_text())
            if rate not in summary[sched]:
                summary[sched][rate] = {}
            if task not in data:
                continue
            rates_d = data[task].get('rates', {})
            score, score_std, throughput, p99_ttfb, p99_tpob = _read_bench_metrics(out_dir, task)
            summary[sched][rate][task] = {
                'score':        score,
                'score_std':    score_std,
                'throughput_tok_s': throughput,
                'strict_ttfb':  rates_d.get('strict_ttfb'),
                'strict_tpob':  rates_d.get('strict_tpob'),
                'strict_all':   rates_d.get('strict_all'),
                'relaxed_ttfb': rates_d.get('relaxed_ttfb'),
                'relaxed_tpob': rates_d.get('relaxed_tpob'),
                'relaxed_all':  rates_d.get('relaxed_all'),
                'p99_ttfb_ms':  p99_ttfb,
                'p99_tpob_ms':  p99_tpob,
            }

Path('${SUMMARY_PATH}').write_text(json.dumps(summary, indent=2))
print(f'[summary] saved → ${SUMMARY_PATH}')

header = f\"{'Scheduler':<10} {'Task':<12} {'Rate':>6} {'Acc':>8} {'Tok/s':>10} {'Str-TTFB':>10} {'Str-TPOB':>10} {'Str-All':>9} {'Rel-TTFB':>10} {'Rel-TPOB':>10} {'Rel-All':>9} {'P99-TTFB':>10} {'P99-TPOB':>10}\"
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
            print(f'{sched:<10} {task:<12} {rate:>6} {fmt(r[\"score\"]):>8} {fms(r[\"throughput_tok_s\"]):>10} {fmt(r[\"strict_ttfb\"]):>10} {fmt(r[\"strict_tpob\"]):>10} {fmt(r[\"strict_all\"]):>9} {fmt(r[\"relaxed_ttfb\"]):>10} {fmt(r[\"relaxed_tpob\"]):>10} {fmt(r[\"relaxed_all\"]):>9} {fms(r[\"p99_ttfb_ms\"]):>10} {fms(r[\"p99_tpob_ms\"]):>10}')
"

echo
echo "============================================================"
echo "Step distribution plots"
echo "============================================================"

MODEL_SLUG="${MODEL_PATH//\//_}"

for SCHEDULER in "${SCHEDULERS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        _rates=($(_task_rates "${TASK}"))
        for RATE in "${_rates[@]}"; do
            OUT_DIR="${OUTPUT_ROOT}/scheduler_${SCHEDULER}/request_rate_${RATE}"

            echo
            echo "Step distribution: scheduler=${SCHEDULER}, task=${TASK}, rate=${RATE}"
            echo "------------------------------------------------------------"

            python test/plot_step_dist.py \
                --log-dir "${OUT_DIR}" \
                --tasks "${TASK}" \
                --output "${OUT_DIR}/step_dist_${MODEL_SLUG}_${TASK}.png"
        done
    done
done

echo
echo "============================================================"
echo "Throughput Summary"
echo "============================================================"
python3 -c "
import json
from pathlib import Path

output_root = '${OUTPUT_ROOT}'
schedulers = '${SCHEDULERS[*]}'.split()
tasks = '${TASKS[*]}'.split()
model_tag = '${MODEL_PATH}'.replace('/', '_')
task_rate_map = {
    'gsm8k':        '${RATES_GSM8K}'.split(),
    'humaneval':    '${RATES_HUMANEVAL}'.split(),
    'math':         '${RATES_MATH}'.split(),
    'gpqa':         '${RATES_GPQA}'.split(),
    'mmlu':         '${RATES_MMLU}'.split(),
    'longbench_v2': '${RATES_LONGBENCH_V2}'.split(),
    'ruler_1_4k':   '${RATES_RULER_1_4K}'.split(),
    'ruler_1k':     '${RATES_RULER_1K}'.split(),
    'ruler_2k':     '${RATES_RULER_2K}'.split(),
    'ruler_3k':     '${RATES_RULER_3K}'.split(),
    'ruler_4k':     '${RATES_RULER_4K}'.split(),
    'ruler_8k':     '${RATES_RULER_8K}'.split(),
    'ruler_16k':    '${RATES_RULER_16K}'.split(),
    'sharegpt':     '${RATES_SHAREGPT}'.split(),
}

print(f\"{'Scheduler':<10} {'Task':<14} {'Rate':>6} {'Score':>8} {'Tok/s':>10}\")
print('-' * 52)
for sched in schedulers:
    for task in tasks:
        for rate in task_rate_map.get(task, []):
            p = Path(output_root) / f'scheduler_{sched}' / f'request_rate_{rate}' / f'{task}_{model_tag}.json'
            if not p.exists():
                print(f'{sched:<10} {task:<14} {rate:>6} {\"N/A\":>8} {\"N/A\":>10}')
                continue
            try:
                d = json.loads(p.read_text())
                score = d.get('score')
                tput  = d.get('output_throughput_tok_s')
                s = f'{score:.4f}' if score is not None else 'N/A'
                t = f'{tput:.1f}' if tput is not None else 'N/A'
                print(f'{sched:<10} {task:<14} {rate:>6} {s:>8} {t:>10}')
            except Exception as e:
                print(f'{sched:<10} {task:<14} {rate:>6} {\"ERR\":>8} {str(e)[:10]:>10}')
"

echo
echo "Done. Results are under ${OUTPUT_ROOT}/scheduler_*/request_rate_*/"
echo "Consolidated SLO summary: ${SUMMARY_PATH}"
