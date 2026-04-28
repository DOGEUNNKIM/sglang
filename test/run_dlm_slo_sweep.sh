#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-inclusionAI/LLaDA2.0-mini}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/dlm_results}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
WARMUP="${WARMUP:-16}"
NUM_OUTPUT_BLOCKS="${NUM_OUTPUT_BLOCKS:-0}"
REQUEST_RATES=(${REQUEST_RATES:-2 4 8})
TASKS=(${TASKS:-gsm8k humaneval math})
NUM_EXAMPLES="${NUM_EXAMPLES:-200}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
PORT="${PORT:-30000}"
BASE_URL="${BASE_URL:-http://localhost:${PORT}}"
THRESHOLD="${THRESHOLD:-0.95}"
CONFIG_PATH="${CONFIG_PATH:-/tmp/dlm_algo_config.yaml}"
STEP_LOG_FILE="${STEP_LOG_FILE:-/tmp/dlm_step_stats.jsonl}"
REQUEST_LATENCY_LOG_FILE="${REQUEST_LATENCY_LOG_FILE:-/tmp/dlm_request_latency.jsonl}"
BATCH_LATENCY_LOG_FILE="${BATCH_LATENCY_LOG_FILE:-/tmp/dlm_batch_latency.jsonl}"

MODEL_TAG="${MODEL_PATH//\//_}"
SERVER_PID=""

write_dllm_config() {
    cat > "${CONFIG_PATH}" <<EOF
threshold: ${THRESHOLD}
step_log_file: ${STEP_LOG_FILE}
request_latency_log_file: ${REQUEST_LATENCY_LOG_FILE}
batch_latency_log_file: ${BATCH_LATENCY_LOG_FILE}
EOF
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
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
    SERVER_PID=""
}

trap stop_server EXIT

for RATE in "${REQUEST_RATES[@]}"; do
    OUT_DIR="${OUTPUT_ROOT}/request_rate_${RATE}"

    mkdir -p "${OUT_DIR}"

    echo
    echo "============================================================"
    echo "DLM benchmark: request_rate=${RATE}, output_dir=${OUT_DIR}"
    echo "A fresh server will be started for this rate."
    echo "============================================================"

    write_dllm_config

    PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6 \
    python -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --port "${PORT}" \
        --trust-remote-code \
        --dllm-algorithm LowConfidence \
        --dllm-algorithm-config "${CONFIG_PATH}" \
        --attention-backend flashinfer \
        --max-running-requests "${MAX_RUNNING_REQUESTS}" \
        --disable-cuda-graph-padding \
        >> /tmp/dlm_results/run_dlm_slo_server_log.txt 2>&1 &
    SERVER_PID=$!

    echo "[server] pid=${SERVER_PID}, waiting for ${BASE_URL}/health"
    wait_server_ready
    echo "[server] ready"

    BENCH_ARGS=(
        test/dlm_benchmark.py
        --base-url "${BASE_URL}"
        --model "${MODEL_PATH}"
        --tasks "${TASKS[@]}"
        --block-size "${BLOCK_SIZE}"
        --log
        --request-rate "${RATE}"
        --warmup "${WARMUP}"
        --num-output-blocks "${NUM_OUTPUT_BLOCKS}"
        --output-dir "${OUT_DIR}"
    )

    if [[ -n "${NUM_EXAMPLES}" ]]; then
        BENCH_ARGS+=(--num-examples "${NUM_EXAMPLES}")
    fi

    PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6 \
    python "${BENCH_ARGS[@]}"

    stop_server
done

echo
echo "============================================================"
echo "DLM SLO rates"
echo "============================================================"

for RATE in "${REQUEST_RATES[@]}"; do
    OUT_DIR="${OUTPUT_ROOT}/request_rate_${RATE}"
    SUMMARY_PATH="${OUT_DIR}/summary_${MODEL_TAG}.json"
    SLO_PATH="${OUT_DIR}/slo_rates.json"

    echo
    echo "DLM SLO rate: request_rate=${RATE}"
    echo "------------------------------------------------------------"

    python test/dlm_slorate.py \
        --summary "${SUMMARY_PATH}" \
        --output-json "${SLO_PATH}"
done

echo
echo "Done. Results are under ${OUTPUT_ROOT}/request_rate_{${REQUEST_RATES[*]}}"
