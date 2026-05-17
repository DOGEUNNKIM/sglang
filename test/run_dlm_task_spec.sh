#!/usr/bin/env bash
# Measure task characteristics: input length distribution and steps-per-block distribution.
# Starts a fresh server per task, runs a short benchmark, then plots combined figures.
set -euo pipefail

###### model config ######
BLOCK_SIZE="${BLOCK_SIZE:-32}" # 16 SDAR 는 config.py도 변경 필요
MODEL_PATH="${MODEL_PATH:-JetLM/SDAR-8B-Chat}" #JetLM/SDAR-8B-Chat
######

SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/nvme0/kdg6245}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/dlm_task_spec}"
TASKS=(${TASKS:-sharegpt humaneval math gsm8k gpqa mmlu ruler_1k ruler_2k ruler_3k ruler_4k ruler_1_4k})
NUM_EXAMPLES="${NUM_EXAMPLES:-200}"
WARMUP="${WARMUP:-16}"
REQUEST_RATE="${REQUEST_RATE:-200}"
NUM_THREADS="${NUM_THREADS:-200}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
PORT="${PORT:-31000}"
BASE_URL="${BASE_URL:-http://localhost:${PORT}}"
TP_SIZE="${TP_SIZE:-1}"
NUM_OUTPUT_BLOCKS="${NUM_OUTPUT_BLOCKS:-0}"

THRESHOLD="${THRESHOLD:-0.85}"
FORWARD_TIME_S="${FORWARD_TIME_S:-0.030}"
CONFIG_PATH="${CONFIG_PATH:-${OUTPUT_ROOT}/dlm_algo_config.yaml}"

SERVER_PID=""

# Per-task log files (overwritten each task run, then copied to OUT_DIR).
_LATENCY_LOG="${OUTPUT_ROOT}/tmp_request_latency.jsonl"
_STEP_LOG="${OUTPUT_ROOT}/tmp_step_stats.jsonl"
export REQUEST_LATENCY_LOG_FILE="${_LATENCY_LOG}"
export STEP_LOG_FILE="${_STEP_LOG}"
export BATCH_LATENCY_LOG_FILE="/dev/null"

write_config() {
    mkdir -p "$(dirname "${CONFIG_PATH}")"
    cat > "${CONFIG_PATH}" <<EOF
threshold: ${THRESHOLD}
dllm_admission_window: ${NUM_THREADS}
forward_time_s: ${FORWARD_TIME_S}
scheduler_mode: fcfs
request_latency_log_file: ${_LATENCY_LOG}
step_log_file: ${_STEP_LOG}
batch_latency_log_file: /dev/null
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

# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: collect per-task data
# ──────────────────────────────────────────────────────────────────────────────
for TASK in "${TASKS[@]}"; do
    OUT_DIR="${OUTPUT_ROOT}/${TASK}"
    mkdir -p "${OUT_DIR}"

    echo
    echo "============================================================"
    echo "Task spec: task=${TASK}  output=${OUT_DIR}"
    echo "============================================================"

    write_config
    rm -f "${_LATENCY_LOG}"

    echo "===== task=${TASK} =====" >> "${SERVER_LOG}"
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

    echo "[server] pid=${SERVER_PID}, waiting for ready..."
    wait_server_ready
    echo "[server] ready"

    BENCH_ARGS=(
        test/dlm_benchmark.py
        --base-url "${BASE_URL}"
        --model "${MODEL_PATH}"
        --tasks "${TASK}"
        --block-size "${BLOCK_SIZE}"
        --request-rate "${REQUEST_RATE}"
        --num-threads "${NUM_THREADS}"
        --warmup "${WARMUP}"
        --num-output-blocks "${NUM_OUTPUT_BLOCKS}"
        --num-examples "${NUM_EXAMPLES}"
        --output-dir "${OUT_DIR}"
        --tp-size "${TP_SIZE}"
    )

    PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6 \
    python "${BENCH_ARGS[@]}"

    [[ -f "${_LATENCY_LOG}" ]] && cp "${_LATENCY_LOG}" "${OUT_DIR}/request_latency_${TASK}.jsonl"
    [[ -f "${_STEP_LOG}" ]]   && cp "${_STEP_LOG}"   "${OUT_DIR}/step_stats_${TASK}.jsonl"

    stop_server
done

# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: plot combined task spec figure
# ──────────────────────────────────────────────────────────────────────────────
echo
echo "============================================================"
echo "Plotting task spec..."
echo "============================================================"

MODEL_SLUG="${MODEL_PATH//\//_}"

python3 - <<PYEOF
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

output_root = "${OUTPUT_ROOT}"
tasks       = "${TASKS[*]}".split()
model_slug  = "${MODEL_SLUG}"

input_lens       = {}  # task -> list[int]
block_steps      = {}  # task -> list[int]  (flattened across all requests)
output_blocks    = {}  # task -> list[int]  (num output blocks per request)
within_batch_cvs = {}  # task -> list[float] (CoV of block_steps across co-running unmask reqs, per batch step)

for task in tasks:
    # ── request_latency: input len, steps/block, output blocks ──
    lat_path = os.path.join(output_root, task, f"request_latency_{task}.jsonl")
    if not os.path.exists(lat_path):
        print(f"[warn] missing {lat_path}", file=sys.stderr)
    else:
        ilens, steps, oblocks = [], [], []
        with open(lat_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("input_len") is not None:
                    ilens.append(rec["input_len"])
                bsl = rec.get("block_steps_list", [])
                steps.extend(bsl)
                oblocks.append(len(bsl))
        if ilens:
            input_lens[task]    = ilens
        if steps:
            block_steps[task]   = steps
        if oblocks:
            output_blocks[task] = oblocks

    # ── step_stats: within-batch CoV of block_steps for co-running unmask reqs ──
    step_path = os.path.join(output_root, task, f"step_stats_{task}.jsonl")
    if not os.path.exists(step_path):
        print(f"[warn] missing {step_path}", file=sys.stderr)
    else:
        cvs = []
        with open(step_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                modes  = rec.get("req_modes", [])
                bsteps = rec.get("block_steps", [])
                # Collect block_steps only for requests currently unmasking (decode)
                unmask_vals = [
                    bsteps[i] for i, m in enumerate(modes)
                    if m == "unmask" and i < len(bsteps) and bsteps[i] > 0
                ]
                if len(unmask_vals) >= 2:
                    mean_v = float(np.mean(unmask_vals))
                    if mean_v > 0:
                        cvs.append(float(np.std(unmask_vals)) / mean_v * 100)
        if cvs:
            within_batch_cvs[task] = cvs

tasks_with_data = [t for t in tasks if t in input_lens or t in block_steps]
if not tasks_with_data:
    print("No data found, skipping plot.", file=sys.stderr)
    sys.exit(0)

n = len(tasks_with_data)
fig, axes = plt.subplots(4, n, figsize=(4.5 * n, 16), squeeze=False)

for col, task in enumerate(tasks_with_data):
    # ── Row 0: input length distribution ──
    ax = axes[0][col]
    ilens = input_lens.get(task, [])
    if ilens:
        ax.hist(ilens, bins=40, color="steelblue", edgecolor="white")
        mean_il = float(np.mean(ilens))
        ax.axvline(mean_il, color="red", linestyle="--", linewidth=1.2,
                   label=f"mean={mean_il:.0f}")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(task)
    ax.set_xlabel("Input tokens")
    ax.set_ylabel("Count")

    # ── Row 1: steps per block distribution ──
    ax = axes[1][col]
    steps = block_steps.get(task, [])
    if steps:
        max_step = max(steps)
        bins = list(range(1, max_step + 2))
        ax.hist(steps, bins=bins, align="left", rwidth=0.8,
                color="darkorange", edgecolor="white")
        mean_s = float(np.mean(steps))
        std_s  = float(np.std(steps))
        ax.axvline(mean_s, color="red", linestyle="--", linewidth=1.2,
                   label=f"mean={mean_s:.1f}\nstd={std_s:.1f}")
        ax.set_xticks(range(1, min(max_step + 1, 33)))
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Steps per block")
    ax.set_ylabel("Count")

    # ── Row 2: output block count distribution ──
    ax = axes[2][col]
    oblocks = output_blocks.get(task, [])
    if oblocks:
        max_ob = max(oblocks)
        bins = list(range(0, max_ob + 2))
        ax.hist(oblocks, bins=bins, align="left", rwidth=0.8,
                color="mediumseagreen", edgecolor="white")
        mean_ob = float(np.mean(oblocks))
        std_ob  = float(np.std(oblocks))
        ax.axvline(mean_ob, color="red", linestyle="--", linewidth=1.2,
                   label=f"mean={mean_ob:.1f}\nstd={std_ob:.1f}")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Output blocks per request")
    ax.set_ylabel("Count")

    # ── Row 3: within-batch CoV of block_steps (bubble opportunity) ──
    ax = axes[3][col]
    cvs = within_batch_cvs.get(task, [])
    if cvs:
        ax.hist(cvs, bins=40, color="mediumpurple", edgecolor="white")
        mean_cv = float(np.mean(cvs))
        std_cv  = float(np.std(cvs))
        ax.axvline(mean_cv, color="red", linestyle="--", linewidth=1.2,
                   label=f"mean={mean_cv:.1f}%\nstd={std_cv:.1f}%")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No data\n(re-run needed)", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
    ax.set_xlabel("Within-batch CoV of steps (%)")
    ax.set_ylabel("Count")

axes[0][0].set_ylabel("Input length\nCount")
axes[1][0].set_ylabel("Steps per block\nCount")
axes[2][0].set_ylabel("Output blocks\nCount")
axes[3][0].set_ylabel("Within-batch CoV\n(bubble opp.) Count")

fig.suptitle(f"Task characteristics — {model_slug} (block_size=${BLOCK_SIZE})", fontsize=13)
plt.tight_layout()

out_path = os.path.join(output_root, f"task_spec_{model_slug}.png")
os.makedirs(output_root, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# Also print summary stats
print()
print(f"{'Task':<12} {'InputLen mean':>14} {'InputLen std':>12} {'Steps mean':>11} {'Steps std':>10} {'Steps CoV':>10} {'OutBlk mean':>12} {'OutBlk std':>11} {'WB-CoV mean':>12} {'WB-CoV std':>11}")
print("-" * 122)
for task in tasks_with_data:
    il = input_lens.get(task, [])
    st = block_steps.get(task, [])
    ob = output_blocks.get(task, [])
    cv = within_batch_cvs.get(task, [])
    il_mean  = f"{np.mean(il):.0f}"  if il else "N/A"
    il_std   = f"{np.std(il):.0f}"   if il else "N/A"
    st_mean  = f"{np.mean(st):.2f}"  if st else "N/A"
    st_std   = f"{np.std(st):.2f}"   if st else "N/A"
    st_cov   = f"{np.std(st)/np.mean(st)*100:.1f}%" if st else "N/A"
    ob_mean  = f"{np.mean(ob):.2f}"  if ob else "N/A"
    ob_std   = f"{np.std(ob):.2f}"   if ob else "N/A"
    cv_mean  = f"{np.mean(cv):.1f}%" if cv else "N/A"
    cv_std   = f"{np.std(cv):.1f}%"  if cv else "N/A"
    print(f"{task:<12} {il_mean:>14} {il_std:>12} {st_mean:>11} {st_std:>10} {st_cov:>10} {ob_mean:>12} {ob_std:>11} {cv_mean:>12} {cv_std:>11}")
PYEOF

echo
echo "Done. Results under ${OUTPUT_ROOT}/"
