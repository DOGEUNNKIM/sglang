#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nvme0/kdg6245/dlm_sched_comparison_SDAR}"
MODEL_PATH="${MODEL_PATH:-JetLM/SDAR-8B-Chat}"

python ./test/finalize_dlm_scheduler_comparison.py \
    --output-root "${OUTPUT_ROOT}" \
    --model-path "${MODEL_PATH}" \
    "$@"
