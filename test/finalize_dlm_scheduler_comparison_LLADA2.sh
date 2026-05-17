#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nvme0/kdg6245/dlm_sched_comparison_LLADA2}"
MODEL_PATH="${MODEL_PATH:-inclusionAI/LLaDA2.0-mini}"

python ./test/finalize_dlm_scheduler_comparison.py \
    --output-root "${OUTPUT_ROOT}" \
    --model-path "${MODEL_PATH}" \
    "$@"
