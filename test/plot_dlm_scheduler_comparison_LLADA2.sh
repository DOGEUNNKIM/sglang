#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nvme0/kdg6245/dlm_sched_comparison_LLADA2}"

python ./test/plot_dlm_scheduler_comparison.py \
    --summary "${OUTPUT_ROOT}/slo_summary.json" \
    --output "${OUTPUT_ROOT}/slo_attainment_comparison.png" \
    --slo-config "${OUTPUT_ROOT}/slo_config.json" \
    --p99-normalize-baseline LST \
    "$@"
