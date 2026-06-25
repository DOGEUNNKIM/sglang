#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nvme0/kdg6245/dlm_sched_comparison}"
LLADA2_ROOT="${LLADA2_ROOT:-/mnt/nvme0/kdg6245/dlm_sched_comparison_LLADA2}"
SDAR_ROOT="${SDAR_ROOT:-/mnt/nvme0/kdg6245/dlm_sched_comparison_SDAR}"
SCATTER_ROOT="${SCATTER_ROOT:-/mnt/nvme0/kdg6245/dlm_scatter_LLADA2_gsm8k_r10_p50}"
SCATTER_ROOT_SDAR="${SCATTER_ROOT_SDAR:-/mnt/nvme0/kdg6245/dlm_scatter_SDAR_gsm8k_r4_p50}"
SCATTER_RATE_SDAR="${SCATTER_RATE_SDAR:-3.5}"
BELLMAN_ROOT="${BELLMAN_ROOT:-/mnt/nvme0/kdg6245/dlm_bellman_slo_change/slo_tight_all}"
BLOCK_UNMASK_SUMMARY="${BLOCK_UNMASK_SUMMARY:-${SCATTER_ROOT}/scheduler_TTFB/request_rate_10/gsm8k/summary_inclusionAI_LLaDA2.0-mini.json}"
BLOCK_UNMASK_BELLMAN="${BLOCK_UNMASK_BELLMAN:-/mnt/nvme0/kdg6245/dlm_bellman_test/rc_mode_zero/request_rate_9/gsm8k/bellman_log_gsm8k.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/miniconda3/userenvs/kdg6245/envs/dlm/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_dlm_scheduler_comparison.py" \
    --summary "${LLADA2_ROOT}/slo_summary.json" \
    --summary-label $'LLaDA2.0\n-mini' \
    --summary2 "${SDAR_ROOT}/slo_summary.json" \
    --summary2-label $'SDAR\n-8B-Chat' \
    --output "${OUTPUT_ROOT}/slo_attainment_comparison.pdf" \
    --slo-config "${LLADA2_ROOT}/slo_config.json" \
    --scatter-root "${SCATTER_ROOT_SDAR}" \
    --scatter-task gsm8k \
    --scatter-rate "${SCATTER_RATE_SDAR}" \
    --scatter-output "${OUTPUT_ROOT}/slo_attainment_comparison_scatter_multi_slo.pdf" \
    --single-slo-scatter-output "${OUTPUT_ROOT}/slo_attainment_comparison_scatter.pdf" \
    --p99-normalize-baseline LST \
    --user-experiment-task gsm8k \
    --bar-task gsm8k \
    --bar-rate 9.5 \
    --block-unmask-summary "${BLOCK_UNMASK_SUMMARY}" \
    --block-unmask-task gsm8k \
    --block-unmask-output "${OUTPUT_ROOT}/block_unmask_steps_by_masked_tokens.pdf" \
    --no-sr \
    --no-bar \
    --no-scatter-combined

"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_throughput_figures.py" \
    --llada-json "${SCRIPT_DIR}/throughput_llada2_results.json" \
    --sdar-json "${SCRIPT_DIR}/throughput_sdar_results.json" \
    --multigpu-json "${SCRIPT_DIR}/throughput_multigpu_results.json" \
    --output-dir "${OUTPUT_ROOT}"

#"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_dlm_bellman_test.py" \
#    --output-root "${BELLMAN_ROOT}" \
#    --tasks sharegpt ruler_4k \
#    --model-path inclusionAI/LLaDA2.0-mini \
#    --slo-config "${BELLMAN_ROOT}/slo_config.json" \
#    --output-dir "${BELLMAN_ROOT}/plots" \
#    --report-tasks ruler_4k sharegpt gsm8k \
#    --report-json "${SCRIPT_DIR}/bellman_strict_slo_report_values.json" \
#    --report-output "${OUTPUT_ROOT}/strict_slo_attainment_report.pdf" \
#    --skip-task-plots \
#    --slo-attainment-only
