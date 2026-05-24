# DLM Scheduler Script Guide

This document describes the DLM run and plot scripts and how they relate to each other.

```text
run_*.sh
  -> launches sglang.launch_server
  -> sends requests via dlm_benchmark.py
  -> computes SLO attainment via dlm_slorate.py
  -> optionally re-generates plots with finalize/plot scripts
```

## File Overview

| File | Role |
|------|------|
| `run_dlm_scheduler_comparison_LLADA2.sh` | Scheduler comparison experiment for LLaDA2.0-mini |
| `run_dlm_scheduler_comparison_SDAR.sh` | Scheduler comparison experiment for SDAR-8B-Chat |
| `plot_dlm_scheduler_comparison_LLADA2.sh` | Re-plot wrapper for LLADA2 results |
| `plot_dlm_scheduler_comparison_SDAR.sh` | Re-plot wrapper for SDAR results |
| `run_dlm_task_spec.sh` | Measures per-task input length, output block, and step distributions |
| `run_dlm_tb_update_test.sh` | Verifies that the Bellman table update converges to the actual step distribution |
| `dlm_benchmark.py` | Benchmark runner — sends requests to a running SGLang server and saves per-task results |
| `dlm_slorate.py` | Computes SLO attainment rates from `request_latency_<task>.jsonl` |
| `plot_dlm_scheduler_comparison.py` | Generates scheduler comparison plots from `slo_summary.json` |
| `plot_bellman_convergence.py` | Plots Bellman table convergence from `bellman_log_<task>.jsonl` |

## Script and Python File Relationships

### Scripts

```text
run_dlm_scheduler_comparison_LLADA2.sh
  -> python -m sglang.launch_server
  -> test/dlm_benchmark.py
  -> test/dlm_slorate.py

run_dlm_scheduler_comparison_SDAR.sh
  -> python -m sglang.launch_server
  -> test/dlm_benchmark.py
  -> test/dlm_slorate.py

plot_dlm_scheduler_comparison_LLADA2.sh
  -> test/plot_dlm_scheduler_comparison.py

plot_dlm_scheduler_comparison_SDAR.sh
  -> test/plot_dlm_scheduler_comparison.py

run_dlm_task_spec.sh
  -> python -m sglang.launch_server
  -> test/dlm_benchmark.py
  -> inline plot code  (produces task_spec_<model>.png)

run_dlm_tb_update_test.sh
  -> python -m sglang.launch_server
  -> test/dlm_benchmark.py
  -> test/dlm_slorate.py
  -> test/plot_bellman_convergence.py
```

### Python Files

```text
dlm_benchmark.py
  - Benchmark runner for an already-running SGLang server
  - Outputs:
    <output-dir>/<task>_<model_tag>.json
    <output-dir>/summary_<model_tag>.json
    <output-dir>/request_latency_<task>.jsonl
    <output-dir>/batch_latency_<task>.jsonl
    <output-dir>/steps_<task>.jsonl

dlm_slorate.py
  - Computes SLO attainment from request_latency_<task>.jsonl
  - Outputs:
    <latency-dir>/slo_rates.json

plot_dlm_scheduler_comparison.py
  - Generates scheduler comparison plots from slo_summary.json
  - Outputs:
    slo_attainment_comparison.png
    slo_attainment_comparison_p99_latency.png
    slo_attainment_comparison_scatter.png
    slo_attainment_comparison_scatter_combined.png
    slo_attainment_comparison_strict_release.png
    slo_attainment_comparison_bar_p99.png

plot_bellman_convergence.py
  - Plots Bellman table convergence from bellman_log_<task>.jsonl
  - Outputs:
    PNG file specified by --output
```

## 1. Run Scheduler Comparison Script

### Run

```bash
./test/run_dlm_scheduler_comparison_LLADA2.sh
./test/run_dlm_scheduler_comparison_SDAR.sh
```

### Execution Modes

The scripts support three execution modes controlled by `RETRY_TASKS`:

| Mode | Condition | Behavior |
|------|-----------|----------|
| **Retry (partial)** | `RETRY_TASKS` is a strict subset of `TASKS` | Deletes and re-runs only the specified tasks across all schedulers; summary covers only `RETRY_TASKS` |
| **Full rerun** | `RETRY_TASKS` equals `TASKS` (same set) | Deletes all `scheduler_*` directories and re-runs everything |
| **Plot-only** | `RETRY_TASKS` is empty (`RETRY_TASKS=""`) | Skips all benchmarks; reads existing `slo_config.json` and `slo_rates.json` as-is; only regenerates `slo_summary.json` |

```bash
# Retry only math and gsm8k across all schedulers
RETRY_TASKS="math gsm8k" ./test/run_dlm_scheduler_comparison_LLADA2.sh

# Full rerun (same as default when RETRY_TASKS == TASKS)
./test/run_dlm_scheduler_comparison_LLADA2.sh

# Plot-only: regenerate slo_summary.json from existing files without running any benchmark
RETRY_TASKS="" ./test/run_dlm_scheduler_comparison_LLADA2.sh
```

`SUMMARY_TASKS` controls which tasks appear in the final `slo_summary.json`. It defaults to `RETRY_TASKS` in retry/full-rerun mode and to all `TASKS` in plot-only mode. It can also be set explicitly.

### Key Parameters

The two scripts share the same structure and differ only in model-specific defaults.

| Variable | LLADA2 default | SDAR default | Description |
|----------|---------------|-------------|-------------|
| `MODEL_PATH` | `inclusionAI/LLaDA2.0-mini` | `JetLM/SDAR-8B-Chat` | Model path for the server |
| `FORWARD_TIME_S` | `0.04` | `0.08` | Expected time per forward pass (s) |
| `OUTPUT_ROOT` | `.../dlm_sched_comparison_LLADA2` | `.../dlm_sched_comparison_SDAR` | Root directory for all outputs |
| `BLOCK_SIZE` | `32` | `32` | Output block size (tokens) |
| `TASKS` | `math mmlu gsm8k sharegpt ruler_4k gpqa humaneval` | `humaneval sharegpt math mmlu gsm8k gpqa ruler_4k` | Full task list |
| `RETRY_TASKS` | same as `TASKS` | same as `TASKS` | Tasks to re-run (empty = plot-only, equals TASKS = full rerun) |
| `SUMMARY_TASKS` | (auto) | (auto) | Tasks included in the summary; defaults to `RETRY_TASKS` if set, else `TASKS` |
| `RATES_GSM8K` | `8 9 10 11` | `3 3.5 4 4.5` | Request rate sweep for gsm8k (req/s) |
| `RATES_MMLU` | `2 2.2 2.4 2.6` | `1 1.2 1.4 1.6` | Request rate sweep for mmlu (req/s) |
| `RATES_MATH` | `2.6 2.8 3 3.2` | `1 1.1 1.2 1.3` | Request rate sweep for math (req/s) |
| `RATES_SHAREGPT` | `1.5 1.7 1.9 2.1` | `1.6 1.8 2.0 2.2` | Request rate sweep for sharegpt (req/s) |
| `RATES_RULER_4K` | `2 3 4 5` | `2 2.5 3 3.5` | Request rate sweep for ruler_4k (req/s) |
| `RATES_HUMANEVAL` | `20 25 30 35` | `10 20 30 40` | Request rate sweep for humaneval (req/s) |
| `RATES_GPQA` | `1.5 2 2.5 3` | `0.8 1.2 1.6 2` | Request rate sweep for gpqa (req/s) |
| `NUM_EXAMPLES_GSM8K/MATH/MMLU/SHAREGPT` | `1000` | `1000` | Number of requests for large tasks |
| `NUM_EXAMPLES_RULER_4K` | `400` | `400` | Number of requests for ruler_4k |
| `NUM_EXAMPLES_HUMANEVAL/GPQA` | `200` | `200` | Number of requests for small tasks |
| `SCHEDULERS` | `TTFB DECODE FCFS PREFILL SOLA LST` | same | Schedulers to compare |
| `STRICT_MULTIPLIER` | `10.0` | `10.0` | strict SLO = multiplier × ideal latency |
| `RELEASE_MULTIPLIER` | `20.0` | `20.0` | release SLO = multiplier × ideal latency |
| `MAX_RUNNING_REQUESTS` | `32` | `32` | Max concurrent requests on the server |
| `WARMUP` | `32` | `32` | Number of warmup requests |
| `TP_SIZE` | `1` | `1` | Tensor parallelism size |
| `THRESHOLD` | `0.95` | `0.95` | Low-confidence masking threshold |
| `PORT` | `30002` | `30001` | Server port |

### Internal Flow

#### Retry / Full-rerun mode (`RETRY_TASKS` non-empty)

```text
Cleanup:
  touch .retry_run_started          (RUN_MARKER — used to check log freshness)
  rm slo_summary.json slo_config.json server_log.txt
  if RETRY_TASKS == TASKS:
    rm -rf scheduler_*/             (full wipe)
  else:
    rm -rf scheduler_*/request_rate_*/<task>/ for each task in RETRY_TASKS

for scheduler in SCHEDULERS:
  for task in RETRY_TASKS:
    for rate in per-task request rates:
      1. Create OUT_DIR for this run
      2. Write dllm_algo_config.yaml
           scheduler_mode: ttfb | fcfs | lst | sola | prefill
           admission_window: MAX_RUNNING_REQUESTS (DECODE only) | dataset size (others)
           strict/release SLO thresholds (from calibration, if available)
      3. Launch sglang.launch_server
      4. Run dlm_benchmark.py
      5. Stop server

  if scheduler == TTFB:
    Read p50_ideal_ttfb_ms from scheduler_TTFB/.../highest_rate/<task>_<model>.json
    Store in IDEAL_TTFB_MS[task]  (used to compute SLO thresholds for later schedulers)

  if scheduler == DECODE:
    Read p50_ideal_tpob_ms from scheduler_DECODE/.../highest_rate/<task>_<model>.json
    Store in IDEAL_TPOB_MS[task]

Validate (exit 1 if any log is missing or older than RUN_MARKER):
  for each (scheduler, task, rate) in SCHEDULERS × SUMMARY_TASKS × rates:
    check dlm_request_latency_<task>.jsonl exists and is newer than RUN_MARKER

Build slo_config.json from IDEAL_TTFB_MS / IDEAL_TPOB_MS × STRICT/RELEASE_MULTIPLIER

Run dlm_slorate.py for each (scheduler, task, rate) in SCHEDULERS × SUMMARY_TASKS × rates
  -> writes slo_rates.json per run

Aggregate into slo_summary.json
```

#### Plot-only mode (`RETRY_TASKS` empty)

```text
No cleanup, no benchmark runs.

Check slo_config.json exists (exit 1 if missing)
Check all slo_rates.json exist for SCHEDULERS × SUMMARY_TASKS × rates (exit 1 if any missing)

Aggregate existing slo_rates.json into slo_summary.json  (only this file is rewritten)
```

### Output Structure

```text
OUTPUT_ROOT/
  .retry_run_started        (RUN_MARKER — touched at start of each retry run)
  server_log.txt
  slo_config.json
  slo_summary.json

  scheduler_LST/
    request_rate_14/
      humaneval/
        dllm_algo_config.yaml
        server.log

        humaneval_<model_tag>.json
        summary_<model_tag>.json
        dlm_step_stats_humaneval.jsonl
        dlm_request_latency_humaneval.jsonl
        dlm_batch_latency_humaneval.jsonl

        slo_rates.json
```

Key files:

| File | Description |
|------|-------------|
| `<task>_<model_tag>.json` | Run summary: accuracy, throughput, p99 TTFB/TPOB, ideal latency |
| `dlm_request_latency_<task>.jsonl` | Per-request latency records; read by `dlm_slorate.py` and scatter plots |
| `dlm_step_stats_<task>.jsonl` | Per-batch-step DLM logs written directly by the server |
| `dlm_batch_latency_<task>.jsonl` | Per-batch latency logs written directly by the server |
| `slo_rates.json` | Strict/relaxed SLO attainment rates for this scheduler/task/rate |
| `slo_config.json` | Per-task strict/relaxed SLO thresholds derived from TTFB/DECODE calibration |
| `slo_summary.json` | All scheduler/task/rate results aggregated; input to the plot script |

## 2. Plot Scheduler Comparison Script

Regenerates plots from an existing `slo_summary.json` and `slo_config.json` without re-running the benchmark. Both wrappers call `plot_dlm_scheduler_comparison.py` directly.

### Run

```bash
./test/plot_dlm_scheduler_comparison_LLADA2.sh --bar-task humaneval --bar-rate 14
./test/plot_dlm_scheduler_comparison_SDAR.sh --bar-task humaneval --bar-rate 14
```

### Key Parameters

| Variable | LLADA2 default | SDAR default | Description |
|----------|---------------|-------------|-------------|
| `OUTPUT_ROOT` | `.../dlm_sched_comparison_LLADA2` | `.../dlm_sched_comparison_SDAR` | Root to read `slo_summary.json` / `slo_config.json` from and save PNGs to |

All other CLI arguments (e.g. `--bar-task`, `--bar-rate`) are passed through via `$@` to `plot_dlm_scheduler_comparison.py`.

### Internal Flow

```text
1. Check that required files exist
   - OUTPUT_ROOT/slo_summary.json  (created by run_dlm_scheduler_comparison_*.sh)
   - OUTPUT_ROOT/slo_config.json   (created by run_dlm_scheduler_comparison_*.sh)

2. Run plot_dlm_scheduler_comparison.py
   - --summary    OUTPUT_ROOT/slo_summary.json
   - --output     OUTPUT_ROOT/slo_attainment_comparison.png
   - --slo-config OUTPUT_ROOT/slo_config.json
   - p99 latency is normalized against the LST scheduler baseline
   - --bar-task and --bar-rate select the task/rate for scatter and bar plots
```

This wrapper does not recompute SLO thresholds, does not run `dlm_slorate.py`, and does not rewrite `slo_summary.json`.

### Output

| File | Description |
|------|-------------|
| `slo_attainment_comparison.png` | Strict/relaxed SLO attainment curves |
| `slo_attainment_comparison_p99_latency.png` | p99 TTFB/TPOB comparison |
| `slo_attainment_comparison_scatter.png` | TTFB vs TPOB scatter per scheduler |
| `slo_attainment_comparison_scatter_combined.png` | Combined scatter across schedulers |
| `slo_attainment_comparison_strict_release.png` | Strict vs relaxed bar for a specific task/rate |
| `slo_attainment_comparison_bar_p99.png` | p99 TTFB/TPOB bar for a specific task/rate |

## 3. Task Spec Script

Measures per-task distributions of input length, steps per block, output block count, and within-batch CoV. Starts a fresh server for each task, runs a short benchmark, then combines all logs into a single PNG using inline Python.

### Run

```bash
./test/run_dlm_task_spec.sh
```

### Key Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `JetLM/SDAR-8B-Chat` | Model path for the server |
| `BLOCK_SIZE` | `32` | Output block size (tokens) |
| `TASKS` | `sharegpt humaneval math gsm8k gpqa mmlu ruler_1k … ruler_1_4k` | Tasks to measure |
| `NUM_EXAMPLES` | `200` | Number of requests per task |
| `REQUEST_RATE` | `200` | Request rate (req/s, effectively saturated) |
| `NUM_THREADS` | `200` | Number of concurrent client threads |
| `WARMUP` | `16` | Number of warmup requests |
| `MAX_RUNNING_REQUESTS` | `32` | Max concurrent requests on the server |
| `OUTPUT_ROOT` | `/mnt/nvme0/kdg6245/dlm_task_spec` | Root directory for all outputs |

### Internal Flow

```text
Delete previous OUTPUT_ROOT
  default OUTPUT_ROOT = /mnt/nvme0/kdg6245/dlm_task_spec

Phase 1 — per-task data collection (iterates over TASKS)
  for TASK in TASKS:
    1. Write dlm_algo_config.yaml
       scheduler_mode=fcfs, threshold=0.85
       request_latency_log_file / step_log_file -> OUTPUT_ROOT/tmp_*.jsonl
    2. Launch sglang.launch_server
       --dllm-algorithm LowConfidence
       --max-running-requests 32
    3. Run dlm_benchmark.py
       --request-rate 200 --num-threads 200 --warmup 16 --num-examples 200
    4. Copy logs
       tmp_request_latency.jsonl -> OUT_DIR/request_latency_<task>.jsonl
       tmp_step_stats.jsonl      -> OUT_DIR/step_stats_<task>.jsonl
    5. Stop server

Phase 2 — combined plot (inline Python)
  request_latency_<task>.jsonl : collect input_len, block_steps_list, output block count
  step_stats_<task>.jsonl      : compute within-batch CoV of block_steps for unmasking requests
  Build 4-row × n-task subplot grid -> save OUTPUT_ROOT/task_spec_<model_slug>.png
  Print summary stats to stdout
    (InputLen mean/std, Steps mean/std/CoV, OutBlk mean/std, WB-CoV mean/std)
```

### Output Structure

```text
/mnt/nvme0/kdg6245/dlm_task_spec/
  server_log.txt
  dlm_algo_config.yaml
  task_spec_<model_slug>.png        <- 4-row combined plot

  <task>/
    <task>_<model_tag>.json
    summary_<model_tag>.json
    request_latency_<task>.jsonl
    step_stats_<task>.jsonl
```

Key files:

| File | Description |
|------|-------------|
| `task_spec_<model_slug>.png` | Row 0: input length / Row 1: steps per block / Row 2: output blocks per request / Row 3: within-batch CoV of block steps (bubble opportunity) |
| `request_latency_<task>.jsonl` | Per-request records containing `input_len`, `block_steps_list`, TTFB/TPOB; primary input for the Phase 2 plot |
| `step_stats_<task>.jsonl` | Per-batch-step records containing `req_modes` and `block_steps`; used for CoV computation |
| `<task>_<model_tag>.json` | Summary of accuracy, throughput, and p99 TTFB/TPOB |

## 4. Bellman Table Update Script

Verifies that the Bellman table update converges to the actual step distribution. Iterates over all REQUEST_RATES × NUM_THREADS_SWEEP × TASKS combinations, starting a fresh server for each task, then generates SLO rates and Bellman convergence plots after all runs finish.

### Run

```bash
./test/run_dlm_tb_update_test.sh
```

### Key Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `JetLM/SDAR-8B-Chat` | Model path for the server |
| `BLOCK_SIZE` | `32` | Output block size (tokens) |
| `REQUEST_RATES` | `200` | Request rate list (space-separated for sweep) |
| `NUM_THREADS_SWEEP` | `200` | Admission window values to sweep (e.g. `50 100 150 200`) |
| `SCHEDULER` | `FCFS` | Scheduler to use (LST / PREFILL / DECODE / FCFS / SOLA / TTFB) |
| `STRICT_MULTIPLIER` | `10.0` | strict SLO = multiplier × ideal latency |
| `RELEASE_MULTIPLIER` | `20.0` | release SLO = multiplier × ideal latency |
| `NUM_EXAMPLES` | `200` | Number of requests per task |
| `WARMUP` | `32` | Number of warmup requests |
| `MAX_RUNNING_REQUESTS` | `32` | Max concurrent requests on the server |
| `OUTPUT_ROOT` | `/mnt/nvme0/kdg6245/dlm_tb_update_test` | Root directory for all outputs |

### Internal Flow

```text
Delete previous OUTPUT_ROOT
  default OUTPUT_ROOT = /mnt/nvme0/kdg6245/dlm_tb_update_test

Phase 1 — benchmark collection (iterates over RATE × THREADS × TASK)
  for RATE in REQUEST_RATES:
    for THREADS in NUM_THREADS_SWEEP:
      for TASK in TASKS:
        1. Compute SLO values from hardcoded ideal latencies
             strict_ttfb  = ideal_ttfb_ms(TASK) × STRICT_MULTIPLIER  / 1000  [s]
             strict_tpob  = ideal_tpob_ms(TASK) × STRICT_MULTIPLIER  / 1000  [s]
             release_ttfb = ideal_ttfb_ms(TASK) × RELEASE_MULTIPLIER / 1000  [s]
             release_tpob = ideal_tpob_ms(TASK) × RELEASE_MULTIPLIER / 1000  [s]
        2. Write dlm_algo_config.yaml
             bellman_log_file: OUT_DIR/bellman_log_<task>.jsonl
             SCHEDULER env var -> scheduler_mode (FCFS -> fcfs, LST -> lst, ...)
        3. Launch sglang.launch_server
             --dllm-algorithm LowConfidence  --max-running-requests 32
        4. Run dlm_benchmark.py (with --log)
        5. Copy STEP_LOG_FILE -> OUT_DIR/step_stats_<task>.jsonl
        6. Stop server

Phase 2 — SLO rate computation
  for TASK:
    dlm_slorate.py --latency-dir <all per-rate OUT_DIRs for this task>
                   --strict-factor STRICT_MULTIPLIER --relaxed-factor RELEASE_MULTIPLIER
                   -> OUTPUT_ROOT/slo_rates_<task>.json   (combined across all rates)
    for RATE:
      dlm_slorate.py --latency-dir OUT_DIR  (single rate)
                     -> OUT_DIR/slo_rates.json

Phase 3 — Bellman convergence plots
  for RATE, TASK:
    plot_bellman_convergence.py
      --log-dir OUT_DIR  --tasks TASK
      --output  OUT_DIR/bellman_convergence.png

Phase 4 — Throughput summary (stdout)
  Reads score and output_throughput_tok_s from <task>_<model_tag>.json and prints a table
```

### Output Structure

```text
/mnt/nvme0/kdg6245/dlm_tb_update_test/
  server_log.txt
  dlm_algo_config.yaml
  slo_rates_<task>.json             <- combined SLO rates across all REQUEST_RATES for this task

  request_rate_<R>/
    <task>/
      <task>_<model_tag>.json
      summary_<model_tag>.json
      request_latency_<task>.jsonl
      batch_latency_<task>.jsonl
      steps_<task>.jsonl
      step_stats_<task>.jsonl
      bellman_log_<task>.jsonl
      slo_rates.json                <- SLO rates for this rate only
      bellman_convergence.png
```

Key files:

| File | Description |
|------|-------------|
| `bellman_log_<task>.jsonl` | Per-step Bellman table update log written by the server; input to `plot_bellman_convergence.py` |
| `step_stats_<task>.jsonl` | Per-batch-step log containing `req_modes` and `block_steps` |
| `slo_rates.json` | Strict/relaxed SLO attainment rates |
| `bellman_convergence.png` | Visualization of Bellman table estimates converging to the actual step distribution |
