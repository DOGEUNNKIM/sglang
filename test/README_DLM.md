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

### Key Parameters

The two scripts differ only in model-specific defaults.

| Variable | LLADA2 default | SDAR default | Description |
|----------|---------------|-------------|-------------|
| `MODEL_PATH` | `inclusionAI/LLaDA2.0-mini` | `JetLM/SDAR-8B-Chat` | Model path for the server |
| `FORWARD_TIME_S` | `0.04` | `0.08` | Expected time per forward pass (s) |
| `OUTPUT_ROOT` | `.../dlm_sched_comparison_LLADA2` | `.../dlm_sched_comparison_SDAR` | Root directory for all outputs |
| `BLOCK_SIZE` | `32` | `32` | Output block size (tokens) |
| `TASKS` | `humaneval math gsm8k gpqa mmlu sharegpt ruler_4k` | same | Tasks to benchmark |
| `RATES_<TASK>` | 4 values per task (e.g. humaneval `10 12 14 16`) | 4 values per task (e.g. humaneval `8 10 12 14`) | Request rate sweep values per task |
| `NUM_EXAMPLES_<TASK>` | `200` | `200` | Number of requests per task |
| `SCHEDULERS` | `TTFB DECODE LST SOLA FCFS PREFILL` | same | Schedulers to compare |
| `STRICT_MULTIPLIER` | `10.0` | `10.0` | strict SLO = multiplier × ideal latency |
| `RELEASE_MULTIPLIER` | `20.0` | `20.0` | release SLO = multiplier × ideal latency |
| `MAX_RUNNING_REQUESTS` | `32` | `32` | Max concurrent requests on the server |
| `WARMUP` | `32` | `32` | Number of warmup requests |
| `TP_SIZE` | `1` | `1` | Tensor parallelism size |

### Internal Flow

```text
Delete previous OUTPUT_ROOT
  LLADA2 default OUTPUT_ROOT = /mnt/nvme0/kdg6245/dlm_sched_comparison_LLADA2
  SDAR   default OUTPUT_ROOT = /mnt/nvme0/kdg6245/dlm_sched_comparison_SDAR

for scheduler in SCHEDULERS:
  for task in TASKS:
    for rate in per-task request rates:
      1. Create OUT_DIR for this run
      2. Write dlm_algo_config.yaml for this run
      3. Launch sglang.launch_server
      4. Run dlm_benchmark.py
      5. Stop server

  if scheduler == TTFB:
    Extract ideal TTFB from the highest-rate result for each task

  if scheduler == DECODE:
    Extract ideal TPOB from the highest-rate result for each task

After all schedulers finish:
  1. Build slo_config.json from the extracted ideal TTFB/TPOB values
  2. Run dlm_slorate.py for each run -> slo_rates.json
  3. Aggregate all results into slo_summary.json
```

### Output Structure

```text
OUTPUT_ROOT/
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
        request_latency_humaneval.jsonl
        batch_latency_humaneval.jsonl
        steps_humaneval.jsonl
        dlm_step_stats_humaneval.jsonl
        dlm_request_latency_humaneval.jsonl
        dlm_batch_latency_humaneval.jsonl

        slo_rates.json
```

Key files:

| File | Description |
|------|-------------|
| `<task>_<model_tag>.json` | Run summary: accuracy, throughput, p99 TTFB/TPOB, ideal latency |
| `request_latency_<task>.jsonl` | Request-level latency records; read by `dlm_slorate.py` and scatter plots |
| `dlm_*_<task>.jsonl` | Raw DLM logs written directly by the server |
| `slo_rates.json` | Strict/relaxed SLO attainment rates for this scheduler/task/rate |
| `slo_config.json` | Per-task strict/relaxed SLO thresholds |
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
  for RATE, TASK:
    dlm_slorate.py --latency-dir OUT_DIR -> OUT_DIR/slo_rates.json

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

  request_rate_<R>/
    <task>/
      <task>_<model_tag>.json
      summary_<model_tag>.json
      request_latency_<task>.jsonl
      batch_latency_<task>.jsonl
      steps_<task>.jsonl
      step_stats_<task>.jsonl
      bellman_log_<task>.jsonl
      slo_rates.json
      bellman_convergence.png
```

Key files:

| File | Description |
|------|-------------|
| `bellman_log_<task>.jsonl` | Per-step Bellman table update log written by the server; input to `plot_bellman_convergence.py` |
| `step_stats_<task>.jsonl` | Per-batch-step log containing `req_modes` and `block_steps` |
| `slo_rates.json` | Strict/relaxed SLO attainment rates |
| `bellman_convergence.png` | Visualization of Bellman table estimates converging to the actual step distribution |
