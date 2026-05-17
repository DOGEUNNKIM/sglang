# DLM Scheduler Script Guide

이 문서는 DLM 관련 Run 및 Plot Script의 관계를 정리한다. 

```text
run_*.sh
  -> sglang.launch_server를 띄움
  -> dlm_benchmark.py로 실제 request를 보냄
  -> dlm_slorate.py로 SLO 달성률을 계산
  -> 필요하면 finalize/plot 스크립트로 결과를 다시 정리
```

## 파일 역할 요약

| 파일 | 역할 |
|------|------|
| `run_dlm_scheduler_comparison_LLADA2.sh` | LLaDA2.0-mini scheduler 비교 실험 |
| `run_dlm_scheduler_comparison_SDAR.sh` | SDAR-8B-Chat scheduler 비교 실험 |
| `plot_dlm_scheduler_comparison_LLADA2.sh` | LLADA2 결과 plot wrapper |
| `plot_dlm_scheduler_comparison_SDAR.sh` | SDAR 결과 plot wrapper |
| `run_dlm_task_spec.sh` | task별 입력 길이, output block, step 분포 분석 |
| `run_dlm_tb_update_test.sh` | Bellman table/TB update 동작 검증 |
| `dlm_benchmark.py` | 실제 benchmark 실행 엔진. 서버에 요청을 보내고 per-task 결과를 저장 |
| `dlm_slorate.py` | `request_latency_<task>.jsonl`에서 SLO 달성률 계산 |
| `plot_dlm_scheduler_comparison.py` | `slo_summary.json` 기반 scheduler 비교 plot 생성 |
| `plot_bellman_convergence.py` | `bellman_log_<task>.jsonl` 수렴 plot 생성 |

## Script와 Python File

### Script

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
  -> plot code for task_spec_<model>.png

run_dlm_tb_update_test.sh
  -> python -m sglang.launch_server
  -> test/dlm_benchmark.py
  -> test/dlm_slorate.py
  -> test/plot_bellman_convergence.py
```

### Python File

```text
dlm_benchmark.py
  - 이미 떠 있는 SGLang server에 request를 보내는 benchmark runner
  - 결과:
    <output-dir>/<task>_<model_tag>.json
    <output-dir>/summary_<model_tag>.json
    <output-dir>/request_latency_<task>.jsonl
    <output-dir>/batch_latency_<task>.jsonl
    <output-dir>/steps_<task>.jsonl

dlm_slorate.py
  - request_latency_<task>.jsonl에서 SLO 달성률 계산
  - 결과:
    <latency-dir>/slo_rates.json

plot_dlm_scheduler_comparison.py
  - slo_summary.json에서 scheduler 비교 plot 생성
  - 결과:
    slo_attainment_comparison.png
    slo_attainment_comparison_p99_latency.png
    slo_attainment_comparison_scatter.png
    slo_attainment_comparison_scatter_combined.png
    slo_attainment_comparison_strict_release.png
    slo_attainment_comparison_bar_p99.png

plot_bellman_convergence.py
  - bellman_log_<task>.jsonl에서 Bellman table 수렴 plot 생성
  - 결과:
    사용자가 --output으로 지정한 PNG
```

## 1. Run scheduler_comparison Script

### 실행

```bash
./test/run_dlm_scheduler_comparison_LLADA2.sh
./test/run_dlm_scheduler_comparison_SDAR.sh
```

### 주요 파라미터

두 스크립트는 모델 관련 기본값만 다르고 나머지는 동일하다.

| 변수 | LLADA2 기본값 | SDAR 기본값 | 설명 |
|------|--------------|------------|------|
| `MODEL_PATH` | `inclusionAI/LLaDA2.0-mini` | `JetLM/SDAR-8B-Chat` | 서버 모델 경로 |
| `FORWARD_TIME_S` | `0.04` | `0.08` | 한 번의 forward pass 예상 소요 시간 (s) |
| `OUTPUT_ROOT` | `.../dlm_sched_comparison_LLADA2` | `.../dlm_sched_comparison_SDAR` | 결과 저장 루트 |
| `BLOCK_SIZE` | `32` | `32` | output block 크기 (토큰 수) |
| `TASKS` | `humaneval math gsm8k gpqa mmlu sharegpt ruler_4k` | 동일 | 비교 대상 태스크 목록 |
| `RATES_<TASK>` | 태스크별 4개 값 (예: humaneval `10 12 14 16`) | 태스크별 4개 값 (예: humaneval `8 10 12 14`) | 태스크별 request rate sweep 값 |
| `NUM_EXAMPLES_<TASK>` | `200` | `200` | 태스크당 request 수 |
| `SCHEDULERS` | `TTFB DECODE LST SOLA FCFS PREFILL` | 동일 | 비교할 scheduler 목록 |
| `STRICT_MULTIPLIER` | `10.0` | `10.0` | strict SLO = multiplier × ideal latency |
| `RELEASE_MULTIPLIER` | `20.0` | `20.0` | release SLO = multiplier × ideal latency |
| `MAX_RUNNING_REQUESTS` | `32` | `32` | 서버 최대 동시 request 수 |
| `WARMUP` | `32` | `32` | warmup request 수 |
| `TP_SIZE` | `1` | `1` | tensor parallelism 크기 |

### 내부 흐름

```text
이전 실행 OUTPUT_ROOT를 삭제
  LLADA2 기본 OUTPUT_ROOT = /mnt/nvme0/kdg6245/dlm_sched_comparison_LLADA2
  SDAR   기본 OUTPUT_ROOT = /mnt/nvme0/kdg6245/dlm_sched_comparison_SDAR

for scheduler in SCHEDULERS:
  for task in TASKS:
    for rate in task별 request rates:
      1. run별 OUT_DIR 생성
      2. run별 dllm config 작성
      3. sglang.launch_server 실행
      4. dlm_benchmark.py 실행
      5. 서버 종료

  if scheduler == TTFB:
    task별 highest-rate 결과에서 ideal TTFB 추출

  if scheduler == DECODE:
    task별 highest-rate 결과에서 ideal TPOB 추출

for scheduler loop가 모두 끝난 후:
  1. TTFB/DECODE에서 추출한 ideal 값으로 slo_config.json 생성
  2. 각 run에 대해 dlm_slorate.py 실행 -> slo_rates.json
  3. 전체 결과를 모아 slo_summary.json 생성
```

### Run별 출력 구조

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

중요한 파일:

| 파일 | 설명 |
|------|------|
| `<task>_<model_tag>.json` | accuracy, throughput, p99 TTFB/TPOB, ideal latency 등 run 요약 |
| `request_latency_<task>.jsonl` | `dlm_slorate.py`와 scatter plot이 읽는 request-level latency |
| `dlm_*_<task>.jsonl` | 서버가 직접 쓴 raw DLM 로그 |
| `slo_rates.json` | 해당 scheduler/task/rate의 strict/relaxed SLO 달성률 |
| `slo_config.json` | task별 strict/relaxed SLO threshold |
| `slo_summary.json` | 모든 scheduler/task/rate 결과를 모은 plot 입력 |

## 2. Plot dlm scheduler comparison Script

Scheduler comparison Script가 만든 `slo_summary.json`과 `slo_config.json`으로 결과 plot을 다시 생성할 때 사용한다. 두 wrapper는 내부적으로 `plot_dlm_scheduler_comparison.py`를 바로 호출한다.

### 실행
```bash
./test/plot_dlm_scheduler_comparison_LLADA2.sh --bar-task humaneval --bar-rate 14
./test/plot_dlm_scheduler_comparison_SDAR.sh --bar-task humaneval --bar-rate 14
```

### 주요 파라미터

| 변수 | LLADA2 기본값 | SDAR 기본값 | 설명 |
|------|--------------|------------|------|
| `OUTPUT_ROOT` | `.../dlm_sched_comparison_LLADA2` | `.../dlm_sched_comparison_SDAR` | `slo_summary.json`과 `slo_config.json`을 읽고 PNG를 저장하는 루트 |

CLI 인자(`--bar-task`, `--bar-rate` 등)는 `$@`로 `plot_dlm_scheduler_comparison.py`에 그대로 전달된다.

### 내부 흐름

```text
1. 기존 결과 파일 확인
   - OUTPUT_ROOT/slo_summary.json이 있어야 한다.
   - OUTPUT_ROOT/slo_config.json이 있어야 한다.
   - 이 두 파일은 run_dlm_scheduler_comparison_*.sh가 정상 완료되면 생성된다.

2. plot_dlm_scheduler_comparison.py 실행
   - --summary OUTPUT_ROOT/slo_summary.json
   - --output OUTPUT_ROOT/slo_attainment_comparison.png
   - --slo-config OUTPUT_ROOT/slo_config.json
   - p99 latency는 LST scheduler를 baseline으로 normalize한다.
   - --bar-task, --bar-rate 값은 scatter/bar plot 대상 task/rate로 plot_dlm_scheduler_comparison.py에 전달된다.
```

### 출력 구조

| 파일 | 설명 |
|------|------|
| `slo_attainment_comparison.png` | strict/relaxed SLO 달성률 curve |
| `slo_attainment_comparison_p99_latency.png` | p99 TTFB/TPOB 비교 |
| `slo_attainment_comparison_scatter.png` | scheduler별 TTFB vs TPOB scatter |
| `slo_attainment_comparison_scatter_combined.png` | scheduler 통합 scatter |
| `slo_attainment_comparison_strict_release.png` | 특정 task/rate의 strict vs relaxed bar |
| `slo_attainment_comparison_bar_p99.png` | 특정 task/rate의 p99 TTFB/TPOB bar |


## 3. Task Spec Script

태스크별 입력 길이·steps/block·output block 수·within-batch CoV 분포를 측정한다. 태스크마다 서버를 새로 띄우고 짧은 벤치마크를 돌린 뒤, 수집된 로그를 inline Python으로 한 장의 PNG에 합친다.

### 실행

```bash
./test/run_dlm_task_spec.sh
```

### 주요 파라미터

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_PATH` | `JetLM/SDAR-8B-Chat` | 서버 모델 경로 |
| `BLOCK_SIZE` | `32` | output block 크기 (토큰 수) |
| `TASKS` | `sharegpt humaneval math gsm8k gpqa mmlu ruler_1k … ruler_1_4k` | 측정 태스크 목록 |
| `NUM_EXAMPLES` | `200` | 태스크당 request 수 |
| `REQUEST_RATE` | `200` | 요청 속도 (req/s, 사실상 포화) |
| `NUM_THREADS` | `200` | 클라이언트 동시 스레드 수 |
| `WARMUP` | `16` | warmup request 수 |
| `MAX_RUNNING_REQUESTS` | `32` | 서버 최대 동시 request 수 |
| `OUTPUT_ROOT` | `/mnt/nvme0/kdg6245/dlm_task_spec` | 결과 저장 루트 |

### 내부 흐름

```text
이전 실행 OUTPUT_ROOT를 삭제
  기본 OUTPUT_ROOT = /mnt/nvme0/kdg6245/dlm_task_spec

Phase 1 — per-task 데이터 수집 (TASKS 순으로 반복)
  for TASK in TASKS:
    1. dlm_algo_config.yaml 작성
       scheduler_mode=fcfs, threshold=0.85
       request_latency_log_file / step_log_file -> OUTPUT_ROOT/tmp_*.jsonl
    2. sglang.launch_server 실행
       --dllm-algorithm LowConfidence
       --max-running-requests 32
    3. dlm_benchmark.py 실행
       --request-rate 200 --num-threads 200 --warmup 16 --num-examples 200
    4. 로그 복사
       tmp_request_latency.jsonl -> OUT_DIR/request_latency_<task>.jsonl
       tmp_step_stats.jsonl      -> OUT_DIR/step_stats_<task>.jsonl
    5. 서버 종료

Phase 2 — combined plot (inline Python)
  request_latency_<task>.jsonl  에서: input_len, block_steps_list, output block 수 집계
  step_stats_<task>.jsonl       에서: unmasking 중인 요청들의 block_steps CoV 계산
  4행 × n태스크 subplot 생성 -> OUTPUT_ROOT/task_spec_<model_slug>.png 저장
  summary 통계 테이블 stdout 출력
    (InputLen mean/std, Steps mean/std/CoV, OutBlk mean/std, WB-CoV mean/std)
```

### 출력 구조

```text
/mnt/nvme0/kdg6245/dlm_task_spec/
  server_log.txt
  dlm_algo_config.yaml
  task_spec_<model_slug>.png        ← 4행 combined plot

  <task>/
    <task>_<model_tag>.json
    summary_<model_tag>.json
    request_latency_<task>.jsonl
    step_stats_<task>.jsonl
```

중요한 파일:

| 파일 | 설명 |
|------|------|
| `task_spec_<model_slug>.png` | Row 0: input length 분포 / Row 1: steps/block 분포 / Row 2: output blocks/request 분포 / Row 3: within-batch CoV of block_steps (bubble 기회 지표) |
| `request_latency_<task>.jsonl` | per-request 레코드. `input_len`, `block_steps_list`, TTFB/TPOB 포함. Phase 2 plot의 주 입력 |
| `step_stats_<task>.jsonl` | per-batch-step 레코드. `req_modes`, `block_steps` 포함. CoV 계산에 사용 |
| `<task>_<model_tag>.json` | accuracy, throughput, p99 TTFB/TPOB 요약 |

## 4. Bellman Table Update Script

Bellman table update가 실제 step 분포에 수렴하는지 확인한다. REQUEST_RATES × NUM_THREADS_SWEEP × TASKS 조합을 순회하며 태스크마다 서버를 새로 띄우고, 완료 후 SLO rate·Bellman 수렴 plot·step 분포 plot을 순서대로 생성한다.

### 실행

```bash
./test/run_dlm_tb_update_test.sh
```

### 주요 파라미터

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_PATH` | `JetLM/SDAR-8B-Chat` | 서버 모델 경로 |
| `BLOCK_SIZE` | `32` | output block 크기 (토큰 수) |
| `REQUEST_RATES` | `200` | 요청 속도 리스트 (공백 구분으로 sweep 가능) |
| `NUM_THREADS_SWEEP` | `200` | admission window 값 리스트 (예: `50 100 150 200`) |
| `SCHEDULER` | `FCFS` | scheduler 종류 (LST / PREFILL / DECODE / FCFS / SOLA / TTFB) |
| `STRICT_MULTIPLIER` | `10.0` | strict SLO = multiplier × ideal latency |
| `RELEASE_MULTIPLIER` | `20.0` | release SLO = multiplier × ideal latency |
| `NUM_EXAMPLES` | `200` | 태스크당 request 수 |
| `WARMUP` | `32` | warmup request 수 |
| `MAX_RUNNING_REQUESTS` | `32` | 서버 최대 동시 request 수 |
| `OUTPUT_ROOT` | `/mnt/nvme0/kdg6245/dlm_tb_update_test` | 결과 저장 루트 |

### 내부 흐름

```text
이전 실행 OUTPUT_ROOT를 삭제
  기본 OUTPUT_ROOT = /mnt/nvme0/kdg6245/dlm_tb_update_test

Phase 1 — Benchmark 수집 (RATE × THREADS × TASK 순으로 반복)
  for RATE in REQUEST_RATES:
    for THREADS in NUM_THREADS_SWEEP:
      for TASK in TASKS:
        1. SLO 값 계산 (하드코딩된 ideal latency 기반)
             strict_ttfb  = ideal_ttfb_ms(TASK)  × STRICT_MULTIPLIER  / 1000  [s]
             strict_tpob  = ideal_tpob_ms(TASK)  × STRICT_MULTIPLIER  / 1000  [s]
             release_ttfb = ideal_ttfb_ms(TASK)  × RELEASE_MULTIPLIER / 1000  [s]
             release_tpob = ideal_tpob_ms(TASK)  × RELEASE_MULTIPLIER / 1000  [s]
        2. dlm_algo_config.yaml 작성
             bellman_log_file: OUT_DIR/bellman_log_<task>.jsonl 포함
             SCHEDULER env var -> scheduler_mode (FCFS -> fcfs, LST -> lst, …)
        3. sglang.launch_server 실행
             --dllm-algorithm LowConfidence  --max-running-requests 32
        4. dlm_benchmark.py 실행 (--log 포함)
        5. STEP_LOG_FILE 복사 -> OUT_DIR/step_stats_<task>.jsonl
        6. 서버 종료

Phase 2 — SLO rate 계산
  for RATE, TASK:
    dlm_slorate.py --latency-dir OUT_DIR -> OUT_DIR/slo_rates.json

Phase 3 — Bellman 수렴 plot
  for RATE, TASK:
    plot_bellman_convergence.py
      --log-dir OUT_DIR  --tasks TASK
      --output OUT_DIR/bellman_convergence.png

Phase 4 — Throughput summary (stdout)
  <task>_<model_tag>.json에서 score, output_throughput_tok_s 집계 후 출력
```

### 출력 구조

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

중요한 파일:

| 파일 | 설명 |
|------|------|
| `bellman_log_<task>.jsonl` | 서버가 매 step마다 기록하는 Bellman table 업데이트 로그. `plot_bellman_convergence.py`의 입력 |
| `step_stats_<task>.jsonl` | per-batch-step 로그. `req_modes`, `block_steps` 포함 |
| `slo_rates.json` | strict / relaxed SLO 달성률 |
| `bellman_convergence.png` | Bellman table 추정값이 실제 step 분포에 수렴하는 과정 시각화 |