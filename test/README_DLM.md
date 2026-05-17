# DLM Benchmark & Scheduler Comparison Guide

이 문서는 DLM 관련 실행 스크립트와 후처리 스크립트의 관계를 정리한다. 

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
| `finalize_dlm_scheduler_comparison_LLADA2.sh` | LLADA2 결과 후처리 wrapper |
| `finalize_dlm_scheduler_comparison_SDAR.sh` | SDAR 결과 후처리 wrapper |
| `run_dlm_task_spec.sh` | task별 입력 길이, output block, step 분포 분석 |
| `run_dlm_tb_update_test.sh` | Bellman table/TB update 동작 검증 |
| `dlm_benchmark.py` | 실제 benchmark 실행 엔진. 서버에 요청을 보내고 per-task 결과를 저장 |
| `dlm_slorate.py` | `request_latency_<task>.jsonl`에서 SLO 달성률 계산 |
| `finalize_dlm_scheduler_comparison.py` | 끝난 scheduler 비교 결과를 다시 스캔해서 `slo_config.json`, `slo_summary.json`, plot 재생성 |
| `plot_dlm_slo_summary.py` | `slo_summary.json` 기반 scheduler 비교 plot 생성 |
| `plot_bellman_convergence.py` | `bellman_log_<task>.jsonl` 수렴 plot 생성 |
| `plot_step_dist.py` | `step_stats_<task>.jsonl` 기반 step distribution plot |
| `dlm_plot_steps.py` | standalone step plot utility. `summary_*.json` 또는 `steps_*.jsonl` 사용 |

## 스크립트 호출 관계

자동으로 호출되는 관계와 사람이 수동으로 실행하는 관계를 분리해서 보면 다음과 같다.

### 자동 호출 관계

```text
run_dlm_scheduler_comparison_LLADA2.sh
  -> python -m sglang.launch_server
  -> test/dlm_benchmark.py
  -> test/dlm_slorate.py

run_dlm_scheduler_comparison_SDAR.sh
  -> python -m sglang.launch_server
  -> test/dlm_benchmark.py
  -> test/dlm_slorate.py

finalize_dlm_scheduler_comparison.py
  -> test/dlm_slorate.py
  -> test/plot_dlm_slo_summary.py
  -> test/plot_step_dist.py          (--step-plots 옵션을 줄 때만)

finalize_dlm_scheduler_comparison_LLADA2.sh
  -> test/finalize_dlm_scheduler_comparison.py

finalize_dlm_scheduler_comparison_SDAR.sh
  -> test/finalize_dlm_scheduler_comparison.py

run_dlm_task_spec.sh
  -> python -m sglang.launch_server
  -> test/dlm_benchmark.py
  -> inline Python plot code         (task_spec_<model>.png 생성)

run_dlm_tb_update_test.sh
  -> python -m sglang.launch_server
  -> test/dlm_benchmark.py
  -> test/dlm_slorate.py
  -> test/plot_bellman_convergence.py
  -> test/plot_step_dist.py
```

### 수동 실행용 스크립트

아래 파일들은 보통 다른 스크립트가 호출하거나, 결과를 다시 보고 싶을 때 직접 실행한다.

```text
dlm_benchmark.py
  - 이미 떠 있는 SGLang server에 request를 보내는 benchmark runner
  - run_dlm_*.sh들이 주로 호출

dlm_slorate.py
  - request_latency_<task>.jsonl에서 SLO 달성률 계산
  - run_dlm_scheduler_comparison_*.sh, run_dlm_tb_update_test.sh, finalize가 호출

plot_dlm_slo_summary.py
  - slo_summary.json에서 scheduler 비교 plot 생성
  - finalize가 호출하거나 사용자가 직접 실행

plot_step_dist.py
  - step_stats_<task>.jsonl에서 step histogram 생성
  - run_dlm_tb_update_test.sh와 finalize --step-plots가 호출

plot_bellman_convergence.py
  - bellman_log_<task>.jsonl에서 Bellman table 수렴 plot 생성
  - run_dlm_tb_update_test.sh가 호출

dlm_plot_steps.py
  - summary_*.json 또는 steps_<task>.jsonl 기반 standalone step plotter
  - 현재 메인 scheduler comparison 흐름에서는 자동 호출되지 않음
```

## 1. Scheduler 비교 실험

### 실행

```bash
./test/run_dlm_scheduler_comparison_LLADA2.sh
./test/run_dlm_scheduler_comparison_SDAR.sh
```

두 스크립트는 시작할 때 기존 `OUTPUT_ROOT`를 삭제하고 새로 시작한다.

```text
LLADA2 기본 OUTPUT_ROOT = /mnt/nvme0/kdg6245/dlm_sched_comparison_LLADA2
SDAR   기본 OUTPUT_ROOT = /mnt/nvme0/kdg6245/dlm_sched_comparison_SDAR
```

### 내부 흐름

```text
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

예시:

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
| `slo_rates.json` | 해당 scheduler/task/rate의 strict/relaxed SLO 달성률 |
| `slo_config.json` | task별 strict/relaxed SLO threshold |
| `slo_summary.json` | 모든 scheduler/task/rate 결과를 모은 plot 입력 |
| `server.log` | 해당 run의 server stdout/stderr |
| `dlm_*_<task>.jsonl` | 서버가 직접 쓴 raw DLM 로그 |

## 2. 후처리와 Plot 재생성

Scheduler comparison 스크립트는 `slo_summary.json`까지 만든다. 이미 끝난 결과를 다시 정리하거나 plot만 다시 만들고 싶으면 `finalize_dlm_scheduler_comparison.py`를 사용한다.

### LLADA2

```bash
./test/finalize_dlm_scheduler_comparison_LLADA2.sh
```

### SDAR

```bash
./test/finalize_dlm_scheduler_comparison_SDAR.sh
```

추가 옵션은 wrapper 뒤에 그대로 붙이면 된다.

```bash
./test/finalize_dlm_scheduler_comparison_LLADA2.sh --step-plots
./test/finalize_dlm_scheduler_comparison_SDAR.sh --skip-summary-plots
```

기본 경로를 바꾸고 싶으면 환경변수로 지정한다.

```bash
OUTPUT_ROOT=/path/to/results ./test/finalize_dlm_scheduler_comparison_LLADA2.sh
MODEL_PATH=custom/model ./test/finalize_dlm_scheduler_comparison_SDAR.sh
```

`finalize_dlm_scheduler_comparison.py`가 하는 일:

```text
1. scheduler_*/request_rate_*/*/<task>_<model>.json 탐색
2. TTFB/DECODE 결과에서 ideal TTFB/TPOB 추출
3. slo_config.json 재생성
4. request_latency_<task>.jsonl이 있으면 dlm_slorate.py 재실행
5. slo_summary.json 재생성
6. plot_dlm_slo_summary.py 실행
```

기본 finalize는 summary plot과 p99 plot만 만든다. 현재 호출 옵션에 `--no-scatter --no-bar`가 들어가므로 scatter/bar는 만들지 않는다.

생성되는 기본 plot:

```text
slo_attainment_comparison.png
slo_attainment_comparison_p99_latency.png
```

## 3. plot_dlm_slo_summary.py 단독 실행

`slo_summary.json`이 있을 때 plot만 다시 그리고 싶으면 직접 실행한다.

```bash
python ./test/plot_dlm_slo_summary.py \
  --summary /mnt/nvme0/kdg6245/dlm_sched_comparison_LLADA2/slo_summary.json \
  --output /mnt/nvme0/kdg6245/dlm_sched_comparison_LLADA2/slo_attainment_comparison.png \
  --slo-config /mnt/nvme0/kdg6245/dlm_sched_comparison_LLADA2/slo_config.json \
  --p99-normalize-baseline LST \
  --bar-task humaneval \
  --bar-rate 14
```

출력:

| 파일 | 설명 |
|------|------|
| `slo_attainment_comparison.png` | strict/relaxed SLO 달성률 curve |
| `slo_attainment_comparison_p99_latency.png` | p99 TTFB/TPOB 비교 |
| `slo_attainment_comparison_scatter.png` | scheduler별 TTFB vs TPOB scatter |
| `slo_attainment_comparison_scatter_combined.png` | scheduler 통합 scatter |
| `slo_attainment_comparison_strict_release.png` | 특정 task/rate의 strict vs relaxed bar |
| `slo_attainment_comparison_bar_p99.png` | 특정 task/rate의 p99 TTFB/TPOB bar |

Scatter plot은 다음 파일이 있어야 생성된다.

```text
OUTPUT_ROOT/scheduler_<S>/request_rate_<R>/<TASK>/request_latency_<TASK>.jsonl
```

## 4. dlm_slorate.py 단독 실행

특정 run의 SLO 달성률만 다시 계산할 때 사용한다.

```bash
python ./test/dlm_slorate.py \
  --latency-dir /mnt/nvme0/kdg6245/dlm_sched_comparison_LLADA2/scheduler_LST/request_rate_14/humaneval \
  --tasks humaneval \
  --slo-config /mnt/nvme0/kdg6245/dlm_sched_comparison_LLADA2/slo_config.json \
  --output-json /mnt/nvme0/kdg6245/dlm_sched_comparison_LLADA2/scheduler_LST/request_rate_14/humaneval/slo_rates.json
```

입력:

```text
request_latency_<task>.jsonl
slo_config.json
```

출력:

```text
slo_rates.json
```

## 5. dlm_benchmark.py

`dlm_benchmark.py`는 실제 workload runner다. 서버를 직접 띄우지 않고, 이미 떠 있는 SGLang server에 요청을 보낸다.

보통 shell script에서 다음 형태로 호출된다.

```bash
python test/dlm_benchmark.py \
  --base-url http://localhost:30000 \
  --model inclusionAI/LLaDA2.0-mini \
  --tasks humaneval \
  --request-rate 14 \
  --num-threads 200 \
  --log \
  --output-dir <OUT_DIR>
```

`--log`가 있으면 DLM server log를 읽어서 다음 파일들을 만든다.

```text
<OUT_DIR>/<task>_<model_tag>.json
<OUT_DIR>/summary_<model_tag>.json
<OUT_DIR>/request_latency_<task>.jsonl
<OUT_DIR>/batch_latency_<task>.jsonl
<OUT_DIR>/steps_<task>.jsonl
```

## 6. Task 특성 분석

```bash
./test/run_dlm_task_spec.sh
```

목적은 scheduler 비교가 아니라 task 자체의 특성을 보는 것이다.

출력 기본 위치:

```text
/mnt/nvme0/kdg6245/dlm_task_spec/
```

구조:

```text
dlm_task_spec/
  server_log.txt
  task_spec_<model_tag>.png

  <task>/
    <task>_<model_tag>.json
    summary_<model_tag>.json
    request_latency_<task>.jsonl
    step_stats_<task>.jsonl
```

`task_spec_<model_tag>.png`에는 다음 항목이 같이 들어간다.

```text
input length distribution
steps per block distribution
output blocks per request distribution
within-batch CoV of block steps
```

## 7. Bellman Table / TB Update 테스트

```bash
./test/run_dlm_tb_update_test.sh
```

목적은 Bellman table update가 실제 step 분포에 수렴하는지 확인하는 것이다.

기본 출력 위치:

```text
/mnt/nvme0/kdg6245/dlm_tb_update_test/
```

구조:

```text
dlm_tb_update_test/
  server_log.txt

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
      step_dist_<model_tag>.png
```

내부 호출:

```text
sglang.launch_server
dlm_benchmark.py
dlm_slorate.py
plot_bellman_convergence.py
plot_step_dist.py
```

## 8. Step Plot 스크립트

### plot_step_dist.py

`step_stats_<task>.jsonl`을 읽어서 단순 histogram을 만든다.

```bash
python ./test/plot_step_dist.py \
  --log-dir /mnt/nvme0/kdg6245/dlm_tb_update_test/request_rate_200/humaneval \
  --tasks humaneval \
  --output step_dist.png
```

### dlm_plot_steps.py

`summary_*.json` 또는 `steps_<task>.jsonl` 기반 standalone plotter다.

```bash
python ./test/dlm_plot_steps.py \
  --summary /tmp/dlm_results/summary_inclusionAI_LLaDA2.0-mini.json \
  --block-size 32 \
  --output-dir /tmp/dlm_results
```

출력:

```text
step_dist_<model_tag>.png
step_boxplot_<model_tag>.png
```

## 자주 헷갈리는 점

### `request_latency_<task>.jsonl`과 `dlm_request_latency_<task>.jsonl`

`dlm_request_latency_<task>.jsonl`은 서버가 직접 쓰는 raw log다.  
`request_latency_<task>.jsonl`은 `dlm_benchmark.py`가 읽어서 후처리용으로 저장한 표준 파일이다.

대부분의 후처리 스크립트는 `request_latency_<task>.jsonl`을 읽는다.

## JSONL 파일 예시

JSONL은 한 줄에 JSON object 하나가 들어가는 형식이다. 즉 파일 전체가 하나의 JSON 배열이 아니라, 아래처럼 줄 단위로 읽는다.

```jsonl
{"request_id": 0, "ttfb_ms": 1240.5, "tpob_ms": 315.2}
{"request_id": 1, "ttfb_ms": 1522.8, "tpob_ms": 401.7}
```

### `request_latency_<task>.jsonl`

`dlm_benchmark.py`가 저장하는 request-level latency 파일이다. `dlm_slorate.py`와 `plot_dlm_slo_summary.py` scatter plot이 주로 읽는다.

대표 형태:

```jsonl
{"request_id": 0, "task": "humaneval", "input_len": 287, "output_len": 96, "ttfb_ms": 3182.4, "tpob_ms": 412.8, "latency_ms": 19840.1, "slo_type": "strict", "block_steps_list": [4, 3, 5, 2]}
{"request_id": 1, "task": "humaneval", "input_len": 311, "output_len": 80, "ttfb_ms": 2740.9, "tpob_ms": 366.5, "latency_ms": 16602.7, "slo_type": "release", "block_steps_list": [3, 3, 4]}
```

자주 쓰는 필드:

| 필드 | 의미 |
|------|------|
| `ttfb_ms` | first block/token까지 걸린 시간 |
| `tpob_ms` | output block 사이 평균 시간 |
| `input_len` | 입력 token 길이 |
| `output_len` | 출력 token 길이 |
| `slo_type` | strict/release 요청 구분 |
| `block_steps_list` | output block별 unmask step 수 |

### `batch_latency_<task>.jsonl`

batch/phase 단위 latency 기록이다. request-level SLO 계산보다는 batch phase 분석에 가깝다.

대표 형태:

```jsonl
{"batch_id": 12, "phase": "initial_prefill", "duration_ms": 95.4, "batch_size": 8, "num_output_blocks": 0}
{"batch_id": 13, "phase": "decode", "duration_ms": 42.1, "batch_size": 16, "num_output_blocks": 16}
```

### `steps_<task>.jsonl`

`dlm_benchmark.py`가 서버 step log를 정규화해서 저장한 파일이다. `dlm_plot_steps.py` 같은 standalone step plotter가 읽을 수 있다.

대표 형태:

```jsonl
{"raw_forward_calls": 1, "block_steps": []}
{"raw_forward_calls": 4, "block_steps": [4, 3, 5]}
```

### `dlm_step_stats_<task>.jsonl` 또는 `step_stats_<task>.jsonl`

서버가 직접 쓰거나 task-spec/TB 스크립트가 복사해 둔 step-level raw 기록이다. `plot_step_dist.py`는 `step_stats_<task>.jsonl` 이름을 기본으로 찾는다.

대표 형태:

```jsonl
{"raw_forward_calls": 1, "forward_duration_ms": 10.2, "unmask_steps": 0, "block_steps": [0], "final_block_steps": [], "req_modes": ["prefill"], "kv_saved": [false]}
{"raw_forward_calls": 4, "forward_duration_ms": 38.7, "unmask_steps": 3, "block_steps": [3, 4, 2], "final_block_steps": [3], "req_modes": ["unmask", "unmask", "prefill"], "kv_saved": [true, false, false]}
```

### `bellman_log_<task>.jsonl`

Bellman table/TB update 실험에서 생성된다. `plot_bellman_convergence.py`가 읽는다.

대표 형태:

```jsonl
{"block_id": 0, "table": [0.0, 1.0, 2.1, 3.2], "trajectory": [4, 3, 2, 1, 0]}
{"block_id": 1, "table": [0.0, 1.0, 1.9, 2.8], "trajectory": [3, 2, 1, 0]}
```

필드는 구현 변경에 따라 조금 달라질 수 있지만, 기본적으로 TB table snapshot과 실제 remaining-mask trajectory를 줄 단위로 저장한다고 보면 된다.

### `slo_rates.json`과 `slo_summary.json`

`slo_rates.json`은 run 하나에 대한 결과다.

```text
scheduler_LST/request_rate_14/humaneval/slo_rates.json
```

`slo_summary.json`은 모든 run을 합친 결과다.

```text
OUTPUT_ROOT/slo_summary.json
```

### `finalize`는 benchmark를 다시 돌리지 않는다

`finalize_dlm_scheduler_comparison.py`는 서버를 띄우지 않고 request도 보내지 않는다. 이미 저장된 JSON/JSONL 파일만 사용한다.

### run script 시작 시 기존 결과는 삭제된다

`run_dlm_scheduler_comparison_LLADA2.sh`와 `run_dlm_scheduler_comparison_SDAR.sh`는 시작할 때 해당 `OUTPUT_ROOT`를 삭제한다. 같은 output root에 이전 결과를 보존하고 싶으면 실행 전에 `OUTPUT_ROOT`를 다르게 지정해야 한다.
