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
| `plot_step_dist.py` | `step_stats_<task>.jsonl` 기반 step distribution plot |

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
  -> inline Python plot code         (task_spec_<model>.png 생성)

run_dlm_tb_update_test.sh
  -> python -m sglang.launch_server
  -> test/dlm_benchmark.py
  -> test/dlm_slorate.py
  -> test/plot_bellman_convergence.py
  -> test/plot_step_dist.py
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

plot_step_dist.py
  - step_stats_<task>.jsonl에서 step histogram 생성
  - 결과:
    사용자가 --output으로 지정한 PNG

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

이 wrapper는 SLO 기준값을 다시 만들지 않고, `dlm_slorate.py`도 실행하지 않으며, `slo_summary.json`도 다시 쓰지 않는다.

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

### 실행

```bash
./test/run_dlm_task_spec.sh
```

### 출력 구조

```text
/mnt/nvme0/kdg6245/dlm_task_spec/
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

Bellman table update가 실제 step 분포에 수렴하는지 확인하는 것이다.

### 실행

```bash
./test/run_dlm_tb_update_test.sh
```

### 내부 흐름

```text
sglang.launch_server
dlm_benchmark.py
dlm_slorate.py
plot_bellman_convergence.py
plot_step_dist.py
```

### 출력 구조

```text
/mnt/nvme0/kdg6245/dlm_tb_update_test/
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