# DLM Benchmark & Scheduler Comparison Guide

이 문서는 DLM 관련 실행 스크립트와 후처리 스크립트의 관계를 정리한다. 핵심은 다음 세 단계다.

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
| `dlm_benchmark.py` | 실제 benchmark 실행 엔진. 서버에 요청을 보내고 per-task 결과를 저장 |
| `dlm_slorate.py` | `request_latency_<task>.jsonl`에서 SLO 달성률 계산 |
| `finalize_dlm_scheduler_comparison.py` | 끝난 scheduler 비교 결과를 다시 스캔해서 `slo_config.json`, `slo_summary.json`, plot 재생성 |
| `plot_dlm_slo_summary.py` | `slo_summary.json` 기반 scheduler 비교 plot 생성 |
| `run_dlm_task_spec.sh` | task별 입력 길이, output block, step 분포 분석 |
| `run_dlm_tb_update_test.sh` | Bellman table/TB update 동작 검증 |
| `plot_bellman_convergence.py` | `bellman_log_<task>.jsonl` 수렴 plot 생성 |
| `plot_step_dist.py` | `step_stats_<task>.jsonl` 기반 step distribution plot |
| `dlm_plot_steps.py` | standalone step plot utility. `summary_*.json` 또는 `steps_*.jsonl` 사용 |

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

TTFB scheduler 완료 후:
  task별 ideal TTFB 추출

DECODE scheduler 완료 후:
  task별 ideal TPOB 추출

전체 benchmark 완료 후:
  1. slo_config.json 생성
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
python ./test/finalize_dlm_scheduler_comparison.py \
  --output-root /mnt/nvme0/kdg6245/dlm_sched_comparison_LLADA2 \
  --model-path inclusionAI/LLaDA2.0-mini
```

### SDAR

```bash
python ./test/finalize_dlm_scheduler_comparison.py \
  --output-root /mnt/nvme0/kdg6245/dlm_sched_comparison_SDAR \
  --model-path JetLM/SDAR-8B-Chat
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

