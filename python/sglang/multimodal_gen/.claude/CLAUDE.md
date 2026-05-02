# CLAUDE.md — sglang-diffusion (multimodal_gen)

## What is this?

SGLang's diffusion/multimodal generation subsystem. Separate from the LLM runtime (`srt`). Supports 20+ image/video diffusion models (Wan, FLUX, HunyuanVideo, LTX, Qwen-Image, etc.) with distributed inference, LoRA, and multiple attention backends.

## Quick Start

```bash
# One-shot generation
sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --prompt "A curious raccoon" --save-output

# Start server
sglang serve --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --num-gpus 4

# Python API
from sglang import DiffGenerator
gen = DiffGenerator.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
result = gen.generate(sampling_params_kwargs={"prompt": "A curious raccoon"})
```

## Architecture

```
CLI / Python API / HTTP Server (FastAPI + OpenAI-compatible)
    ↓ ZMQ
Scheduler (request queue, batching, dispatch)
    ↓ multiprocessing pipes
GPU Worker(s) → ComposedPipeline (stages: TextEncode → Denoise → Decode)
```

### Key Directories

```
runtime/
├── entrypoints/        # CLI (generate/serve), HTTP server, Python API (DiffGenerator)
├── managers/           # scheduler.py, gpu_worker.py
├── pipelines_core/     # ComposedPipelineBase, stages/, schedule_batch.py (Req/OutputBatch)
├── pipelines/          # Model-specific pipelines (wan, flux, hunyuan, ltx, qwen_image, ...)
├── models/             # encoders/, dits/, vaes/, schedulers/
├── layers/             # attention/, lora/, quantization/
├── loader/             # Model loading, weight utils
├── server_args.py      # ServerArgs (all CLI/config params)
└── distributed/        # Multi-GPU (TP, SP: ulysses/ring)
configs/
├── pipeline_configs/    # Per-model pipeline configs
├── sample/             # SamplingParams
└── models/             # DiT, VAE, Encoder configs
```

### Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `DiffGenerator` | `runtime/entrypoints/diffusion_generator.py` | Python API entry point |
| `ComposedPipelineBase` | `runtime/pipelines_core/composed_pipeline_base.py` | Pipeline orchestrator (stages) |
| `Scheduler` | `runtime/managers/scheduler.py` | ZMQ event loop, request dispatch |
| `GPUWorker` | `runtime/managers/gpu_worker.py` | GPU inference worker |
| `Req` / `OutputBatch` | `runtime/pipelines_core/schedule_batch.py` | Request/output containers |
| `ServerArgs` | `runtime/server_args.py` | All config params |
| `SamplingParams` | `configs/sample/sampling_params.py` | Generation params |
| `PipelineConfig` | `configs/pipeline_configs/base.py` | Model structure config |

### Key Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `build_pipeline()` | `runtime/pipelines_core/__init__.py` | Instantiate pipeline from model_path |
| `get_model_info()` | `registry.py` | Resolve pipeline + config classes |
| `launch_server()` | `runtime/launch_server.py` | Start multi-process server |

## Adding a New Model

1. Create pipeline in `runtime/pipelines/` extending `ComposedPipelineBase`
2. Define stages via `create_pipeline_stages()` (TextEncoding → Denoising → Decoding)
3. Add config in `configs/pipeline_configs/`
4. Register in `registry.py` via `register_configs()`

## Multi-GPU

```bash
# Sequence parallelism (video frames across GPUs)
sglang serve --model-path ... --num-gpus 4 --ulysses-degree 2 --ring-degree 2

# Tensor parallelism (model layers across GPUs)
sglang serve --model-path ... --num-gpus 2 --tp-size 2
```

## Testing

```bash
# Tests live in test/ subdirectory
python -m pytest python/sglang/multimodal_gen/test/

# No need to pre-download models — auto-downloaded at runtime
# Dependencies assumed already installed via `pip install -e "python[diffusion]"`
```

## Performance Tuning

For questions about optimal performance, fastest commands, VRAM reduction, or best flag combinations for a given model/GPU setup, **read the [sglang-diffusion-performance skill](skills/sglang-diffusion-performance/SKILL.md)**. It contains a complete table of lossless and lossy optimization flags with trade-offs, quick recipes, and tuning tips.

### Perf Measurement

Look for `Pixel data generated successfully in xxxx seconds` in console output. With warmup enabled, use the line containing `warmup excluded` for accurate timing.

## Env Vars

Defined in `envs.py` (300+ vars). Key ones:
- `SGLANG_DIFFUSION_ATTENTION_BACKEND` — attention backend override
- `SGLANG_CACHE_DIT_ENABLED` — enable Cache-DiT acceleration
- `SGLANG_CLOUD_STORAGE_TYPE` — cloud output storage (s3, etc.)

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.