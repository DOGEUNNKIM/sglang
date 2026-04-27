import json
import time
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class LowConfidence(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        # When set, append per-block step counts (JSONL) to this file for analysis.
        self.step_log_file = config.algorithm_config.get("step_log_file", None)

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor],
        List[torch.Tensor],
        bool,
        int,
        List[int],
        List[Union[torch.Tensor, None]],
        List[bool],
        List[bool],
        List[str],
    ]:
        batch_size = forward_batch.batch_size

        if self.step_log_file is not None:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        raw_forward_calls = 1

        if self.step_log_file is not None:
            torch.cuda.synchronize()
            _forward_duration_ms = (time.perf_counter() - _t0) * 1000

        req_modes = getattr(forward_batch, "dllm_req_modes", None)
        if req_modes is None:
            req_modes = []
            for batch_id in range(batch_size):
                block_start = batch_id * self.block_size
                block_end = block_start + self.block_size
                block_input_ids = forward_batch.input_ids[block_start:block_end]
                req_modes.append(
                    "unmask"
                    if torch.any(block_input_ids == self.mask_id).item()
                    else "prefill"
                )

        active_starts = getattr(forward_batch, "dllm_active_starts", None)
        if active_starts is None:
            active_starts = [0] * batch_size

        prev_block_steps = getattr(forward_batch, "dllm_active_block_steps", None)
        if prev_block_steps is None:
            prev_block_steps = [0] * batch_size

        next_token_ids: List[torch.Tensor] = []
        updated_ids: List[Union[torch.Tensor, None]] = []
        pending_kv_save = [False] * batch_size
        kv_saved = [False] * batch_size
        block_steps = list(prev_block_steps)
        needs_cpu_block = [False] * batch_size

        empty = torch.empty(0, dtype=forward_batch.input_ids.dtype, device="cpu")

        for batch_id in range(batch_size):
            curr_block_start = batch_id * self.block_size
            curr_block_end = curr_block_start + self.block_size
            block_input_ids = forward_batch.input_ids[curr_block_start:curr_block_end]
            mode = req_modes[batch_id]

            if mode == "kv_save":
                next_token_ids.append(empty)
                updated_ids.append(None)
                kv_saved[batch_id] = True
                needs_cpu_block[batch_id] = True
                continue

            if mode != "unmask":
                next_token_ids.append(empty)
                updated_ids.append(None)
                block_steps[batch_id] = 0
                continue

            block_mask_index = block_input_ids == self.mask_id
            curr_logits = logits_output.full_logits[curr_block_start:curr_block_end]
            x = torch.argmax(curr_logits, dim=-1)
            p = torch.squeeze(
                torch.gather(
                    F.softmax(curr_logits, dim=-1),
                    dim=-1,
                    index=torch.unsqueeze(x, -1),
                ),
                -1,
            )
            x = torch.where(block_mask_index, x, block_input_ids)
            confidence = torch.where(block_mask_index, p, -np.inf)

            transfer_index = confidence > self.threshold
            select_index = torch.argmax(confidence)
            transfer_index[select_index] = True

            block_input_ids[transfer_index] = x[transfer_index]
            block_steps[batch_id] = int(prev_block_steps[batch_id]) + 1
            next_token_ids.append(empty)
            updated_ids.append(None)
            needs_cpu_block[batch_id] = True

        if any(needs_cpu_block):
            block_cpu = (
                forward_batch.input_ids.reshape(batch_size, self.block_size)
                .detach()
                .cpu()
            )
            remaining_masks = (block_cpu == self.mask_id).any(dim=1).tolist()
            for batch_id, needs_cpu in enumerate(needs_cpu_block):
                if not needs_cpu:
                    continue

                mode = req_modes[batch_id]
                if mode == "kv_save":
                    start = int(active_starts[batch_id])
                    next_token_ids[batch_id] = block_cpu[batch_id, start:].clone()
                elif mode == "unmask":
                    pending_kv_save[batch_id] = not bool(remaining_masks[batch_id])
                    updated_ids[batch_id] = block_cpu[batch_id].clone()

        if self.step_log_file is not None:
            final_block_steps = [
                int(step) for step, saved in zip(block_steps, kv_saved) if saved
            ]
            with open(self.step_log_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "raw_forward_calls": raw_forward_calls,
                            "forward_duration_ms": round(_forward_duration_ms, 3),
                            "unmask_steps": 1
                            if any(mode == "unmask" for mode in req_modes)
                            else 0,
                            "block_steps": block_steps,
                            "final_block_steps": final_block_steps,
                            "req_modes": req_modes,
                            "pending_kv_save": pending_kv_save,
                            "kv_saved": kv_saved,
                        }
                    )
                    + "\n"
                )

        return (
            logits_output,
            next_token_ids,
            can_run_cuda_graph,
            raw_forward_calls,
            block_steps,
            updated_ids,
            pending_kv_save,
            kv_saved,
            req_modes,
        )


Algorithm = LowConfidence
