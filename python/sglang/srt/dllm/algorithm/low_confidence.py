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
        # When set, append one record per individual GPU forward during decode.
        self.batch_log_file = config.algorithm_config.get("batch_latency_log_file", None)
        self._inner_seq = 0

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
    ]:
        batch_size = forward_batch.batch_size
        raw_forward_calls = 0
        # Here, the forward_batch full logits contains all the blocks
        # such as [dllm_block_size * batch_size, hidden_size]
        start_list = []
        mask_index = forward_batch.input_ids == self.mask_id

        # Fast path: if there is no mask token, forward and save kv cache
        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            raw_forward_calls += 1
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            if self.step_log_file is not None:
                with open(self.step_log_file, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "raw_forward_calls": raw_forward_calls,
                                "unmask_steps": 0,
                                "block_steps": [],
                            }
                        )
                        + "\n"
                    )

            next_token_ids = []
            return (
                logits_output,
                next_token_ids,
                can_run_cuda_graph,
                raw_forward_calls,
                [],
            )

        # Calculate start positions for each block
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            block_mask_index = block_input_ids == self.mask_id
            start = self.block_size - torch.sum(block_mask_index).item()
            start_list.append(start)

        # block_steps[i] = actual forward passes needed to fully unmask block i
        block_steps = [0] * batch_size
        unmask_steps = 0

        for step in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            pre_forward_mask_index = mask_index.clone()
            _step_t0 = time.perf_counter()
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            _step_ms = (time.perf_counter() - _step_t0) * 1000
            raw_forward_calls += 1
            unmask_steps += 1
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            assert batch_size == forward_batch.input_ids.shape[0] // self.block_size
            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end,
                ]
                block_mask_index = block_input_ids == self.mask_id
                if torch.sum(block_mask_index).item() == 0:
                    continue

                block_steps[batch_id] = step + 1

                curr_logits = logits_output.full_logits[
                    curr_block_start:curr_block_end,
                ]

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

                if transfer_index.sum().item() == 0:
                    _, select_index = torch.topk(confidence, k=1)
                    transfer_index[select_index] = True

                block_input_ids[transfer_index] = x[transfer_index]

            if self.batch_log_file is not None:
                self._log_inner_forward(
                    batch_size,
                    _step_ms,
                    forward_batch,
                    step,
                    pre_forward_mask_index,
                )

        _final_t0 = time.perf_counter()
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        _final_ms = (time.perf_counter() - _final_t0) * 1000
        raw_forward_calls += 1
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        if self.batch_log_file is not None:
            self._log_inner_forward(
                batch_size,
                _final_ms,
                forward_batch,
                unmask_steps,
                count_all_reqs=True,
            )

        if self.step_log_file is not None:
            with open(self.step_log_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "raw_forward_calls": raw_forward_calls,
                            "unmask_steps": unmask_steps,
                            "block_steps": block_steps,
                        }
                    )
                    + "\n"
                )

        # Here next token ids is tricky to implement the dynamic lengths,
        # so we return a list of tensors
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]

        return (
            logits_output,
            next_token_ids_list,
            can_run_cuda_graph,
            raw_forward_calls,
            block_steps,
        )


    def _log_inner_forward(
        self,
        batch_size: int,
        forward_ms: float,
        forward_batch: ForwardBatch,
        step: int,
        mask_index_for_budget: torch.Tensor = None,
        count_all_reqs: bool = False,
    ) -> None:
        if mask_index_for_budget is None:
            mask_index_for_budget = forward_batch.input_ids == self.mask_id

        per_req_has_mask = [
            bool(
                mask_index_for_budget[
                    b * self.block_size : (b + 1) * self.block_size
                ].any()
            )
            for b in range(batch_size)
        ]
        if count_all_reqs:
            per_req_extend = [self.block_size] * batch_size
            effective_per_req_extend = per_req_extend
        else:
            # The baseline algorithm forwards the whole fixed batch on every
            # inner step.  For ETB, count only blocks that still needed unmask
            # compute before this forward; completed blocks are intra-batch
            # bubbles.
            per_req_extend = [
                self.block_size if has_mask else 0 for has_mask in per_req_has_mask
            ]
            effective_per_req_extend = per_req_extend
        num_masked_reqs = sum(1 for has_mask in per_req_has_mask if has_mask)
        record = {
            "seq": self._inner_seq,
            "phase": "decode",
            "decode_inner_step": step,
            "per_req_extend_input_len": per_req_extend,
            "physical_per_req_extend_input_len": [self.block_size] * batch_size,
            "effective_per_req_extend_input_len": effective_per_req_extend,
            "per_req_has_mask": per_req_has_mask,
            "num_reqs": batch_size,
            "num_masked_reqs": num_masked_reqs,
            "forward_result_ms": forward_ms,
            "duration_ms": forward_ms,
        }
        with open(self.batch_log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        self._inner_seq += 1


Algorithm = LowConfidence
