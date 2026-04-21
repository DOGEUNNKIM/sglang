from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Optional

from sglang.srt.dllm.config import DllmConfig

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DllmReqPhase(str, enum.Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


class ReqDllmMixin:
    def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
        self.dllm_phase: Optional[DllmReqPhase] = None
        self.dllm_block_offset = 0
        self.dllm_config = dllm_config
        self.dllm_first_block_time: Optional[float] = None
        self.dllm_last_block_time: Optional[float] = None
        self.dllm_tpob_sum = 0.0
        self.dllm_tpob_count = 0
        self.dllm_output_block_count = 0
        self.dllm_latency_logged = False
        self.dllm_scheduled_source_phase: Optional[DllmReqPhase] = None
        self.dllm_scheduled_exec_phase: Optional[DllmReqPhase] = None

        if self.dllm_config is not None:
            if len(self.origin_input_ids) < self.dllm_config.block_size:
                self.dllm_phase = DllmReqPhase.INCOMING_DECODE
            else:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL

    def is_dllm(self: Req) -> bool:
        return self.dllm_config is not None

    def record_dllm_block_emit_time(self: Req, ts: float) -> None:
        if self.dllm_first_block_time is None:
            self.dllm_first_block_time = ts
        elif self.dllm_last_block_time is not None:
            self.dllm_tpob_sum += ts - self.dllm_last_block_time
            self.dllm_tpob_count += 1

        self.dllm_last_block_time = ts
        self.dllm_output_block_count += 1

    def get_dllm_ttfb(self: Req) -> Optional[float]:
        if self.dllm_first_block_time is None:
            return None

        start_time = getattr(self.time_stats, "wait_queue_entry_time", 0.0)
        if start_time == 0.0:
            start_time = getattr(self.time_stats, "scheduler_recv_time", 0.0)
        if start_time == 0.0:
            return None

        return self.dllm_first_block_time - start_time

    def get_dllm_tpob(self: Req) -> Optional[float]:
        if self.dllm_tpob_count == 0:
            return None
        return self.dllm_tpob_sum / self.dllm_tpob_count

    def is_dllm_prefill(self: Req) -> bool:
        return self.dllm_phase in [
            DllmReqPhase.STAGING_PREFILL,
            DllmReqPhase.INCOMING_PREFILL,
        ]

    def determine_dllm_phase(self: Req):
        prefix_length = len(self.prefix_indices)
        min_required_length = prefix_length + self.dllm_config.block_size

        if len(self.fill_ids) < min_required_length:
            # still incoming stage
            return

        input_block = self.fill_ids[prefix_length:min_required_length]
        is_prefill_phase = self.dllm_config.mask_id not in input_block

        if is_prefill_phase:
            self.dllm_phase = DllmReqPhase.STAGING_PREFILL
        else:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE

    def _init_fill_ids_for_dllm(self: Req):
        self.dllm_block_offset = (
            0
            if not self.fill_ids
            else self.dllm_block_offset + self.dllm_config.block_size
        )
        self.fill_ids = (
            self.origin_input_ids
            + self.output_ids
            + [self.dllm_config.mask_id] * self.dllm_config.block_size
        )

    def _update_block_offset_for_dllm(self):
        prefix_len = len(self.prefix_indices)
        assert (
            prefix_len % self.dllm_config.block_size == 0
        ), f"Unexpected prefix len: {prefix_len}"
        if prefix_len > self.dllm_block_offset:
            self.dllm_block_offset = prefix_len
