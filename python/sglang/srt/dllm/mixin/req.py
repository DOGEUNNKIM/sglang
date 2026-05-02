from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING, Optional

from sglang.srt.dllm.config import DllmConfig

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DllmReqPhase(str, enum.Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    QUEUING_PREFILL = "queuing_prefill"
    QUEUING_DECODE = "queuing_decode"


class ReqDllmMixin:
    def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
        self.dllm_phase: Optional[DllmReqPhase] = None
        self.dllm_block_offset = 0
        self.dllm_config = dllm_config
        self.dllm_first_block_time: Optional[float] = None
        self.dllm_last_block_time: Optional[float] = None
        self.dllm_tpob_sum = 0.0
        self.dllm_tpob_count = 0
        self.dllm_tpob_list: list[float] = []  # per-block intervals (s), index = block_idx - 1
        self.dllm_block_steps_list: list[int] = []  # decode forward steps per output block
        self.dllm_output_block_count = 0
        self.dllm_latency_logged = False
        self.dllm_scheduled_source_phase: Optional[DllmReqPhase] = None
        self.dllm_scheduled_exec_phase: Optional[DllmReqPhase] = None
        self.dllm_active_prefix_ids: Optional[list[int]] = None
        self.dllm_active_ids: Optional[list[int]] = None
        self.dllm_active_cache_locs = None
        self.dllm_active_prefix_len: Optional[int] = None
        self.dllm_active_seq_len: Optional[int] = None
        self.dllm_active_block_offset: Optional[int] = None
        self.dllm_active_start = 0
        self.dllm_active_block_steps = 0
        self.dllm_pending_kv_save = False
        # cumulative decode forward steps across all completed blocks
        self.dllm_total_block_steps: int = 0
        # decode delay: time from block completion → next block first enters a batch
        self.dllm_block_ready_time: Optional[float] = None
        self.dllm_decode_delay_sum: float = 0.0
        self.dllm_decode_delay_count: int = 0
        # prefill→first_unmask gap: time from last prefill forward → first unmask batch
        self.dllm_prefill_end_time: Optional[float] = None
        self.dllm_first_unmask_gap: Optional[float] = None
        # LST scheduling: per-phase deadline and admission timestamp
        self.dllm_current_deadline: Optional[float] = None
        self.dllm_admission_time: Optional[float] = None
        # Prefix length from a lightweight cache peek done in _fetch_waiting_reqs
        # before the request is fully scheduled.  Used only in compute_dllm_slack
        # so that cache hits are reflected in remaining_prefill even before
        # init_next_round_input sets prefix_indices.
        self.dllm_prefetched_prefix_len: int = 0
        # Per-request EMA of unmask tokens per decode forward pass.
        # Warm-started from the global EMA so new requests get a reasonable prior;
        # updated on each completed block.  Used instead of the global EMA in
        # compute_dllm_slack so that per-request variation is reflected.
        self.dllm_ema_unmask_per_fwd: float = (
            dllm_config.ema_unmask_per_forward if dllm_config is not None else 1.0
        )

        if self.dllm_config is not None:
            if len(self.origin_input_ids) < self.dllm_config.block_size:
                self.dllm_phase = DllmReqPhase.QUEUING_DECODE
            else:
                self.dllm_phase = DllmReqPhase.QUEUING_PREFILL

    def is_dllm(self: Req) -> bool:
        return self.dllm_config is not None

    def record_dllm_block_emit_time(self: Req, ts: float) -> None:
        if self.dllm_first_block_time is None:
            self.dllm_first_block_time = ts
            # Switch from TTFB deadline to TPOB deadline
            if self.dllm_config is not None and self.dllm_config.tpob_slo is not None:
                self.dllm_current_deadline = ts + self.dllm_config.tpob_slo
            else:
                self.dllm_current_deadline = None
        elif self.dllm_last_block_time is not None:
            interval = ts - self.dllm_last_block_time
            self.dllm_tpob_sum += interval
            self.dllm_tpob_count += 1
            self.dllm_tpob_list.append(interval)
            # Refresh TPOB deadline for the next block
            if self.dllm_config is not None and self.dllm_config.tpob_slo is not None:
                self.dllm_current_deadline = ts + self.dllm_config.tpob_slo

        self.dllm_last_block_time = ts
        self.dllm_output_block_count += 1

    def set_dllm_admission_time(self: Req, ts: float) -> None:
        """Record admission time and set initial TTFB deadline (called once on first admission).

        Uses the request's actual arrival time (wait_queue_entry_time or
        scheduler_recv_time) so that time spent in the outer Scheduler.waiting_queue
        is already counted against the TTFB SLO.  Falls back to `ts` (perf_counter
        at DLLM admission) only when no arrival timestamp is available.
        """
        if self.dllm_admission_time is not None:
            return  # Already admitted; don't reset deadline mid-flight

        # Prefer the earliest recorded arrival timestamp; all use perf_counter.
        arrival = getattr(self.time_stats, "wait_queue_entry_time", 0.0)
        if not arrival:
            arrival = getattr(self.time_stats, "scheduler_recv_time", 0.0)
        if not arrival:
            arrival = ts  # Fallback: DLLM admission time

        self.dllm_admission_time = arrival
        if self.dllm_config is not None and self.dllm_config.ttfb_slo is not None:
            self.dllm_current_deadline = arrival + self.dllm_config.ttfb_slo

    def compute_dllm_slack(self: Req, now: float) -> float:
        """Remaining slack = deadline - now - estimated remaining compute.

        Least-slack-first scheduling: smaller value → higher priority.
        Returns inf when no deadline is active (LST disabled or TTFB already met
        and no tpob_slo configured).

        Remaining compute:
          TTFB phase: remaining_prefill_blocks * prefill_forward_time_s
                    + ceil(block_size / expected_unmask_per_forward) * decode_forward_time_s
          TPOB phase: ceil(block_size / expected_unmask_per_forward) * decode_forward_time_s

        completed uses len(prefix_indices) so that pre-fetched prefix cache hits
        (done by _fetch_waiting_reqs for new QUEUING_PREFILL requests) are
        reflected in remaining_prefill.
        """
        if self.dllm_current_deadline is None:
            return float("inf")

        cfg = self.dllm_config
        block_size = cfg.block_size
        decode_forwards = math.ceil(block_size / self.dllm_ema_unmask_per_fwd)

        if self.dllm_first_block_time is None:
            # TTFB phase
            total_prefill = math.ceil(len(self.origin_input_ids) / block_size)
            # Use the larger of the live prefix_indices and the pre-fetched peek
            # so cache hits are reflected even before init_next_round_input runs.
            completed = max(len(self.prefix_indices), self.dllm_prefetched_prefix_len) // block_size
            remaining_prefill = max(0, total_prefill - completed) if self.is_dllm_prefill() else 0
            remaining_compute = (
                remaining_prefill * cfg.ema_prefill_forward_time_s
                + decode_forwards * cfg.ema_decode_forward_time_s
            )
        else:
            # TPOB phase
            remaining_compute = decode_forwards * cfg.ema_decode_forward_time_s

        return self.dllm_current_deadline - now - remaining_compute

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

    def get_dllm_decode_delay(self: Req) -> Optional[float]:
        """Mean inter-block scheduling delay (block ready → block enters next batch)."""
        if self.dllm_decode_delay_count == 0:
            return None
        return self.dllm_decode_delay_sum / self.dllm_decode_delay_count

    def is_dllm_prefill(self: Req) -> bool:
        return self.dllm_phase in [
            DllmReqPhase.STAGING_PREFILL,
            DllmReqPhase.QUEUING_PREFILL,
        ]

    def has_dllm_active_block(self: Req) -> bool:
        return self.dllm_active_ids is not None

    def clear_dllm_active_block(self: Req) -> None:
        self.dllm_active_prefix_ids = None
        self.dllm_active_ids = None
        self.dllm_active_cache_locs = None
        self.dllm_active_prefix_len = None
        self.dllm_active_seq_len = None
        self.dllm_active_block_offset = None
        self.dllm_active_start = 0
        self.dllm_active_block_steps = 0
        self.dllm_pending_kv_save = False

    def determine_dllm_phase(self: Req):
        if self.has_dllm_active_block():
            self.dllm_phase = DllmReqPhase.STAGING_DECODE
            return

        # No active block: either a brand-new request or a completed block.
        # _init_fill_ids_for_dllm() sets dllm_block_offset=0 only on the very
        # first call (when fill_ids is still empty).  After any completed block,
        # dllm_block_offset is already > 0.
        prefix_length = len(self.prefix_indices)
        min_required_length = prefix_length + self.dllm_config.block_size

        if len(self.fill_ids) < min_required_length:
            return

        input_block = self.fill_ids[prefix_length:min_required_length]
        is_prefill_phase = self.dllm_config.mask_id not in input_block

        if is_prefill_phase:
            # dllm_block_offset==0 means fill_ids was empty before this call →
            # brand-new request entering the system for the first time.
            # dllm_block_offset>0 means at least one block already completed →
            # continuation prefill, keep staging priority.
            if self.dllm_block_offset == 0:
                self.dllm_phase = DllmReqPhase.QUEUING_PREFILL
            else:
                self.dllm_phase = DllmReqPhase.STAGING_PREFILL
        else:
            # Block contains masks → kv_save-completed decode returning from staging.
            # Must stay STAGING_DECODE so it goes through process_dllm_staging_reqs,
            # which preserves prefix_indices without tree_cache re-matching.
            # (QUEUING_DECODE is only set in init_diffusion_llm for brand-new short inputs.)
            self.dllm_phase = DllmReqPhase.STAGING_DECODE

    def _init_fill_ids_for_dllm(self: Req):
        if self.has_dllm_active_block():
            prefix_ids = self.dllm_active_prefix_ids or []
            self.fill_ids = prefix_ids + self.dllm_active_ids
            if self.dllm_active_block_offset is not None:
                self.dllm_block_offset = self.dllm_active_block_offset
            return

        self.dllm_block_offset = (
            0
            if not self.fill_ids
            else self.dllm_block_offset + self.dllm_config.block_size
        )
        self.fill_ids = self.origin_input_ids + self.output_ids
        self._pad_fill_ids_for_dllm_block()

    def _pad_fill_ids_for_dllm_block(self: Req):
        block_size = self.dllm_config.block_size
        prefix_length = len(self.prefix_indices)
        next_block_end = prefix_length + block_size

        # Add masks only for the block that is about to be processed.  Earlier
        # pure-prefill blocks should not carry placeholder decode masks.
        if len(self.fill_ids) < next_block_end:
            self.fill_ids = self.fill_ids + [self.dllm_config.mask_id] * (
                next_block_end - len(self.fill_ids)
            )

    def _update_block_offset_for_dllm(self):
        prefix_len = len(self.prefix_indices)
        assert (
            prefix_len % self.dllm_config.block_size == 0
        ), f"Unexpected prefix len: {prefix_len}"
        if prefix_len > self.dllm_block_offset:
            self.dllm_block_offset = prefix_len
