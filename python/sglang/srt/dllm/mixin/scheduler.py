from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, List, Optional, Set, Union

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.utils import broadcast_pyobj

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler


class SchedulerDllmMixin:
    def init_diffusion_llm(self: Scheduler):
        # dllm_config was already created in init_model_config() so that
        # admission_window could bump max_running_requests before the token
        # pool was allocated.  Fall back to creating it here only if absent.
        if not hasattr(self, "dllm_config"):
            self.dllm_config = (
                DllmConfig.from_server_args(self.server_args)
                if self.server_args.dllm_algorithm is not None
                else None
            )
        self.dllm_manager = DllmManager(dllm_config=self.dllm_config)
        self.dllm_batch_seq = 0
        self.dllm_last_batch_phase = None
        self.dllm_phase_run_length = 0
        self.dllm_phase_switch_count = 0
        self.dllm_prefill_batch_count = 0
        self.dllm_decode_batch_count = 0
        self.dllm_mixed_prefill_decode_batch_count = 0
        self.dllm_prefill_req_time_sum = 0.0
        self.dllm_prefill_req_count = 0
        self.dllm_prefill_batch_time_sum = 0.0
        self.dllm_decode_block_time_sum = 0.0
        self.dllm_decode_block_count = 0
        self.dllm_decode_batch_time_sum = 0.0
        self.dllm_mixed_prefill_decode_batch_time_sum = 0.0

    def get_new_batch_dllm(self: Scheduler) -> Optional[ScheduleBatch]:
        """Generate a new batch for DLLM (Diffusion LLM) scheduling."""
        schedule_start_time = time.perf_counter()
        schedule_timing = {
            "fetch_waiting_s": 0.0,
            "sort_laxity_s": 0.0,
            "tp_sync_order_s": 0.0,
            "select_batch_s": 0.0,
            "update_state_s": 0.0,
            "batch_build_s": 0.0,
        }
        if self.enable_priority_preemption:
            self.running_batch.batch_is_full = False

        # Early exit if batch is full or no requests available
        if self._should_skip_prefill():
            return None

        running_bs = len(self.running_batch.reqs)
        self.policy.calc_priority(self.waiting_queue)

        # Create prefill adder with resource constraints
        adder = self._create_dllm_prefill_adder(running_bs)

        # Initialize DLLM manager and transfer requests
        self.dllm_manager.init_next_round()
        phase_start_time = time.perf_counter()
        self._fetch_waiting_reqs()
        schedule_timing["fetch_waiting_s"] = (
            time.perf_counter() - phase_start_time
        )
        phase_start_time = time.perf_counter()
        self._sort_dllm_by_laxity()
        schedule_timing["sort_laxity_s"] = time.perf_counter() - phase_start_time
        phase_start_time = time.perf_counter()
        self._sync_dllm_lst_order_across_tp()
        schedule_timing["tp_sync_order_s"] = time.perf_counter() - phase_start_time

        # Process batches
        phase_start_time = time.perf_counter()
        forward_mode = self._process_dllm_batches(adder)
        schedule_timing["select_batch_s"] = time.perf_counter() - phase_start_time

        can_run_list = adder.can_run_list
        if not can_run_list:
            return None

        # Record metrics and update state
        set_time_batch(can_run_list, "set_forward_entry_time")
        phase_start_time = time.perf_counter()
        self._update_state_for_batch(can_run_list, adder, running_bs)
        schedule_timing["update_state_s"] = time.perf_counter() - phase_start_time

        # Create and prepare batch
        batch_phase = self._get_dllm_batch_phase(can_run_list)
        phase_start_time = time.perf_counter()
        new_batch = self._create_dllm_batch(can_run_list, forward_mode)
        schedule_timing["batch_build_s"] = time.perf_counter() - phase_start_time
        self._annotate_dllm_batch_metrics(
            new_batch,
            batch_phase=batch_phase,
            schedule_start_time=schedule_start_time,
            schedule_timing=schedule_timing,
        )
        return new_batch

    def process_batch_result_dllm(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        updated_ids = getattr(result, "dllm_updated_ids", None) or []
        pending_kv_save = getattr(result, "dllm_pending_kv_save", None) or []
        kv_saved = getattr(result, "dllm_kv_saved", None) or []
        block_steps = getattr(result, "dllm_block_steps", None) or []
        next_token_ids_list = result.next_token_ids or []
        block_emit_time = time.perf_counter()
        has_output = False
        has_finished = False

        if next_token_ids_list or updated_ids or pending_kv_save or kv_saved:
            self.token_to_kv_pool_allocator.free_group_begin()
            req_modes_in_batch = getattr(batch, "dllm_req_modes", []) or []

            for idx in range(batch.batch_size()):
                req = batch.reqs[idx]

                # track last prefill forward end time (updated every prefill batch)
                if idx < len(req_modes_in_batch) and req_modes_in_batch[idx] == "prefill":
                    req.dllm_prefill_end_time = block_emit_time

                if idx < len(updated_ids) and updated_ids[idx] is not None:
                    req.dllm_active_ids = updated_ids[idx].tolist()
                    if idx < len(block_steps) and block_steps[idx] is not None:
                        req.dllm_active_block_steps = int(block_steps[idx])

                if idx < len(pending_kv_save) and pending_kv_save[idx]:
                    req.dllm_pending_kv_save = True

                next_token_ids = []
                if idx < len(next_token_ids_list):
                    next_token_ids = next_token_ids_list[idx].tolist()
                new_tokens = len(next_token_ids)
                if new_tokens == 0:
                    if idx < len(kv_saved) and kv_saved[idx]:
                        self._commit_dllm_saved_block_prefix(req)
                        req.clear_dllm_active_block()
                    req.check_finished(new_accepted_len=0)
                    if req.finished():
                        req.clear_dllm_active_block()
                        release_kv_cache(req, self.tree_cache)
                        req.time_stats.set_completion_time()
                        self._maybe_log_dllm_request_latency(req)
                        has_finished = True
                    continue

                req.fill_ids[-new_tokens:] = next_token_ids[:]
                self.num_generated_tokens += new_tokens
                req.record_dllm_block_emit_time(block_emit_time)

                req.output_ids.extend(next_token_ids)
                req.check_finished(new_accepted_len=new_tokens)
                self._commit_dllm_saved_block_prefix(req)

                # Resolve final step count for this completed block.
                # Prefer block_steps[idx] from the result (may be updated in the
                # final step even when updated_ids[idx] is None); fall back to the
                # accumulated req field.
                final_steps = req.dllm_active_block_steps
                if idx < len(block_steps) and block_steps[idx] is not None:
                    final_steps = int(block_steps[idx])
                if final_steps > 0:
                    req.dllm_total_block_steps += final_steps
                    req.dllm_block_steps_list.append(final_steps)
                    # Bellman TD update at block completion (no GPU sync needed).
                    # Multi-step target: starting from block_size masks, observed final_steps.
                    cfg = self.dllm_config
                    r = cfg.block_size
                    alpha = cfg.bellman_alpha
                    cfg.decode_forwards_by_remaining[r] = (
                        (1.0 - alpha) * cfg.decode_forwards_by_remaining[r]
                        + alpha * float(final_steps)
                    )

                req.clear_dllm_active_block()
                has_output = True

                if not req.finished():
                    req.dllm_block_ready_time = block_emit_time

                if req.finished():
                    release_kv_cache(req, self.tree_cache)
                    req.time_stats.set_completion_time()
                    self._maybe_log_dllm_request_latency(req)
                    has_finished = True

            if has_output or has_finished:
                self.stream_output(batch.reqs, batch.return_logprob)
                # Block completion changes deadlines/phase → re-sort next schedule.
                if has_output:
                    self.dllm_manager.queue_dirty = True
            self.token_to_kv_pool_allocator.free_group_end()

        can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
        self._maybe_log_dllm_batch_metrics(batch, result)
        self.report_prefill_stats(
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def _maybe_log_dllm_request_latency(self: Scheduler, req: Req) -> None:
        """Append per-request TTFB/TPOB metrics when configured."""
        if req.dllm_latency_logged:
            return

        log_file = self.dllm_config.algorithm_config.get("request_latency_log_file")
        if log_file is None:
            log_file = self.dllm_config.algorithm_config.get("ttfb_tpob_log_file")
        if log_file is None:
            return

        ttfb = req.get_dllm_ttfb()
        tpob = req.get_dllm_tpob()
        decode_delay = req.get_dllm_decode_delay()
        first_unmask_gap = req.dllm_first_unmask_gap
        sched_wait_s = req.time_stats.get_queueing_time() if req.time_stats else None
        record = {
            "rid": req.rid,
            "input_len": len(req.origin_input_ids),
            "output_len": len(req.output_ids),
            "block_size": self.dllm_config.block_size,
            "num_output_blocks": req.dllm_output_block_count,
            # average decode forward steps per output block (total_steps / num_blocks)
            "avg_block_steps": (
                req.dllm_total_block_steps / req.dllm_output_block_count
                if req.dllm_output_block_count > 0
                else None
            ),
            "ttfb_s": ttfb,
            "ttfb_ms": None if ttfb is None else ttfb * 1000,
            # TPOB is the mean of adjacent output-block intervals.
            "tpob_s": tpob,
            "tpob_ms": None if tpob is None else tpob * 1000,
            "tpob_count": req.dllm_tpob_count,
            # Per-block intervals: index i = gap between block i and block i+1 (ms).
            # Length == num_output_blocks - 1.
            "tpob_list_ms": [v * 1000 for v in req.dllm_tpob_list],
            "block_steps_list": list(req.dllm_block_steps_list),
            # mean inter-block scheduling delay (block ready → enters next batch)
            "decode_delay_s": decode_delay,
            "decode_delay_ms": None if decode_delay is None else decode_delay * 1000,
            # prefill completion → first unmask batch assigned
            "first_unmask_gap_s": first_unmask_gap,
            "first_unmask_gap_ms": None if first_unmask_gap is None else first_unmask_gap * 1000,
            # time from entering waiting_queue to first forward pass
            "sched_wait_s": sched_wait_s,
            "sched_wait_ms": None if sched_wait_s is None else sched_wait_s * 1000,
            "slo_type": req.dllm_slo_type,
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        req.dllm_latency_logged = True

    def _get_dllm_batch_metric_log_file(self: Scheduler) -> Optional[str]:
        log_file = self.dllm_config.algorithm_config.get("batch_latency_log_file")
        if log_file is None:
            log_file = self.dllm_config.algorithm_config.get("phase_log_file")
        return log_file

    @staticmethod
    def _get_dllm_batch_phase(reqs: List[Req]) -> str:
        has_prefill = any(req.is_dllm_prefill() for req in reqs)
        has_decode = any(not req.is_dllm_prefill() for req in reqs)
        if has_prefill and has_decode:
            return "mixed_prefill_decode"
        if has_prefill:
            return "prefill"
        return "decode"

    def _annotate_dllm_batch_metrics(
        self: Scheduler,
        batch: ScheduleBatch,
        batch_phase: str,
        schedule_start_time: float,
        schedule_timing: dict[str, float],
    ) -> None:
        if self.dllm_last_batch_phase == batch_phase:
            self.dllm_phase_run_length += 1
        else:
            if self.dllm_last_batch_phase is not None:
                self.dllm_phase_switch_count += 1
            self.dllm_last_batch_phase = batch_phase
            self.dllm_phase_run_length = 1

        if batch_phase == "prefill":
            self.dllm_prefill_batch_count += 1
        elif batch_phase == "decode":
            self.dllm_decode_batch_count += 1
        elif batch_phase == "mixed_prefill_decode":
            self.dllm_mixed_prefill_decode_batch_count += 1

        batch.dllm_batch_seq = self.dllm_batch_seq
        batch.dllm_batch_phase = batch_phase
        batch.dllm_phase_run_length = self.dllm_phase_run_length
        batch.dllm_phase_switch_count = self.dllm_phase_switch_count
        batch.dllm_schedule_start_time = schedule_start_time
        batch.dllm_schedule_timing = dict(schedule_timing)
        batch.dllm_schedule_total_s = max(
            0.0, sum(float(value) for value in schedule_timing.values())
        )
        batch.dllm_forward_start_time = time.perf_counter()
        self._annotate_dllm_per_req_metrics(batch)
        self.dllm_batch_seq += 1

    def _annotate_dllm_per_req_metrics(self: Scheduler, batch: ScheduleBatch) -> None:
        """Snapshot request-level DLLM metadata before forward mutates inputs."""
        mask_id = self.dllm_config.mask_id

        rids = []
        phases = []
        exec_phases = []
        effective_phases = []
        has_mask = []
        extend_input_lens = []
        req_modes = []
        active_starts = []
        active_block_steps = []
        offset = 0

        for req in batch.reqs:
            rids.append(req.rid)
            source_phase = getattr(req, "dllm_scheduled_source_phase", None)
            exec_phase = getattr(req, "dllm_scheduled_exec_phase", None)
            current_phase = getattr(req, "dllm_phase", None)
            if source_phase is None:
                source_phase = current_phase
            if exec_phase is None:
                exec_phase = current_phase
            phases.append(source_phase.value if source_phase is not None else None)
            exec_phases.append(exec_phase.value if exec_phase is not None else None)

            extend_len = int(getattr(req, "extend_input_len", 0) or 0)
            extend_input_lens.append(extend_len)
            prefix_len = len(getattr(req, "prefix_indices", []))
            chunk = req.fill_ids[prefix_len : prefix_len + extend_len]
            req_has_mask = mask_id in chunk
            has_mask.append(req_has_mask)
            effective_phases.append("decode" if req_has_mask else "prefill")

            if getattr(req, "dllm_pending_kv_save", False):
                req_mode = "kv_save"
            elif req_has_mask:
                req_mode = "unmask"
            else:
                req_mode = "prefill"
            req_modes.append(req_mode)

            if req_mode in ("unmask", "kv_save"):
                if req.dllm_active_ids is None:
                    now = time.perf_counter()
                    # inter-block decode delay (block 2+)
                    if req.dllm_block_ready_time is not None:
                        delay = now - req.dllm_block_ready_time
                        req.dllm_decode_delay_sum += delay
                        req.dllm_decode_delay_count += 1
                        req.dllm_block_ready_time = None
                    # prefill → first unmask gap (first block only)
                    if req.dllm_prefill_end_time is not None and req.dllm_first_unmask_gap is None:
                        req.dllm_first_unmask_gap = now - req.dllm_prefill_end_time
                        req.dllm_prefill_end_time = None
                    req.dllm_active_prefix_ids = list(req.fill_ids[:prefix_len])
                    req.dllm_active_ids = list(chunk)
                    _remaining = chunk.count(mask_id)
                    req.dllm_active_start = self.dllm_config.block_size - _remaining
                    req.dllm_active_block_steps = 0
                    req.dllm_active_remaining_masks = _remaining
                else:
                    req.dllm_active_remaining_masks = chunk.count(mask_id)
                if req.dllm_active_cache_locs is None and batch.out_cache_loc is not None:
                    req.dllm_active_cache_locs = batch.out_cache_loc[
                        offset : offset + extend_len
                    ].detach().clone()
                req.dllm_active_prefix_len = len(getattr(req, "prefix_indices", []))
                req.dllm_active_seq_len = len(req.fill_ids)
                req.dllm_active_block_offset = req.dllm_block_offset

            active_starts.append(getattr(req, "dllm_active_start", 0))
            active_block_steps.append(getattr(req, "dllm_active_block_steps", 0))
            offset += extend_len

        batch.dllm_rids = rids
        batch.dllm_per_req_phase = phases
        batch.dllm_per_req_exec_phase = exec_phases
        batch.dllm_per_req_effective_phase = effective_phases
        batch.dllm_per_req_has_mask = has_mask
        batch.dllm_per_req_extend_input_len = extend_input_lens
        batch.dllm_req_modes = req_modes
        batch.dllm_active_starts = active_starts
        batch.dllm_active_block_steps = active_block_steps

    def _maybe_log_dllm_batch_metrics(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        log_file = self._get_dllm_batch_metric_log_file()
        if log_file is None:
            return

        end_time = time.perf_counter()
        start_time = getattr(batch, "dllm_forward_start_time", end_time)
        schedule_start_time = getattr(batch, "dllm_schedule_start_time", start_time)
        schedule_timing = dict(getattr(batch, "dllm_schedule_timing", {}) or {})
        forward_result_s = end_time - start_time
        duration_s = end_time - schedule_start_time
        schedule_prepare_s = start_time - schedule_start_time
        schedule_total_s = float(
            getattr(batch, "dllm_schedule_total_s", schedule_prepare_s)
        )
        fetch_waiting_s = float(schedule_timing.get("fetch_waiting_s", 0.0))
        sort_laxity_s = float(schedule_timing.get("sort_laxity_s", 0.0))
        tp_sync_order_s = float(schedule_timing.get("tp_sync_order_s", 0.0))
        select_batch_s = float(schedule_timing.get("select_batch_s", 0.0))
        update_state_s = float(schedule_timing.get("update_state_s", 0.0))
        batch_build_s = float(schedule_timing.get("batch_build_s", schedule_prepare_s))
        phase = getattr(batch, "dllm_batch_phase", "unknown")
        num_reqs = batch.batch_size()
        num_output_blocks = 0
        if result.next_token_ids:
            num_output_blocks = sum(
                1 for token_ids in result.next_token_ids if len(token_ids) > 0
            )

        per_req_phase = list(getattr(batch, "dllm_per_req_phase", []))
        per_req_exec_phase = list(getattr(batch, "dllm_per_req_exec_phase", []))
        per_req_effective_phase = list(
            getattr(batch, "dllm_per_req_effective_phase", [])
        )
        per_req_has_mask = list(getattr(batch, "dllm_per_req_has_mask", []))
        per_req_extend_input_len = list(
            getattr(batch, "dllm_per_req_extend_input_len", [])
        )
        per_req_block_steps = getattr(result, "dllm_block_steps", None)
        per_req_mode = getattr(result, "dllm_req_modes", None)
        if per_req_mode is None:
            per_req_mode = list(getattr(batch, "dllm_req_modes", []))
        else:
            per_req_mode = list(per_req_mode)
        if per_req_block_steps is None:
            per_req_block_steps = [None] * num_reqs
        else:
            per_req_block_steps = list(per_req_block_steps)
        if not per_req_block_steps and num_reqs > 0:
            per_req_block_steps = [0] * num_reqs
        elif len(per_req_block_steps) < num_reqs:
            per_req_block_steps.extend([None] * (num_reqs - len(per_req_block_steps)))
        elif len(per_req_block_steps) > num_reqs:
            per_req_block_steps = per_req_block_steps[:num_reqs]

        num_initial_prefill_reqs = per_req_phase.count(
            DllmReqPhase.QUEUING_PREFILL.value
        )
        num_staging_prefill_reqs = per_req_phase.count(
            DllmReqPhase.STAGING_PREFILL.value
        )
        num_initial_decode_reqs = per_req_phase.count(DllmReqPhase.QUEUING_DECODE.value)
        num_staging_decode_reqs = per_req_phase.count(DllmReqPhase.STAGING_DECODE.value)
        num_masked_reqs = sum(1 for value in per_req_has_mask if value)
        num_unmasked_reqs = sum(1 for value in per_req_has_mask if not value)
        is_mixed_mask_batch = num_masked_reqs > 0 and num_unmasked_reqs > 0

        if phase == "prefill":
            self.dllm_prefill_batch_time_sum += duration_s
            self.dllm_prefill_req_time_sum += duration_s * num_reqs
            self.dllm_prefill_req_count += num_reqs
        elif phase == "decode":
            self.dllm_decode_batch_time_sum += duration_s
            if num_output_blocks > 0:
                self.dllm_decode_block_time_sum += duration_s * num_output_blocks
                self.dllm_decode_block_count += num_output_blocks
        elif phase == "mixed_prefill_decode":
            self.dllm_mixed_prefill_decode_batch_time_sum += duration_s
            prefill_reqs = num_initial_prefill_reqs + num_staging_prefill_reqs
            if prefill_reqs > 0:
                self.dllm_prefill_req_time_sum += duration_s * prefill_reqs
                self.dllm_prefill_req_count += prefill_reqs
            if num_output_blocks > 0:
                self.dllm_decode_block_time_sum += duration_s * num_output_blocks
                self.dllm_decode_block_count += num_output_blocks

        avg_prefill_req_s = (
            self.dllm_prefill_req_time_sum / self.dllm_prefill_req_count
            if self.dllm_prefill_req_count > 0
            else None
        )
        avg_decode_block_s = (
            self.dllm_decode_block_time_sum / self.dllm_decode_block_count
            if self.dllm_decode_block_count > 0
            else None
        )
        avg_prefill_batch_s = (
            self.dllm_prefill_batch_time_sum / self.dllm_prefill_batch_count
            if self.dllm_prefill_batch_count > 0
            else None
        )
        avg_decode_batch_s = (
            self.dllm_decode_batch_time_sum / self.dllm_decode_batch_count
            if self.dllm_decode_batch_count > 0
            else None
        )
        dllm_waiting_queue_size = len(self.dllm_manager.waiting_queue)
        dllm_pending_next_round_reqs_size = len(
            self.dllm_manager.pending_next_round_reqs
        )
        dllm_admitted_reqs = len(
            {
                req.rid
                for req in (
                    self.dllm_manager.waiting_queue
                    + self.dllm_manager.pending_next_round_reqs
                )
            }
        )

        record = {
            "seq": getattr(batch, "dllm_batch_seq", None),
            "phase": phase,
            "phase_run_length": getattr(batch, "dllm_phase_run_length", None),
            "phase_switch_count": getattr(batch, "dllm_phase_switch_count", None),
            "rids": list(getattr(batch, "dllm_rids", [])),
            "per_req_phase": per_req_phase,
            "per_req_source_phase": per_req_phase,
            "per_req_exec_phase": per_req_exec_phase,
            "per_req_effective_phase": per_req_effective_phase,
            "per_req_mode": per_req_mode,
            "per_req_has_mask": per_req_has_mask,
            "per_req_extend_input_len": per_req_extend_input_len,
            "per_req_block_steps": per_req_block_steps,
            "raw_forward_calls": getattr(result, "dllm_raw_forward_calls", None),
            "num_reqs": num_reqs,
            "num_output_blocks": num_output_blocks,
            "num_initial_prefill_reqs": num_initial_prefill_reqs,
            "num_staging_prefill_reqs": num_staging_prefill_reqs,
            "num_initial_decode_reqs": num_initial_decode_reqs,
            "num_staging_decode_reqs": num_staging_decode_reqs,
            "num_masked_reqs": num_masked_reqs,
            "num_unmasked_reqs": num_unmasked_reqs,
            "is_mixed_mask_batch": is_mixed_mask_batch,
            "dllm_admission_window": self.dllm_config.admission_window,
            "dllm_max_running_requests": self.dllm_config.max_running_requests,
            "dllm_waiting_queue_size": dllm_waiting_queue_size,
            "dllm_pending_next_round_reqs_size": dllm_pending_next_round_reqs_size,
            "dllm_admitted_reqs": dllm_admitted_reqs,
            "duration_s": duration_s,
            "duration_ms": duration_s * 1000,
            "forward_result_s": forward_result_s,
            "forward_result_ms": forward_result_s * 1000,
            "schedule_total_s": schedule_total_s,
            "schedule_total_ms": schedule_total_s * 1000,
            "schedule_prepare_s": schedule_prepare_s,
            "schedule_prepare_ms": schedule_prepare_s * 1000,
            "fetch_waiting_s": fetch_waiting_s,
            "fetch_waiting_ms": fetch_waiting_s * 1000,
            "sort_laxity_s": sort_laxity_s,
            "sort_laxity_ms": sort_laxity_s * 1000,
            "tp_sync_order_s": tp_sync_order_s,
            "tp_sync_order_ms": tp_sync_order_s * 1000,
            "select_batch_s": select_batch_s,
            "select_batch_ms": select_batch_s * 1000,
            "update_state_s": update_state_s,
            "update_state_ms": update_state_s * 1000,
            "batch_build_s": batch_build_s,
            "batch_build_ms": batch_build_s * 1000,
            "prefill_batch_count": self.dllm_prefill_batch_count,
            "decode_batch_count": self.dllm_decode_batch_count,
            "mixed_prefill_decode_batch_count": (
                self.dllm_mixed_prefill_decode_batch_count
            ),
            "prefill_req_count": self.dllm_prefill_req_count,
            "decode_block_count": self.dllm_decode_block_count,
            "avg_prefill_req_s": avg_prefill_req_s,
            "avg_prefill_req_ms": (
                None if avg_prefill_req_s is None else avg_prefill_req_s * 1000
            ),
            "avg_prefill_batch_s": avg_prefill_batch_s,
            "avg_prefill_batch_ms": (
                None if avg_prefill_batch_s is None else avg_prefill_batch_s * 1000
            ),
            "avg_decode_block_s": avg_decode_block_s,
            "avg_decode_block_ms": (
                None if avg_decode_block_s is None else avg_decode_block_s * 1000
            ),
            "avg_decode_batch_s": avg_decode_batch_s,
            "avg_decode_batch_ms": (
                None if avg_decode_batch_s is None else avg_decode_batch_s * 1000
            ),
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _fetch_waiting_reqs(self: Scheduler):
        # Calculate how many requests can be added to DLLM manager
        max_dllm_capacity = self.dllm_config.admission_window - len(
            self.dllm_manager.waiting_queue
        )
        num_requests_to_add = min(max_dllm_capacity, len(self.waiting_queue))

        if num_requests_to_add > 0:
            requests_to_add = self.waiting_queue[:num_requests_to_add]
            if self.dllm_config.use_unified_traversal():
                now = time.perf_counter()
                for req in requests_to_add:
                    req.set_dllm_admission_time(now)
            self.dllm_manager.add_waiting_reqs(requests_to_add)
            self.waiting_queue = self.waiting_queue[num_requests_to_add:]

            # Lightweight prefix peek for new QUEUING_PREFILL requests so that
            # _sort_dllm_by_laxity() sees actual cache hits in compute_dllm_slack.
            # Only updates dllm_prefetched_prefix_len — does NOT touch fill_ids,
            # dllm_block_offset, or extend_input_len, so no scheduling state is
            # mutated and the normal init_next_round_input path stays intact.
            if self.dllm_config.use_lst() and not self.tree_cache.disable:
                for req in requests_to_add:
                    if req.dllm_phase == DllmReqPhase.QUEUING_PREFILL:
                        temp_ids = req.origin_input_ids
                        max_prefix_len = max(len(temp_ids) - 1, 0)
                        if max_prefix_len > 0:
                            match_result = self.tree_cache.match_prefix(
                                MatchPrefixParams(
                                    key=RadixKey(
                                        token_ids=temp_ids[:max_prefix_len],
                                        extra_key=req.extra_key,
                                    ),
                                )
                            )
                            req.dllm_prefetched_prefix_len = len(match_result.device_indices)

    def _sort_dllm_by_laxity(self: Scheduler) -> None:
        """Sort waiting_queue for unified-traversal modes.

        LST:  ascending slack (most urgent first).
        FCFS: ascending admission time (first arrived first served).
        Prefill-priority: no-op (order is irrelevant for that path).

        Skipped when queue_dirty is False — the ordering hasn't changed since
        the last sort (no new requests, no block completions).
        """
        if not self.dllm_manager.queue_dirty:
            return
        cfg = self.dllm_config
        if cfg.use_lst():
            now = time.perf_counter()
            self.dllm_manager.waiting_queue.sort(
                key=lambda req: req.compute_dllm_slack(now)
            )
        elif cfg.use_fcfs():
            self.dllm_manager.waiting_queue.sort(
                key=lambda req: req.dllm_admission_time or 0.0
            )
        self.dllm_manager.queue_dirty = False

    def _sync_dllm_lst_order_across_tp(self: Scheduler) -> None:
        """Use TP0's LST order on all TP ranks.

        LST slack depends on wall-clock time, cache-prefix observations, and EMA
        state, any of which can drift slightly across TP ranks.  The forward
        path requires every TP rank to schedule the same requests in the same
        order, so broadcast only the sorted request ids and keep local Req
        objects/resource checks unchanged.
        """
        if (
            not self.dllm_config.use_unified_traversal()
            or getattr(self, "tp_size", 1) == 1
        ):
            return

        ordered_rids = [req.rid for req in self.dllm_manager.waiting_queue]
        ordered_rids = broadcast_pyobj(
            ordered_rids,
            self.tp_group.rank,
            self.tp_cpu_group,
            src=self.tp_group.ranks[0],
        )

        req_by_rid = {req.rid: req for req in self.dllm_manager.waiting_queue}
        synced_queue = [req_by_rid[rid] for rid in ordered_rids if rid in req_by_rid]
        if len(synced_queue) != len(self.dllm_manager.waiting_queue):
            logger.warning(
                "DLM LST TP order sync saw mismatched waiting queues: "
                "rank=%s local=%s broadcast=%s",
                getattr(self, "tp_rank", None),
                len(self.dllm_manager.waiting_queue),
                len(ordered_rids),
            )
            synced_rids = {req.rid for req in synced_queue}
            synced_queue.extend(
                req
                for req in self.dllm_manager.waiting_queue
                if req.rid not in synced_rids
            )
        self.dllm_manager.waiting_queue = synced_queue

    def _should_skip_prefill(self: Scheduler) -> bool:
        """Check if DLLM prefill should be skipped."""
        if (
            self.running_batch.batch_is_full or not self.waiting_queue
        ) and self.dllm_manager.is_empty():
            return True

        running_bs = len(self.running_batch.reqs)
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.dllm_manager.is_empty()
            and not self.enable_priority_preemption
        ):
            self.running_batch.batch_is_full = True
            return True

        return False

    def _create_dllm_prefill_adder(self: Scheduler, running_bs: int) -> PrefillAdder:
        """Create a prefill adder configured for DLLM scheduling."""
        return PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            prefill_max_requests=self.server_args.prefill_max_requests,
            dllm_config=self.dllm_config,
        )

    def _compute_sola_pressure(self: Scheduler) -> tuple:
        """Return (p_ttfb, p_tpob): max normalized elapsed/SLO across waiting requests.

        P_TTFB: max over QUEUING requests (first_block=None) of elapsed/ttfb_slo.
                STAGING requests are excluded — they are already committed (KV
                allocated) and will be processed regardless of the batch-type decision.
        P_TPOB: max over inter-block-delay requests (first_block set, no active block)
                of elapsed_since_last_block/tpob_slo.
                Mid-block STAGING_DECODE (has_active_block=True) is excluded because
                it is already in progress and the scheduling decision cannot help it.
        Both values are in [0, 1]; 1.0 means the SLO deadline has been reached.
        """
        now = time.perf_counter()
        p_ttfb = 0.0
        p_tpob = 0.0
        cfg = self.dllm_config

        for req in self.dllm_manager.waiting_queue:
            if req.dllm_admission_time is None:
                continue
            slo_type = req.dllm_slo_type or "strict"

            if req.dllm_first_block_time is None:
                # TTFB pressure: only QUEUING requests contribute — STAGING requests
                # (STAGING_PREFILL, STAGING_DECODE first block) are already committed
                # (KV allocated) and will be processed regardless of the scheduling
                # decision, so they should not drive the admit-first signal.
                if req.dllm_phase not in (
                    DllmReqPhase.QUEUING_PREFILL,
                    DllmReqPhase.QUEUING_DECODE,
                ):
                    continue
                ttfb_slo = (
                    cfg.strict_ttfb_slo if slo_type == "strict" else cfg.release_ttfb_slo
                )
                if ttfb_slo and ttfb_slo > 0:
                    elapsed = now - req.dllm_admission_time
                    p_ttfb = max(p_ttfb, min(elapsed / ttfb_slo, 1.0))
            elif (
                not req.has_dllm_active_block()
                and req.dllm_last_block_time is not None
            ):
                # TPOB pressure: inter-block delay only — first block already emitted
                # and no active block in progress.  Mid-block STAGING_DECODE
                # (has_active_block=True) is excluded: it is already being denoised
                # and the decode_first decision cannot accelerate it.
                tpob_slo = (
                    cfg.strict_tpob_slo if slo_type == "strict" else cfg.release_tpob_slo
                )
                if tpob_slo and tpob_slo > 0:
                    elapsed = now - req.dllm_last_block_time
                    p_tpob = max(p_tpob, min(elapsed / tpob_slo, 1.0))

        return p_ttfb, p_tpob

    def _process_dllm_batches_sola(
        self: Scheduler, adder: PrefillAdder, decode_first: bool
    ) -> None:
        """SOLA-adapted two-group traversal driven by system-level pressure.

        Groups are split by TTFB/TPOB phase, matching SOLA's original semantics:
          ttfb_group: requests that have NOT yet emitted their first block
                      (dllm_first_block_time is None) — maps to AR prefill priority.
          tpob_group: requests that have already emitted ≥1 block
                      (dllm_first_block_time is not None) — maps to AR decode priority.

        decode_first=True  (P_TPOB > P_TTFB): tpob_group first, then ttfb_group.
        decode_first=False (P_TTFB >= P_TPOB): ttfb_group first, then tpob_group.
        Within each group requests are ordered by ascending slack (LST-style).
        The is_staging check inside the loop handles KV-memory routing per request.
        """
        scheduled_rids: Set[str] = {req.rid for req in adder.can_run_list}
        queuing_capacity_exhausted = False

        # Reuse the slack-sorted waiting_queue from _sort_dllm_by_laxity().
        # O(n) partition preserves the sorted order — no additional sort needed.
        ttfb_group: List[Req] = []
        tpob_group: List[Req] = []
        for r in self.dllm_manager.waiting_queue:
            if r.rid in scheduled_rids:
                continue
            if r.dllm_first_block_time is None:
                ttfb_group.append(r)
            else:
                tpob_group.append(r)

        ordered = (tpob_group + ttfb_group) if decode_first else (ttfb_group + tpob_group)

        for req in ordered:
            is_staging = req.dllm_phase in (
                DllmReqPhase.STAGING_PREFILL,
                DllmReqPhase.STAGING_DECODE,
            )

            if is_staging:
                req.dllm_scheduled_source_phase = req.dllm_phase
                self._restore_dllm_active_prefix_indices(req)
                res = adder.add_dllm_staging_req(req)
                req.dllm_scheduled_exec_phase = req.dllm_phase
                if res == AddReqResult.NO_TOKEN:
                    break  # KV memory exhausted
                scheduled_rids.add(req.rid)
            else:
                if queuing_capacity_exhausted:
                    continue

                running_bs = len(self.running_batch.reqs)
                if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                    self.running_batch.batch_is_full = True
                    if (
                        not self.enable_priority_preemption
                        or not adder.preempt_to_schedule(req, self.server_args)
                    ):
                        queuing_capacity_exhausted = True
                        continue

                req.dllm_scheduled_source_phase = req.dllm_phase
                req.init_next_round_input(self.tree_cache)
                req.dllm_scheduled_exec_phase = req.dllm_phase
                res = adder.add_one_req(
                    req,
                    has_chunked_req=True,
                    truncation_align_size=self.truncation_align_size,
                )
                if res != AddReqResult.CONTINUE:
                    if res == AddReqResult.NO_TOKEN:
                        self.running_batch.batch_is_full = True
                        break
                    queuing_capacity_exhausted = True
                else:
                    scheduled_rids.add(req.rid)

    def _process_dllm_batches(self: Scheduler, adder: PrefillAdder) -> ForwardMode:
        """Process DLLM batches.

        prefill: prefill-priority (original behavior, zero overhead).
        fcfs:    unified traversal in arrival-time order.
        lst:     unified traversal in ascending slack order.
        sola:    two-group traversal driven by system-level TTFB/TPOB pressure.
        """
        forward_mode = ForwardMode.DLLM_EXTEND

        if self.dllm_config.use_sola():
            p_ttfb, p_tpob = self._compute_sola_pressure()
            decode_first = p_tpob > p_ttfb
            logger.debug(
                "SOLA pressure: p_ttfb=%.3f p_tpob=%.3f decode_first=%s",
                p_ttfb, p_tpob, decode_first,
            )
            # Broadcast the binary decision from TP0 so every rank picks the
            # same traversal order despite minor wall-clock timing differences.
            if getattr(self, "tp_size", 1) > 1:
                decode_first = broadcast_pyobj(
                    [decode_first],
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )[0]
            self._process_dllm_batches_sola(adder, decode_first)
        elif self.dllm_config.use_unified_traversal():
            self._process_dllm_batches_unified(adder)
        else:
            # Original prefill-priority path (unchanged)
            prefill_reqs = self.dllm_manager.get_prefill_requests()
            prefill_result = AddReqResult.CONTINUE
            if prefill_reqs:
                prefill_result = self._process_batch_by_phase(
                    adder,
                    prefill_reqs,
                    DllmReqPhase.STAGING_PREFILL,
                    DllmReqPhase.QUEUING_PREFILL,
                )

            if prefill_result == AddReqResult.CONTINUE:
                scheduled_reqs = {req.rid for req in adder.can_run_list}
                decode_reqs = [
                    req
                    for req in self.dllm_manager.get_decode_requests()
                    if req.rid not in scheduled_reqs
                ]
                if decode_reqs:
                    self._process_batch_by_phase(
                        adder,
                        decode_reqs,
                        DllmReqPhase.STAGING_DECODE,
                        DllmReqPhase.QUEUING_DECODE,
                    )

        return forward_mode

    def _process_dllm_batches_unified(self: Scheduler, adder: PrefillAdder) -> None:
        """Unified batch selection: iterate waiting_queue in pre-sorted order.

        Used by both LST (slack order) and FCFS (arrival-time order).
        Staging requests (continuation of an active block) are added whenever KV
        memory allows, independent of request-count capacity, because they must
        finish the block they started.  Queuing requests (newly scheduled blocks)
        are gated by request-count capacity; when capacity is exhausted for new
        requests, queuing requests are skipped but the traversal continues so that
        urgent staging requests further down the sorted list are not blocked.
        """
        scheduled_rids: Set[str] = {req.rid for req in adder.can_run_list}
        # Once the batch is full for new requests, flip this flag to skip
        # subsequent queuing reqs without re-evaluating capacity each time.
        queuing_capacity_exhausted = False

        for req in self.dllm_manager.waiting_queue:
            if req.rid in scheduled_rids:
                continue

            is_staging = req.dllm_phase in (
                DllmReqPhase.STAGING_PREFILL,
                DllmReqPhase.STAGING_DECODE,
            )

            if is_staging:
                req.dllm_scheduled_source_phase = req.dllm_phase
                self._restore_dllm_active_prefix_indices(req)
                res = adder.add_dllm_staging_req(req)
                req.dllm_scheduled_exec_phase = req.dllm_phase
                if res == AddReqResult.NO_TOKEN:
                    break  # KV memory exhausted; no further requests can be added
                scheduled_rids.add(req.rid)
            else:
                # QUEUING_PREFILL or QUEUING_DECODE
                if queuing_capacity_exhausted:
                    continue  # Batch full for new reqs; still scan for staging reqs

                running_bs = len(self.running_batch.reqs)
                if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                    self.running_batch.batch_is_full = True
                    if (
                        not self.enable_priority_preemption
                        or not adder.preempt_to_schedule(req, self.server_args)
                    ):
                        queuing_capacity_exhausted = True
                        continue  # Skip this queuing req; keep scanning for staging
                    # Preemption succeeded; batch is no longer full

                req.dllm_scheduled_source_phase = req.dllm_phase
                req.init_next_round_input(self.tree_cache)
                req.dllm_scheduled_exec_phase = req.dllm_phase
                res = adder.add_one_req(
                    req,
                    has_chunked_req=True,
                    truncation_align_size=self.truncation_align_size,
                )
                if res != AddReqResult.CONTINUE:
                    if res == AddReqResult.NO_TOKEN:
                        self.running_batch.batch_is_full = True
                        break  # KV memory exhausted
                    queuing_capacity_exhausted = True
                else:
                    scheduled_rids.add(req.rid)

    def _process_batch_by_phase(
        self,
        adder: PrefillAdder,
        batch: List[Req],
        staging_phase: DllmReqPhase,
        incoming_phase: DllmReqPhase,
    ) -> AddReqResult:
        """Process a batch, separating staging and incoming requests."""
        res = AddReqResult.CONTINUE
        staging_reqs = [req for req in batch if req.dllm_phase == staging_phase]
        if staging_reqs:
            res = self.process_dllm_staging_reqs(adder, staging_reqs)
            if res != AddReqResult.CONTINUE:
                return res

        incoming_reqs = [req for req in batch if req.dllm_phase == incoming_phase]
        if incoming_reqs:
            res = self.process_dllm_incoming_reqs(adder, incoming_reqs)

        return res

    def _update_state_for_batch(
        self: Scheduler, can_run_list: List[Req], adder: PrefillAdder, running_bs: int
    ) -> None:
        """Update state for the batch."""

        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if can_run_list:
            self.dllm_manager.set_pending_next_round_reqs(can_run_list)
            self.dllm_manager.increment_chunked_count()

        self.adder = adder
        self.can_run_list = can_run_list
        self.running_bs = len(self.running_batch.reqs)

    def _create_dllm_batch(
        self: Scheduler, can_run_list: List[Req], forward_mode: ForwardMode
    ) -> ScheduleBatch:
        """Create and prepare a new DLLM batch."""
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            dllm_config=self.dllm_config,
        )
        new_batch.prepare_for_extend()
        new_batch.forward_mode = forward_mode
        new_batch.decoding_reqs = None

        # Record prefill stats for logging after forward
        from sglang.srt.observability.scheduler_metrics_mixin import PrefillStats

        new_batch.prefill_stats = PrefillStats.from_adder(
            self.adder, self.running_batch.reqs, self.enable_priority_scheduling
        )

        return new_batch

    def _commit_dllm_saved_block_prefix(self: Scheduler, req: Req) -> None:
        """Expose a KV-saved DLLM block as committed prefix for later blocks."""
        if req.req_pool_idx is None:
            return

        committed_len = len(req.fill_ids)
        if committed_len == 0:
            req.prefix_indices = self.req_to_token_pool.req_to_token.new_empty(
                (0,), dtype=self.req_to_token_pool.req_to_token.dtype
            )
            return

        req.prefix_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :committed_len
        ].detach().clone()

    def _restore_dllm_active_prefix_indices(self: Scheduler, req: Req) -> None:
        """Set prefix_indices to physical KV locations for an active DLLM block."""
        if not getattr(req, "has_dllm_active_block", lambda: False)():
            return

        prefix_len = getattr(req, "dllm_active_prefix_len", None)
        if prefix_len is None:
            return

        if prefix_len == 0:
            req.prefix_indices = self.req_to_token_pool.req_to_token.new_empty(
                (0,), dtype=self.req_to_token_pool.req_to_token.dtype
            )
        else:
            if req.req_pool_idx is None:
                return
            req.prefix_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :prefix_len
            ].detach().clone()

        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))

    def process_dllm_incoming_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        """Process incoming DLLM requests with resource allocation and preemption."""
        res = AddReqResult.CONTINUE
        for req in reqs:
            # Check if batch is full
            running_bs = len(self.running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True

            # Try preemption if batch is full
            if self.running_batch.batch_is_full:
                if (
                    not self.enable_priority_preemption
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    break

            # Snapshot the incoming phase before init_next_round_input converts it
            # into a staging phase for the actual executable block.
            req.dllm_scheduled_source_phase = req.dllm_phase
            req.init_next_round_input(self.tree_cache)
            req.dllm_scheduled_exec_phase = req.dllm_phase
            res = adder.add_one_req(
                req,
                has_chunked_req=True,
                truncation_align_size=self.truncation_align_size,
            )

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    self.running_batch.batch_is_full = True
                break

        return res

    def process_dllm_staging_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        """Process staging DLLM requests with resource allocation."""
        for req in reqs:
            req.dllm_scheduled_source_phase = req.dllm_phase
            self._restore_dllm_active_prefix_indices(req)
            res = adder.add_dllm_staging_req(req)
            req.dllm_scheduled_exec_phase = req.dllm_phase
            if res == AddReqResult.NO_TOKEN:
                return res

        return AddReqResult.CONTINUE


class DllmManager:
    """
    Manager for Diffusion LLM request scheduling.

    Maintains request sets:
    - waiting_queue: The requests waiting to be scheduled with max running requests limit
    - pending_next_round_reqs: Requests from the previous forward that need
      init_next_round_input() before they can be scheduled again
    """

    def __init__(self, dllm_config: Optional[DllmConfig] = None):
        self.dllm_config = dllm_config
        self.max_running_reqs = (
            dllm_config.max_running_requests if dllm_config is not None else 1
        )
        self.waiting_queue: List[Req] = []
        self.pending_next_round_reqs: List[Req] = []
        # Dirty flag: True when waiting_queue membership or request state has
        # changed since the last sort.  Avoids redundant O(n log n) sorts on
        # forward steps where nothing has changed.
        self.queue_dirty: bool = True

    def get_prefill_requests(self) -> List[Req]:
        """Get all prefill requests from waiting queue."""
        return [req for req in self.waiting_queue if req.is_dllm_prefill()]

    def get_decode_requests(self) -> List[Req]:
        """Get all decode requests from waiting queue."""
        return [req for req in self.waiting_queue if not req.is_dllm_prefill()]

    def add_waiting_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        """Add requests to waiting queue with redundancy check."""
        assert self.dllm_config is not None, "Diffusion LLM config is not set."

        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]

        # Check for duplicate request IDs
        if self._has_duplicate_reqs(reqs_to_add):
            raise RuntimeError("Redundant requests detected in dLLM requests.")

        self.waiting_queue.extend(reqs_to_add)
        self.queue_dirty = True

    def set_pending_next_round_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        """Remember requests that need next-round initialization."""
        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]
        self.pending_next_round_reqs = list(reqs_to_add)

    def _has_duplicate_reqs(self, reqs: List[Req]) -> bool:
        """Check if any request ID already exists in waiting queue."""
        existing_rids: Set[str] = {r.rid for r in self.waiting_queue}
        return any(req.rid in existing_rids for req in reqs)

    def any_pending_next_round_reqs(self) -> bool:
        """Check if any previous-forward requests need next-round initialization."""
        return (
            self.dllm_config is not None and len(self.pending_next_round_reqs) > 0
        )

    def is_empty(self) -> bool:
        """Check if both queues are empty or DLLM is not configured."""
        if self.dllm_config is None:
            return True
        return (
            len(self.waiting_queue) == 0
            and len(self.pending_next_round_reqs) == 0
        )

    def increment_chunked_count(self) -> None:
        """Increment chunked count for requests selected for the last forward."""
        for req in self.pending_next_round_reqs:
            req.is_chunked += 1

    def filter_finished_reqs(self) -> None:
        """Remove finished requests from both queues."""
        self.waiting_queue = [req for req in self.waiting_queue if not req.finished()]
        self.pending_next_round_reqs = [
            req for req in self.pending_next_round_reqs if not req.finished()
        ]

    def init_next_round(self) -> None:
        """Initialize previous-forward requests for next round and clear them."""
        for req in self.pending_next_round_reqs:
            req.init_next_round_input()
            if not req.finished() and all(
                waiting_req.rid != req.rid for waiting_req in self.waiting_queue
            ):
                self.waiting_queue.append(req)
                self.queue_dirty = True
        self.pending_next_round_reqs = []
