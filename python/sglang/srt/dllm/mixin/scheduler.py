from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, List, Optional, Set, Union

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler


class SchedulerDllmMixin:
    def init_diffusion_llm(self: Scheduler):
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
        self.dllm_prefill_req_time_sum = 0.0
        self.dllm_prefill_req_count = 0
        self.dllm_prefill_batch_time_sum = 0.0
        self.dllm_decode_block_time_sum = 0.0
        self.dllm_decode_block_count = 0
        self.dllm_decode_batch_time_sum = 0.0

    def get_new_batch_dllm(self: Scheduler) -> Optional[ScheduleBatch]:
        """Generate a new batch for DLLM (Diffusion LLM) scheduling."""
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
        self._fetch_waiting_reqs()

        # Process batches
        forward_mode = self._process_dllm_batches(adder)

        can_run_list = adder.can_run_list
        if not can_run_list:
            return None

        # Record metrics and update state
        set_time_batch(can_run_list, "set_forward_entry_time")
        self._update_state_for_batch(can_run_list, adder, running_bs)

        # Create and prepare batch
        schedule_start_time = time.perf_counter()
        batch_phase = self._get_dllm_batch_phase(can_run_list)
        new_batch = self._create_dllm_batch(can_run_list, forward_mode)
        self._annotate_dllm_batch_metrics(
            new_batch,
            batch_phase=batch_phase,
            schedule_start_time=schedule_start_time,
        )
        return new_batch

    def process_batch_result_dllm(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        if result.next_token_ids:
            block_emit_time = time.perf_counter()
            self.token_to_kv_pool_allocator.free_group_begin()

            for idx in range(batch.batch_size()):
                req = batch.reqs[idx]

                next_token_ids = result.next_token_ids[idx].tolist()
                new_tokens = len(next_token_ids)
                if new_tokens == 0:
                    continue

                req.fill_ids[-new_tokens:] = next_token_ids[:]
                self.num_generated_tokens += new_tokens
                req.record_dllm_block_emit_time(block_emit_time)

                req.output_ids.extend(next_token_ids)
                req.check_finished(new_accepted_len=new_tokens)

                if req.finished():
                    release_kv_cache(req, self.tree_cache)
                    req.time_stats.set_completion_time()
                    self._maybe_log_dllm_request_latency(req)

            self.stream_output(batch.reqs, batch.return_logprob)
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
        record = {
            "rid": req.rid,
            "input_len": len(req.origin_input_ids),
            "output_len": len(req.output_ids),
            "block_size": self.dllm_config.block_size,
            "num_output_blocks": req.dllm_output_block_count,
            "ttfb_s": ttfb,
            "ttfb_ms": None if ttfb is None else ttfb * 1000,
            # TPOB is the mean of adjacent output-block intervals.
            "tpob_s": tpob,
            "tpob_ms": None if tpob is None else tpob * 1000,
            "tpob_count": req.dllm_tpob_count,
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
        if any(req.is_dllm_prefill() for req in reqs):
            return "prefill"
        return "decode"

    def _annotate_dllm_batch_metrics(
        self: Scheduler,
        batch: ScheduleBatch,
        batch_phase: str,
        schedule_start_time: float,
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
        else:
            self.dllm_decode_batch_count += 1

        batch.dllm_batch_seq = self.dllm_batch_seq
        batch.dllm_batch_phase = batch_phase
        batch.dllm_phase_run_length = self.dllm_phase_run_length
        batch.dllm_phase_switch_count = self.dllm_phase_switch_count
        batch.dllm_schedule_start_time = schedule_start_time
        batch.dllm_forward_start_time = time.perf_counter()
        if self._get_dllm_batch_metric_log_file() is not None:
            self._annotate_dllm_per_req_metrics(batch)
        self.dllm_batch_seq += 1

    def _annotate_dllm_per_req_metrics(self: Scheduler, batch: ScheduleBatch) -> None:
        """Snapshot request-level DLLM metadata before forward mutates inputs."""
        mask_id = self.dllm_config.mask_id
        input_ids = getattr(batch, "input_ids", None)
        flat_input_ids = (
            input_ids.detach().cpu().tolist() if input_ids is not None else []
        )

        rids = []
        phases = []
        exec_phases = []
        effective_phases = []
        has_mask = []
        extend_input_lens = []
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
            chunk = flat_input_ids[offset : offset + extend_len]
            if len(chunk) < extend_len:
                prefix_len = len(getattr(req, "prefix_indices", []))
                chunk = req.fill_ids[prefix_len : prefix_len + extend_len]
            req_has_mask = mask_id in chunk
            has_mask.append(req_has_mask)
            effective_phases.append("decode" if req_has_mask else "prefill")
            offset += extend_len

        batch.dllm_rids = rids
        batch.dllm_per_req_phase = phases
        batch.dllm_per_req_exec_phase = exec_phases
        batch.dllm_per_req_effective_phase = effective_phases
        batch.dllm_per_req_has_mask = has_mask
        batch.dllm_per_req_extend_input_len = extend_input_lens

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
        forward_result_s = end_time - start_time
        duration_s = end_time - schedule_start_time
        schedule_prepare_s = start_time - schedule_start_time
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
            DllmReqPhase.INCOMING_PREFILL.value
        )
        num_staging_prefill_reqs = per_req_phase.count(
            DllmReqPhase.STAGING_PREFILL.value
        )
        num_initial_decode_reqs = per_req_phase.count(DllmReqPhase.INCOMING_DECODE.value)
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
            "duration_s": duration_s,
            "duration_ms": duration_s * 1000,
            "forward_result_s": forward_result_s,
            "forward_result_ms": forward_result_s * 1000,
            "schedule_prepare_s": schedule_prepare_s,
            "schedule_prepare_ms": schedule_prepare_s * 1000,
            "prefill_batch_count": self.dllm_prefill_batch_count,
            "decode_batch_count": self.dllm_decode_batch_count,
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
        max_dllm_capacity = self.dllm_config.max_running_requests - len(
            self.dllm_manager.waiting_queue
        )
        num_requests_to_add = min(max_dllm_capacity, len(self.waiting_queue))

        if num_requests_to_add > 0:
            requests_to_add = self.waiting_queue[:num_requests_to_add]
            self.dllm_manager.add_waiting_reqs(requests_to_add)
            self.waiting_queue = self.waiting_queue[num_requests_to_add:]

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

    def _process_dllm_batches(self: Scheduler, adder: PrefillAdder) -> ForwardMode:
        """Process prefill or decode batches for DLLM."""
        forward_mode = ForwardMode.DLLM_EXTEND

        # Try prefill batch first
        prefill_reqs = self.dllm_manager.get_prefill_requests()
        if prefill_reqs:
            self._process_batch_by_phase(
                adder,
                prefill_reqs,
                DllmReqPhase.STAGING_PREFILL,
                DllmReqPhase.INCOMING_PREFILL,
            )
        else:
            # Fall back to decode batch
            decode_reqs = self.dllm_manager.get_decode_requests()
            self._process_batch_by_phase(
                adder,
                decode_reqs,
                DllmReqPhase.STAGING_DECODE,
                DllmReqPhase.INCOMING_DECODE,
            )

        return forward_mode

    def _process_batch_by_phase(
        self,
        adder: PrefillAdder,
        batch: List[Req],
        staging_phase: DllmReqPhase,
        incoming_phase: DllmReqPhase,
    ) -> None:
        """Process a batch, separating staging and incoming requests."""
        staging_reqs = [req for req in batch if req.dllm_phase == staging_phase]
        if staging_reqs:
            staging_result = self.process_dllm_staging_reqs(adder, staging_reqs)
            if staging_result != AddReqResult.CONTINUE:
                return

        incoming_reqs = [req for req in batch if req.dllm_phase == incoming_phase]
        if incoming_reqs:
            self.process_dllm_incoming_reqs(adder, incoming_reqs)

    def _update_state_for_batch(
        self: Scheduler, can_run_list: List[Req], adder: PrefillAdder, running_bs: int
    ) -> None:
        """Update state for the batch."""

        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if can_run_list:
            self.dllm_manager.add_staging_reqs(can_run_list)
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
            res = adder.add_dllm_staging_req(req)
            req.dllm_scheduled_exec_phase = req.dllm_phase
            if res == AddReqResult.NO_TOKEN:
                return res

        return AddReqResult.CONTINUE


class DllmManager:
    """
    Manager for Diffusion LLM request scheduling.

    Maintains two queues:
    - waiting_queue: The requests waiting to be scheduled with max running requests limit
    - staging_queue: Requests allocated resources by PrefillAdder
    """

    def __init__(self, dllm_config: Optional[DllmConfig] = None):
        self.dllm_config = dllm_config
        self.max_running_reqs = (
            dllm_config.max_running_requests if dllm_config is not None else 1
        )
        self.waiting_queue: List[Req] = []
        self.staging_queue: List[Req] = []

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

    def add_staging_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        """Add requests to staging queue (allocated by PrefillAdder)."""
        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]
        self.staging_queue.extend(reqs_to_add)

    def _has_duplicate_reqs(self, reqs: List[Req]) -> bool:
        """Check if any request ID already exists in waiting queue."""
        existing_rids: Set[str] = {r.rid for r in self.waiting_queue}
        return any(req.rid in existing_rids for req in reqs)

    def any_staging_reqs(self) -> bool:
        """Check if there are requests in staging queue."""
        return self.dllm_config is not None and len(self.staging_queue) > 0

    def is_empty(self) -> bool:
        """Check if both queues are empty or DLLM is not configured."""
        if self.dllm_config is None:
            return True
        return len(self.waiting_queue) == 0

    def increment_chunked_count(self) -> None:
        """Increment chunked count for all staging requests."""
        for req in self.staging_queue:
            req.is_chunked += 1

    def filter_finished_reqs(self) -> None:
        """Remove finished requests from both queues."""
        self.waiting_queue = [req for req in self.waiting_queue if not req.finished()]
        self.staging_queue = [req for req in self.staging_queue if not req.finished()]

    def init_next_round(self) -> None:
        """Initialize staging requests for next round and clear staging queue."""
        for req in self.staging_queue:
            req.init_next_round_input()
        self.staging_queue = []
