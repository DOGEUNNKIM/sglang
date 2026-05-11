import math
from typing import Any

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs


class DllmConfig:
    def __init__(
        self,
        algorithm: str,
        algorithm_config: dict[str, Any],
        block_size: int,
        mask_id: int,
        max_running_requests: int,
        admission_window: int | None = None,
        strict_ttfb_slo: float | None = None,
        strict_tpob_slo: float | None = None,
        release_ttfb_slo: float | None = None,
        release_tpob_slo: float | None = None,
        strict_prob: float = 0.5,
        scheduler_mode: str = "prefill",
        prefill_forward_time_s: float = 0.030,
        decode_forward_time_s: float = 0.030,
        expected_unmask_per_forward: float = 1.0,
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.block_size = block_size
        self.mask_id = mask_id
        self.max_running_requests = max_running_requests
        self.admission_window = admission_window or max_running_requests
        self.strict_ttfb_slo = strict_ttfb_slo
        self.strict_tpob_slo = strict_tpob_slo
        self.release_ttfb_slo = release_ttfb_slo
        self.release_tpob_slo = release_tpob_slo
        self.strict_prob = max(0.0, min(1.0, strict_prob))
        self.scheduler_mode = scheduler_mode  # "prefill" | "fcfs" | "lst"
        self.prefill_forward_time_s = prefill_forward_time_s
        self.decode_forward_time_s = decode_forward_time_s
        self.expected_unmask_per_forward = max(expected_unmask_per_forward, 1e-6)

        self.bellman_alpha0: float = max(
            0.0,
            float(algorithm_config.get("bellman_alpha0", 0.1)),
        )
        self.bellman_alpha_min: float = min(
            1.0,
            max(0.0, float(algorithm_config.get("bellman_alpha_min", 0.03))),
        )
        self.bellman_alpha: float = max(self.bellman_alpha_min, self.bellman_alpha0)

        # V[r] = estimated forward passes to finish a block with r masks remaining.
        # Initialized to r+1; updated online via adaptive stochastic median.
        self.decode_forwards_by_remaining: list[float] = [
            float(remaining + 1)
            for remaining in range(self.block_size + 1)
        ]
        self._update_counts: list[int] = [0] * (self.block_size + 1)

        self.decode_safety_factor: float = float(
            algorithm_config.get("decode_safety_factor", 1.0)
        )

        self.bellman_log_file: str | None = algorithm_config.get("bellman_log_file") or None
        self.bellman_block_count: int = 0

        if self.bellman_log_file:
            import json, time as _time
            with open(self.bellman_log_file, "w") as _bf:
                _bf.write(json.dumps({
                    "block": 0,
                    "time": _time.perf_counter(),
                    "table": list(self.decode_forwards_by_remaining),
                    "traj": [],
                }) + "\n")

    def _clamp_remaining_masks(self, remaining_masked_tokens: int | None) -> int:
        if remaining_masked_tokens is None:
            return self.block_size
        return max(0, min(self.block_size, int(remaining_masked_tokens)))

    def update_decode_forwards_estimate(
        self,
        remaining: int | None,
        observed_steps: float,
    ) -> None:
        """Stochastic median update: step = max(0.1, 2/sqrt(n)) per r, skips r=0."""
        r = self._clamp_remaining_masks(remaining)
        if r <= 0:
            return
        self._update_counts[r] += 1
        step = max(0.1, 2.0 / math.sqrt(self._update_counts[r]))
        if self.decode_forwards_by_remaining[r] > observed_steps:
            self.decode_forwards_by_remaining[r] -= step
        else:
            self.decode_forwards_by_remaining[r] += step

    def estimate_decode_forwards(self, remaining_masked_tokens: int) -> int:
        """Return ceil(safety_factor * V[remaining]), minimum 1."""
        remaining = self._clamp_remaining_masks(remaining_masked_tokens)
        if remaining <= 0:
            return 0
        estimated_forwards = self.decode_forwards_by_remaining[remaining]
        return max(1, math.ceil(self.decode_safety_factor * estimated_forwards))

    def use_lst(self) -> bool:
        """True if Least Slack Time scheduling is active."""
        return self.scheduler_mode == "lst" and self.strict_ttfb_slo is not None

    def use_fcfs(self) -> bool:
        """True if First-Come First-Served scheduling is active."""
        return self.scheduler_mode == "fcfs"

    def use_sola(self) -> bool:
        """True if SOLA-adapted pressure-based scheduling is active."""
        return self.scheduler_mode == "sola"

    def use_ttfb(self) -> bool:
        """True if TTFB-first scheduling is active.

        Prioritises requests by ascending remaining compute before first block
        emission.  TPOB-phase requests (first block already emitted) are placed
        last and only scheduled when capacity remains.
        """
        return self.scheduler_mode == "ttfb"

    def use_unified_traversal(self) -> bool:
        """True when the non-prefill-priority traversal path is used (LST, FCFS, SOLA, or TTFB)."""
        return self.use_lst() or self.use_fcfs() or self.use_sola() or self.use_ttfb()

    @staticmethod
    def from_server_args(
        server_args: ServerArgs,
    ):
        if server_args.dllm_algorithm is None:
            return None

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
            model_revision=server_args.revision,
        )
        DLLM_PARAMS = {
            "LLaDA2MoeModelLM": {"block_size": 32, "mask_id": 156895},
            "SDARForCausalLM": {"block_size": 16, "mask_id": 151669},
            "SDARMoeForCausalLM": {"block_size": 16, "mask_id": 151669},
        }

        arch = model_config.hf_config.architectures[0]
        if arch in DLLM_PARAMS:
            params = DLLM_PARAMS[arch]
            block_size = params["block_size"]
            mask_id = params["mask_id"]
        else:
            raise RuntimeError(f"Unknown diffusion LLM: {arch}")

        max_running_requests = (
            1
            if server_args.max_running_requests is None
            else server_args.max_running_requests
        )

        algorithm_config = {}
        if server_args.dllm_algorithm_config is not None:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "Please install PyYAML to use YAML config files. "
                    "`pip install pyyaml`"
                )
            with open(server_args.dllm_algorithm_config, "r") as f:
                algorithm_config = yaml.safe_load(f) or {}

            block_size = algorithm_config.get("block_size", block_size)

        admission_window = int(
            algorithm_config.get("dllm_admission_window", max_running_requests)
        )
        #if admission_window < max_running_requests:
        #    admission_window = max_running_requests

        def _parse_slo(key: str) -> float | None:
            raw = algorithm_config.get(key)
            return float(raw) if raw is not None else None

        strict_ttfb_slo = _parse_slo("strict_ttfb_slo")
        strict_tpob_slo = _parse_slo("strict_tpob_slo")
        release_ttfb_slo = _parse_slo("release_ttfb_slo")
        release_tpob_slo = _parse_slo("release_tpob_slo")
        strict_prob = float(algorithm_config.get("strict_prob", 0.5))
        # scheduler_mode takes precedence; fall back to legacy lst_enabled field.
        raw_mode = algorithm_config.get("scheduler_mode", None)
        if raw_mode is not None:
            scheduler_mode = str(raw_mode)
        else:
            lst_enabled = bool(algorithm_config.get("lst_enabled", True))
            scheduler_mode = "lst" if lst_enabled else "prefill"

        # forward_time_s is a shared fallback; prefill/decode can be set separately.
        forward_time_s = float(algorithm_config.get("forward_time_s", 0.030))
        prefill_forward_time_s = float(
            algorithm_config.get("prefill_forward_time_s", forward_time_s)
        )
        decode_forward_time_s = float(
            algorithm_config.get("decode_forward_time_s", forward_time_s)
        )
        expected_unmask_per_forward = float(
            algorithm_config.get("expected_unmask_per_forward", 1.0)
        )

        return DllmConfig(
            algorithm=server_args.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            mask_id=mask_id,
            max_running_requests=max_running_requests,
            admission_window=admission_window,
            strict_ttfb_slo=strict_ttfb_slo,
            strict_tpob_slo=strict_tpob_slo,
            release_ttfb_slo=release_ttfb_slo,
            release_tpob_slo=release_tpob_slo,
            strict_prob=strict_prob,
            scheduler_mode=scheduler_mode,
            prefill_forward_time_s=prefill_forward_time_s,
            decode_forward_time_s=decode_forward_time_s,
            expected_unmask_per_forward=expected_unmask_per_forward,
        )
