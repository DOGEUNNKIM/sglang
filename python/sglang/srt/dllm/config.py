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
        ttfb_slo: float | None = None,
        tpob_slo: float | None = None,
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
        self.ttfb_slo = ttfb_slo
        self.tpob_slo = tpob_slo
        self.prefill_forward_time_s = prefill_forward_time_s
        self.decode_forward_time_s = decode_forward_time_s
        self.expected_unmask_per_forward = max(expected_unmask_per_forward, 1e-6)

    def use_lst(self) -> bool:
        """True if Least Slack Time scheduling is enabled."""
        return self.ttfb_slo is not None or self.tpob_slo is not None

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
            "SDARForCausalLM": {"block_size": 4, "mask_id": 151669},
            "SDARMoeForCausalLM": {"block_size": 4, "mask_id": 151669},
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

            # Parse common algorithm configurations
            block_size = algorithm_config.get("block_size", block_size)

        admission_window = int(
            algorithm_config.get("dllm_admission_window", max_running_requests)
        )
        #if admission_window < max_running_requests:
        #    admission_window = max_running_requests

        ttfb_slo_raw = algorithm_config.get("ttfb_slo", None)
        tpob_slo_raw = algorithm_config.get("tpob_slo", None)
        ttfb_slo = float(ttfb_slo_raw) if ttfb_slo_raw is not None else None
        tpob_slo = float(tpob_slo_raw) if tpob_slo_raw is not None else None

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
            ttfb_slo=ttfb_slo,
            tpob_slo=tpob_slo,
            prefill_forward_time_s=prefill_forward_time_s,
            decode_forward_time_s=decode_forward_time_s,
            expected_unmask_per_forward=expected_unmask_per_forward,
        )
