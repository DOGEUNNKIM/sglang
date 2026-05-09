from types import SimpleNamespace
import unittest

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import ReqDllmMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _Req(ReqDllmMixin):
    def __init__(self):
        self.origin_input_ids = [1, 2]
        self.time_stats = SimpleNamespace(
            wait_queue_entry_time=10.0,
            scheduler_recv_time=9.0,
        )
        self.init_diffusion_llm(
            DllmConfig(
                algorithm="LowConfidence",
                algorithm_config={},
                block_size=4,
                mask_id=0,
                max_running_requests=1,
            )
        )


class TestDllmLatencyMetrics(unittest.TestCase):
    def test_ttfb_and_max_tpob(self):
        req = _Req()

        req.record_dllm_block_emit_time(12.0)
        self.assertEqual(req.get_dllm_ttfb(), 2.0)
        self.assertIsNone(req.get_dllm_tpob())

        req.record_dllm_block_emit_time(15.0)
        req.record_dllm_block_emit_time(21.0)

        self.assertEqual(req.dllm_output_block_count, 3)
        self.assertEqual(req.dllm_tpob_count, 2)
        self.assertEqual(req.get_dllm_tpob(), 6.0)
        self.assertEqual(req.get_dllm_mean_tpob(), 4.5)


if __name__ == "__main__":
    unittest.main()
