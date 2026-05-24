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

    def test_decode_wait_tracks_tpob_interval_waits(self):
        req = _Req()

        req.record_dllm_block_emit_time(12.0)

        req.record_dllm_block_start_delay(0.2)
        req.mark_dllm_decode_batch_end(12.5)
        req.record_dllm_decode_batch_start(13.0)
        req.mark_dllm_decode_batch_end(13.4)
        req.record_dllm_decode_batch_start(14.0)
        req.record_dllm_completed_block_decode_wait()
        req.record_dllm_block_emit_time(15.0)

        self.assertAlmostEqual(req.get_dllm_decode_delay(), 0.2)
        self.assertAlmostEqual(req.get_dllm_decode_inter_batch_gap(), 0.55)
        self.assertAlmostEqual(req.get_dllm_decode_wait(), 1.3)
        self.assertEqual(req.dllm_decode_wait_count, 1)
        self.assertEqual(len(req.dllm_decode_wait_list), 1)
        self.assertAlmostEqual(req.dllm_decode_wait_list[0], 1.3)


if __name__ == "__main__":
    unittest.main()
