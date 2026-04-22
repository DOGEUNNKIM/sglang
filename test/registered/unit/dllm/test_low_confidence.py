import json
import tempfile
import types
import unittest

import torch

from sglang.srt.dllm.algorithm.low_confidence import LowConfidence
from sglang.srt.dllm.config import DllmConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class FakeForwardBatch:
    def __init__(self, input_ids: torch.Tensor, batch_size: int):
        self.input_ids = input_ids
        self.batch_size = batch_size
        self.dllm_req_modes = None
        self.dllm_active_starts = None
        self.dllm_active_block_steps = None


class FakeModelRunner:
    def __init__(self, logits_per_call):
        self.logits_per_call = logits_per_call
        self.call_count = 0

    def forward(self, forward_batch, pp_proxy_tensors=None):
        logits = self.logits_per_call[min(self.call_count, len(self.logits_per_call) - 1)]
        self.call_count += 1
        return types.SimpleNamespace(
            logits_output=types.SimpleNamespace(full_logits=logits),
            can_run_graph=False,
        )


class TestLowConfidenceForwardCounting(unittest.TestCase):
    def test_logs_raw_forward_calls_for_masked_block(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            config = DllmConfig(
                algorithm="LowConfidence",
                algorithm_config={"threshold": 0.95, "step_log_file": tmp.name},
                block_size=2,
                mask_id=0,
                max_running_requests=1,
            )
            algo = LowConfidence(config)
            forward_batch = FakeForwardBatch(
                input_ids=torch.tensor([0, 0], dtype=torch.long),
                batch_size=1,
            )
            logits_per_call = [
                torch.tensor([[0.0, 10.0, 0.0], [0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]),
                torch.tensor([[0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]),
            ]
            runner = FakeModelRunner(logits_per_call)

            (
                _,
                next_token_ids,
                can_run_cuda_graph,
                raw_forward_calls,
                block_steps,
                updated_ids,
                pending_kv_save,
                kv_saved,
                req_modes,
            ) = algo.run(runner, forward_batch)

        self.assertFalse(can_run_cuda_graph)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(next_token_ids[0].tolist(), [])
        self.assertEqual(updated_ids[0].tolist(), [1, 0])
        self.assertEqual(pending_kv_save, [False])
        self.assertEqual(kv_saved, [False])
        self.assertEqual(req_modes, ["unmask"])
        self.assertEqual(raw_forward_calls, 1)
        self.assertEqual(block_steps, [1])

        with open(tmp.name) as f:
            record = json.loads(f.read().strip())

        self.assertEqual(
            record,
            {
                "raw_forward_calls": 1,
                "unmask_steps": 1,
                "block_steps": [1],
                "final_block_steps": [],
                "req_modes": ["unmask"],
                "pending_kv_save": [False],
                "kv_saved": [False],
            },
        )

    def test_logs_fast_path_forward_call(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            config = DllmConfig(
                algorithm="LowConfidence",
                algorithm_config={"threshold": 0.95, "step_log_file": tmp.name},
                block_size=2,
                mask_id=0,
                max_running_requests=1,
            )
            algo = LowConfidence(config)
            forward_batch = FakeForwardBatch(
                input_ids=torch.tensor([1, 2], dtype=torch.long),
                batch_size=1,
            )
            runner = FakeModelRunner([torch.tensor([[0.0, 1.0], [1.0, 0.0]])])

            (
                _,
                next_token_ids,
                can_run_cuda_graph,
                raw_forward_calls,
                block_steps,
                updated_ids,
                pending_kv_save,
                kv_saved,
                req_modes,
            ) = algo.run(runner, forward_batch)

        self.assertFalse(can_run_cuda_graph)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(next_token_ids[0].tolist(), [])
        self.assertEqual(updated_ids, [None])
        self.assertEqual(pending_kv_save, [False])
        self.assertEqual(kv_saved, [False])
        self.assertEqual(req_modes, ["prefill"])
        self.assertEqual(raw_forward_calls, 1)
        self.assertEqual(block_steps, [0])

        with open(tmp.name) as f:
            record = json.loads(f.read().strip())

        self.assertEqual(
            record,
            {
                "raw_forward_calls": 1,
                "unmask_steps": 0,
                "block_steps": [0],
                "final_block_steps": [],
                "req_modes": ["prefill"],
                "pending_kv_save": [False],
                "kv_saved": [False],
            },
        )

    def test_mixed_kv_save_and_unmask_forward(self):
        config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={"threshold": 0.95},
            block_size=2,
            mask_id=0,
            max_running_requests=2,
        )
        algo = LowConfidence(config)
        forward_batch = FakeForwardBatch(
            input_ids=torch.tensor([1, 2, 0, 0], dtype=torch.long),
            batch_size=2,
        )
        forward_batch.dllm_req_modes = ["kv_save", "unmask"]
        forward_batch.dllm_active_starts = [0, 0]
        forward_batch.dllm_active_block_steps = [4, 0]
        runner = FakeModelRunner(
            [
                torch.tensor(
                    [
                        [0.0, 10.0, 0.0],
                        [0.0, 0.0, 10.0],
                        [0.0, 10.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                )
            ]
        )

        (
            _,
            next_token_ids,
            can_run_cuda_graph,
            raw_forward_calls,
            block_steps,
            updated_ids,
            pending_kv_save,
            kv_saved,
            req_modes,
        ) = algo.run(runner, forward_batch)

        self.assertFalse(can_run_cuda_graph)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(raw_forward_calls, 1)
        self.assertEqual(req_modes, ["kv_save", "unmask"])
        self.assertEqual(next_token_ids[0].tolist(), [1, 2])
        self.assertEqual(next_token_ids[1].tolist(), [])
        self.assertIsNone(updated_ids[0])
        self.assertEqual(updated_ids[1].tolist(), [1, 0])
        self.assertEqual(block_steps, [4, 1])
        self.assertEqual(pending_kv_save, [False, False])
        self.assertEqual(kv_saved, [True, False])
