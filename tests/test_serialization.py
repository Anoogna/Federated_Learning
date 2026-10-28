"""Unit tests for wire serialization and communication telemetry tracking."""

import unittest
import torch
from src.communication.serialization import (
    compute_tensor_dense_bytes,
    compute_tensor_sparse_wire_bytes,
    serialize_adapter_delta,
)
from src.communication.metrics import CommunicationTracker


class TestSerializationAndMetrics(unittest.TestCase):
    def test_dense_wire_bytes(self):
        # 1000 float elements
        t = torch.randn(10, 100)
        fp32_bytes = compute_tensor_dense_bytes(t, fp16=False)
        fp16_bytes = compute_tensor_dense_bytes(t, fp16=True)

        # FP32: 1000 * 4 + 12 header = 4012
        self.assertEqual(fp32_bytes, 4012)
        # FP16: 1000 * 2 + 12 header = 2012
        self.assertEqual(fp16_bytes, 2012)

    def test_sparse_wire_bytes_scaling(self):
        shape = torch.Size([100, 100])  # 10,000 elements
        dense_fp16 = compute_tensor_dense_bytes(torch.empty(shape), fp16=True)

        # 100 non-zero elements (1% density, 99% sparse)
        sparse_bytes_1pct = compute_tensor_sparse_wire_bytes(shape, non_zero_count=100, fp16=True)
        # Must be drastically smaller than dense
        self.assertLess(sparse_bytes_1pct, dense_fp16 * 0.1)

        # 0 non-zero elements
        sparse_zero = compute_tensor_sparse_wire_bytes(shape, non_zero_count=0, fp16=True)
        self.assertLess(sparse_zero, 100)

    def test_serialize_adapter_delta(self):
        delta = {
            "layer.0.lora_A": torch.randn(8, 128),
            "layer.0.lora_B": torch.zeros(128, 8),  # 100% sparse
        }
        wire_bytes, dense_bytes, stats = serialize_adapter_delta(delta, fp16=True)
        self.assertLess(wire_bytes, dense_bytes)
        self.assertEqual(stats["layer.0.lora_B"]["non_zero_count"], 0)
        self.assertEqual(stats["layer.0.lora_B"]["sparsity_ratio"], 1.0)

    def test_communication_tracker(self):
        tracker = CommunicationTracker()
        rec = tracker.log_round(
            round_idx=1,
            client_uplink_bytes_list=[1000, 1000, 1000],
            client_dense_bytes_list=[10000, 10000, 10000],
            downlink_bytes_per_client=500,
        )
        self.assertEqual(rec.total_wire_bytes, 3000 + 1500)
        self.assertEqual(rec.dense_equivalent_bytes, 30000 + 1500)
        self.assertGreater(rec.round_compression_ratio, 6.0)

        summary = tracker.get_summary()
        self.assertEqual(summary["num_rounds"], 1)
        self.assertGreater(summary["overall_savings_percent"], 80.0)


if __name__ == "__main__":
    unittest.main()
