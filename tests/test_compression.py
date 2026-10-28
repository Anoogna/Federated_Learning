"""Unit tests for fixed and adaptive communication compression algorithms."""

import unittest
import torch
from src.compression.fixed_compression import FixedTopKCompressor, topk_sparsify_tensor
from src.compression.adaptive_compression import AdaCompCompressor


class TestCompression(unittest.TestCase):
    def test_topk_sparsify_tensor(self):
        # 100-element tensor with 10 prominent spikes
        t = torch.zeros(100)
        t[0:10] = 5.0
        sparse_t, mask = topk_sparsify_tensor(t, k_ratio=0.10)

        self.assertEqual(int(mask.sum().item()), 10)
        self.assertTrue(torch.all(sparse_t[0:10] == 5.0))
        self.assertTrue(torch.all(sparse_t[10:] == 0.0))

    def test_fixed_topk_error_feedback(self):
        compressor = FixedTopKCompressor(k_ratio=0.5, error_feedback=True)
        # Step 1: tensor with 4 coordinates [4, 3, 2, 1]
        delta_1 = {"w": torch.tensor([4.0, 3.0, 2.0, 1.0])}
        comp_1, telem_1 = compressor.compress(delta_1, round_idx=0, client_id=1)
        # Should keep top 2: [4.0, 3.0, 0.0, 0.0]
        self.assertTrue(torch.equal(comp_1["w"], torch.tensor([4.0, 3.0, 0.0, 0.0])))
        # Error buffer should retain [0.0, 0.0, 2.0, 1.0]
        err = compressor.error_buffers[1]["w"]
        self.assertTrue(torch.equal(err, torch.tensor([0.0, 0.0, 2.0, 1.0])))

        # Step 2: new delta is zeros; error feedback should now transmit previously omitted elements
        delta_2 = {"w": torch.tensor([0.0, 0.0, 0.0, 0.0])}
        comp_2, telem_2 = compressor.compress(delta_2, round_idx=1, client_id=1)
        self.assertTrue(torch.equal(comp_2["w"], torch.tensor([0.0, 0.0, 2.0, 1.0])))

    def test_adacomp_round_decay(self):
        compressor = AdaCompCompressor(min_k_ratio=0.05, max_k_ratio=0.30, error_feedback=False)
        d = 1000
        delta = {"w": torch.randn(d)}

        # Round 0 (early exploration): ratio near max_k_ratio (0.30)
        _, telem_early = compressor.compress(delta, round_idx=0, max_rounds=10)
        # Round 10 (late convergence): ratio near min_k_ratio (0.05)
        _, telem_late = compressor.compress(delta, round_idx=10, max_rounds=10)

        self.assertGreater(telem_early["total_retained"], telem_late["total_retained"])
        self.assertGreater(telem_late["effective_sparsity"], telem_early["effective_sparsity"])

    def test_adacomp_bounds_enforcement(self):
        compressor = AdaCompCompressor(min_k_ratio=0.02, max_k_ratio=0.20, error_feedback=False)
        # Extreme high norm delta
        huge_delta = {"w": torch.randn(1000) * 1000.0}
        _, telem = compressor.compress(huge_delta, round_idx=5, max_rounds=10, norm_bound=1.0)

        retained = telem["total_retained"]
        # Must not exceed max_k_ratio (20% of 1000 = 200)
        self.assertLessEqual(retained, 200)
        # Must not drop below min_k_ratio (2% of 1000 = 20)
        self.assertGreaterEqual(retained, 20)


if __name__ == "__main__":
    unittest.main()
