"""Unit tests for differential privacy mechanism and RDP accounting."""

import unittest
import torch
from src.privacy.dp_mechanism import (
    compute_global_l2_norm,
    clip_parameter_delta,
    add_gaussian_noise,
    apply_differential_privacy,
)
from src.privacy.privacy_accountant import PrivacyAccountant
from src.config import PrivacyConfig


class TestDifferentialPrivacy(unittest.TestCase):
    def test_l2_norm_clipping(self):
        # Tensor with norm = 5.0
        delta = {"w": torch.tensor([3.0, 4.0])}
        initial_norm = compute_global_l2_norm(delta)
        self.assertAlmostEqual(initial_norm, 5.0, places=4)

        # Clip to C = 2.0
        clipped, norm_before, was_clipped = clip_parameter_delta(delta, clipping_norm=2.0)
        clipped_norm = compute_global_l2_norm(clipped)

        self.assertTrue(was_clipped)
        self.assertAlmostEqual(clipped_norm, 2.0, places=4)
        # Direction should be preserved: [1.2, 1.6]
        self.assertAlmostEqual(float(clipped["w"][0].item()), 1.2, places=4)
        self.assertAlmostEqual(float(clipped["w"][1].item()), 1.6, places=4)

    def test_clip_unaffected_when_under_bound(self):
        delta = {"w": torch.tensor([0.5, 0.5])}
        clipped, _, was_clipped = clip_parameter_delta(delta, clipping_norm=2.0)
        self.assertFalse(was_clipped)
        self.assertTrue(torch.equal(delta["w"], clipped["w"]))

    def test_gaussian_noise_std(self):
        delta = {"w": torch.zeros(10000)}
        # Without dimension normalization: std = 1.0
        noisy_raw = add_gaussian_noise(delta, clipping_norm=1.0, noise_multiplier=1.0, normalize_by_dimension=False, seed=42)
        std_raw = float(torch.std(noisy_raw["w"]).item())
        self.assertAlmostEqual(std_raw, 1.0, places=1)

        # With dimension normalization: total L2 norm equals approx clipping_norm (1.0)
        noisy_norm = add_gaussian_noise(delta, clipping_norm=1.0, noise_multiplier=1.0, normalize_by_dimension=True, seed=42)
        total_norm = float(torch.norm(noisy_norm["w"]).item())
        self.assertAlmostEqual(total_norm, 1.0, places=1)

    def test_rdp_accountant(self):
        accountant = PrivacyAccountant(noise_multiplier=1.0, target_delta=1e-5)
        eps_1, _ = accountant.step()
        eps_2, _ = accountant.step()
        eps_3, _ = accountant.step()

        # Epsilon must strictly increase monotonically with rounds
        self.assertGreater(eps_2, eps_1)
        self.assertGreater(eps_3, eps_2)

        # Higher noise multiplier must yield strictly lower epsilon (stronger privacy)
        accountant_high_noise = PrivacyAccountant(noise_multiplier=2.0, target_delta=1e-5)
        eps_hn_1, _ = accountant_high_noise.step()
        eps_hn_2, _ = accountant_high_noise.step()

        self.assertLess(eps_hn_2, eps_2)


if __name__ == "__main__":
    unittest.main()
