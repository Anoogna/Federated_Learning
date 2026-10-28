"""End-to-end integration smoke test running a full multi-round federated training loop."""

import os
import unittest
from src.config import ExperimentConfig
from src.federated.training_loop import run_federated_training


class TestEndToEndSmoke(unittest.TestCase):
    def test_full_federated_smoke_run(self):
        """Execute a rapid 2-round federated training run on CPU."""
        config_path = "configs/smoke_test_cpu.yaml"
        self.assertTrue(os.path.exists(config_path), f"Config file not found: {config_path}")

        config = ExperimentConfig.from_yaml(config_path)
        # Override to ensure it finishes in under 10 seconds
        config.federated.num_rounds = 2
        config.data.num_samples = 30
        config.output_dir = "outputs/test_smoke"

        results = run_federated_training(config)

        # Assertions
        self.assertIn("final_test_eval", results)
        self.assertIn("history", results)
        self.assertEqual(len(results["history"]), 3)  # Round 0, 1, 2

        # Check loss and perplexity validity
        final_loss = results["final_test_eval"]["loss"]
        final_ppl = results["final_test_eval"]["perplexity"]
        self.assertGreater(final_loss, 0.0)
        self.assertGreater(final_ppl, 1.0)
        self.assertFalse(float(final_loss) != float(final_loss))  # Not NaN

        # Check communication tracking
        comm = results["communication_summary"]
        self.assertGreater(comm["cumulative_total_wire_bytes"], 0)
        self.assertGreater(comm["overall_compression_ratio"], 1.0)

        # Check differential privacy
        self.assertTrue(config.privacy.enabled)
        self.assertGreater(results["privacy_summary"]["spent_epsilon"], 0.0)

        # Check artifact persistence
        expected_json = os.path.join(config.output_dir, f"{config.experiment_id}_results.json")
        self.assertTrue(os.path.exists(expected_json))


if __name__ == "__main__":
    unittest.main()
