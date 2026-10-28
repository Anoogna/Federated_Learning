"""Unit tests for dataset loading, partitioning, and heterogeneity quantification."""

import unittest
from src.data.load_dataset import generate_synthetic_instruction_dataset, load_dolly_subset
from src.data.partition_clients import (
    split_train_val_test,
    partition_iid,
    partition_non_iid_dirichlet,
    compute_heterogeneity_metric,
)


class TestDataPartitioning(unittest.TestCase):
    def setUp(self):
        # Generate 100 deterministic test samples
        self.samples = generate_synthetic_instruction_dataset(num_samples=100, seed=42)

    def test_split_no_leakage(self):
        """Verify train, val, and test sets are mutually disjoint and cover all data."""
        train, val, test = split_train_val_test(
            self.samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
        )
        self.assertEqual(len(train), 70)
        self.assertEqual(len(val), 15)
        self.assertEqual(len(test), 15)

        train_instructions = {x["instruction"] for x in train}
        val_instructions = {x["instruction"] for x in val}
        test_instructions = {x["instruction"] for x in test}

        # Assert no overlap
        self.assertTrue(train_instructions.isdisjoint(val_instructions))
        self.assertTrue(train_instructions.isdisjoint(test_instructions))
        self.assertTrue(val_instructions.isdisjoint(test_instructions))

    def test_partition_iid_balance(self):
        """Verify IID partitioning yields equal allocations and preserves all samples."""
        train, _, _ = split_train_val_test(self.samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        client_data = partition_iid(train, num_clients=3, seed=42)

        self.assertEqual(len(client_data), 3)
        total_allocated = sum(len(c) for c in client_data.values())
        self.assertEqual(total_allocated, len(train))

        # Check balance: max difference between clients is at most 1
        client_lens = [len(c) for c in client_data.values()]
        self.assertLessEqual(max(client_lens) - min(client_lens), 1)

    def test_partition_non_iid_heterogeneity(self):
        """Verify Dirichlet Non-IID creates measurable category heterogeneity."""
        train, _, _ = split_train_val_test(self.samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

        # IID partitioning
        iid_clients = partition_iid(train, num_clients=3, seed=42)
        iid_metric = compute_heterogeneity_metric(iid_clients)

        # Non-IID Dirichlet with alpha=0.2 (strong skew)
        non_iid_clients = partition_non_iid_dirichlet(train, num_clients=3, alpha=0.2, seed=42)
        non_iid_metric = compute_heterogeneity_metric(non_iid_clients)

        # Total allocated samples must match
        self.assertEqual(sum(len(c) for c in non_iid_clients.values()), len(train))

        # Non-IID TVD must be strictly greater than IID TVD
        self.assertGreater(non_iid_metric["mean_tvd"], iid_metric["mean_tvd"])
        self.assertTrue(non_iid_metric["is_heterogeneous"])


if __name__ == "__main__":
    unittest.main()
