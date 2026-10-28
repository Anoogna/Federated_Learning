"""Client dataset partitioning and heterogeneity quantification module.

Splits data into train/val/test without leakage, partitions client training data using
IID or Dirichlet Non-IID distributions, and computes Total Variation Distance / EMD
to mathematically quantify data heterogeneity across federated clients.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Any, Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def split_train_val_test(
    data: List[Dict[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Deterministically split data into train, val, and test partitions before client allocation."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    test_data = [data[i] for i in test_idx]

    logger.info(
        f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test "
        f"(total: {n_total})"
    )
    return train_data, val_data, test_data


def partition_iid(
    train_data: List[Dict[str, str]],
    num_clients: int = 3,
    seed: int = 42,
) -> Dict[int, List[Dict[str, str]]]:
    """Uniformly and randomly partition training data across clients (IID benchmark)."""
    rng = random.Random(seed)
    shuffled = list(train_data)
    rng.shuffle(shuffled)

    client_data: Dict[int, List[Dict[str, str]]] = {i: [] for i in range(num_clients)}
    for idx, item in enumerate(shuffled):
        client_data[idx % num_clients].append(item)

    for i in range(num_clients):
        logger.info(f"Client {i} (IID) assigned {len(client_data[i])} samples.")

    return client_data


def partition_non_iid_dirichlet(
    train_data: List[Dict[str, str]],
    num_clients: int = 3,
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, List[Dict[str, str]]]:
    """Partition training data across clients using a Dirichlet distribution over task categories.

    Args:
        train_data: List of instruction examples with 'category' field.
        num_clients: Number of simulated federated clients.
        alpha: Dirichlet concentration parameter. Smaller alpha (< 1.0) creates higher non-IID skew.
        seed: Random seed for reproducibility.

    Returns:
        Mapping from client_id to list of assigned examples.
    """
    rng = np.random.default_rng(seed)

    # Group sample indices by category
    category_indices: Dict[str, List[int]] = {}
    for idx, item in enumerate(train_data):
        cat = item.get("category", "default")
        category_indices.setdefault(cat, []).append(idx)

    client_sample_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    for cat, idxs in category_indices.items():
        np_idxs = np.array(idxs)
        rng.shuffle(np_idxs)

        # Sample Dirichlet proportions
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Scale to counts
        counts = (proportions * len(np_idxs)).astype(int)

        # Handle rounding discrepancies
        remainder = len(np_idxs) - counts.sum()
        for r in range(remainder):
            counts[r % num_clients] += 1

        # Slice and allocate
        start_idx = 0
        for client_id in range(num_clients):
            allocated = np_idxs[start_idx : start_idx + counts[client_id]].tolist()
            client_sample_indices[client_id].extend(allocated)
            start_idx += counts[client_id]

    client_data: Dict[int, List[Dict[str, str]]] = {}
    for client_id, idxs in client_sample_indices.items():
        client_data[client_id] = [train_data[i] for i in idxs]
        logger.info(
            f"Client {client_id} (Non-IID alpha={alpha}) assigned {len(client_data[client_id])} samples."
        )

    return client_data


def compute_heterogeneity_metric(
    client_data: Dict[int, List[Dict[str, str]]],
) -> Dict[str, Any]:
    """Quantify data heterogeneity across clients using Total Variation Distance (TVD) on category distributions.

    Returns:
        Dictionary with per-client category counts, category probability vectors,
        pairwise TVD matrix, and mean pairwise TVD in range [0, 1] (0 = identical, 1 = disjoint).
    """
    num_clients = len(client_data)
    all_categories = sorted(
        list({ex.get("category", "default") for c_data in client_data.values() for ex in c_data})
    )

    client_category_dist: Dict[int, np.ndarray] = {}
    client_counts: Dict[int, Dict[str, int]] = {}

    for client_id, samples in client_data.items():
        counts = Counter(s.get("category", "default") for s in samples)
        client_counts[client_id] = dict(counts)
        total = max(len(samples), 1)
        prob_vec = np.array([counts.get(cat, 0) / total for cat in all_categories])
        client_category_dist[client_id] = prob_vec

    # Compute pairwise Total Variation Distance: TVD(P, Q) = 0.5 * sum(|P - Q|)
    tvd_matrix = np.zeros((num_clients, num_clients))
    pairwise_distances: List[float] = []

    for i in range(num_clients):
        for j in range(num_clients):
            dist = 0.5 * float(np.sum(np.abs(client_category_dist[i] - client_category_dist[j])))
            tvd_matrix[i, j] = dist
            if i < j:
                pairwise_distances.append(dist)

    mean_tvd = float(np.mean(pairwise_distances)) if pairwise_distances else 0.0

    return {
        "categories": all_categories,
        "client_counts": client_counts,
        "pairwise_tvd_matrix": tvd_matrix.tolist(),
        "mean_tvd": mean_tvd,
        "is_heterogeneous": mean_tvd > 0.05,
    }
