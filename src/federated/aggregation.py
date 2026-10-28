"""Federated aggregation module.

Implements sample-weighted Federated Averaging (FedAvg) and robust aggregation of sparse
and dense client parameter updates.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
import torch

logger = logging.getLogger(__name__)


def aggregate_client_deltas_fedavg(
    client_deltas: List[Dict[str, torch.Tensor]],
    client_sample_counts: List[int],
) -> Dict[str, torch.Tensor]:
    """Aggregate client parameter deltas using sample-weighted Federated Averaging.

    Formula:
        Delta_global = sum_{i=1}^M (n_i / N) * Delta_i
        where n_i is the number of local samples on client i and N = sum_i n_i.

    Args:
        client_deltas: List of delta dictionaries from participating clients.
        client_sample_counts: Number of samples belonging to each client.

    Returns:
        Aggregated global delta dictionary.
    """
    assert len(client_deltas) == len(client_sample_counts), "Mismatched client deltas and sample counts"
    assert len(client_deltas) > 0, "No client deltas provided for aggregation"

    total_samples = max(sum(client_sample_counts), 1)
    weights = [count / total_samples for count in client_sample_counts]

    # Initialize aggregated delta with zeros shaped like the first client's tensors
    sample_delta = client_deltas[0]
    aggregated_delta: Dict[str, torch.Tensor] = {}

    for name, tensor in sample_delta.items():
        # Accumulate in float32 for numerical stability
        weighted_sum = torch.zeros_like(tensor, dtype=torch.float32)

        for client_idx, c_delta in enumerate(client_deltas):
            if name not in c_delta:
                logger.warning("Tensor '%s' missing from client %d update. Skipping.", name, client_idx)
                continue
            client_tensor = c_delta[name].to(weighted_sum.device).to(torch.float32)
            weighted_sum += weights[client_idx] * client_tensor

        aggregated_delta[name] = weighted_sum.to(tensor.dtype)

    return aggregated_delta
