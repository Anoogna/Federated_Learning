"""Fixed Top-k communication compression module with Error Feedback (EF21).

Compresses parameter updates by retaining only the top-k largest elements by magnitude,
accumulating uncommunicated residual gradients in a local error feedback buffer to guarantee
convergence for biased contractive operators.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import torch

from src.compression.base import BaseCompressor

logger = logging.getLogger(__name__)


def topk_sparsify_tensor(
    tensor: torch.Tensor,
    k_ratio: float = 0.10,
    min_k: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparsify tensor by retaining top-k coordinates by absolute magnitude.

    Returns:
        Tuple of (sparsified_tensor, mask_tensor).
    """
    flat = tensor.view(-1)
    d = flat.numel()
    k = max(min_k, min(int(round(d * k_ratio)), d))

    if k >= d:
        return tensor.clone(), torch.ones_like(tensor, dtype=torch.bool)

    # Top-k by absolute magnitude
    _, top_indices = torch.topk(torch.abs(flat), k=k, sorted=False)
    mask = torch.zeros(d, dtype=torch.bool, device=tensor.device)
    mask[top_indices] = True

    sparse_flat = torch.zeros_like(flat)
    sparse_flat[mask] = flat[mask]

    return sparse_flat.view_as(tensor), mask.view_as(tensor)


class FixedTopKCompressor(BaseCompressor):
    """Fixed-ratio Top-k compressor with local error feedback accumulation."""

    def __init__(self, k_ratio: float = 0.10, error_feedback: bool = True):
        self.k_ratio = max(0.001, min(k_ratio, 1.0))
        self.error_feedback = error_feedback
        # Maps client_id -> {tensor_name: residual_error_tensor}
        self.error_buffers: Dict[int, Dict[str, torch.Tensor]] = {}

    def reset_client_state(self, client_id: int) -> None:
        if client_id in self.error_buffers:
            del self.error_buffers[client_id]

    def compress(
        self,
        tensor_dict: Dict[str, torch.Tensor],
        round_idx: int = 0,
        client_id: int = 0,
        norm_bound: float = 1.0,
        max_rounds: int = 10,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        client_errors = self.error_buffers.setdefault(client_id, {})
        compressed_dict: Dict[str, torch.Tensor] = {}
        total_elements = 0
        total_retained = 0

        for name, delta in tensor_dict.items():
            # Error feedback: v = delta + error
            if self.error_feedback and name in client_errors:
                v = delta + client_errors[name].to(delta.device)
            else:
                v = delta.clone()

            # Top-k sparsification
            sparse_v, mask = topk_sparsify_tensor(v, k_ratio=self.k_ratio)
            compressed_dict[name] = sparse_v

            # Update error accumulator: error = v - sparse_v
            if self.error_feedback:
                client_errors[name] = (v - sparse_v).detach().clone()

            total_elements += delta.numel()
            total_retained += int(mask.sum().item())

        telemetry = {
            "strategy": "fixed_topk",
            "k_ratio": self.k_ratio,
            "error_feedback": self.error_feedback,
            "total_elements": total_elements,
            "total_retained": total_retained,
            "effective_sparsity": round(1.0 - (total_retained / max(total_elements, 1)), 4),
        }
        return compressed_dict, telemetry
