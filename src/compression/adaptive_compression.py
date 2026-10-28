"""Adaptive Communication Compression module: AdaComp-DP.

Dynamically adjusts sparsity based on training round progression and local update energy (SNR),
retaining informative gradients during early exploration and maximizing compression during
late refinement, integrated with EF21 error feedback.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import torch

from src.compression.base import BaseCompressor
from src.compression.fixed_compression import topk_sparsify_tensor

logger = logging.getLogger(__name__)


class AdaCompCompressor(BaseCompressor):
    """SNR-Aware and Convergence-Guided Adaptive Sparsification (AdaComp-DP)."""

    def __init__(
        self,
        min_k_ratio: float = 0.02,
        max_k_ratio: float = 0.25,
        energy_threshold: float = 0.85,
        use_energy_threshold: bool = False,
        error_feedback: bool = True,
    ):
        self.min_k_ratio = min_k_ratio
        self.max_k_ratio = max_k_ratio
        self.energy_threshold = energy_threshold
        self.use_energy_threshold = use_energy_threshold
        self.error_feedback = error_feedback
        # Maps client_id -> {tensor_name: residual_error_tensor}
        self.error_buffers: Dict[int, Dict[str, torch.Tensor]] = {}

    def reset_client_state(self, client_id: int) -> None:
        if client_id in self.error_buffers:
            del self.error_buffers[client_id]

    def compute_scheduled_ratio(self, round_idx: int, max_rounds: int) -> float:
        """Compute cosine round decay schedule: starts at max_k_ratio, decays to min_k_ratio."""
        progress = min(max(round_idx / max(max_rounds, 1), 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduled_k = self.min_k_ratio + (self.max_k_ratio - self.min_k_ratio) * cosine_decay
        return scheduled_k

    def compress(
        self,
        tensor_dict: Dict[str, torch.Tensor],
        round_idx: int = 0,
        client_id: int = 0,
        norm_bound: float = 1.0,
        max_rounds: int = 10,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        client_errors = self.error_buffers.setdefault(client_id, {})
        base_k_ratio = self.compute_scheduled_ratio(round_idx, max_rounds)

        compressed_dict: Dict[str, torch.Tensor] = {}
        total_elements = 0
        total_retained = 0
        per_tensor_k_ratios: Dict[str, float] = {}

        for name, delta in tensor_dict.items():
            # Error feedback: v = delta + error
            if self.error_feedback and name in client_errors:
                v = delta + client_errors[name].to(delta.device)
            else:
                v = delta.clone()

            d = v.numel()
            v_norm = float(torch.norm(v).item())

            if self.use_energy_threshold and v_norm > 1e-7:
                # Energy threshold mode: select k to capture energy_threshold fraction of ||v||^2
                flat_abs = torch.abs(v.view(-1))
                sorted_sq, _ = torch.sort(flat_abs ** 2, descending=True)
                cum_energy = torch.cumsum(sorted_sq, dim=0)
                target_energy = self.energy_threshold * (v_norm ** 2)
                # Find smallest index satisfying target energy
                indices_meeting_target = (cum_energy >= target_energy).nonzero(as_tuple=True)[0]
                if len(indices_meeting_target) > 0:
                    k_energy = int(indices_meeting_target[0].item()) + 1
                else:
                    k_energy = d
                # Enforce bounds [min_k, max_k]
                k_min = max(1, int(round(d * self.min_k_ratio)))
                k_max = max(1, int(round(d * self.max_k_ratio)))
                effective_k = min(max(k_energy, k_min), k_max)
                effective_k_ratio = effective_k / d
            else:
                # Energy modulation mode: scale base_k_ratio by normalized update energy
                energy_scale = min(max(v_norm / max(norm_bound, 1e-5), 0.2), 1.5)
                effective_k_ratio = min(max(base_k_ratio * energy_scale, self.min_k_ratio), self.max_k_ratio)

            # Sparsify using dynamic effective_k_ratio
            sparse_v, mask = topk_sparsify_tensor(v, k_ratio=effective_k_ratio)
            compressed_dict[name] = sparse_v

            # Update error accumulator
            if self.error_feedback:
                client_errors[name] = (v - sparse_v).detach().clone()

            retained = int(mask.sum().item())
            total_elements += d
            total_retained += retained
            per_tensor_k_ratios[name] = round(effective_k_ratio, 4)

        telemetry = {
            "strategy": "adaptive_adacomp",
            "round_idx": round_idx,
            "base_k_ratio": round(base_k_ratio, 4),
            "total_elements": total_elements,
            "total_retained": total_retained,
            "effective_sparsity": round(1.0 - (total_retained / max(total_elements, 1)), 4),
            "per_tensor_k_ratios": per_tensor_k_ratios,
        }
        return compressed_dict, telemetry
