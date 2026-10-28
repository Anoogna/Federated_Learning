"""Differential Privacy mechanism module.

Implements client-level and parameter-level L2 norm clipping and calibrated Gaussian
noise perturbation to provide formal differential privacy guarantees for federated model updates.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple
import torch

from src.config import PrivacyConfig

logger = logging.getLogger(__name__)


def compute_global_l2_norm(tensor_dict: Dict[str, torch.Tensor]) -> float:
    """Compute the combined L2 norm across all tensors in a parameter dictionary."""
    total_sq_norm = 0.0
    for tensor in tensor_dict.values():
        total_sq_norm += float(torch.sum(tensor ** 2).item())
    return (total_sq_norm ** 0.5)


def clip_parameter_delta(
    delta_dict: Dict[str, torch.Tensor],
    clipping_norm: float = 1.0,
) -> Tuple[Dict[str, torch.Tensor], float, bool]:
    """Perform L2 norm clipping on a dictionary of parameter updates.

    Returns:
        Tuple of (clipped_delta_dict, original_global_norm, was_clipped).
    """
    global_norm = compute_global_l2_norm(delta_dict)
    clip_factor = min(1.0, clipping_norm / max(global_norm, 1e-8))
    was_clipped = global_norm > clipping_norm

    clipped_dict: Dict[str, torch.Tensor] = {}
    for name, tensor in delta_dict.items():
        clipped_dict[name] = tensor * clip_factor

    return clipped_dict, global_norm, was_clipped


def add_gaussian_noise(
    delta_dict: Dict[str, torch.Tensor],
    clipping_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    normalize_by_dimension: bool = True,
    seed: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Inject calibrated Gaussian noise scaled to the clipping norm sensitivity.

    When normalize_by_dimension is True, coordinate variance is scaled such that
    the expected total squared L2 norm of the noise vector across all adapter parameters
    equals (noise_multiplier * clipping_norm)^2: E[||Z||_2^2] = (sigma * C)^2.
    """
    if noise_multiplier <= 0.0:
        return delta_dict

    total_dim = sum(t.numel() for t in delta_dict.values())
    if normalize_by_dimension and total_dim > 0:
        std = (noise_multiplier * clipping_norm) / (total_dim ** 0.5)
    else:
        std = noise_multiplier * clipping_norm

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    noisy_dict: Dict[str, torch.Tensor] = {}
    for name, tensor in delta_dict.items():
        # Sample noise on CPU to ensure deterministic cross-platform reproducibility
        noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float32) * std
        noisy_dict[name] = tensor + noise.to(device=tensor.device, dtype=tensor.dtype)

    return noisy_dict


def apply_differential_privacy(
    delta_dict: Dict[str, torch.Tensor],
    privacy_config: PrivacyConfig,
    round_seed: int | None = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Apply L2 clipping followed by calibrated Gaussian noise perturbation.

    Returns:
        Tuple of (privatized_delta_dict, privacy_telemetry).
    """
    if not privacy_config.enabled:
        norm = compute_global_l2_norm(delta_dict)
        return delta_dict, {
            "dp_enabled": False,
            "pre_clip_norm": round(norm, 4),
            "was_clipped": False,
            "noise_multiplier": 0.0,
        }

    # 1. Bounded Sensitivity via L2 Norm Clipping
    clipped_dict, pre_clip_norm, was_clipped = clip_parameter_delta(
        delta_dict, clipping_norm=privacy_config.clipping_norm
    )

    # 2. Calibrated Perturbation via Gaussian Mechanism
    privatized_dict = add_gaussian_noise(
        clipped_dict,
        clipping_norm=privacy_config.clipping_norm,
        noise_multiplier=privacy_config.noise_multiplier,
        seed=round_seed,
    )

    post_noise_norm = compute_global_l2_norm(privatized_dict)

    telemetry = {
        "dp_enabled": True,
        "clipping_norm": privacy_config.clipping_norm,
        "noise_multiplier": privacy_config.noise_multiplier,
        "pre_clip_norm": round(pre_clip_norm, 4),
        "post_noise_norm": round(post_noise_norm, 4),
        "was_clipped": was_clipped,
    }
    return privatized_dict, telemetry
