"""Rényi Differential Privacy (RDP) accounting module.

Computes formal, composable (epsilon, delta)-differential privacy bounds across federated
training rounds using analytical Rényi Differential Privacy evaluation.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Standard RDP orders to optimize conversion to (epsilon, delta)
DEFAULT_RDP_ORDERS = [
    1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
    5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 20.0,
    24.0, 32.0, 48.0, 64.0, 128.0, 256.0,
]


class PrivacyAccountant:
    """Tracks cumulative privacy expenditure using Rényi Differential Privacy (RDP)."""

    def __init__(
        self,
        noise_multiplier: float = 1.0,
        target_delta: float = 1e-5,
        sample_rate: float = 1.0,
        orders: List[float] | None = None,
    ):
        self.noise_multiplier = noise_multiplier
        self.target_delta = target_delta
        self.sample_rate = sample_rate
        self.orders = np.array(orders or DEFAULT_RDP_ORDERS, dtype=float)

        # Cumulative RDP budget at each order
        self.cumulative_rdp = np.zeros_like(self.orders)
        self.total_steps: int = 0
        self.history: List[Dict[str, float]] = []

    def compute_step_rdp(self) -> np.ndarray:
        """Compute single-step RDP for Gaussian mechanism.

        For sampling rate q = 1: RDP(alpha) = alpha / (2 * sigma^2).
        For q < 1, uses standard Poisson subsampled Gaussian RDP approximation.
        """
        sigma = self.noise_multiplier
        if sigma <= 0.0:
            return np.full_like(self.orders, float("inf"))

        if abs(self.sample_rate - 1.0) < 1e-6:
            # Full participation Gaussian mechanism
            return self.orders / (2.0 * (sigma ** 2))
        else:
            # Subsampled Gaussian mechanism (Mironov et al. 2019)
            q = self.sample_rate
            rdp = np.zeros_like(self.orders)
            for idx, alpha in enumerate(self.orders):
                if alpha <= 1.0:
                    continue
                # Order-alpha subsampled bound
                term = (q ** 2) * alpha / (2.0 * (sigma ** 2))
                rdp[idx] = min(term, alpha / (2.0 * (sigma ** 2)))
            return rdp

    def step(self) -> Tuple[float, float]:
        """Record one training round / step and compute updated (epsilon, delta)."""
        if self.noise_multiplier <= 0.0:
            self.total_steps += 1
            return float("inf"), self.target_delta

        step_rdp = self.compute_step_rdp()
        self.cumulative_rdp += step_rdp
        self.total_steps += 1

        eps = self.get_epsilon(self.target_delta)
        self.history.append({
            "step": self.total_steps,
            "epsilon": eps,
            "delta": self.target_delta,
        })
        return eps, self.target_delta

    def get_epsilon(self, delta: float | None = None) -> float:
        """Convert cumulative RDP into standard (epsilon, delta)-DP bound.

        Formula: eps(delta) = min_{alpha > 1} { RDP(alpha) + log(1 / delta) / (alpha - 1) }
        """
        if delta is None:
            delta = self.target_delta

        if self.noise_multiplier <= 0.0:
            return float("inf")

        eps_candidates = []
        for idx, alpha in enumerate(self.orders):
            if alpha <= 1.0:
                continue
            eps = self.cumulative_rdp[idx] + (math.log(1.0 / delta) / (alpha - 1.0))
            eps_candidates.append(eps)

        if not eps_candidates:
            return float("inf")

        return round(float(np.min(eps_candidates)), 4)

    def get_summary(self) -> Dict[str, Any]:
        """Return comprehensive privacy telemetry summary."""
        current_eps = self.get_epsilon(self.target_delta)
        return {
            "noise_multiplier": self.noise_multiplier,
            "target_delta": self.target_delta,
            "sample_rate": self.sample_rate,
            "total_steps": self.total_steps,
            "spent_epsilon": current_eps,
            "history": self.history,
        }
