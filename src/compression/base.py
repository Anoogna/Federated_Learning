"""Base compressor abstraction module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import torch


class BaseCompressor(ABC):
    """Abstract interface for parameter update compression algorithms."""

    @abstractmethod
    def compress(
        self,
        tensor_dict: Dict[str, torch.Tensor],
        round_idx: int = 0,
        client_id: int = 0,
        norm_bound: float = 1.0,
        max_rounds: int = 10,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Compress parameter delta dictionary.

        Args:
            tensor_dict: Mapping of parameter names to update tensors.
            round_idx: Current communication round index.
            client_id: Unique identifier of the local client.
            norm_bound: DP clipping norm or gradient scale for energy normalization.
            max_rounds: Total scheduled federated rounds.

        Returns:
            Tuple of (compressed_tensor_dict, telemetry_metadata).
        """
        pass

    @abstractmethod
    def reset_client_state(self, client_id: int) -> None:
        """Reset internal accumulators (e.g., error feedback) for a specific client."""
        pass
