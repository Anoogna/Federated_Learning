"""Wire serialization and exact payload byte calculation module.

Implements realistic network serialization for dense and sparse parameter updates, supporting
FP32/FP16 numerical encodings, coordinate indexing (COO), and 1-bit bitmask compression to
accurately measure transmitted network payload sizes.
"""

from __future__ import annotations

import io
import struct
from typing import Any, Dict, List, Tuple
import torch


def compute_tensor_dense_bytes(tensor: torch.Tensor, fp16: bool = True) -> int:
    """Calculate wire bytes for dense tensor transmission including shape metadata."""
    elem_bytes = 2 if fp16 else 4
    # 4 bytes for ndim, 4 bytes per dim shape header, + data bytes
    header_bytes = 4 + (4 * len(tensor.shape))
    return header_bytes + (tensor.numel() * elem_bytes)


def compute_tensor_sparse_wire_bytes(
    dense_shape: torch.Size,
    non_zero_count: int,
    fp16: bool = True,
) -> int:
    """Calculate the exact wire payload bytes for a sparse tensor update.

    Evaluates both Coordinate List (COO) and 1-Bit Bitmask representations,
    selecting the optimal (minimal payload) format.

    Args:
        dense_shape: Original shape of the tensor.
        non_zero_count: Number of retained non-zero coordinates (k).
        fp16: Whether values are transmitted in FP16 (2 bytes) or FP32 (4 bytes).

    Returns:
        Exact number of wire bytes required to transmit the sparse update.
    """
    total_elements = 1
    for dim in dense_shape:
        total_elements *= dim

    k = non_zero_count
    val_bytes_per_elem = 2 if fp16 else 4
    header_bytes = 8 + (4 * len(dense_shape))  # shape header + k count

    if k == 0:
        return header_bytes

    # Representation 1: Coordinate List (COO)
    # Uses 2-byte indices if total_elements <= 65535, else 4-byte indices
    index_bytes_per_elem = 2 if total_elements <= 65535 else 4
    coo_bytes = header_bytes + (k * (index_bytes_per_elem + val_bytes_per_elem))

    # Representation 2: 1-Bit Bitmask + Packed Values
    # 1 bit per coordinate in dense tensor + k packed values
    bitmask_bytes = (total_elements + 7) // 8
    mask_format_bytes = header_bytes + bitmask_bytes + (k * val_bytes_per_elem)

    # Return the optimal serialization size
    return min(coo_bytes, mask_format_bytes)


def serialize_adapter_delta(
    delta_dict: Dict[str, torch.Tensor],
    fp16: bool = True,
) -> Tuple[int, int, Dict[str, Any]]:
    """Measure serialized wire bytes vs dense equivalent for an entire adapter delta.

    Returns:
        Tuple of (actual_wire_bytes, dense_equivalent_bytes, per_tensor_stats).
    """
    actual_wire_bytes = 0
    dense_equivalent_bytes = 0
    per_tensor_stats: Dict[str, Any] = {}

    for name, tensor in delta_dict.items():
        dense_bytes = compute_tensor_dense_bytes(tensor, fp16=fp16)
        dense_equivalent_bytes += dense_bytes

        # Count non-zero elements
        non_zeros = int(torch.count_nonzero(tensor).item())
        is_sparse = non_zeros < tensor.numel()

        if is_sparse:
            wire_bytes = compute_tensor_sparse_wire_bytes(
                dense_shape=tensor.shape,
                non_zero_count=non_zeros,
                fp16=fp16,
            )
        else:
            wire_bytes = dense_bytes

        actual_wire_bytes += wire_bytes
        per_tensor_stats[name] = {
            "shape": list(tensor.shape),
            "total_elements": tensor.numel(),
            "non_zero_count": non_zeros,
            "sparsity_ratio": round(1.0 - (non_zeros / max(tensor.numel(), 1)), 4),
            "wire_bytes": wire_bytes,
            "dense_bytes": dense_bytes,
        }

    return actual_wire_bytes, dense_equivalent_bytes, per_tensor_stats
