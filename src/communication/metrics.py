"""Communication telemetry and efficiency tracking module.

Tracks round-level and cumulative uplink, downlink, and aggregate network transmission volumes,
computing effective compression ratios and bandwidth reduction statistics.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class RoundCommunicationRecord:
    """Telemetry record for a single communication round."""
    round_idx: int
    num_participating_clients: int
    uplink_wire_bytes: int
    downlink_wire_bytes: int
    total_wire_bytes: int
    dense_equivalent_bytes: int
    round_compression_ratio: float
    round_savings_percent: float


class CommunicationTracker:
    """Accumulates and analyzes end-to-end communication volume across federated training."""

    def __init__(self):
        self.round_records: List[RoundCommunicationRecord] = []
        self.cumulative_uplink_bytes: int = 0
        self.cumulative_downlink_bytes: int = 0
        self.cumulative_total_wire_bytes: int = 0
        self.cumulative_dense_equivalent_bytes: int = 0

    def log_round(
        self,
        round_idx: int,
        client_uplink_bytes_list: List[int],
        client_dense_bytes_list: List[int],
        downlink_bytes_per_client: int,
    ) -> RoundCommunicationRecord:
        """Record communication telemetry for one federated round."""
        num_clients = len(client_uplink_bytes_list)
        total_uplink = sum(client_uplink_bytes_list)
        total_downlink = downlink_bytes_per_client * num_clients
        round_total_wire = total_uplink + total_downlink

        # Dense equivalent is dense uplink + dense downlink
        total_dense_uplink = sum(client_dense_bytes_list)
        total_dense_downlink = downlink_bytes_per_client * num_clients
        round_dense_equivalent = total_dense_uplink + total_dense_downlink

        compression_ratio = (
            round_dense_equivalent / max(round_total_wire, 1)
        )
        savings_percent = max(0.0, (1.0 - (round_total_wire / max(round_dense_equivalent, 1))) * 100.0)

        # Update cumulative totals
        self.cumulative_uplink_bytes += total_uplink
        self.cumulative_downlink_bytes += total_downlink
        self.cumulative_total_wire_bytes += round_total_wire
        self.cumulative_dense_equivalent_bytes += round_dense_equivalent

        record = RoundCommunicationRecord(
            round_idx=round_idx,
            num_participating_clients=num_clients,
            uplink_wire_bytes=total_uplink,
            downlink_wire_bytes=total_downlink,
            total_wire_bytes=round_total_wire,
            dense_equivalent_bytes=round_dense_equivalent,
            round_compression_ratio=round(compression_ratio, 2),
            round_savings_percent=round(savings_percent, 2),
        )
        self.round_records.append(record)
        return record

    def get_summary(self) -> Dict[str, Any]:
        """Return comprehensive communication efficiency summary statistics."""
        overall_compression_ratio = (
            self.cumulative_dense_equivalent_bytes / max(self.cumulative_total_wire_bytes, 1)
        )
        overall_savings_percent = max(
            0.0,
            (1.0 - (self.cumulative_total_wire_bytes / max(self.cumulative_dense_equivalent_bytes, 1))) * 100.0,
        )

        return {
            "num_rounds": len(self.round_records),
            "cumulative_uplink_bytes": self.cumulative_uplink_bytes,
            "cumulative_downlink_bytes": self.cumulative_downlink_bytes,
            "cumulative_total_wire_bytes": self.cumulative_total_wire_bytes,
            "cumulative_dense_equivalent_bytes": self.cumulative_dense_equivalent_bytes,
            "overall_compression_ratio": round(overall_compression_ratio, 2),
            "overall_savings_percent": round(overall_savings_percent, 2),
            "history": [asdict(r) for r in self.round_records],
        }
