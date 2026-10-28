"""Federated central server module.

Maintains the authoritative global model, broadcasts parameters, aggregates client updates
using sample-weighted FedAvg, and computes global validation loss, perplexity, and probes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn

from src.config import ExperimentConfig
from src.model.lora_config import (
    extract_adapter_parameters,
    load_adapter_parameters,
    apply_adapter_delta,
)
from src.federated.aggregation import aggregate_client_deltas_fedavg
from src.model.evaluation import evaluate_model, run_instruction_probes

logger = logging.getLogger(__name__)


class FederatedServer:
    """Orchestrates federated aggregation and global model evaluation."""

    def __init__(
        self,
        global_model: nn.Module,
        tokenizer: Any,
        val_data: List[Dict[str, str]],
        test_data: List[Dict[str, str]],
        config: ExperimentConfig,
        device: str = "cpu",
    ):
        self.global_model = global_model
        self.tokenizer = tokenizer
        self.val_data = val_data
        self.test_data = test_data
        self.config = config
        self.device = device

    def get_global_adapter_params(self) -> Dict[str, torch.Tensor]:
        """Extract the current global adapter parameters for broadcast to clients."""
        return extract_adapter_parameters(self.global_model)

    def aggregate_and_update(
        self,
        client_deltas: List[Dict[str, torch.Tensor]],
        client_sample_counts: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client updates via sample-weighted FedAvg and update global model."""
        aggregated_delta = aggregate_client_deltas_fedavg(
            client_deltas=client_deltas,
            client_sample_counts=client_sample_counts,
        )

        current_params = self.get_global_adapter_params()
        new_params = apply_adapter_delta(current_params, aggregated_delta, step_size=1.0)
        load_adapter_parameters(self.global_model, new_params)

        return aggregated_delta

    def evaluate(self, use_test_set: bool = False, run_probes: bool = False) -> Dict[str, Any]:
        """Evaluate the global model on held-out validation or test datasets."""
        eval_data = self.test_data if use_test_set else self.val_data
        metrics = evaluate_model(
            model=self.global_model,
            eval_examples=eval_data,
            tokenizer=self.tokenizer,
            batch_size=self.config.federated.local_batch_size,
            max_length=self.config.model.max_seq_length,
            device=self.device,
        )
        probes = []
        if run_probes:
            probes = run_instruction_probes(
                model=self.global_model,
                tokenizer=self.tokenizer,
                device=self.device,
            )
        return {
            **metrics,
            "probes": probes,
        }
