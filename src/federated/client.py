"""Federated client module.

Manages private local dataset training, adapter delta extraction, differential privacy perturbation,
communication compression with local error feedback, and wire byte calculation.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.config import ExperimentConfig
from src.data.preprocess import tokenize_batch
from src.model.lora_config import (
    load_adapter_parameters,
    extract_adapter_parameters,
    compute_adapter_delta,
)
from src.privacy.dp_mechanism import apply_differential_privacy
from src.compression.base import BaseCompressor
from src.communication.serialization import serialize_adapter_delta

logger = logging.getLogger(__name__)


class FederatedClient:
    """Represents a simulated decentralized client in federated training."""

    def __init__(
        self,
        client_id: int,
        local_data: List[Dict[str, str]],
        model: nn.Module,
        tokenizer: Any,
        config: ExperimentConfig,
        compressor: Optional[BaseCompressor] = None,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.local_data = local_data
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.compressor = compressor
        self.device = device
        self.num_samples = len(local_data)

    def train_round(
        self,
        global_adapter_params: Dict[str, torch.Tensor],
        round_idx: int,
        max_rounds: int,
    ) -> Tuple[Dict[str, torch.Tensor], int, int, float, Dict[str, Any]]:
        """Execute one local training round on client's private partition.

        Returns:
            Tuple of (transmitted_delta, wire_bytes, dense_bytes, avg_loss, telemetry).
        """
        # 1. Synchronize local adapter with global broadcast
        load_adapter_parameters(self.model, global_adapter_params)

        # Record initial parameter state
        initial_params = extract_adapter_parameters(self.model)

        # 2. Local Training Loop
        self.model.train()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.config.federated.local_lr, weight_decay=0.01)

        batch_size = max(self.config.federated.local_batch_size, 1)
        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.config.federated.local_epochs):
            # Mini-batch training
            for i in range(0, len(self.local_data), batch_size):
                batch_examples = self.local_data[i : i + batch_size]
                batch = tokenize_batch(
                    batch_examples,
                    tokenizer=self.tokenizer,
                    max_length=self.config.model.max_seq_length,
                    mask_prompt_labels=True,
                )
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"]
                if loss is not None:
                    loss.backward()
                    # Gradient clipping to stabilize local transformer step
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # 3. Extract Local Parameter Delta: Delta = theta_local - theta_global
        local_params = extract_adapter_parameters(self.model)
        raw_delta = compute_adapter_delta(local_params, initial_params)

        # 4. Differential Privacy Perturbation
        dp_seed = self.config.seed + (round_idx * 1000) + self.client_id
        privatized_delta, dp_telemetry = apply_differential_privacy(
            raw_delta,
            privacy_config=self.config.privacy,
            round_seed=dp_seed,
        )

        # 5. Communication Compression with Error Feedback
        if self.compressor is not None and self.config.compression.enabled:
            transmitted_delta, comp_telemetry = self.compressor.compress(
                privatized_delta,
                round_idx=round_idx,
                client_id=self.client_id,
                norm_bound=self.config.privacy.clipping_norm,
                max_rounds=max_rounds,
            )
        else:
            transmitted_delta = privatized_delta
            comp_telemetry = {"strategy": "none", "effective_sparsity": 0.0}

        # 6. Wire Serialization and Byte Measurement
        fp16_wire = (self.config.compression.quantization_bits == 16)
        wire_bytes, dense_bytes, _ = serialize_adapter_delta(transmitted_delta, fp16=fp16_wire)

        telemetry = {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "local_loss": round(avg_loss, 4),
            "wire_bytes": wire_bytes,
            "dense_bytes": dense_bytes,
            "dp": dp_telemetry,
            "compression": comp_telemetry,
        }

        return transmitted_delta, wire_bytes, dense_bytes, avg_loss, telemetry
