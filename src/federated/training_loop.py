"""Federated training orchestration module.

Executes end-to-end multi-round federated fine-tuning, synchronizing clients, server aggregation,
privacy tracking, wire communication telemetry, and saving structured experiment outputs.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from src.config import ExperimentConfig
from src.data.load_dataset import load_dolly_subset
from src.data.partition_clients import (
    split_train_val_test,
    partition_iid,
    partition_non_iid_dirichlet,
    compute_heterogeneity_metric,
)
from src.model.load_model import load_model_and_tokenizer
from src.model.lora_config import apply_lora_to_model, get_trainable_parameter_stats
from src.compression.fixed_compression import FixedTopKCompressor
from src.compression.adaptive_compression import AdaCompCompressor
from src.communication.metrics import CommunicationTracker
from src.communication.serialization import compute_tensor_dense_bytes
from src.privacy.privacy_accountant import PrivacyAccountant
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer

logger = logging.getLogger(__name__)


def run_federated_training(config: ExperimentConfig) -> Dict[str, Any]:
    """Execute complete federated LoRA fine-tuning experiment according to ExperimentConfig."""
    logger.info("Starting Federated Experiment '%s'...", config.experiment_id)

    # 1. Deterministic Seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available() and config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    os.makedirs(config.output_dir, exist_ok=True)

    # 2. Dataset Loading & Partitioning
    raw_data = load_dolly_subset(
        num_samples=config.data.num_samples,
        seed=config.data.seed,
        cache_dir=config.data.cache_dir,
    )
    train_data, val_data, test_data = split_train_val_test(
        raw_data,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        seed=config.data.seed,
    )

    if config.data.partition_strategy == "iid":
        client_partitions = partition_iid(train_data, num_clients=config.data.num_clients, seed=config.data.seed)
    else:
        client_partitions = partition_non_iid_dirichlet(
            train_data,
            num_clients=config.data.num_clients,
            alpha=config.data.dirichlet_alpha,
            seed=config.data.seed,
        )

    heterogeneity_stats = compute_heterogeneity_metric(client_partitions)
    logger.info("Data heterogeneity (mean pairwise TVD): %.4f", heterogeneity_stats["mean_tvd"])

    # 3. Model & LoRA Initialization
    global_model, tokenizer = load_model_and_tokenizer(config.model, device=device)
    global_model = apply_lora_to_model(global_model, config.lora, use_peft_library=not config.model.is_surrogate)
    param_stats = get_trainable_parameter_stats(global_model)
    logger.info("Trainable parameters: %d / %d (%.2f%%)",
                param_stats["trainable_params"], param_stats["total_params"], param_stats["trainable_percent"])

    # 4. Compressor Setup
    compressor = None
    if config.compression.enabled:
        if config.compression.strategy == "fixed_topk":
            compressor = FixedTopKCompressor(
                k_ratio=config.compression.fixed_k_ratio,
                error_feedback=config.compression.error_feedback,
            )
        elif config.compression.strategy == "adaptive_adacomp":
            compressor = AdaCompCompressor(
                min_k_ratio=config.compression.adaptive_min_ratio,
                max_k_ratio=config.compression.adaptive_max_ratio,
                energy_threshold=config.compression.energy_threshold,
                error_feedback=config.compression.error_feedback,
            )

    # 5. Telemetry & Privacy Tracking
    comm_tracker = CommunicationTracker()
    privacy_accountant = PrivacyAccountant(
        noise_multiplier=config.privacy.noise_multiplier if config.privacy.enabled else 0.0,
        target_delta=config.privacy.target_delta,
        sample_rate=config.federated.clients_per_round / max(config.data.num_clients, 1),
    )

    # 6. Instantiate Server & Clients
    server = FederatedServer(
        global_model=global_model,
        tokenizer=tokenizer,
        val_data=val_data,
        test_data=test_data,
        config=config,
        device=device,
    )

    # 6. Instantiate Server & Shared Client Worker Model (Saves 3x GPU VRAM)
    server = FederatedServer(
        global_model=global_model,
        tokenizer=tokenizer,
        val_data=val_data,
        test_data=test_data,
        config=config,
        device=device,
    )

    worker_model, _ = load_model_and_tokenizer(config.model, device=device)
    worker_model = apply_lora_to_model(worker_model, config.lora, use_peft_library=not config.model.is_surrogate)

    clients: Dict[int, FederatedClient] = {}
    for c_id, c_data in client_partitions.items():
        clients[c_id] = FederatedClient(
            client_id=c_id,
            local_data=c_data,
            model=worker_model,
            tokenizer=tokenizer,
            config=config,
            compressor=compressor,
            device=device,
        )

    # 7. Initial Evaluation (Round 0 Baseline)
    init_eval = server.evaluate(run_probes=True)
    logger.info("Round 0 (Initial) -> Val Loss: %.4f | Perplexity: %.4f", init_eval["loss"], init_eval["perplexity"])
    print(f"[{config.experiment_id}] Round 0 (Initial) -> Val Loss: {init_eval['loss']:.4f} | Perplexity: {init_eval['perplexity']:.2f}", flush=True)

    history: List[Dict[str, Any]] = [{
        "round": 0,
        "val_loss": init_eval["loss"],
        "val_perplexity": init_eval["perplexity"],
        "cumulative_wire_bytes": 0,
        "cumulative_dense_bytes": 0,
        "overall_compression_ratio": 1.0,
        "spent_epsilon": 0.0,
        "probes": init_eval["probes"],
    }]

    # Compute dense downlink byte size per client (broadcast of adapter parameters)
    global_adapter_params = server.get_global_adapter_params()
    downlink_bytes_per_client = sum(
        compute_tensor_dense_bytes(p, fp16=(config.compression.quantization_bits == 16))
        for p in global_adapter_params.values()
    )

    # 8. Federated Multi-Round Optimization Loop
    for r in range(1, config.federated.num_rounds + 1):
        logger.info("--- Communication Round %d / %d ---", r, config.federated.num_rounds)

        # Select participating clients (all clients if clients_per_round == num_clients)
        all_client_ids = list(clients.keys())
        participating_ids = sorted(random.sample(all_client_ids, k=config.federated.clients_per_round))

        client_deltas: List[Dict[str, torch.Tensor]] = []
        client_sample_counts: List[int] = []
        client_uplink_bytes: List[int] = []
        client_dense_bytes: List[int] = []
        client_losses: List[float] = []

        # Local Training on Each Participating Client
        for c_id in participating_ids:
            client = clients[c_id]
            delta, wire_b, dense_b, local_loss, _ = client.train_round(
                global_adapter_params=global_adapter_params,
                round_idx=r,
                max_rounds=config.federated.num_rounds,
            )
            client_deltas.append(delta)
            client_sample_counts.append(client.num_samples)
            client_uplink_bytes.append(wire_b)
            client_dense_bytes.append(dense_b)
            client_losses.append(local_loss)

        # Update Privacy Accountant
        spent_eps = 0.0
        if config.privacy.enabled:
            spent_eps, _ = privacy_accountant.step()

        # Server Aggregation & Global Model Update
        server.aggregate_and_update(
            client_deltas=client_deltas,
            client_sample_counts=client_sample_counts,
        )
        global_adapter_params = server.get_global_adapter_params()

        # Log Communication Telemetry
        comm_rec = comm_tracker.log_round(
            round_idx=r,
            client_uplink_bytes_list=client_uplink_bytes,
            client_dense_bytes_list=client_dense_bytes,
            downlink_bytes_per_client=downlink_bytes_per_client,
        )

        # Global Evaluation (run generative probes only on the final round)
        is_final_round = (r == config.federated.num_rounds)
        round_eval = server.evaluate(run_probes=is_final_round)
        logger.info(
            "Round %d -> Val Loss: %.4f | PPL: %.4f | Comm Ratio: %.2fx | Epsilon: %.2f",
            r, round_eval["loss"], round_eval["perplexity"], comm_rec.round_compression_ratio, spent_eps
        )
        print(
            f"  [{config.experiment_id}] Round {r}/{config.federated.num_rounds} -> "
            f"Val Loss: {round_eval['loss']:.4f} | PPL: {round_eval['perplexity']:.2f} | "
            f"Comm Ratio: {comm_rec.round_compression_ratio:.2f}x | Epsilon: {spent_eps:.2f}",
            flush=True,
        )

        round_record = {
            "round": r,
            "avg_client_loss": round(float(np.mean(client_losses)), 4),
            "val_loss": round_eval["loss"],
            "val_perplexity": round_eval["perplexity"],
            "round_wire_bytes": comm_rec.total_wire_bytes,
            "round_dense_bytes": comm_rec.dense_equivalent_bytes,
            "round_compression_ratio": comm_rec.round_compression_ratio,
            "cumulative_wire_bytes": comm_tracker.cumulative_total_wire_bytes,
            "cumulative_dense_bytes": comm_tracker.cumulative_dense_equivalent_bytes,
            "overall_compression_ratio": round(
                comm_tracker.cumulative_dense_equivalent_bytes / max(comm_tracker.cumulative_total_wire_bytes, 1), 2
            ),
            "spent_epsilon": spent_eps,
            "probes": round_eval["probes"],
        }
        history.append(round_record)

    # 9. Final Evaluation on Global Test Set
    final_test_eval = server.evaluate(use_test_set=True)
    logger.info("=== Final Test Results === -> Test Loss: %.4f | Test Perplexity: %.4f",
                final_test_eval["loss"], final_test_eval["perplexity"])

    # 10. Assemble & Persist Complete Results
    comm_summary = comm_tracker.get_summary()
    privacy_summary = privacy_accountant.get_summary() if config.privacy.enabled else {}

    results = {
        "experiment_id": config.experiment_id,
        "config": config.to_dict(),
        "parameter_stats": param_stats,
        "heterogeneity_stats": heterogeneity_stats,
        "initial_eval": init_eval,
        "final_test_eval": final_test_eval,
        "communication_summary": comm_summary,
        "privacy_summary": privacy_summary,
        "history": history,
    }

    results_file = os.path.join(config.output_dir, f"{config.experiment_id}_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results successfully saved to %s", results_file)

    return results
