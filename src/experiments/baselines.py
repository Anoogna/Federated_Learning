"""Baseline and ablation experiment orchestration suite.

Automates execution of the core experimental matrix:
1. FedAvg Dense (no compression, no DP)
2. FedAvg Fixed Top-k (static compression, no DP)
3. FedAvg AdaComp (adaptive compression, no DP)
4. DP-FedAvg Dense (DP, uncompressed)
5. DP-FedAvg Fixed Top-k (DP, static compression)
6. AdaComp-DP (Ours: DP, adaptive compression)
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from typing import Any, Dict, List
import torch

from src.config import ExperimentConfig
from src.federated.training_loop import run_federated_training
from src.reporting.plots import plot_convergence_comparison, plot_pareto_efficiency
from src.reporting.tables import generate_comparison_markdown_table

logger = logging.getLogger(__name__)


def run_comparative_baseline_suite(
    base_config_path: str = "configs/smoke_test_cpu.yaml",
    num_rounds: int = 3,
    output_dir: str = "outputs/baselines_suite",
) -> Dict[str, Dict[str, Any]]:
    """Execute the systematic suite of federated baselines and proposed AdaComp-DP method."""
    os.makedirs(output_dir, exist_ok=True)
    base_cfg = ExperimentConfig.from_yaml(base_config_path)
    base_cfg.federated.num_rounds = num_rounds

    suite_results: Dict[str, Dict[str, Any]] = {}

    configurations = [
        {
            "id": "FedAvg-Dense",
            "dp_enabled": False,
            "comp_enabled": False,
            "comp_strategy": "none",
        },
        {
            "id": "FedAvg-FixedTopK",
            "dp_enabled": False,
            "comp_enabled": True,
            "comp_strategy": "fixed_topk",
        },
        {
            "id": "FedAvg-AdaComp",
            "dp_enabled": False,
            "comp_enabled": True,
            "comp_strategy": "adaptive_adacomp",
        },
        {
            "id": "DP-FedAvg-Dense",
            "dp_enabled": True,
            "comp_enabled": False,
            "comp_strategy": "none",
        },
        {
            "id": "DP-FedAvg-FixedTopK",
            "dp_enabled": True,
            "comp_enabled": True,
            "comp_strategy": "fixed_topk",
        },
        {
            "id": "AdaComp-DP-Proposed",
            "dp_enabled": True,
            "comp_enabled": True,
            "comp_strategy": "adaptive_adacomp",
        },
    ]

    for idx, item in enumerate(configurations):
        cfg = copy.deepcopy(base_cfg)
        cfg.experiment_id = item["id"]
        cfg.output_dir = os.path.join(output_dir, item["id"])
        cfg.privacy.enabled = item["dp_enabled"]
        cfg.compression.enabled = item["comp_enabled"]
        cfg.compression.strategy = item["comp_strategy"]

        print(
            f"\n{'='*70}\n[BENCHMARK {idx+1}/{len(configurations)}]: Running {item['id']} "
            f"({cfg.federated.num_rounds} rounds, {cfg.data.num_clients} clients)...\n{'='*70}\n",
            flush=True,
        )
        logger.info("Executing Benchmark %d/%d: %s", idx+1, len(configurations), cfg.experiment_id)
        res = run_federated_training(cfg)
        suite_results[item["id"]] = res

        # Clear GPU memory between benchmark runs
        import gc
        del res
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Generate comparative plots and tables
    plots_dir = os.path.join(output_dir, "plots")
    plot_convergence_comparison(suite_results, os.path.join(plots_dir, "convergence_comparison.png"))
    plot_pareto_efficiency(suite_results, os.path.join(plots_dir, "pareto_efficiency.png"))

    # Generate markdown table summary
    table_md = generate_comparison_markdown_table(suite_results)
    table_file = os.path.join(output_dir, "baseline_results_summary.md")
    with open(table_file, "w", encoding="utf-8") as f:
        f.write(table_md)
    logger.info("Saved baseline summary table to %s", table_file)

    return suite_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Comparative Baseline Suite")
    parser.add_argument("--config", type=str, default="configs/smoke_test_cpu.yaml")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="outputs/baselines_suite")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    run_comparative_baseline_suite(
        base_config_path=args.config,
        num_rounds=args.rounds,
        output_dir=args.output_dir,
    )
