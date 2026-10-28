"""Plotting and visualization module for federated learning experiments.

Generates publication-quality figures comparing convergence rates, communication efficiency,
Pareto frontiers (communication volume vs model perplexity), and privacy expenditure.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_convergence_comparison(
    results_dict_map: Dict[str, Dict[str, Any]],
    output_path: str = "outputs/plots/convergence_comparison.png",
) -> None:
    """Plot Validation Loss and Perplexity across communication rounds for multiple experiments."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    for exp_label, results in results_dict_map.items():
        history = results.get("history", [])
        rounds = [h["round"] for h in history]
        losses = [h["val_loss"] for h in history]
        ppls = [h["val_perplexity"] for h in history]

        ax1.plot(rounds, losses, marker="o", markersize=4, label=exp_label)
        ax2.plot(rounds, ppls, marker="s", markersize=4, label=exp_label)

    ax1.set_title("Global Validation Loss vs. Rounds", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Communication Round", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)

    ax2.set_title("Global Perplexity vs. Rounds", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Communication Round", fontsize=11)
    ax2.set_ylabel("Perplexity (exp(loss))", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved convergence comparison plot to %s", output_path)


def plot_pareto_efficiency(
    results_dict_map: Dict[str, Dict[str, Any]],
    output_path: str = "outputs/plots/pareto_efficiency.png",
) -> None:
    """Plot the Pareto trade-off between Transmitted Wire Bytes (KB) and Final Model Perplexity."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=300)

    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for idx, (exp_label, results) in enumerate(results_dict_map.items()):
        total_kb = results["communication_summary"]["cumulative_total_wire_bytes"] / 1024.0
        final_ppl = results["final_test_eval"]["perplexity"]
        m = markers[idx % len(markers)]

        plt.scatter(total_kb, final_ppl, s=120, marker=m, label=exp_label)
        plt.annotate(
            exp_label,
            (total_kb, final_ppl),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    plt.title("Communication Volume vs. Model Quality Trade-off", fontsize=13, fontweight="bold")
    plt.xlabel("Total Transmitted Wire Data (KB)", fontsize=11)
    plt.ylabel("Final Test Perplexity (Lower is Better)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved Pareto efficiency plot to %s", output_path)
