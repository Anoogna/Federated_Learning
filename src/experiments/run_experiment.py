"""CLI entry point for executing federated experiments."""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import ExperimentConfig
from src.federated.training_loop import run_federated_training


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Run Federated LLM Fine-Tuning Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/smoke_test_cpu.yaml",
        help="Path to experiment YAML configuration file",
    )
    parser.add_argument("--rounds", type=int, default=None, help="Override number of rounds")
    parser.add_argument("--device", type=str, default=None, help="Override compute device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")

    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    if args.rounds is not None:
        config.federated.num_rounds = args.rounds
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed

    results = run_federated_training(config)
    print("\n" + "=" * 60)
    print(f"EXPERIMENT '{config.experiment_id}' COMPLETED SUCCESSFULLY!")
    print(f"Final Test Loss:       {results['final_test_eval']['loss']}")
    print(f"Final Test Perplexity: {results['final_test_eval']['perplexity']}")
    print(f"Total Transmitted KB:  {results['communication_summary']['cumulative_total_wire_bytes'] / 1024:.2f} KB")
    print(f"Compression Ratio:     {results['communication_summary']['overall_compression_ratio']:.2f}x")
    if config.privacy.enabled:
        print(f"Spent DP Epsilon:      {results['privacy_summary'].get('spent_epsilon')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
