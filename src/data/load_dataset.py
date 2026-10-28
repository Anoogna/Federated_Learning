"""Dataset loading and caching module for federated LLM fine-tuning.

Loads the Databricks Dolly 15K instruction-following dataset with deterministic subsetting,
filtering, and an offline reproducible fallback generator for compute-constrained and
offline testing environments.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Standard Dolly 15K categories
DOLLY_CATEGORIES = [
    "creative_writing",
    "closed_qa",
    "open_qa",
    "summarization",
    "information_extraction",
    "brainstorming",
    "classification",
    "general_qa",
]

# Curated synthetic seed examples for deterministic offline fallback
OFFLINE_SEED_EXAMPLES = [
    {
        "instruction": "Explain the concept of Federated Learning in one sentence.",
        "context": "",
        "response": "Federated Learning is a decentralized machine learning approach where clients train models on local data and share only model updates with a central server.",
        "category": "open_qa",
    },
    {
        "instruction": "Summarize the key advantages of Low-Rank Adaptation (LoRA).",
        "context": "LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer, substantially reducing the number of trainable parameters and GPU memory requirements.",
        "response": "LoRA reduces trainable parameter count and memory consumption by injecting lightweight low-rank matrices while freezing pre-trained weights.",
        "category": "summarization",
    },
    {
        "instruction": "Classify whether the following sentiment is positive, negative, or neutral.",
        "context": "The adaptive communication compression algorithm converged with 70% fewer transmitted bytes while maintaining target perplexity.",
        "response": "positive",
        "category": "classification",
    },
    {
        "instruction": "Draft an email inviting a researcher to give a guest seminar on Differential Privacy.",
        "context": "",
        "response": "Dear Dr. Smith, We would be honored if you could deliver a guest seminar on your recent research in Differential Privacy and machine learning at our university colloquium next month.",
        "category": "creative_writing",
    },
    {
        "instruction": "List three metrics used to evaluate communication efficiency in distributed training.",
        "context": "",
        "response": "1. Total transmitted payload bytes per round\n2. Effective compression ratio\n3. Per-client uplink and downlink bandwidth consumption.",
        "category": "brainstorming",
    },
    {
        "instruction": "Extract the privacy budget values from the text.",
        "context": "Our mechanism was evaluated under a differential privacy guarantee of epsilon = 2.5 and delta = 1e-5 across 20 communication rounds.",
        "response": "Epsilon: 2.5, Delta: 1e-5",
        "category": "information_extraction",
    },
    {
        "instruction": "What is the primary trade-off in differentially private federated learning?",
        "context": "Adding Gaussian noise to client updates guarantees privacy but can degrade global model convergence and instruction-following fidelity.",
        "response": "The primary trade-off is between privacy preservation (lower epsilon) and model utility/convergence rate.",
        "category": "closed_qa",
    },
    {
        "instruction": "Why does non-IID data present challenges for federated averaging?",
        "context": "",
        "response": "Non-IID data causes local client optima to drift away from the global objective, leading to slow convergence and sub-optimal aggregation.",
        "category": "general_qa",
    },
]


def generate_synthetic_instruction_dataset(num_samples: int, seed: int = 42) -> List[Dict[str, str]]:
    """Generate a high-quality, category-balanced synthetic instruction dataset for offline testing."""
    rng = random.Random(seed)
    samples: List[Dict[str, str]] = []
    num_seeds = len(OFFLINE_SEED_EXAMPLES)

    for i in range(num_samples):
        base = OFFLINE_SEED_EXAMPLES[i % num_seeds]
        cat = base["category"]
        samples.append({
            "instruction": f"[ID:{i}] {base['instruction']}",
            "context": base["context"],
            "response": f"{base['response']} (Sample index {i}).",
            "category": cat,
        })
    rng.shuffle(samples)
    return samples


def load_dolly_subset(
    num_samples: int = 1000,
    seed: int = 42,
    cache_dir: str = "data/cache",
    use_huggingface: bool = True,
) -> List[Dict[str, str]]:
    """Load and curate a subset of Databricks Dolly 15K.

    Args:
        num_samples: Number of curated examples to return.
        seed: Random seed for deterministic reproducibility.
        cache_dir: Local path to cache the dataset.
        use_huggingface: Whether to attempt loading from Hugging Face Datasets hub.

    Returns:
        List of cleaned dictionaries containing instruction, context, response, and category.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"dolly_subset_{num_samples}_seed_{seed}.json")

    # Return cached file if available
    if os.path.exists(cache_file):
        logger.info(f"Loading cached Dolly subset from {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    raw_data: List[Dict[str, Any]] = []

    if use_huggingface:
        try:
            logger.info("Attempting to load 'databricks/databricks-dolly-15k' via Hugging Face...")
            from datasets import load_dataset
            hf_ds = load_dataset("databricks/databricks-dolly-15k", split="train")
            raw_data = [dict(item) for item in hf_ds]
            logger.info(f"Successfully fetched {len(raw_data)} samples from Hugging Face.")
        except Exception as e:
            logger.warning(f"Could not load from Hugging Face ({e}). Falling back to deterministic generator.")
            raw_data = []

    # Fallback to deterministic synthetic dataset if Hugging Face is unreachable
    if not raw_data:
        logger.info(f"Generating synthetic offline dataset with {num_samples} samples (seed={seed})")
        raw_data = generate_synthetic_instruction_dataset(num_samples=max(num_samples, 100), seed=seed)

    # Clean and filter
    cleaned: List[Dict[str, str]] = []
    for item in raw_data:
        inst = str(item.get("instruction", "")).strip()
        resp = str(item.get("response", "")).strip()
        ctx = str(item.get("context", "")).strip()
        cat = str(item.get("category", "general_qa")).strip()

        # Reject malformed or empty examples
        if not inst or not resp:
            continue
        cleaned.append({
            "instruction": inst,
            "context": ctx,
            "response": resp,
            "category": cat,
        })

    # Subsample deterministically
    rng = random.Random(seed)
    rng.shuffle(cleaned)
    final_samples = cleaned[:num_samples]

    # Save to cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(final_samples, f, indent=2)
    logger.info(f"Cached {len(final_samples)} curated samples to {cache_file}")

    return final_samples
