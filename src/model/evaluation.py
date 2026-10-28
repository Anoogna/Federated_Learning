"""Model evaluation module for federated LLM fine-tuning.

Evaluates validation cross-entropy loss, perplexity, and generates responses on a fixed set
of instruction-following probes to monitor qualitative model progression.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from src.data.preprocess import tokenize_batch, format_instruction_prompt

logger = logging.getLogger(__name__)

# Standard instruction probes to track qualitative progress across federated rounds
DEFAULT_PROBES = [
    {
        "instruction": "Explain the goal of Federated Learning in one concise sentence.",
        "context": "",
    },
    {
        "instruction": "What is the primary function of Differential Privacy?",
        "context": "",
    },
    {
        "instruction": "Summarize how LoRA reduces memory usage during fine-tuning.",
        "context": "LoRA freezes the pre-trained weights and adds low-rank decomposition matrices.",
    },
]


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    eval_examples: List[Dict[str, str]],
    tokenizer: Any,
    batch_size: int = 4,
    max_length: int = 128,
    device: str = "cpu",
) -> Dict[str, float]:
    """Compute validation loss and perplexity on a held-out dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    if not eval_examples:
        return {"loss": 0.0, "perplexity": 1.0}

    for i in range(0, len(eval_examples), batch_size):
        batch_examples = eval_examples[i : i + batch_size]
        batch = tokenize_batch(
            batch_examples,
            tokenizer=tokenizer,
            max_length=max_length,
            mask_prompt_labels=True,
        )
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.get("loss")
        if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
            total_loss += loss.item()
            num_batches += 1

    avg_loss = (total_loss / num_batches) if num_batches > 0 else 20.0
    # Clip loss to prevent numerical overflow in exp()
    clipped_loss = min(avg_loss, 20.0)
    perplexity = math.exp(clipped_loss)

    return {
        "loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 4),
    }


@torch.no_grad()
def run_instruction_probes(
    model: nn.Module,
    tokenizer: Any,
    probes: Optional[List[Dict[str, str]]] = None,
    max_new_tokens: int = 24,
    device: str = "cpu",
) -> List[Dict[str, str]]:
    """Generate completions on fixed instruction probes to observe generation quality."""
    model.eval()
    if probes is None:
        probes = DEFAULT_PROBES

    results: List[Dict[str, str]] = []

    for probe in probes:
        prompt_str = format_instruction_prompt(probe, include_response=False)
        if hasattr(tokenizer, "__call__"):
            encoded = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=128)
            input_ids = encoded["input_ids"].to(device)
        else:
            token_ids = tokenizer.encode(prompt_str, max_length=128, truncation=True)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        generated_text = ""
        try:
            if hasattr(model, "generate"):
                pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)
                gen_tokens = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
                # Slice newly generated tokens
                new_tokens = gen_tokens[0][input_ids.shape[1] :].tolist()
                if hasattr(tokenizer, "decode"):
                    generated_text = tokenizer.decode(new_tokens)
                else:
                    generated_text = str(new_tokens)
            else:
                generated_text = "[Generation not supported for this model type]"
        except Exception as e:
            generated_text = f"[Probe generation skipped due to: {type(e).__name__}]"

        results.append({
            "instruction": probe["instruction"],
            "generated_response": generated_text.strip(),
        })

    return results
