"""Instruction prompt formatting, tokenization, and causal LM batching module.

Formats instruction-following examples into structured prompts and prepares tokenized
tensors with proper attention masks and response-focused loss masking.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_WITH_CONTEXT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Context:\n{context}\n\n"
    "### Response:\n{response}"
)

PROMPT_TEMPLATE_WITHOUT_CONTEXT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}"
)

INFERENCE_PROMPT_WITH_CONTEXT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Context:\n{context}\n\n"
    "### Response:\n"
)

INFERENCE_PROMPT_WITHOUT_CONTEXT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def format_instruction_prompt(
    example: Dict[str, str],
    include_response: bool = True,
) -> str:
    """Format an instruction example into a standard prompt string."""
    instruction = example.get("instruction", "").strip()
    context = example.get("context", "").strip()
    response = example.get("response", "").strip()

    if include_response:
        if context:
            return PROMPT_TEMPLATE_WITH_CONTEXT.format(
                instruction=instruction, context=context, response=response
            )
        return PROMPT_TEMPLATE_WITHOUT_CONTEXT.format(
            instruction=instruction, response=response
        )
    else:
        if context:
            return INFERENCE_PROMPT_WITH_CONTEXT.format(
                instruction=instruction, context=context
            )
        return INFERENCE_PROMPT_WITHOUT_CONTEXT.format(instruction=instruction)


class SimpleSurrogateTokenizer:
    """A lightweight, deterministic whitespace/character hash tokenizer for CPU smoke testing."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3

    def encode(
        self,
        text: str,
        max_length: int = 128,
        truncation: bool = True,
        padding: bool = False,
    ) -> List[int]:
        tokens = [self.bos_token_id]
        words = text.split()
        for w in words:
            # Deterministic hash to token ID in range [4, vocab_size-1]
            token_id = 4 + (abs(hash(w)) % (self.vocab_size - 4))
            tokens.append(token_id)
        tokens.append(self.eos_token_id)

        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
            tokens[-1] = self.eos_token_id

        if padding and len(tokens) < max_length:
            tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        words = [f"[tok_{tid}]" for tid in token_ids if tid not in (self.pad_token_id, self.bos_token_id, self.eos_token_id)]
        return " ".join(words)

    def __call__(
        self,
        texts: Union[str, List[str]],
        max_length: int = 128,
        truncation: bool = True,
        padding: str = "max_length",
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]

        input_ids_list = [
            self.encode(t, max_length=max_length, truncation=truncation, padding=True)
            for t in texts
        ]
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def tokenize_batch(
    examples: List[Dict[str, str]],
    tokenizer: Any,
    max_length: int = 128,
    mask_prompt_labels: bool = True,
) -> Dict[str, torch.Tensor]:
    """Tokenize a batch of instruction examples for causal language modeling."""
    full_prompts = [format_instruction_prompt(ex, include_response=True) for ex in examples]

    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    encoded = tokenizer(
        full_prompts,
        max_length=max_length,
        truncation=True,
        padding="max_length" if hasattr(tokenizer, "__call__") else True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = input_ids.clone()

    # Mask pad tokens in labels so they don't contribute to loss
    pad_id = getattr(tokenizer, "pad_token_id", 0)
    labels[labels == pad_id] = -100

    if mask_prompt_labels:
        # Optionally mask the instruction prompt so loss is computed primarily on the response
        inference_prompts = [format_instruction_prompt(ex, include_response=False) for ex in examples]
        for i, prompt_text in enumerate(inference_prompts):
            if hasattr(tokenizer, "encode"):
                try:
                    prompt_ids = tokenizer.encode(prompt_text, truncation=True, max_length=max_length, padding=False)
                except TypeError:
                    prompt_ids = tokenizer.encode(prompt_text, truncation=True, max_length=max_length)
                
                # Number of valid tokens before pad
                valid_mask = (labels[i] != -100)
                total_valid_tokens = int(valid_mask.sum().item())
                # Mask up to prompt_len, ensuring at least one response token is active
                prompt_len = min(len(prompt_ids), max(total_valid_tokens - 1, 0))
                labels[i, :prompt_len] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
