"""Model loading and surrogate causal language model definition module.

Provides a dual-mode model architecture:
1. TinyCausalLM: A parameter-matched, lightweight causal transformer for fast local CPU testing.
2. Hugging Face AutoModelForCausalLM: Full model loading for GPU-accelerated Qwen2.5-0.5B training.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig
from src.data.preprocess import SimpleSurrogateTokenizer

logger = logging.getLogger(__name__)


class TinyAttentionBlock(nn.Module):
    """Multi-Head Self-Attention block with explicit target linear projections for LoRA."""

    def __init__(self, hidden_size: int = 128, num_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Named projections that match standard LoRA target module names
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask (prevent looking forward)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            # attention_mask: (batch_size, seq_len) with 0 for pad
            pad_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
            scores = scores + pad_mask

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return self.o_proj(context)


class TinyTransformerLayer(nn.Module):
    """Single Transformer decoder layer with Pre-LayerNorm architecture."""

    def __init__(self, hidden_size: int = 128, num_heads: int = 4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.attn = TinyAttentionBlock(hidden_size=hidden_size, num_heads=num_heads)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN residual connections
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyCausalLM(nn.Module):
    """Lightweight, parameter-matched causal language model for fast CPU verification."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TinyTransformerLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
            )
            for _ in range(config.num_hidden_layers)
        ])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift tokens for next-token causal prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        temperature: float = 0.7,
        eos_token_id: int = 1,
    ) -> torch.Tensor:
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            outputs = self.forward(generated)
            next_token_logits = outputs["logits"][:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break
        return generated


def load_model_and_tokenizer(
    config: ModelConfig,
    device: str = "cpu",
) -> Tuple[nn.Module, Any]:
    """Load model and tokenizer based on ModelConfig (dual-mode).

    Returns:
        Tuple of (model, tokenizer) moved to target device.
    """
    if config.is_surrogate:
        logger.info("Initializing surrogate TinyCausalLM on %s...", device)
        model = TinyCausalLM(config).to(device)
        tokenizer = SimpleSurrogateTokenizer(vocab_size=config.vocab_size)
        return model, tokenizer

    # Real Hugging Face model
    logger.info("Loading Hugging Face model '%s' on %s...", config.model_name_or_path, device)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            dtype=torch_dtype,
            device_map=device if device != "cpu" else None,
        )
    except (TypeError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device if device != "cpu" else None,
        )
    if device == "cpu":
        model = model.to("cpu")

    return model, tokenizer
