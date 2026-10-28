"""LoRA (Low-Rank Adaptation) adapter injection, extraction, and management module.

Applies parameter-efficient low-rank adaptation to target attention projections, freezes base
parameters, and provides extraction and aggregation utilities for adapter parameter deltas.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn

from src.config import LoRAConfig

logger = logging.getLogger(__name__)


class NativeLoRALinear(nn.Module):
    """Native PyTorch LoRA wrapper for linear layers.

    Implements: W_new(x) = W_base(x) + (alpha / r) * B(A(x))
    where W_base is frozen, A is initialized with Gaussian, and B is initialized to 0.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_linear = base_linear
        # Freeze base parameters
        for param in self.base_linear.parameters():
            param.requires_grad = False

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Determine base layer's dtype and device
        weight = base_linear.weight
        self.dtype = getattr(weight, "dtype", torch.float32)
        self.device = getattr(weight, "device", torch.device("cpu"))

        # LoRA factor matrices matching base layer dtype and device
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=self.dtype, device=self.device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=self.dtype, device=self.device))

        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        # Ensure factor matrices match input x dtype and device
        lora_A = self.lora_A.to(dtype=x.dtype, device=x.device)
        lora_B = self.lora_B.to(dtype=x.dtype, device=x.device)
        lora_out = self.lora_dropout(x)
        lora_out = torch.matmul(lora_out, lora_A.t())
        lora_out = torch.matmul(lora_out, lora_B.t()) * self.scaling
        return base_out + lora_out


def apply_lora_to_model(
    model: nn.Module,
    lora_config: LoRAConfig,
    use_peft_library: bool = True,
) -> nn.Module:
    """Inject LoRA adapters into target modules and freeze non-adapter parameters."""
    if use_peft_library:
        try:
            from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
            peft_cfg = PeftLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                target_modules=lora_config.target_modules,
                bias=lora_config.bias,
            )
            peft_model = get_peft_model(model, peft_cfg)
            logger.info("Successfully applied LoRA using PEFT library.")
            return peft_model
        except Exception as e:
            logger.warning("PEFT library application failed (%s). Falling back to native LoRA injection.", e)

    # Native injection for surrogate or fallback
    target_names = set(lora_config.target_modules)
    injected_count = 0

    for name, module in list(model.named_modules()):
        for child_name, child_module in list(module.named_children()):
            if child_name in target_names and isinstance(child_module, nn.Linear):
                wrapped = NativeLoRALinear(
                    base_linear=child_module,
                    r=lora_config.r,
                    lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                )
                setattr(module, child_name, wrapped)
                injected_count += 1

    # Freeze all parameters that are not LoRA parameters
    for p_name, param in model.named_parameters():
        if "lora_" not in p_name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    logger.info("Injected native LoRA into %d linear modules.", injected_count)
    return model


def get_trainable_parameter_stats(model: nn.Module) -> Dict[str, Any]:
    """Calculate trainable vs total parameter statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percent = 100.0 * (trainable_params / max(total_params, 1))

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": round(trainable_percent, 4),
    }


def extract_adapter_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract a clone of all trainable adapter parameters."""
    adapter_params: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad or "lora" in name.lower():
            adapter_params[name] = param.detach().clone()
    return adapter_params


def load_adapter_parameters(model: nn.Module, adapter_params: Dict[str, torch.Tensor]) -> None:
    """Load adapter parameters back into the model."""
    with torch.no_grad():
        for name, tensor in adapter_params.items():
            param = dict(model.named_parameters()).get(name)
            if param is not None:
                param.copy_(tensor.to(device=param.device, dtype=param.dtype))


def compute_adapter_delta(
    local_state: Dict[str, torch.Tensor],
    global_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute parameter delta: Delta = theta_local - theta_global."""
    delta: Dict[str, torch.Tensor] = {}
    for name, local_p in local_state.items():
        if name in global_state:
            delta[name] = local_p - global_state[name]
        else:
            delta[name] = local_p.clone()
    return delta


def apply_adapter_delta(
    global_state: Dict[str, torch.Tensor],
    delta: Dict[str, torch.Tensor],
    step_size: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Apply parameter delta: theta_new = theta_global + step_size * Delta."""
    new_state: Dict[str, torch.Tensor] = {}
    for name, global_p in global_state.items():
        if name in delta:
            new_state[name] = global_p + step_size * delta[name]
        else:
            new_state[name] = global_p.clone()
    return new_state
