"""Configuration module for Adaptive Communication Compression in Federated Fine-Tuning.

Defines strongly-typed, validated dataclasses for model, LoRA, data, differential privacy,
compression, and federated orchestration configurations with YAML serialization support.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional
import yaml


@dataclass
class ModelConfig:
    """Configuration for LLM architecture and loading."""
    model_name_or_path: str = "tiny"
    is_surrogate: bool = True
    vocab_size: int = 1000
    hidden_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    max_seq_length: int = 128
    torch_dtype: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoRAConfig:
    """Configuration for LoRA parameter-efficient adapter fine-tuning."""
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataConfig:
    """Configuration for instruction dataset loading and federated partitioning."""
    dataset_name: str = "databricks/databricks-dolly-15k"
    num_samples: int = 1000
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    partition_strategy: str = "non_iid_dirichlet"  # "iid" or "non_iid_dirichlet"
    dirichlet_alpha: float = 0.5
    num_clients: int = 3
    seed: int = 42
    cache_dir: str = "data/cache"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrivacyConfig:
    """Configuration for differential privacy mechanism and accounting."""
    enabled: bool = False
    privacy_unit: str = "client"  # "client" or "sample"
    clipping_norm: float = 1.0
    noise_multiplier: float = 1.0
    target_delta: float = 1e-5
    accountant_type: str = "rdp"  # Rényi Differential Privacy

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompressionConfig:
    """Configuration for communication compression strategies and error feedback."""
    enabled: bool = False
    strategy: str = "none"  # "none", "fixed_topk", "adaptive_adacomp"
    fixed_k_ratio: float = 0.10
    adaptive_min_ratio: float = 0.02
    adaptive_max_ratio: float = 0.25
    energy_threshold: float = 0.85
    error_feedback: bool = True
    quantization_bits: int = 16  # FP16 payload representation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FederatedConfig:
    """Configuration for federated optimization rounds and local client training."""
    num_rounds: int = 5
    clients_per_round: int = 3
    local_epochs: int = 1
    local_batch_size: int = 4
    local_lr: float = 2e-4
    aggregation_strategy: str = "fedavg"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Root configuration uniting all experimental components."""
    experiment_id: str = "exp_baseline"
    description: str = "Federated LoRA fine-tuning experiment"
    output_dir: str = "outputs"
    seed: int = 42
    device: str = "cpu"  # "cpu" or "cuda"

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Construct ExperimentConfig from a dictionary with nested sub-dataclasses."""
        cfg = cls()
        for key, val in data.items():
            if key == "model" and isinstance(val, dict):
                cfg.model = ModelConfig(**val)
            elif key == "lora" and isinstance(val, dict):
                cfg.lora = LoRAConfig(**val)
            elif key == "data" and isinstance(val, dict):
                cfg.data = DataConfig(**val)
            elif key == "privacy" and isinstance(val, dict):
                cfg.privacy = PrivacyConfig(**val)
            elif key == "compression" and isinstance(val, dict):
                cfg.compression = CompressionConfig(**val)
            elif key == "federated" and isinstance(val, dict):
                cfg.federated = FederatedConfig(**val)
            elif hasattr(cfg, key):
                setattr(cfg, key, val)
        return cfg

    @classmethod
    def from_yaml(cls, yaml_path: str) -> ExperimentConfig:
        """Load and parse an ExperimentConfig from a YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config YAML not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_yaml(self, yaml_path: str) -> None:
        """Serialize ExperimentConfig to a YAML file."""
        os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False, indent=2)
