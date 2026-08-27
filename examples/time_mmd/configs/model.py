"""Model configuration for Time-MMD dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from tsfmx.utils.yaml import load_yaml


@dataclass
class AdapterConfig:
    """Configuration for the time series foundation model adapter."""

    type: Literal["chronos", "timesfm"] = "timesfm"
    pretrained_repo: str = "google/timesfm-2.5-200m-pytorch"
    patch_len: int = 32


@dataclass
class FusionConfig:
    """Configuration for the multimodal fusion layer."""

    text_encoder_type: Literal["english", "japanese"] = "english"
    text_embedding_dims: int = 384
    num_fusion_layers: int = 1
    fusion_hidden_dims: list[int] = field(default_factory=list)
    fusion_normalize: bool = False


@dataclass
class ModelConfig:
    """Configuration for the multimodal model."""

    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> ModelConfig:
        config_dict = load_yaml(path)
        return cls(
            adapter=AdapterConfig(**config_dict.get("adapter", {})),
            fusion=FusionConfig(**config_dict.get("fusion", {})),
        )
