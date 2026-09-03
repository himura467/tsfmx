"""Adapter, decoder, and text encoder construction from a Time-MMD model config."""

from typing import Literal

import torch

from examples.time_mmd.configs.model import ModelConfig
from tsfmx.decoder import MultimodalDecoder, MultimodalDecoderConfig
from tsfmx.text_encoder.base import TextEncoderBase
from tsfmx.text_encoder.english import EnglishTextEncoder
from tsfmx.text_encoder.japanese import JapaneseTextEncoder
from tsfmx.tsfm.base import TsfmAdapter
from tsfmx.tsfm.chronos import Chronos2Adapter
from tsfmx.tsfm.timesfm import TimesFM2p5Adapter
from tsfmx.utils.logging import get_logger

_logger = get_logger()


def build_text_encoder(text_encoder_type: Literal["english", "japanese"], device: torch.device) -> TextEncoderBase:
    """Instantiate the text encoder named by the config.

    Args:
        text_encoder_type: Which encoder to use — "english" or "japanese".
        device: Device to run the encoder on.

    Returns:
        Initialized TextEncoderBase instance.

    Raises:
        ValueError: If text_encoder_type is not recognized.
    """
    match text_encoder_type:
        case "english":
            _logger.info("Initializing EnglishTextEncoder")
            return EnglishTextEncoder(device=device)
        case "japanese":
            _logger.info("Initializing JapaneseTextEncoder")
            return JapaneseTextEncoder(device=device)
        case _:
            raise ValueError(f"Unknown text encoder type: {text_encoder_type!r}")


def build_adapter(model_config: ModelConfig, device: torch.device) -> TsfmAdapter:
    """Load the pretrained TSFM adapter named by the config.

    Args:
        model_config: Model configuration naming the adapter type and pretrained repo.
        device: Device to load the pretrained weights onto.

    Returns:
        Initialized adapter.

    Raises:
        NotImplementedError: If the configured adapter type is unsupported.
        ValueError: If the adapter's patch length does not match the config value, which
            would silently misalign text patches against the cached time series patches.
    """
    _logger.info("Loading pretrained adapter from %s on %s", model_config.adapter.pretrained_repo, device)

    adapter: TsfmAdapter
    match model_config.adapter.type:
        case "chronos":
            adapter = Chronos2Adapter.from_pretrained(device, repo_id=model_config.adapter.pretrained_repo)
        case "timesfm":
            adapter = TimesFM2p5Adapter.from_pretrained(device, repo_id=model_config.adapter.pretrained_repo)
        case _ as t:
            raise NotImplementedError(f"Unsupported adapter type: {t!r}")

    if adapter.patch_len != model_config.adapter.patch_len:
        raise ValueError(
            f"adapter.patch_len ({adapter.patch_len}) does not match "
            f"model_config.adapter.patch_len ({model_config.adapter.patch_len}); "
            "the cached dataset was built with the config value — rebuild the cache or fix the config."
        )

    return adapter


def build_decoder(
    model_config: ModelConfig,
    device: torch.device,
    num_fusion_layers: int | None = None,
    fusion_hidden_dims: list[int] | None = None,
) -> MultimodalDecoder:
    """Build a MultimodalDecoder with a pretrained adapter and an initialized fusion head.

    Args:
        model_config: Model configuration naming the adapter and fusion defaults.
        device: Device to load the model onto.
        num_fusion_layers: Overrides the config value when given, as sweep trials do.
        fusion_hidden_dims: Overrides the config value when given, as sweep trials do.

    Returns:
        MultimodalDecoder on device.
    """
    resolved_layers = num_fusion_layers if num_fusion_layers is not None else model_config.fusion.num_fusion_layers
    resolved_hidden_dims = (
        fusion_hidden_dims if fusion_hidden_dims is not None else model_config.fusion.fusion_hidden_dims
    )

    _logger.info(
        "Creating MultimodalDecoder: num_fusion_layers=%d, fusion_hidden_dims=%s",
        resolved_layers,
        resolved_hidden_dims,
    )
    config = MultimodalDecoderConfig(
        text_embedding_dims=model_config.fusion.text_embedding_dims,
        num_fusion_layers=resolved_layers,
        fusion_hidden_dims=resolved_hidden_dims,
        fusion_normalize=model_config.fusion.fusion_normalize,
    )
    return MultimodalDecoder(build_adapter(model_config, device), config).to(device)
