"""Wrappers for loading self-supervised vision encoders.

These encoders adapt popular self-supervised checkpoints (DINOv2, MAE, BEiT,
CLIP, MoCo-v3) to the project's latent access contract so they can participate
in multi-stage training pipelines as encoder stages.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from spectramr.models.interfaces.latent_access import LatentAccessMixin

if torch.__version__ < "1.12":  # pragma: no cover - environment guard
    raise ImportError("SSL encoders require torch>=1.12 for ViT checkpoints.")


@dataclass(slots=True)
class EncoderOutputs:
    """Container for standardized encoder outputs."""

    embedding: torch.Tensor
    sequence: torch.Tensor | None = None
    pooled: torch.Tensor | None = None


class HuggingFaceVisionEncoder(LatentAccessMixin, nn.Module):
    """Base class for Hugging Face vision encoders."""

    def __init__(
        self,
        *,
        model_name: str,
        feature_key: str = "last_hidden_state",
        pooling: str = "tokens",
        trainable: bool = False,
        model_loader: Callable[..., Any] | None = None,
        processor_loader: Callable[..., Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """__init__.

        Args:
            model_name (str): Description.
            feature_key (str): Description.
            pooling (str): Description.
            trainable (bool): Description.
            model_loader (Callable[..., Any] | None): Description.
            processor_loader (Callable[..., Any] | None): Description.
            model_kwargs (dict[str, Any] | None): Description.
            processor_kwargs (dict[str, Any] | None): Description.
        """
        super().__init__()
        model_kwargs = model_kwargs or {}
        processor_kwargs = processor_kwargs or {}

        if model_loader is None:
            from transformers import AutoModel

            model_loader = AutoModel.from_pretrained
        if processor_loader is None:
            from transformers import AutoImageProcessor

            processor_loader = AutoImageProcessor.from_pretrained

        self.model_name = model_name
        self.pooling = pooling
        self.feature_key = feature_key
        self.model = model_loader(model_name, **model_kwargs)
        self._processor = processor_loader(model_name, **processor_kwargs)

        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    @property
    def image_processor(self) -> Any:
        """Return the associated image processor."""

        return self._processor

    @property
    def name(self) -> str:
        """Get the model name."""
        return "hf_ssl"

    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def preprocess(self, images: Any, **kwargs: Any) -> torch.Tensor:
        """Preprocess raw images using the encoder's processor."""

        processed = self._processor(images, return_tensors="pt", **kwargs)
        pixel_values = processed["pixel_values"]
        if isinstance(images, torch.Tensor):
            return pixel_values.to(images.device)
        return pixel_values

    def _extract_sequence_tensor(self, outputs: Any) -> torch.Tensor:
        """_extract_sequence_tensor.

        Args:
            outputs (Any): Description.
        Returns:
            torch.Tensor: Description.
        """
        if isinstance(outputs, dict) and self.feature_key in outputs:
            return outputs[self.feature_key]
        if hasattr(outputs, self.feature_key):
            return getattr(outputs, self.feature_key)
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0]
        raise ValueError(
            f"Unable to extract feature '{self.feature_key}' from outputs of type {type(outputs)}",
        )

    def _extract_pooled_tensor(self, outputs: Any) -> torch.Tensor | None:
        """_extract_pooled_tensor.

        Args:
            outputs (Any): Description.
        Returns:
            torch.Tensor | None: Description.
        """
        if isinstance(outputs, dict) and "pooler_output" in outputs:
            return outputs["pooler_output"]
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output
        return None

    def _apply_pooling(self, sequence: torch.Tensor) -> torch.Tensor:
        """_apply_pooling.

        Args:
            sequence (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        if self.pooling == "tokens":
            return sequence
        if self.pooling == "cls":
            return sequence[:, 0]
        if self.pooling == "mean":
            return sequence.mean(dim=1)
        raise ValueError(f"Unsupported pooling strategy: {self.pooling}")

    def forward(self, pixel_values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward.

        Args:
            pixel_values (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for HuggingFaceVisionEncoder.

        Executes PyTorch tensor operations.

        Args:
            pixel_values (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        outputs = self.model(pixel_values=pixel_values, **kwargs)
        sequence = self._extract_sequence_tensor(outputs)
        pooled = self._extract_pooled_tensor(outputs)
        embedding = self._apply_pooling(sequence)

        extras: dict[str, torch.Tensor] = {"sequence": sequence}
        if pooled is not None:
            extras["pooled"] = pooled

        self._set_latent_state(embedding=embedding, **extras)
        return embedding


class Dinov2SmallEncoder(HuggingFaceVisionEncoder):
    """DINOv2 small encoder wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        """__init__."""
        super().__init__(model_name="facebook/dinov2-small", **kwargs)


class Dinov3VisionEncoder(HuggingFaceVisionEncoder):
    """Base encoder for DINOv3 ViT backbones hosted on Hugging Face."""

    def __init__(
        self,
        *,
        arch: str,
        dataset: str = "lvd1689m",
        version: str | None = None,
        use_auth_token: str | bool | None = None,
        **kwargs: Any,
    ) -> None:
        """__init__.

        Args:
            arch (str): Description.
            dataset (str): Description.
            version (str | None): Description.
            use_auth_token (str | bool | None): Description.
        """
        model_kwargs = dict(kwargs.pop("model_kwargs", {}))
        processor_kwargs = dict(kwargs.pop("processor_kwargs", {}))
        if use_auth_token is not None:
            model_kwargs.setdefault("use_auth_token", use_auth_token)
            processor_kwargs.setdefault("use_auth_token", use_auth_token)
        kwargs["model_kwargs"] = model_kwargs
        kwargs["processor_kwargs"] = processor_kwargs

        normalized_dataset = dataset.replace("_", "-").lower()
        normalized_version = version.replace("_", "-").lower() if version else ""
        version_suffix = f"-{normalized_version}" if normalized_version else ""
        model_name = f"facebook/dinov3-{arch}-pretrain-{normalized_dataset}{version_suffix}"
        super().__init__(
            model_name=model_name,
            feature_key="last_hidden_state",
            **kwargs,
        )


class Dinov3VitS16Encoder(Dinov3VisionEncoder):
    """DINOv3 ViT-S/16 encoder wrapper."""

    def __init__(self, *, dataset: str = "lvd1689m", **kwargs: Any) -> None:
        """__init__.

        Args:
            dataset (str): Description.
        """
        super().__init__(arch="vits16", dataset=dataset, **kwargs)


class Dinov3VitS16PlusEncoder(Dinov3VisionEncoder):
    """DINOv3 ViT-S+/16 encoder wrapper."""

    def __init__(self, *, dataset: str = "lvd1689m", **kwargs: Any) -> None:
        """__init__.

        Args:
            dataset (str): Description.
        """
        super().__init__(arch="vits16plus", dataset=dataset, **kwargs)


class Dinov3ConvNextTinyEncoder(Dinov3VisionEncoder):
    """DINOv3 ConvNeXt-Tiny encoder wrapper."""

    def __init__(self, *, dataset: str = "lvd1689m", **kwargs: Any) -> None:
        """__init__.

        Args:
            dataset (str): Description.
        """
        super().__init__(arch="convnext-tiny", dataset=dataset, **kwargs)


class Dinov3ConvNextSmallEncoder(Dinov3VisionEncoder):
    """DINOv3 ConvNeXt-Small encoder wrapper."""

    def __init__(self, *, dataset: str = "lvd1689m", **kwargs: Any) -> None:
        """__init__.

        Args:
            dataset (str): Description.
        """
        super().__init__(arch="convnext-small", dataset=dataset, **kwargs)


class MaeBaseEncoder(HuggingFaceVisionEncoder):
    """MAE base encoder wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        """__init__."""
        super().__init__(model_name="facebook/mae-base-patch16-224", **kwargs)


class BeitBaseEncoder(HuggingFaceVisionEncoder):
    """BEiT base encoder wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        """__init__."""
        super().__init__(
            model_name="microsoft/beit-base-patch16-224-in22k",
            **kwargs,
        )


class ClipVitBaseEncoder(HuggingFaceVisionEncoder):
    """CLIP ViT-B/32 encoder wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        """__init__."""
        if "model_loader" not in kwargs:
            from transformers import CLIPModel

            kwargs["model_loader"] = CLIPModel.from_pretrained
        if "processor_loader" not in kwargs:
            from transformers import CLIPProcessor

            kwargs["processor_loader"] = CLIPProcessor.from_pretrained
        super().__init__(model_name="openai/clip-vit-base-patch32", **kwargs)


class MocoV3VitSmallEncoder(LatentAccessMixin, nn.Module):
    """MoCo-v3 ViT-S encoder wrapper loaded via torch.hub."""

    def __init__(
        self,
        *,
        repo_or_dir: str = "facebookresearch/moco-v3",
        model: str = "vit_small",
        pretrained: bool = True,
        trainable: bool = False,
        loader: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """__init__.

        Args:
            repo_or_dir (str): Description.
            model (str): Description.
            pretrained (bool): Description.
            trainable (bool): Description.
            loader (Callable[..., Any] | None): Description.
        """
        super().__init__()
        if loader is None:
            loader = torch.hub.load
        self.model = loader(
            repo_or_dir,
            model,
            pretrained=pretrained,
            **kwargs,
        )
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    @property
    def image_processor(self) -> None:
        """MoCo-v3 does not ship a processor."""

        return None

    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """forward.

        Args:
            pixel_values (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for MocoV3VitSmallEncoder.

        Executes PyTorch tensor operations.

        Args:
            pixel_values (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        outputs = self.model(pixel_values, **kwargs)
        if isinstance(outputs, (tuple, list)):
            embedding = outputs[0]
        else:
            embedding = outputs
        self._set_latent_state(embedding=embedding)
        return embedding


# Alias for backward compatibility
HFSSLEncoder = HuggingFaceVisionEncoder

__all__ = [
    "BeitBaseEncoder",
    "ClipVitBaseEncoder",
    "Dinov2SmallEncoder",
    "Dinov3ConvNextSmallEncoder",
    "Dinov3ConvNextTinyEncoder",
    "Dinov3VisionEncoder",
    "Dinov3VitS16Encoder",
    "Dinov3VitS16PlusEncoder",
    "HFSSLEncoder",  # Alias for backward compatibility
    "MaeBaseEncoder",
    "MocoV3VitSmallEncoder",
]
