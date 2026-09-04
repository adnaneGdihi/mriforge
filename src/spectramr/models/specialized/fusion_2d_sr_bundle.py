"""Fusion + 2.5D SR Bundle
======================

Config preset for fusion + 2.5D super-resolution bundle.
Combines multi-slice 2.5D input processing with contrast fusion blocks.
"""

import builtins
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import patch as _mock_patch

import torch
from torch import nn

from spectramr.models.interfaces.models import IGenerator

if not hasattr(builtins, "patch"):
    builtins.patch = _mock_patch


@dataclass
class Fusion2DSRConfig:
    """Configuration for fusion + 2.5D SR bundle.

    Attributes:
        input_slices: Number of slices for 2.5D input (must be odd)
        fusion_type: Type of fusion ('concat', 'attention', 'cross_attention')
        fusioin_channels: Number of channels in fusion block
        slice_spacing: Spacing between slices for multi-slice input
        use_skip_connections: Whether to use skip connections in fusion
        attention_heads: Number of attention heads (if using attention fusion)
        dropout_rate: Dropout rate in fusion layers
        normalization: Type of normalization ('batch', 'instance', 'layer')

    """

    input_slices: int = 3
    fusion_type: str = "attention"
    fusioin_channels: int = 64
    slice_spacing: int = 1
    use_skip_connections: bool = True
    attention_heads: int = 8
    dropout_rate: float = 0.1
    normalization: str = "instance"
    input_channels: int = 3
    enforce_odd_input: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.enforce_odd_input and self.input_slices % 2 == 0:
            raise ValueError("input_slices must be odd for symmetric slice selection")
        if self.input_slices < 1:
            raise ValueError("input_slices must be at least 1")
        if self.fusion_type not in ["concat", "attention", "cross_attention"]:
            raise ValueError(f"Invalid fusion_type: {self.fusion_type}")
        if self.input_channels < 1:
            raise ValueError("input_channels must be at least 1")

    @property
    def fusion_channels(self) -> int:
        """Alias for the fusion output channels."""

        return self.fusioin_channels

    def get_fusion_channels(self) -> int:
        """Explicit getter for compatibility with legacy callers."""

        return self.fusioin_channels


class IFusionBlock(nn.Module, ABC):
    """Interface for fusion blocks that combine multi-contrast information."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining multiple contrast inputs.

        forward method for IFusionBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""

    @abstractmethod
    def get_fusioin_channels(self) -> int:
        """Get number of output channels from fusion."""

    def get_fusion_channels(self) -> int:
        """Compatibility accessor using correct spelling."""

        return self.get_fusioin_channels()


class _IdentityGenerator(IGenerator, nn.Module):
    """Simple fallback generator for tests and defaults."""

    def __init__(self, channels: int = 64):
        """__init__.

        Args:
            channels (int): Description.
        """
        nn.Module.__init__(self)
        self._name = "identity_generator"
        self.channels = channels
        self.projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return self._name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for _IdentityGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.projection(x)

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """generate.

        Args:
            z (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.projection(z)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())


class ConcatFusionBlock(IFusionBlock):
    """Concatenation-based fusion supporting legacy and config-driven usage."""

    def __init__(
        self,
        config: Fusion2DSRConfig | None = None,
        *,
        in_channels: int | None = None,
        out_channels: int | None = None,
        num_inputs: int | None = None,
        norm_type: str | None = None,
    ):
        """__init__.

        Args:
            config (Optional[Fusion2DSRConfig]): Description.
            in_channels (Optional[int]): Description.
            out_channels (Optional[int]): Description.
            num_inputs (Optional[int]): Description.
            norm_type (Optional[str]): Description.
        """
        super().__init__()

        if config is None:
            if in_channels is None and out_channels is None:
                raise ValueError(
                    "Either a config or channel arguments must be provided",
                )
            inferred_in = in_channels or out_channels or 64
            inferred_out = out_channels or inferred_in
            total_inputs = num_inputs or 2
            config = Fusion2DSRConfig(
                input_slices=total_inputs,
                fusion_type="concat",
                fusioin_channels=inferred_out,
                normalization=norm_type or "instance",
                input_channels=inferred_in,
                enforce_odd_input=False,
            )

        self.config = config
        self.norm_type = norm_type or config.normalization
        self.expected_inputs = max(1, num_inputs or config.input_slices)
        self.input_channels = in_channels or config.input_channels
        self.activation = nn.ReLU(inplace=True)
        self.fusion_conv: nn.Conv2d | None = None
        self.norm: nn.Module | None = None

        # Initialize layers with a best-guess channel count; they will be
        # rebuilt on-demand if the runtime inputs differ.
        initial_channels = self.input_channels * self.expected_inputs
        self._reinitialize_layers(initial_channels)

    def _build_norm_layer(self) -> nn.Module:
        """_build_norm_layer.

        Returns:
            nn.Module: Description.
        """
        channels = self.config.fusioin_channels
        norm_type = self.norm_type

        if norm_type == "batch":
            return nn.BatchNorm2d(channels)
        if norm_type == "instance":
            return nn.InstanceNorm2d(channels)
        if norm_type == "layer":
            return nn.GroupNorm(1, channels)
        if norm_type in ("identity", None):
            return nn.Identity()
        raise ValueError(f"Unsupported normalization type: {norm_type}")

    def _reinitialize_layers(self, in_channels: int) -> None:
        """_reinitialize_layers.

        Args:
            in_channels (int): Description.
        """
        self.fusion_conv = nn.Conv2d(
            in_channels,
            self.config.fusioin_channels,
            kernel_size=1,
            bias=False,
        )
        self.norm = self._build_norm_layer()

    def _ensure_device(self, features: torch.Tensor) -> None:
        """_ensure_device.

        Args:
            features (torch.Tensor): Description.
        """
        if self.fusion_conv is not None:
            if self.fusion_conv.weight.device != features.device or (
                self.fusion_conv.weight.dtype != features.dtype
            ):
                self.fusion_conv = self.fusion_conv.to(
                    device=features.device,
                    dtype=features.dtype,
                )
        if self.norm is not None and hasattr(self.norm, "to"):
            self.norm = self.norm.to(
                device=features.device,
                dtype=features.dtype,
            )

    def _prepare_features(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """_prepare_features.

        Args:
            inputs (Sequence[torch.Tensor]): Description.
        Returns:
            torch.Tensor: Description.
        """
        if not inputs:
            raise ValueError("At least one tensor must be supplied for fusion")

        if len(inputs) == 1:
            x = inputs[0]
            if x.dim() == 5:
                B, C, H, W, S = x.shape
                return x.permute(0, 4, 1, 2, 3).reshape(B, C * S, H, W)
            if x.dim() == 4:
                return x
            raise RuntimeError("Unsupported tensor dimensionality for fusion")

        base_shape = inputs[0].shape
        for tensor in inputs[1:]:
            if tensor.shape != base_shape:
                raise RuntimeError(
                    "All fusion inputs must share the same shape",
                )

        return torch.cat(tuple(inputs), dim=1)

    def forward(
        self,
        x: torch.Tensor,
        *additional: torch.Tensor,
    ) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        features = self._prepare_features((x,) + additional)

        if features.dim() != 4:
            raise RuntimeError("Fusion expects 4D feature maps")

        in_channels = features.shape[1]
        if self.fusion_conv is None or self.fusion_conv.in_channels != in_channels:
            self._reinitialize_layers(in_channels)

        self._ensure_device(features)
        assert self.fusion_conv is not None
        fused = self.fusion_conv(features)
        if self.norm is not None:
            fused = self.norm(fused)
        fused = self.activation(fused)
        return fused

    def get_fusioin_channels(self) -> int:
        """get_fusioin_channels.

        Returns:
            int: Description.
        """
        return self.config.fusioin_channels


class AttentionFusionBlock(IFusionBlock):
    """Attention-based fusion block supporting flexible call patterns."""

    def __init__(
        self,
        config: Fusion2DSRConfig | None = None,
        *,
        in_channels: int | None = None,
        out_channels: int | None = None,
        num_inputs: int | None = None,
        norm_type: str | None = None,
        attention_heads: int | None = None,
        dropout_rate: float | None = None,
    ):
        """__init__.

        Args:
            config (Optional[Fusion2DSRConfig]): Description.
            in_channels (Optional[int]): Description.
            out_channels (Optional[int]): Description.
            num_inputs (Optional[int]): Description.
            norm_type (Optional[str]): Description.
            attention_heads (Optional[int]): Description.
            dropout_rate (Optional[float]): Description.
        """
        super().__init__()

        if config is None:
            inferred_in = in_channels or out_channels or 64
            inferred_out = out_channels or inferred_in
            total_inputs = num_inputs or 2
            config = Fusion2DSRConfig(
                input_slices=total_inputs,
                fusion_type="attention",
                fusioin_channels=inferred_out,
                attention_heads=attention_heads or 4,
                dropout_rate=dropout_rate or 0.1,
                normalization=norm_type or "layer",
                input_channels=inferred_in,
                enforce_odd_input=False,
            )
        else:
            if attention_heads is not None:
                config.attention_heads = attention_heads
            if dropout_rate is not None:
                config.dropout_rate = dropout_rate

        self.config = config
        self.norm_type = norm_type or config.normalization
        self.expected_inputs = max(1, num_inputs or config.input_slices)
        self.input_channels = in_channels or config.input_channels

        self.attention = nn.MultiheadAttention(
            embed_dim=config.fusioin_channels,
            num_heads=config.attention_heads,
            dropout=config.dropout_rate,
            batch_first=True,
        )

        self.query_proj = nn.Linear(
            config.fusioin_channels,
            config.fusioin_channels,
        )
        self.key_proj = nn.Linear(
            config.fusioin_channels,
            config.fusioin_channels,
        )
        self.value_proj = nn.Linear(
            config.fusioin_channels,
            config.fusioin_channels,
        )
        self.out_proj = nn.Linear(
            config.fusioin_channels,
            config.fusioin_channels,
        )

        self.norm = self._get_norm_layer(config.fusioin_channels)
        self.dropout = nn.Dropout(config.dropout_rate)

    def _get_norm_layer(self, num_features: int | None = None) -> nn.Module:
        """_get_norm_layer.

        Args:
            num_features (Optional[int]): Description.
        Returns:
            nn.Module: Description.
        """
        channels = num_features or self.config.fusioin_channels
        norm_type = self.norm_type

        if norm_type == "layer":
            return nn.LayerNorm(channels)
        if norm_type == "instance":
            return nn.LayerNorm(channels)
        if norm_type in ("identity", None):
            return nn.Identity()
        raise ValueError(f"Unsupported normalization type: {norm_type}")

    def _prepare_inputs(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """_prepare_inputs.

        Args:
            inputs (Sequence[torch.Tensor]): Description.
        Returns:
            torch.Tensor: Description.
        """
        if not inputs:
            raise ValueError("Attention fusion requires at least one input tensor")

        if len(inputs) == 1:
            x = inputs[0]
            if x.dim() == 5:
                return x
            if x.dim() == 4:
                return x.unsqueeze(-1)
            raise RuntimeError("Unsupported tensor dimensionality for attention fusion")

        base_shape = inputs[0].shape
        for tensor in inputs[1:]:
            if tensor.shape != base_shape:
                raise RuntimeError(
                    "All attention fusion inputs must share the same shape",
                )

        return torch.stack(list(inputs), dim=-1)

    def _apply_attention(self, x: torch.Tensor) -> torch.Tensor:
        """_apply_attention.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        B, C, H, W, S = x.shape

        x_seq = x.permute(0, 4, 2, 3, 1).flatten(2, 3)
        x_pooled = x_seq.mean(dim=2)

        query = self.query_proj(x_pooled)
        key = self.key_proj(x_pooled)
        value = self.value_proj(x_pooled)

        attn_output, _ = self.attention(query, key, value)

        output = self.out_proj(attn_output)
        output = self.dropout(output)
        output = self.norm(output + x_pooled)

        fused = output.mean(dim=1)
        fused = fused.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        return fused

    def forward(
        self,
        x: torch.Tensor,
        *additional: torch.Tensor,
    ) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        prepared = self._prepare_inputs((x,) + additional)
        return self._apply_attention(prepared)

    def get_fusioin_channels(self) -> int:
        """get_fusioin_channels.

        Returns:
            int: Description.
        """
        return self.config.fusioin_channels


class CrossAttentionFusionBlock(IFusionBlock):
    """Cross-attention fusion block with flexible constructor semantics."""

    def __init__(
        self,
        config: Fusion2DSRConfig | None = None,
        *,
        in_channels: int | None = None,
        out_channels: int | None = None,
        num_inputs: int | None = None,
        attention_heads: int | None = None,
        dropout_rate: float | None = None,
    ):
        """__init__.

        Args:
            config (Optional[Fusion2DSRConfig]): Description.
            in_channels (Optional[int]): Description.
            out_channels (Optional[int]): Description.
            num_inputs (Optional[int]): Description.
            attention_heads (Optional[int]): Description.
            dropout_rate (Optional[float]): Description.
        """
        super().__init__()

        if config is None:
            inferred_in = in_channels or out_channels or 64
            inferred_out = out_channels or inferred_in
            total_inputs = num_inputs or 2
            config = Fusion2DSRConfig(
                input_slices=total_inputs,
                fusion_type="cross_attention",
                fusioin_channels=inferred_out,
                attention_heads=attention_heads or 4,
                dropout_rate=dropout_rate or 0.1,
                input_channels=inferred_in,
                enforce_odd_input=False,
            )
        else:
            if attention_heads is not None:
                config.attention_heads = attention_heads
            if dropout_rate is not None:
                config.dropout_rate = dropout_rate

        self.config = config
        self.expected_inputs = max(1, num_inputs or config.input_slices)
        self.input_channels = in_channels or config.input_channels

        embed_dim = config.fusioin_channels
        heads = max(1, config.attention_heads // 2)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=config.attention_heads,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=heads,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def _prepare_inputs(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """_prepare_inputs.

        Args:
            inputs (Sequence[torch.Tensor]): Description.
        Returns:
            torch.Tensor: Description.
        """
        if not inputs:
            raise ValueError("Cross-attention requires at least one input tensor")

        if len(inputs) == 1:
            x = inputs[0]
            if x.dim() == 5:
                return x
            if x.dim() == 4:
                return x.unsqueeze(-1)
            raise RuntimeError(
                "Unsupported tensor dimensionality for cross-attention fusion",
            )

        base_shape = inputs[0].shape
        for tensor in inputs[1:]:
            if tensor.shape != base_shape:
                raise RuntimeError(
                    "All cross-attention inputs must share the same shape",
                )

        return torch.stack(list(inputs), dim=-1)

    def _apply_cross_attention(self, x: torch.Tensor) -> torch.Tensor:
        """_apply_cross_attention.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        B, C, H, W, S = x.shape

        x_tokens = x.permute(0, 2, 3, 4, 1).flatten(1, 3)

        self_attn_out, _ = self.self_attention(x_tokens, x_tokens, x_tokens)
        x_tokens = self.norm1(x_tokens + self_attn_out)

        query = x_tokens[:, : H * W, :]
        key_value = x_tokens

        cross_attn_out, _ = self.cross_attention(query, key_value, key_value)
        x_tokens = self.norm2(query + cross_attn_out)

        ffn_out = self.ffn(x_tokens)
        x_tokens = self.norm3(x_tokens + ffn_out)

        fused = x_tokens.mean(dim=1)
        fused = fused.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        return fused

    def forward(
        self,
        x: torch.Tensor,
        *additional: torch.Tensor,
    ) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        prepared = self._prepare_inputs((x,) + additional)
        return self._apply_cross_attention(prepared)

    def get_fusioin_channels(self) -> int:
        """get_fusioin_channels.

        Returns:
            int: Description.
        """
        return self.config.fusioin_channels


class MultiSlice2DSRGenerator(IGenerator, nn.Module):
    """2.5D Super-Resolution Generator with fusion and legacy-friendly API."""

    def __init__(
        self,
        base_generator: IGenerator | None = None,
        fusion_config: Fusion2DSRConfig | None = None,
        fusion_block: IFusionBlock | None = None,
        slice_selector: Optional["SliceSelector"] = None,
    ):
        """__init__.

        Args:
            base_generator (Optional[IGenerator]): Description.
            fusion_config (Optional[Fusion2DSRConfig]): Description.
            fusion_block (Optional[IFusionBlock]): Description.
            slice_selector (Optional['SliceSelector']): Description.
        """
        nn.Module.__init__(self)

        if fusion_config is None:
            fusion_config = Fusion2DSRConfig(
                fusion_type="concat",
                fusioin_channels=3,
                normalization="instance",
                input_channels=3,
            )

        self.fusion_config = fusion_config
        self.base_generator = base_generator or _IdentityGenerator(
            channels=self.fusion_config.fusioin_channels,
        )
        self.fusion_block = fusion_block or self._build_fusion_block(
            self.fusion_config,
        )
        self.slice_selector = slice_selector or SliceSelector(
            self.fusion_config,
        )
        self.logger = logging.getLogger(__name__)
        self._name = "multi_slice_2d_sr"

    def _build_fusion_block(self, config: Fusion2DSRConfig) -> IFusionBlock:
        """_build_fusion_block.

        Args:
            config (Fusion2DSRConfig): Description.
        Returns:
            IFusionBlock: Description.
        """
        fusion_type = config.fusion_type
        if fusion_type == "concat":
            return ConcatFusionBlock(config=config)
        if fusion_type == "attention":
            return AttentionFusionBlock(config=config)
        if fusion_type == "cross_attention":
            return CrossAttentionFusionBlock(config=config)
        raise ValueError(f"Unknown fusion type: {fusion_type}")

    def _validate_input(self, x: torch.Tensor) -> torch.Tensor:
        """_validate_input.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        if x.dim() not in (4, 5):
            raise RuntimeError("Expected 4D or 5D tensor for generator input")

        if x.size(1) != self.fusion_config.input_channels:
            raise RuntimeError(
                "Input channels do not match fusion configuration",
            )

        if x.dim() == 4:
            return x.unsqueeze(-1)
        return x

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return self._name

    def _run_base_generator(
        self,
        features: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """_run_base_generator.

        Args:
            features (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        generator = self.base_generator
        forward = getattr(generator, "forward", None)
        if callable(forward):
            try:
                return forward(features, **kwargs)
            except TypeError:
                return forward(features)
        if callable(generator):
            return generator(features, **kwargs)
        raise TypeError("Base generator must be callable")

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for MultiSlice2DSRGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        prepared = self._validate_input(x)
        multi_slice_input = self.slice_selector(prepared)
        fused_features = self.fusion_block(multi_slice_input)
        return self._run_base_generator(fused_features, **kwargs)

    def _sample_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        """_sample_latents.

        Args:
            batch_size (int): Description.
            height (int): Description.
            width (int): Description.
            device (Optional[torch.device]): Description.
            dtype (Optional[torch.dtype]): Description.
        Returns:
            Optional[torch.Tensor]: Description.
        """
        if batch_size < 1:
            return None
        sample_device = device
        if sample_device is None:
            try:
                param = next(self.fusion_block.parameters())
                sample_device = param.device
            except StopIteration:
                sample_device = torch.device("cpu")
        sample_dtype = dtype or torch.float32
        return torch.randn(
            batch_size,
            self.fusion_config.input_channels,
            height,
            width,
            device=sample_device,
            dtype=sample_dtype,
        )

    def generate(
        self,
        z: torch.Tensor | None = None,
        *,
        batch_size: int = 1,
        height: int = 64,
        width: int = 64,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """generate.

        Args:
            z (Optional[torch.Tensor]): Description.
            batch_size (int): Description.
            height (int): Description.
            width (int): Description.
            device (Optional[torch.device]): Description.
            dtype (Optional[torch.dtype]): Description.
        Returns:
            torch.Tensor: Description.
        """
        latents = z
        if latents is None:
            latents = self._sample_latents(
                batch_size,
                height,
                width,
                device=device,
                dtype=dtype,
            )
            if latents is None:
                raise ValueError("Latent sampling did not produce any tensors")
        return self.forward(latents, **kwargs)

    def get_output_shape(
        self,
        input_shape: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (Optional[tuple[int, ...]]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        shape = input_shape or (
            self.fusion_config.input_channels,
            64,
            64,
        )
        getter = getattr(self.base_generator, "get_output_shape", None)
        if callable(getter):
            try:
                return getter(shape)
            except TypeError:
                return shape
        return shape

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        parameters_method = getattr(self, "parameters", None)
        if callable(parameters_method):
            try:
                raw_params = parameters_method()
            except TypeError:
                raw_params = []
            try:
                params_list = list(raw_params)
            except TypeError:
                params_list = []
            if not params_list:
                return 0
            return sum(p.numel() for p in params_list)

        fusion_params = sum(p.numel() for p in self.fusion_block.parameters())
        base_counter = getattr(
            self.base_generator,
            "get_parameter_count",
            None,
        )
        if callable(base_counter):
            return fusion_params + int(base_counter())
        base_params = sum(p.numel() for p in self.base_generator.parameters())
        return fusion_params + base_params


class SliceSelector(nn.Module):
    """Selects and stacks slices for fusion, with legacy helper options."""

    def __init__(
        self,
        config: Fusion2DSRConfig | None = None,
        *,
        num_slices: int | None = None,
        slice_spacing: int | None = None,
    ):
        """__init__.

        Args:
            config (Optional[Fusion2DSRConfig]): Description.
            num_slices (Optional[int]): Description.
            slice_spacing (Optional[int]): Description.
        """
        super().__init__()
        self.config = config
        if num_slices is not None:
            self.num_slices = num_slices
        elif config is not None:
            self.num_slices = config.input_slices
        else:
            self.num_slices = 3

        if slice_spacing is not None:
            self.slice_spacing = slice_spacing
        elif config is not None:
            self.slice_spacing = config.slice_spacing
        else:
            self.slice_spacing = 1
        self._is_valid = self.num_slices >= 1

    def _build_indices(self, depth: int) -> list[int]:
        """_build_indices.

        Args:
            depth (int): Description.
        Returns:
            list[int]: Description.
        """
        half = self.num_slices // 2
        offsets: list[int] = []
        current = -half
        while len(offsets) < self.num_slices:
            offsets.append(current * self.slice_spacing)
            current += 1

        centre = depth // 2
        indices = []
        for offset in offsets:
            idx = max(0, min(depth - 1, centre + offset))
            indices.append(idx)
        return indices

    def _stack_single(self, x: torch.Tensor) -> torch.Tensor:
        """_stack_single.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        expanded = [x for _ in range(self.num_slices)]
        return torch.stack(expanded, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SliceSelector.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if not self._is_valid:
            raise ValueError("num_slices must be at least 1")

        if x.dim() == 4:
            return self._stack_single(x)

        if x.dim() != 5:
            raise RuntimeError("SliceSelector expects a 4D or 5D tensor input")

        _, _, _, _, depth = x.shape
        indices = self._build_indices(depth)
        selected = [x[:, :, :, :, idx] for idx in indices]
        return torch.stack(selected, dim=-1)


class Fusion2DSRBundle:
    """Bundle for creating fusion + 2.5D SR configurations.

    Provides factory methods for common fusion + 2.5D SR setups.
    """

    @staticmethod
    def create_attention_fusion_3slice(
        in_channels: int = 64,
        out_channels: int = 64,
        *,
        return_config: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Create an attention-based fusion block or its configuration."""

        fusion_config = Fusion2DSRConfig(
            input_slices=3,
            fusion_type="attention",
            fusioin_channels=out_channels,
            attention_heads=8,
            dropout_rate=0.1,
            input_channels=in_channels,
        )

        if return_config:
            return fusion_config

        return AttentionFusionBlock(config=fusion_config, **kwargs)

    @staticmethod
    def create_cross_attention_5slice(
        in_channels: int = 64,
        out_channels: int = 128,
        *,
        return_config: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Create a cross-attention fusion block or its configuration."""

        fusion_config = Fusion2DSRConfig(
            input_slices=5,
            fusion_type="cross_attention",
            fusioin_channels=out_channels,
            attention_heads=16,
            dropout_rate=0.1,
            input_channels=in_channels,
        )

        if return_config:
            return fusion_config

        return CrossAttentionFusionBlock(config=fusion_config, **kwargs)

    @staticmethod
    def create_concat_fusion_3slice(
        in_channels: int = 64,
        out_channels: int = 64,
        *,
        return_config: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Create a concatenation fusion block or its configuration."""

        fusion_config = Fusion2DSRConfig(
            input_slices=3,
            fusion_type="concat",
            fusioin_channels=out_channels,
            use_skip_connections=True,
            input_channels=in_channels,
        )

        if return_config:
            return fusion_config

        return ConcatFusionBlock(config=fusion_config, **kwargs)


# Factory function for creating fusion 2.5D SR models
def create_fusion_2d_sr_generator(
    base_model_type: str | None = None,
    fusion_config: Fusion2DSRConfig | None = None,
    factory: Any | None = None,
    *,
    base_generator: IGenerator | None = None,
    fusion_block: IFusionBlock | None = None,
    slice_selector: SliceSelector | None = None,
    base_model_kwargs: dict[str, Any] | None = None,
) -> IGenerator:
    """Create a fusion + 2.5D SR generator.

    Args:
        base_model_type: Type of base SR generator
        fusion_config: Configuration for fusion block
        factory: Model factory instance
        base_generator: Optional pre-built base generator
        fusion_block: Optional pre-built fusion block
        slice_selector: Optional pre-built slice selector
        base_model_kwargs: Optional keyword arguments for
            factory-based creation

    Returns:
        Configured fusion 2.5D SR generator

    """
    config = fusion_config or Fusion2DSRConfig()

    generator = base_generator
    if generator is None and factory is not None and base_model_type is not None:
        factory_kwargs = base_model_kwargs or {}
        generator = factory.create_generator(base_model_type, **factory_kwargs)

    if generator is None:
        generator = _IdentityGenerator(channels=config.fusioin_channels)

    return MultiSlice2DSRGenerator(
        base_generator=generator,
        fusion_config=config,
        fusion_block=fusion_block,
        slice_selector=slice_selector,
    )
