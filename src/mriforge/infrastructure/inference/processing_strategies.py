"""Processing Strategies for MRI Inference

This module implements the Strategy pattern for different inference processing modes:
- Slice-aware 2D (process 3D volume as sequence of 2D slices)
- Slice-aware 3D (process 3D volume as sequence of 3D chunks/slices)
- Volume-aware 3D (process full 3D volume at once)
- K-space variants (with FFT/IFFT)

These strategies decouple the data traversal logic from the model execution logic.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from mriforge.infrastructure.physics.fft_ops import FFTTransformer


class InferenceStrategy(ABC):
    """Base class for inference processing strategies."""

    @abstractmethod
    @torch.no_grad()
    def infer(
        self,
        batch: torch.Tensor,
        model: nn.Module | Callable,
        device: torch.device,
    ) -> torch.Tensor:
        """Run inference on a batch of data.

        Args:
            batch: Input tensor (B, C, D, H, W) or (B, C, H, W)
            model: PyTorch model or callable to execute inference
            device: Device to run inference on

        Returns:
            Output tensor
        """
        pass


class SliceAware2DStrategy(InferenceStrategy):
    """2D inference, processing slices independently.

    Expects input shape: (B, C, D, H, W)
    Iterates over D dimension, treating each slice as (B, C, H, W).
    """

    @torch.no_grad()
    def infer(
        self,
        batch: torch.Tensor,
        model: nn.Module | Callable,
        device: torch.device,
    ) -> torch.Tensor:
        """Run slice-aware 2D inference."""
        batch = batch.to(device)

        # Check dimensions
        if batch.dim() != 5:
            raise ValueError(
                f"SliceAware2DStrategy expects 5D input (B, C, D, H, W), got {batch.shape}"
            )

        B, C, D, H, W = batch.shape

        # Process each slice along depth dimension
        # Vectorized implementation: Stack along batch dimension
        # (B, C, D, H, W) -> (B, D, C, H, W) -> (B*D, C, H, W)
        batch_reshaped = batch.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W)

        # Run model on full batch (B*D)
        if isinstance(model, nn.Module):
            output_reshaped = model(batch_reshaped)
        else:
            output_reshaped = model(batch_reshaped)

        # Reshape back: (B*D, C_out, H, W) -> (B, D, C_out, H, W) -> (B, C_out, D, H, W)
        # We need to know C_out from output
        _, C_out, H_out, W_out = output_reshaped.shape
        output = output_reshaped.view(B, D, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)

        return output


class SliceAware3DStrategy(InferenceStrategy):
    """3D inference, processing 3D chunks/slices from 3D volume.

    This is typically used when the model is 3D but we want to process
    the volume in chunks (e.g. sliding window) or if the input is
    already a batch of 3D volumes.

    If input is (B, C, D, H, W), this strategy treats it as a batch of 3D volumes.
    """

    @torch.no_grad()
    def infer(
        self,
        batch: torch.Tensor,
        model: nn.Module | Callable,
        device: torch.device,
    ) -> torch.Tensor:
        """Run slice-aware 3D inference."""
        batch = batch.to(device)

        # If the model expects 3D inputs, we just pass the batch
        # But if we are doing "slice aware" 3D, maybe we mean iterating over D?
        # The example code says: "Process depth-wise slices... output_z = model(slice_z)"
        # where slice_z is (B, C, H, W). That sounds like SliceAware2DStrategy.

        # Let's assume SliceAware3DStrategy means we have a 3D model that takes
        # (B, C, D_chunk, H, W) and we want to process a large volume.
        # But for now, let's follow the example which seems to imply
        # iterating over D but using a model that might handle it differently?
        # Actually, the example code for SliceAware3DStrategy in IMPLEMENTATION_EXAMPLES.md
        # iterates over D and calls model(slice_z). This is identical to SliceAware2DStrategy
        # unless the model expects something else.

        # Wait, the example says:
        # class SliceAware3DStrategy(InferenceStrategy):
        #     def infer(self, batch, model, device):
        #         # ...
        #         for z in range(depth):
        #             slice_z = batch[:, :, z, :, :]
        #             output_z = model(slice_z)
        #             outputs.append(output_z)
        #         return torch.stack(outputs, dim=2)

        # This is exactly the same as SliceAware2DStrategy logic.
        # The name might be confusing. Maybe it means "Strategy for 3D data using 2D model"?
        # Or maybe "Strategy for 3D data using 3D model on slices"?

        # Let's implement it as per example, which is effectively iterating over D.

        if batch.dim() != 5:
            raise ValueError(
                f"SliceAware3DStrategy expects 5D input (B, C, D, H, W), got {batch.shape}"
            )

        B, C, D, H, W = batch.shape

        # Vectorized implementation (Same as SliceAware2DStrategy)
        batch_reshaped = batch.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W)

        if isinstance(model, nn.Module):
            output_reshaped = model(batch_reshaped)
        else:
            output_reshaped = model(batch_reshaped)

        _, C_out, H_out, W_out = output_reshaped.shape
        output = output_reshaped.view(B, D, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)

        return output


class VolumeAware3DStrategy(InferenceStrategy):
    """3D inference, processing full volume at once.

    Expects input shape: (B, C, D, H, W)
    Passes the whole 5D tensor to the model.
    """

    @torch.no_grad()
    def infer(
        self,
        batch: torch.Tensor,
        model: nn.Module | Callable,
        device: torch.device,
    ) -> torch.Tensor:
        """Run volume-aware 3D inference."""
        batch = batch.to(device)

        if isinstance(model, nn.Module):
            return model(batch)
        else:
            return model(batch)


class KSpaceSliceAware2DStrategy(InferenceStrategy):
    """2D k-space inference, with FFT/IFFT transformation.

    Expects input in image space (or k-space? The example implies image space input -> FFT -> model -> IFFT).
    """

    def __init__(self, fft_transformer: FFTTransformer | None = None):
        """__init__.

        Args:
            fft_transformer (Optional[FFTTransformer]): Description.
        """
        self.fft = fft_transformer or FFTTransformer()

    @torch.no_grad()
    def infer(
        self,
        batch: torch.Tensor,
        model: nn.Module | Callable,
        device: torch.device,
    ) -> torch.Tensor:
        """Run k-space slice-aware 2D inference."""
        batch = batch.to(device)

        # Transform to k-space
        # batch is (B, C, D, H, W) or (B, C, H, W)
        # fft2 applies to last 2 dims
        kspace = self.fft.fftnc(batch, dims=(-2, -1))

        # Infer in k-space
        # We need to iterate over slices if it's 3D volume
        if batch.dim() == 5:
            B, C, D, H, W = batch.shape

            # Vectorize k-space processing
            # kspace: (B, C, D, H, W) -> (B*D, C, H, W)
            # Use reshape directly after permute
            kspace_reshaped = kspace.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W)

            if isinstance(model, nn.Module):
                output_k_reshaped = model(kspace_reshaped)
            else:
                output_k_reshaped = model(kspace_reshaped)

            _, C_out, H_out, W_out = output_k_reshaped.shape
            kspace_pred = output_k_reshaped.view(B, D, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)
        else:
            if isinstance(model, nn.Module):
                kspace_pred = model(kspace)
            else:
                kspace_pred = model(kspace)

        # Transform back to image space
        image_pred = self.fft.ifftnc(kspace_pred, dims=(-2, -1))

        return image_pred


class KSpaceSliceAware3DStrategy(InferenceStrategy):
    """3D k-space inference with FFT on depth dimension."""

    def __init__(self, fft_transformer: FFTTransformer | None = None):
        """__init__.

        Args:
            fft_transformer (Optional[FFTTransformer]): Description.
        """
        self.fft = fft_transformer or FFTTransformer()

    @torch.no_grad()
    def infer(
        self,
        batch: torch.Tensor,
        model: nn.Module | Callable,
        device: torch.device,
    ) -> torch.Tensor:
        """Run k-space slice-aware 3D inference."""
        batch = batch.to(device)

        # Transform to 3D k-space
        kspace = self.fft.fftnc(batch)  # centred FFT on all spatial dims

        # Process slices in k-space (along depth)
        if batch.dim() == 5:
            B, C, D, H, W = batch.shape

            # Vectorize k-space 3D processing
            kspace_reshaped = kspace.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W)

            if isinstance(model, nn.Module):
                output_k_reshaped = model(kspace_reshaped)
            else:
                output_k_reshaped = model(kspace_reshaped)

            _, C_out, H_out, W_out = output_k_reshaped.shape
            kspace_pred = output_k_reshaped.view(B, D, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)
        else:
            # If not 5D, maybe just run model?
            if isinstance(model, nn.Module):
                kspace_pred = model(kspace)
            else:
                kspace_pred = model(kspace)

        # Transform back to image space
        image_pred = self.fft.ifftnc(kspace_pred)

        return image_pred


class SlidingWindow3DStrategy(InferenceStrategy):
    """3D inference using sliding window with Gaussian weighting.

    [ARCHITECT FIX] Prevents OOM on large 3D volumes by processing
    overlapping patches and stitching with weighted averaging.

    Expects input shape: (B, C, D, H, W)
    """

    def __init__(
        self,
        roi_size: tuple = (32, 256, 256),
        overlap: float = 0.25,
    ):
        """__init__.

        Args:
            roi_size (tuple): Description.
            overlap (float): Description.
        """
        self.roi_size = roi_size
        self.overlap = overlap

    @torch.no_grad()
    def infer(
        self,
        batch: torch.Tensor,
        model: nn.Module | Callable,
        device: torch.device,
    ) -> torch.Tensor:
        """Run sliding window 3D inference."""
        from mriforge.infrastructure.inference.sliding_window import sliding_window_inference

        batch = batch.to(device)

        if batch.dim() != 5:
            raise ValueError(
                f"SlidingWindow3DStrategy expects 5D input (B, C, D, H, W), got {batch.shape}"
            )

        # Use the sliding window utility
        def predictor(x):
            """predictor.

            Args:
                x (Any): Description.
            Returns:
                Any: Description.
            """
            if isinstance(model, nn.Module):
                return model(x)
            return model(x)

        return sliding_window_inference(
            batch,
            roi_size=self.roi_size,
            predictor=predictor,
            overlap=self.overlap,
            mode="gaussian",
            device=device,
        )


class InferenceDispatcher:
    """Dispatches to appropriate strategy based on config."""

    STRATEGY_MAP = {
        ("2d", "slice_aware", "image"): SliceAware2DStrategy,
        ("2d", "slice_aware", "kspace"): KSpaceSliceAware2DStrategy,
        ("3d", "slice_aware", "image"): SliceAware3DStrategy,
        ("3d", "slice_aware", "kspace"): KSpaceSliceAware3DStrategy,
        ("3d", "volume_aware", "image"): VolumeAware3DStrategy,
        ("3d", "sliding_window", "image"): SlidingWindow3DStrategy,  # [ARCHITECT FIX]
    }

    def __init__(self, config: dict[str, Any]):
        """__init__.

        Args:
            config (dict[str, Any]): Description.
        """
        self.config = config
        self.strategy = self._select_strategy()

    def _select_strategy(self) -> InferenceStrategy:
        """Select strategy based on config. O(1) lookup."""
        # Default values
        dim_type = self.config.get("dimension_type", "2d")
        proc_mode = self.config.get("processing_mode", "slice_aware")
        data_fmt = self.config.get("data_format", "image")

        key = (dim_type, proc_mode, data_fmt)

        strategy_class = self.STRATEGY_MAP.get(key)
        if not strategy_class:
            # Fallback or error?
            if dim_type == "3d" and proc_mode == "sliding_window":
                strategy_class = SlidingWindow3DStrategy
            elif dim_type == "3d" and proc_mode == "volume_aware":
                strategy_class = VolumeAware3DStrategy
            elif proc_mode == "slice_aware":
                strategy_class = SliceAware2DStrategy
            else:
                raise ValueError(f"Unknown strategy configuration: {key}")

        import inspect

        # Handle strategies with custom __init__
        if strategy_class == SlidingWindow3DStrategy:
            roi_size = self.config.get("roi_size", (32, 256, 256))
            overlap = self.config.get("overlap", 0.25)
            return strategy_class(roi_size=roi_size, overlap=overlap)

        if hasattr(strategy_class, "__init__"):
            init_method = strategy_class.__init__
            if inspect.isfunction(init_method) or inspect.ismethod(init_method):
                if "fft_transformer" in init_method.__code__.co_varnames:
                    return strategy_class(FFTTransformer())

        return strategy_class()

    @torch.no_grad()
    def infer(
        self,
        batch: torch.Tensor,
        model: nn.Module | Callable,
        device: torch.device,
    ) -> torch.Tensor:
        """Delegate to strategy."""
        return self.strategy.infer(batch, model, device)
