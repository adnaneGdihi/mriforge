"""Utility transforms for ensuring MRI tensors are grayscale.

This module provides lightweight tensor transforms that can be composed to
standardize inputs originating from RGB datasets before they are used in MRI
pipelines. The helpers avoid a hard dependency on torchvision while remaining
PyTorch-native.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass
class EnsureGrayscaleTensor:
    """Convert RGB tensors to single-channel grayscale tensors.

    The transform accepts tensors shaped either as ``(C, H, W)`` or
    ``(N, C, H, W)``. When three channels are detected the tensor is converted
    to a single channel using perceptual luminance weights. Two-dimensional
    tensors are interpreted as ``(H, W)`` slices and receive a channel
    dimension. Tensors that are already single channel are returned unchanged.
    """

    luminance_weights: Sequence[float] = (0.2989, 0.5870, 0.1140)

    def __call__(self, tensor: Tensor) -> Tensor:
        """__call__.

        Args:
            tensor (Tensor): Description.
        Returns:
            Tensor: Description.
        """
        if not torch.is_tensor(tensor):
            raise TypeError(
                "EnsureGrayscaleTensor expects a torch.Tensor input, "
                f"received {type(tensor)!r} instead.",
            )

        if tensor.ndim == 2:
            return tensor.unsqueeze(0)

        if tensor.ndim == 3:
            channels = tensor.shape[0]
            if channels == 1:
                return tensor
            if channels == 3:
                weights = tensor.new_tensor(self.luminance_weights).view(
                    3,
                    1,
                    1,
                )
                grayscale = (tensor * weights).sum(dim=0, keepdim=True)
                return grayscale
            return tensor

        if tensor.ndim == 4:
            channels = tensor.shape[1]
            if channels == 1:
                return tensor
            if channels == 3:
                weights = tensor.new_tensor(self.luminance_weights).view(
                    1,
                    3,
                    1,
                    1,
                )
                grayscale = (tensor * weights).sum(dim=1, keepdim=True)
                return grayscale
            return tensor

        return tensor


class ComposeTensorTransforms:
    """A minimal callable that composes tensor transforms sequentially."""

    def __init__(self, transforms: Iterable):
        """__init__.

        Args:
            transforms (Iterable): Description.
        """
        self._transforms = tuple(transforms)

    def __call__(self, tensor: Tensor) -> Tensor:
        """__call__.

        Args:
            tensor (Tensor): Description.
        Returns:
            Tensor: Description.
        """
        out = tensor
        for transform in self._transforms:
            out = transform(out)
        return out

    def __repr__(self) -> str:  # pragma: no cover - trivial
        """__repr__.

        Returns:
            str: Description.
        """
        names = ", ".join(type(t).__name__ for t in self._transforms)
        return f"ComposeTensorTransforms([{names}])"
