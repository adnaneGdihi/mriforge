"""Latent access interfaces and mixins.

This module defines lightweight helpers that standardize how encoder/decoder
style models expose intermediate representations. Models that cache latent
embeddings or skip feature maps can inherit from :class:`LatentAccessMixin`
to automatically provide a consistent API.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class SupportsLatentAccess(Protocol):
    """Protocol describing models that expose cached latent representations."""

    def get_embedding(self) -> torch.Tensor:
        """Return the latent embedding from the last forward pass."""

    def iter_feature_maps(self) -> Iterable[torch.Tensor]:
        """Yield feature maps (e.g., skip connections) shallow → deep."""

    def export_latent_state(self) -> dict[str, torch.Tensor]:
        """Export a dictionary representation of the cached latent state."""


class LatentAccessMixin:
    """Mixin that simplifies exposing latent embeddings and feature maps."""

    def __init__(self) -> None:
        """__init__."""
        super().__init__()
        self._latent_embedding: torch.Tensor | None = None
        self._latent_feature_maps: tuple[torch.Tensor, ...] = ()
        self._latent_extras: dict[str, torch.Tensor] = {}

    def get_embedding(self) -> torch.Tensor:
        """get_embedding.

        Returns:
            torch.Tensor: Description.
        """
        if self._latent_embedding is None:
            raise RuntimeError(
                "No latent embedding cached – run a forward pass first.",
            )
        return self._latent_embedding

    def iter_feature_maps(self) -> Iterable[torch.Tensor]:
        """iter_feature_maps.

        Returns:
            Iterable[torch.Tensor]: Description.
        """
        return tuple(self._latent_feature_maps)

    def export_latent_state(self) -> dict[str, torch.Tensor]:
        """export_latent_state.

        Returns:
            dict[str, torch.Tensor]: Description.
        """
        state: dict[str, torch.Tensor] = {
            "embedding": self.get_embedding(),
            "skips": tuple(self._latent_feature_maps),
        }
        state.update(self._latent_extras)
        return state

    def _set_latent_state(
        self,
        *,
        embedding: torch.Tensor,
        feature_maps: Iterable[torch.Tensor] = (),
        **extras: torch.Tensor,
    ) -> None:
        """Store the latent cache for later retrieval."""

        self._latent_embedding = embedding
        self._latent_feature_maps = tuple(feature_maps)
        # Filter out ``None`` values to keep the exported state clean
        self._latent_extras = {key: value for key, value in extras.items() if value is not None}


__all__ = ["LatentAccessMixin", "SupportsLatentAccess"]
