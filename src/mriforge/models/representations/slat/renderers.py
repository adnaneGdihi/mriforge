"""Renderer utilities for Structured LATent (SLAT) representations.

These renderers adapt raw generator outputs (e.g., TRELLIS Gaussian and mesh
parameters or X-Diffusion volumetric tensors) into structured artifacts that can
later be promoted to domain entities by orchestration services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .interfaces import GaussianSLATConfig, MeshSLATConfig


@dataclass
class RendererOutput:
    """Container describing the output of a renderer.

    Attributes:
        format: The canonical representation format (e.g., ``"gaussian"``).
        items: A batch of representation payloads, one dictionary per sample in
            the batch. Each payload contains CPU tensors ready for conversion to
            domain entities.
        metadata: Optional metadata describing the rendered representation (for
            example, SLAT configuration hints or heuristic fallbacks).
    """

    format: str
    items: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the renderer output to a serialisable dictionary."""

        return {
            "format": self.format,
            "items": self.items,
            "metadata": self.metadata,
        }


class BaseRenderer:
    """Abstract base renderer for SLAT-derived assets."""

    def render(self, output: Any, **kwargs: Any) -> RendererOutput:
        """Convert raw generator output into a structured representation.

        Subclasses must override this method.  See ``TrellisGaussianRenderer``,
        ``TrellisMeshRenderer``, and ``VolumeRenderer`` for concrete
        implementations.
        """
        raise NotImplementedError

    @staticmethod
    def _detach_cpu(tensor: torch.Tensor) -> torch.Tensor:
        """Detach, clone, and move a tensor to CPU while sanitising NaNs."""

        sanitized = torch.nan_to_num(tensor.detach(), nan=0.0, posinf=1e6, neginf=-1e6)
        return sanitized.cpu().contiguous()


class TrellisGaussianRenderer(BaseRenderer):
    """Renderer for TRELLIS Gaussian SLAT representations."""

    def __init__(self, config: GaussianSLATConfig | None = None) -> None:
        """__init__.

        Args:
            config (GaussianSLATConfig | None): Description.
        """
        self.config = config or GaussianSLATConfig()

    def render(
        self,
        output: torch.Tensor,
        *,
        config: GaussianSLATConfig | dict[str, Any] | None = None,
        source: torch.Tensor | None = None,
    ) -> RendererOutput:
        """Render Gaussian parameters into structured payloads.

        Args:
            output: Tensor shaped ``(B, N, 14)`` containing Gaussian parameters.
            config: Optional override for the SLAT configuration; accepts either
                a ``GaussianSLATConfig`` instance or a dictionary of keyword
                arguments used to instantiate one.
            source: Optional conditioning tensor (unused, but accepted for
                parity with other renderer signatures).
        """

        slat_config = self._resolve_config(config)
        if output.dim() != 3 or output.size(-1) < 14:
            raise ValueError(
                "Expected Gaussian parameter tensor with shape (B, N, 14); "
                f"got {tuple(output.shape)}",
            )

        batch_payload: list[dict[str, Any]] = []
        batch_size = output.size(0)
        num_gaussians = output.size(1)

        for batch_idx in range(batch_size):
            sample = output[batch_idx]
            positions = self._detach_cpu(sample[:, 0:3])
            scales = self._detach_cpu(F.softplus(sample[:, 3:6]))
            rotations = self._detach_cpu(self._normalise_quaternions(sample[:, 6:10]))
            colors = self._detach_cpu(torch.sigmoid(sample[:, 10:13]))
            opacities = self._detach_cpu(torch.sigmoid(sample[:, 13])).unsqueeze(-1)

            batch_payload.append(
                {
                    "positions": positions,
                    "rotations": rotations,
                    "scales": scales,
                    "colors": colors,
                    "opacities": opacities,
                    "sh_degree": slat_config.extra_params.get("sh_degree", 0),
                },
            )

        metadata = {
            "num_points": num_gaussians,
            "feature_channels": 14,
            "slat_config": slat_config.to_dict(),
            "source_spatial_resolution": (tuple(source.shape[-2:]) if source is not None else None),
        }

        return RendererOutput(
            format="gaussian",
            items=batch_payload,
            metadata=metadata,
        )

    def _resolve_config(
        self,
        config: GaussianSLATConfig | dict[str, Any] | None,
    ) -> GaussianSLATConfig:
        """_resolve_config.

        Args:
            config (GaussianSLATConfig | dict[str, Any] | None): Description.
        Returns:
            GaussianSLATConfig: Description.
        """
        if config is None:
            return self.config
        if isinstance(config, GaussianSLATConfig):
            self.config = config
            return self.config
        self.config = GaussianSLATConfig(**config)
        return self.config

    @staticmethod
    def _normalise_quaternions(quats: torch.Tensor) -> torch.Tensor:
        """Normalise quaternion rotations to unit length."""

        norm = torch.linalg.norm(quats, dim=-1, keepdim=True).clamp(min=1e-6)
        normalised = quats / norm
        # Ensure the scalar component is non-negative for a canonical orientation
        scalar = normalised[..., 0]
        mask = scalar < 0
        if mask.any():
            normalised[mask] = -normalised[mask]
        return normalised


class TrellisMeshRenderer(BaseRenderer):
    """Renderer for TRELLIS mesh SLAT representations."""

    def __init__(self, config: MeshSLATConfig | None = None) -> None:
        """__init__.

        Args:
            config (MeshSLATConfig | None): Description.
        """
        self.config = config or MeshSLATConfig()

    def render(
        self,
        output: dict[str, torch.Tensor],
        *,
        config: MeshSLATConfig | dict[str, Any] | None = None,
        source: torch.Tensor | None = None,
    ) -> RendererOutput:
        """Render mesh decoder outputs into structured payloads."""

        slat_config = self._resolve_config(config)
        vertices = output.get("vertices")
        faces = output.get("faces")
        normals = output.get("normals")
        colors = output.get("colors")

        if vertices is None or faces is None:
            raise ValueError("Mesh renderer expects 'vertices' and 'faces' tensors")

        if vertices.dim() != 3 or faces.dim() != 3:
            raise ValueError("Expected batched vertices and faces tensors")

        batch_payload: list[dict[str, Any]] = []
        batch_size = vertices.size(0)

        for batch_idx in range(batch_size):
            sample_vertices = self._detach_cpu(vertices[batch_idx])
            sample_faces = self._detach_cpu(faces[batch_idx]).long()

            sample_payload: dict[str, Any] = {
                "vertices": sample_vertices,
                "faces": sample_faces,
            }

            if normals is not None:
                sample_payload["normals"] = self._detach_cpu(normals[batch_idx])
            if colors is not None:
                sample_payload["colors"] = self._detach_cpu(colors[batch_idx])

            batch_payload.append(sample_payload)

        metadata = {
            "num_vertices": vertices.size(1),
            "num_faces": faces.size(1),
            "slat_config": slat_config.to_dict(),
            "source_spatial_resolution": (tuple(source.shape[-2:]) if source is not None else None),
        }

        return RendererOutput(
            format="mesh",
            items=batch_payload,
            metadata=metadata,
        )

    def _resolve_config(
        self,
        config: MeshSLATConfig | dict[str, Any] | None,
    ) -> MeshSLATConfig:
        """_resolve_config.

        Args:
            config (MeshSLATConfig | dict[str, Any] | None): Description.
        Returns:
            MeshSLATConfig: Description.
        """
        if config is None:
            return self.config
        if isinstance(config, MeshSLATConfig):
            self.config = config
            return self.config
        self.config = MeshSLATConfig(**config)
        return self.config


class VolumeRenderer(BaseRenderer):
    """Renderer for volumetric outputs (e.g., X-Diffusion MRI volumes)."""

    def render(
        self,
        output: torch.Tensor,
        *,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RendererOutput:
        """Render volumetric tensors into structured payloads."""

        if output.dim() not in (4, 5):
            raise ValueError(
                "Expected volume tensor with shape (B, D, H, W) or (B, C, D, H, W)",
            )

        batch_payload: list[dict[str, Any]] = []
        batch_size = output.size(0)
        has_channels = output.dim() == 5

        for batch_idx in range(batch_size):
            sample = output[batch_idx]
            volume_tensor = self._detach_cpu(sample)
            payload = {
                "volume": volume_tensor,
                "spacing": spacing,
                "origin": origin,
                "direction": self._default_direction(direction),
            }
            batch_payload.append(payload)

        render_metadata = {
            "has_channels": has_channels,
            "voxel_spacing": spacing,
        }
        if metadata:
            render_metadata.update(metadata)

        return RendererOutput(
            format="volume",
            items=batch_payload,
            metadata=render_metadata,
        )

    def _default_direction(
        self,
        direction: torch.Tensor | None,
    ) -> torch.Tensor:
        """_default_direction.

        Args:
            direction (torch.Tensor | None): Description.
        Returns:
            torch.Tensor: Description.
        """
        if direction is None:
            return torch.eye(3)
        if direction.dim() != 2:
            raise ValueError("Direction matrix must be 2D")
        return self._detach_cpu(direction)


__all__ = [
    "BaseRenderer",
    "RendererOutput",
    "TrellisGaussianRenderer",
    "TrellisMeshRenderer",
    "VolumeRenderer",
]
