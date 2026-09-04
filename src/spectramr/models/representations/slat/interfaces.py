"""Structured LATent (SLAT) Representations
=========================================

Interfaces and base classes for TRELLIS Structured LATent representations.
SLAT provides a unified representation that can be decoded to different
3D formats.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol

import torch


class ISLATRepresentation(Protocol):
    """Protocol for Structured LATent representations.

    SLAT representations provide a unified way to encode 3D structures
    that can be decoded to various output formats (meshes, gaussians,
    radiance fields).
    """

    @property
    def spatial_resolution(self) -> tuple[int, int, int]:
        """Get the spatial resolution of the SLAT representation."""
        ...

    @property
    def feature_channels(self) -> int:
        """Get the number of feature channels in the SLAT."""
        ...

    def encode(self, data: Any) -> torch.Tensor:
        """Encode input data into SLAT representation."""
        ...

    def decode(self, _slat: torch.Tensor, format: str) -> Any:
        """Decode SLAT to specified output format."""
        ...


class ISLATEcoder(ABC):
    """Abstract base class for SLAT encoders.

    Encoders convert various input formats (point clouds, meshes, etc.)
    into the unified SLAT representation.
    """

    @abstractmethod
    def encode(self, input_data: Any, **kwargs) -> torch.Tensor:
        """Encode input data into SLAT representation.

        Args:
            input_data: Input data (mesh, point cloud, etc.)
            **kwargs: Additional encoding parameters

        Returns:
            SLAT tensor representation

        """

    @property
    @abstractmethod
    def input_format(self) -> str:
        """Get the expected input format for this encoder."""

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, ...]:
        """Get the output shape of the encoded SLAT."""


class ISLATDecoder(ABC):
    """Abstract base class for SLAT decoders.

    Decoders convert SLAT representations into specific output formats
    like meshes, gaussians, or radiance fields.
    """

    @abstractmethod
    def decode(self, _slat: torch.Tensor, **kwargs) -> Any:
        """Decode SLAT representation into output format.

        Args:
            _slat: SLAT tensor representation
            **kwargs: Additional decoding parameters

        Returns:
            Decoded output in target format

        """

    @property
    @abstractmethod
    def output_format(self) -> str:
        """Get the output format produced by this decoder."""

    @property
    @abstractmethod
    def input_shape(self) -> tuple[int, ...]:
        """Get the expected input shape for SLAT tensor."""


class SLATConfig:
    """Configuration class for SLAT representations.

    Encapsulates all parameters needed to create and configure
    SLAT encoders and decoders.
    """

    def __init__(
        self,
        resolution: int = 256,
        num_points: int = 1000,
        spatial_resolution: tuple[int, int, int] = (32, 32, 32),
        feature_channels: int = 64,
        compression_ratio: float = 0.5,
        elastic: bool = False,
        sparse_structure: bool = True,
        **kwargs,
    ):
        """Initialize SLAT configuration.

        Args:
            resolution: Legacy resolution parameter for compatibility
            num_points: Legacy num_points parameter for compatibility
            spatial_resolution: Spatial dimensions of SLAT grid
            feature_channels: Number of feature channels
            compression_ratio: Compression ratio for sparse representations
            elastic: Whether to use elastic SLAT variant
            sparse_structure: Whether to use sparse structure encoding
            **kwargs: Additional configuration parameters

        """
        self.resolution = resolution
        self.num_points = num_points
        self.spatial_resolution = spatial_resolution
        self.feature_channels = feature_channels
        self.compression_ratio = compression_ratio
        self.elastic = elastic
        self.sparse_structure = sparse_structure
        self.extra_params = kwargs

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "spatial_resolution": self.spatial_resolution,
            "feature_channels": self.feature_channels,
            "compression_ratio": self.compression_ratio,
            "elastic": self.elastic,
            "sparse_structure": self.sparse_structure,
            **self.extra_params,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SLATConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


class GaussianSLATConfig(SLATConfig):
    """Configuration for Gaussian SLAT representations.

    Specialized config for 3D Gaussian splatting representations.
    """

    def __init__(
        self,
        splat_size: float = 1.0,
        opacity_threshold: float = 0.01,
        **kwargs,
    ):
        """Initialize Gaussian SLAT configuration.

        Args:
            splat_size: Size of Gaussian splats
            opacity_threshold: Minimum opacity for rendering
            **kwargs: Additional SLATConfig parameters

        """
        super().__init__(**kwargs)
        self.splat_size = splat_size
        self.opacity_threshold = opacity_threshold

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "splat_size": self.splat_size,
                "opacity_threshold": self.opacity_threshold,
            }
        )
        return base_dict


class MeshSLATConfig(SLATConfig):
    """Configuration for Mesh SLAT representations.

    Specialized config for 3D mesh representations.
    """

    def __init__(
        self,
        num_faces: int = 1000,
        simplify_mesh: bool = True,
        manifold_constraint: bool = True,
        **kwargs,
    ):
        """Initialize Mesh SLAT configuration.

        Args:
            num_faces: Target number of faces in mesh
            simplify_mesh: Whether to simplify mesh topology
            manifold_constraint: Whether to enforce manifold constraints
            **kwargs: Additional SLATConfig parameters

        """
        super().__init__(**kwargs)
        self.num_faces = num_faces
        self.simplify_mesh = simplify_mesh
        self.manifold_constraint = manifold_constraint

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "num_faces": self.num_faces,
                "simplify_mesh": self.simplify_mesh,
                "manifold_constraint": self.manifold_constraint,
            }
        )
        return base_dict
