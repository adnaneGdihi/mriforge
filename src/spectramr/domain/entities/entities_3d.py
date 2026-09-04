#!/usr/bin/env python
"""3D Generation Domain Entities
Core business entities for 3D asset generation and medical imaging.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch


@dataclass
class SLAT:
    """
    ### SLAT Representation
    ```mermaid
    classDiagram
    class SLAT {
    +Tensor sparse_structure
    +Tensor slat_latent
    +tuple resolution
    +dict metadata
    +shape() tuple
    +to_dict() dict
    +from_dict(data) SLAT
    }
    ```
    """

    sparse_structure: torch.Tensor
    slat_latent: torch.Tensor
    resolution: tuple[int, int, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the SLAT representation."""
        return self.slat_latent.shape

    def to_dict(self) -> dict[str, Any]:
        """Convert SLAT to dictionary for serialization."""
        return {
            "sparse_structure": self.sparse_structure.cpu().numpy(),
            "slat_latent": self.slat_latent.cpu().numpy(),
            "resolution": self.resolution,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SLAT":
        """Create SLAT from dictionary."""
        return cls(
            sparse_structure=torch.from_numpy(data["sparse_structure"]),
            slat_latent=torch.from_numpy(data["slat_latent"]),
            resolution=tuple(data["resolution"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RadianceField:
    """
    ### Radiance Field structure
    ```mermaid
    classDiagram
    class RadianceField {
    +Tensor density_grid
    +Tensor color_grid
    +tuple resolution
    +tuple bounds
    +shape() tuple
    +sample_points(points) tuple
    }
    ```
    """

    density_grid: torch.Tensor
    color_grid: torch.Tensor
    resolution: tuple[int, int, int]
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the radiance field."""
        return self.density_grid.shape

    def sample_points(self, points):
        """Sample density and color at given 3D points.

        Args:
            points: Tensor of shape (N, 3) with 3D coordinates

        Returns:
            Tuple of (density, color) tensors

        """
        # Simple trilinear sampling implementation
        # In practice, this would use more sophisticated sampling
        return torch.zeros(points.shape[0]), torch.zeros(points.shape[0], 3)


@dataclass
class Gaussian3D:
    """
    ### 3D Gaussian structure
    ```mermaid
    classDiagram
    class Gaussian3D {
    +Tensor positions
    +Tensor rotations
    +Tensor scales
    +Tensor opacities
    +Tensor colors
    +int sh_degree
    +count() int
    +to_ply_format() dict
    }
    ```
    """

    positions: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor
    sh_degree: int = 0

    @property
    def count(self) -> int:
        """Get the number of Gaussians."""
        return self.positions.shape[0]

    def to_ply_format(self) -> dict[str, Any]:
        """Convert to PLY file format for saving."""
        return {
            "positions": self.positions.cpu().numpy(),
            "rotations": self.rotations.cpu().numpy(),
            "scales": self.scales.cpu().numpy(),
            "opacities": self.opacities.cpu().numpy(),
            "colors": self.colors.cpu().numpy(),
            "sh_degree": self.sh_degree,
        }


@dataclass
class Mesh:
    """Triangle mesh representation for 3D objects.

    Represents 3D objects as collections of vertices, faces, and optional
    vertex attributes like normals and texture coordinates.

    Attributes:
        vertices: Vertex positions (N, 3)
        faces: Triangle face indices (M, 3)
        normals: Vertex normals (N, 3), optional
        texcoords: Texture coordinates (N, 2), optional
        colors: Vertex colors (N, 3), optional

    """

    vertices: torch.Tensor
    faces: torch.Tensor
    normals: torch.Tensor | None = None
    texcoords: torch.Tensor | None = None
    colors: torch.Tensor | None = None

    @property
    def vertex_count(self) -> int:
        """Get the number of vertices."""
        return self.vertices.shape[0]

    @property
    def face_count(self) -> int:
        """Get the number of faces."""
        return self.faces.shape[0]

    def compute_normals(self) -> torch.Tensor:
        """Compute vertex normals from face geometry."""
        # Simple normal computation - in practice would use more robust method
        if self.normals is None:
            # Compute face normals
            v0 = self.vertices[self.faces[:, 0]]
            v1 = self.vertices[self.faces[:, 1]]
            v2 = self.vertices[self.faces[:, 2]]

            # Cross product of edges
            e1 = v1 - v0
            e2 = v2 - v0
            face_normals = torch.cross(e1, e2, dim=1)
            face_normals = torch.nn.functional.normalize(face_normals, dim=1)

            # Accumulate face normals to vertices
            vertex_normals = torch.zeros_like(self.vertices)

            # Scatter add face normals to vertices
            for i in range(3):
                vertex_normals.index_add_(0, self.faces[:, i], face_normals)

            # Normalize
            self.normals = torch.nn.functional.normalize(vertex_normals, dim=1)

        return self.normals

    def to_glb_format(self) -> dict[str, Any]:
        """Convert to GLB format for web deployment."""
        return {
            "vertices": self.vertices.cpu().numpy(),
            "faces": self.faces.cpu().numpy(),
            "normals": (self.normals.cpu().numpy() if self.normals is not None else None),
            "texcoords": (self.texcoords.cpu().numpy() if self.texcoords is not None else None),
            "colors": (self.colors.cpu().numpy() if self.colors is not None else None),
        }


@dataclass
class MRIVolume:
    """Medical MRI volume representation.

    Represents 3D medical imaging data with associated metadata
    for cross-sectional diffusion and reconstruction tasks.

    This entity encapsulates volumetric MRI data along with spatial
    and orientation information necessary for accurate 3D reconstruction
    and analysis.

    Attributes:
        volume: 3D volume data (D, H, W) or (C, D, H, W)
            - For single-channel volumes: shape (D, H, W)
            - For multi-channel volumes: shape (C, D, H, W)
            - Data type: torch.Tensor with values in [-1, 1] range
        spacing: Voxel spacing in mm (x, y, z)
            - Defines physical dimensions of each voxel
            - Used for accurate spatial measurements
        origin: Origin coordinates in world space
            - Reference point for coordinate system
            - Used for alignment with other modalities
        direction: Direction matrix for orientation (3, 3)
            - Defines orientation of volume in world space
            - Accounts for patient positioning and scan orientation
        metadata: DICOM/NIfTI metadata dictionary
            - Contains acquisition parameters, patient info, etc.
            - Preserves original imaging metadata

    Example:
        >>> # Create from NIfTI file
        >>> import nibabel as nib
        >>> nii_img = nib.load('patient_scan.nii.gz')
        >>> volume_data = torch.from_numpy(nii_img.get_fdata()).float()
        >>> volume = MRIVolume(
        ...     volume=volume_data,
        ...     spacing=(1.0, 1.0, 1.0),
        ...     origin=(0.0, 0.0, 0.0),
        ...     direction=torch.eye(3),
        ...     metadata={'modality': 'T1w', 'patient_id': 'P001'}
        ... )

        >>> # Extract 2D slice
        >>> axial_slice = volume.get_slice(axis=2, index=50)
        >>> print(f"Slice shape: {axial_slice.shape}")

        >>> # Get volume properties
        >>> print(f"Volume shape: {volume.shape}")
        >>> print(f"Spatial dimensions: {volume.dimensions}")

    """

    volume: torch.Tensor
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the volume."""
        return self.volume.shape

    @property
    def dimensions(self) -> tuple[int, int, int]:
        """Get spatial dimensions (D, H, W)."""
        if len(self.shape) == 3:
            return self.shape
        if len(self.shape) == 4:
            return self.shape[1:]  # Remove channel dimension
        raise ValueError(f"Unexpected volume shape: {self.shape}")

    def get_slice(self, axis: int, index: int) -> torch.Tensor:
        """Extract a 2D slice from the 3D volume."""
        if axis == 0:  # Sagittal
            return self.volume[index] if len(self.shape) == 3 else self.volume[:, index]
        if axis == 1:  # Coronal
            return self.volume[:, index] if len(self.shape) == 3 else self.volume[:, :, index]
        if axis == 2:  # Axial
            return self.volume[:, :, index] if len(self.shape) == 3 else self.volume[:, :, :, index]
        raise ValueError(f"Invalid axis: {axis}")


@dataclass
class CrossSectionalDiffusion:
    """Cross-sectional diffusion representation for 3D reconstruction.

    Represents the diffusion process for generating 3D volumes from
    single 2D images using cross-sectional conditioning.

    Attributes:
        condition_image: 2D conditioning image
        target_volume: Target 3D volume (if available for training)
        timestep: Current diffusion timestep
        noise_schedule: Noise schedule parameters
        conditioning_axis: Axis along which to condition
            (0=sagittal, 1=coronal, 2=axial)

    """

    condition_image: torch.Tensor
    target_volume: torch.Tensor | None = None
    timestep: int = 0
    noise_schedule: dict[str, Any] = field(default_factory=dict)
    conditioning_axis: int = 2  # Default to axial

    @property
    def is_training(self) -> bool:
        """Check if this is a training sample."""
        return self.target_volume is not None


@dataclass
class GenerationRequest:
    """Request entity for 3D generation tasks.

    Encapsulates all parameters needed for 3D asset generation,
    including input modalities, output formats, and generation parameters.

    Attributes:
        request_id: Unique identifier for the request
        input_type: Type of input ('text', 'image', 'volume')
        input_data: Input data (text prompt, image tensor, or volume)
        output_formats: list of desired output formats
        generation_params: Model-specific generation parameters
        created_at: Request creation timestamp

    """

    request_id: str
    input_type: str  # 'text', 'image', 'volume'
    input_data: str | torch.Tensor
    output_formats: list[str]  # ['gaussian', 'radiance_field',
    # 'mesh', 'volume']
    generation_params: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def validate(self) -> bool:
        """Validate the generation request."""
        if self.input_type not in ["text", "image", "volume"]:
            return False
        if not self.output_formats:
            return False
        return True


@dataclass
class GenerationResult:
    """Result entity for 3D generation tasks.

    Contains the generated 3D assets and associated metadata.

    Attributes:
        request_id: Reference to the original request
        outputs: Dictionary of generated outputs by format
        metadata: Generation metadata and statistics
        completed_at: Completion timestamp
        success: Whether generation was successful
        error_message: Error message if generation failed

    """

    request_id: str
    outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: str | None = None

    def add_output(self, format_type: str, data: Any) -> None:
        """Add an output in a specific format."""
        self.outputs[format_type] = data

    def get_output(self, format_type: str) -> Any:
        """Get output in a specific format."""
        return self.outputs.get(format_type)


@dataclass
class SliceToVolumeRequest:
    """Request entity for slice-to-volume generation tasks.

    Encapsulates parameters for generating 3D volumes from 2D slices,
    including conditioning information and generation parameters.

    This entity defines the input requirements and constraints
    for 3D volume generation from individual 2D MRI slices,
    supporting various generation techniques and quality requirements.

    Attributes:
        request_id: Unique identifier for the request
            - UUID or timestamp-based string
            - Used for tracking and result correlation
        slice_data: 2D slice tensor(s) for conditioning
            - Single tensor: shape (H, W) or (C, H, W)
            - list of tensors: multiple slices with consistent dimensions
            - Data type: torch.Tensor with values in [-1, 1] range
        conditioning_axis: Axis along which slices are conditioned
            - 0: sagittal (left-right slices)
            - 1: coronal (front-back slices)
            - 2: axial (top-bottom slices, most common)
        target_shape: Desired output volume shape (D, H, W)
            - Target dimensions for reconstructed volume
            - May differ from input slice dimensions
        generation_params: Model-specific generation parameters
            - Dictionary of parameters for the chosen model
            - Includes guidance scale, steps, noise level, etc.
        metadata: Additional request metadata
            - Patient information, acquisition parameters, etc.
            - Preserves context for medical applications
        created_at: Request creation timestamp
            - Automatic timestamp for request tracking

    Example:
        >>> # Single slice reconstruction
        >>> slice_tensor = torch.randn(1, 256, 256)  # (C, H, W)
        >>> request = SliceToVolumeRequest(
        ...     request_id="req_001",
        ...     slice_data=slice_tensor,
        ...     conditioning_axis=2,  # Axial
        ...     target_shape=(128, 256, 256),
        ...     generation_params={
        ...         'model_type': 'trellis',
        ...         'guidance_scale': 7.5,
        ...         'num_inference_steps': 50
        ...     },
        ...     metadata={'patient_id': 'P001', 'modality': 'T1w'}
        ... )

        >>> # Multiple slice conditioning
        >>> slices = [torch.randn(256, 256) for _ in range(10)]
        >>> multi_slice_request = SliceToVolumeRequest(
        ...     request_id="req_002",
        ...     slice_data=slices,
        ...     conditioning_axis=1,  # Coronal
        ...     target_shape=(64, 128, 128),
        ...     generation_params={
        ...         'model_type': 'x_diffusion',
        ...         'noise_level': 0.1
        ...     }
        ... )

        >>> # Validation and processing
        >>> if request.validate():
        >>>     print(f"Valid request with {request.slice_count} slices")
        >>>     # Process with orchestration service
        >>>     from domain.services.generation_3d_orchestration_service \\
        >>>         import (
        >>>         Generation3DOrchestrationService
        >>>     )
        >>>     service = Generation3DOrchestrationService()
        >>>     response = service.generate_volume(request)

    """

    request_id: str
    slice_data: torch.Tensor | list[torch.Tensor]
    conditioning_axis: int = 2  # Default to axial
    target_shape: tuple[int, int, int] = (64, 64, 64)
    generation_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def validate(self) -> bool:
        """Validate the slice-to-volume request."""
        if self.conditioning_axis not in [0, 1, 2]:
            return False
        if len(self.target_shape) != 3:
            return False
        # Reject if ANY dimension is non-positive (was ``all``, which only
        # rejected an all-non-positive shape — e.g. (-1, 64, 64) slipped through).
        if any(dim <= 0 for dim in self.target_shape):
            return False
        return True

    @property
    def slice_count(self) -> int:
        """Get the number of slices provided."""
        if isinstance(self.slice_data, list):
            return len(self.slice_data)
        shape_len = len(self.slice_data.shape)
        return self.slice_data.shape[0] if shape_len == 3 else 1


@dataclass
class VolumeGenerationResponse:
    """Response entity for volume generation tasks.

    Contains the generated 3D volume and associated metadata,
    including quality metrics and generation statistics.

    Attributes:
        request_id: Reference to the original request
        generated_volume: Generated 3D volume tensor
        confidence_scores: Confidence scores for the generation
        generation_metadata: Generation process metadata
        quality_metrics: Quality assessment metrics
        completed_at: Completion timestamp
        success: Whether generation was successful
        error_message: Error message if generation failed

    """

    request_id: str
    generated_volume: torch.Tensor | None = None
    confidence_scores: torch.Tensor | None = None
    generation_metadata: dict[str, Any] = field(default_factory=dict)
    quality_metrics: dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: str | None = None

    def validate(self) -> bool:
        """Validate the generation response."""
        if not self.success and self.generated_volume is not None:
            return False
        if self.generated_volume is not None and len(self.generated_volume.shape) != 4:
            return False
        return True

    @property
    def volume_shape(self) -> tuple[int, ...] | None:
        """Get the shape of the generated volume."""
        if self.generated_volume is not None:
            return self.generated_volume.shape
        return None

    def add_quality_metric(self, metric_name: str, value: Any) -> None:
        """Add a quality metric to the response."""
        self.quality_metrics[metric_name] = value

    def get_quality_metric(self, metric_name: str) -> Any:
        """Get a quality metric from the response.

        Raises:
            KeyError: if ``metric_name`` was never recorded — silently returning
                None hides a typo'd or missing metric (NN#3 / WS-3).
        """
        if metric_name not in self.quality_metrics:
            raise KeyError(
                f"Quality metric {metric_name!r} not recorded; "
                f"available: {sorted(self.quality_metrics)}"
            )
        return self.quality_metrics[metric_name]
