"""3D Generation Pipeline - Integration of TRELLIS and X-Diffusion
===============================================================

Pipeline-based integration for 3D asset generation combining TRELLIS
and X-Diffusion models following SOLID principles.

.. warning::

   **UNWIRED — removal candidate (2026-06-12, import repaired 2026-08-31).**
   This module now IMPORTS, but nothing calls it. Its only consumer,
   :mod:`spectramr.domain.services.generation_3d_orchestration_service`, has no
   production callers of its own, so the whole 3D-generation chain is still
   unreachable from any training/inference path.

   It was previously *broken* as well as unwired: ``from ....services.device_policy
   import ...`` reached above the ``spectramr`` top-level package (a leftover from
   the ``src -> spectramr`` refactor) and raised ``ImportError``. The consumer's
   ``try/except ImportError`` turned that into ``generation_pipeline_module = None``
   — indistinguishable from an absent optional dependency, so the breakage was
   invisible in every environment (non-negotiable 18: absent is a state to report,
   never a state to infer). The imports are absolute now, which makes the
   remaining deadness *visible* rather than concealed; it does not wire anything.

   Kept pending a keep-or-wire decision — see
   ``TODO/backlog_dead_code_pipelines_audit_2026_06_12.md``.
"""

# pyright: ignore-file

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
from tqdm import tqdm

from spectramr.config.schemas.enums import GeneratorTypes

# The device policy arrives by INJECTION and is typed against a Protocol that
# lives in the model layer.
#
# This used to import ``DevicePolicy`` + two factories from
# ``spectramr.infrastructure.services.device_policy`` -- an upward import
# (``infrastructure/ -> models/``, never the reverse; non-negotiable 5) that
# ``scripts/ci/check_layering.sh`` rejects. Before that it read
# ``from ....services.device_policy import ...``, four dots from
# ``spectramr/models/pipelines/``, which walks past the top-level package and
# raises ImportError; the caller
# (``generation_3d_orchestration_service.py:31``) wrapped it in
# ``except ImportError: generation_pipeline_module = None``, so a broken module
# was indistinguishable from an absent optional dependency in EVERY environment
# (non-negotiable 18). Repairing that import to an absolute one is what made the
# upward dependency real -- the gate had nothing to see while the module could
# not be imported at all.
#
# NOT solved with a function-local import: every grep in ``check_layering.sh``
# is anchored at ``^``, so that would turn the gate green while leaving the
# dependency in place -- the detector-blindness shape non-negotiable 15 exists
# to prevent (#1183).
from spectramr.models.interfaces.device_movement import SupportsDeviceMovement
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.representations.slat import (
    TrellisGaussianRenderer,
    TrellisMeshRenderer,
    VolumeRenderer,
)
from spectramr.models.representations.slat.interfaces import GaussianSLATConfig, MeshSLATConfig


class GenerationPipeline(ABC):
    """Abstract base class for 3D generation pipelines.

    Defines the interface for pipelines that can generate 3D assets
    from various input modalities.
    """

    def __init__(
        self,
        device_policy: SupportsDeviceMovement | None = None,
        cpu_policy: SupportsDeviceMovement | None = None,
    ) -> None:
        """Store the injected policies.

        Both arrive from the composition root
        (``domain.services.generation_3d_orchestration_service``), which is the
        layer allowed to construct infrastructure services. They used to be
        built HERE via ``create_cpu_device_policy()`` and
        ``get_default_device_policy()``, which is what made the model layer
        depend on ``infrastructure/`` at runtime and not merely in its
        annotations.

        Args:
            device_policy: Policy for the compute device. Required at use time;
                see :attr:`device_policy`.
            cpu_policy: Policy that moves data to CPU for downstream consumers.
                Required only if :meth:`_ensure_cpu` is called.
        """
        self._device_policy = device_policy
        self._cpu_policy = cpu_policy

    @property
    def device_policy(self) -> SupportsDeviceMovement:
        """Return the injected device policy.

        RAISES when none was injected, rather than falling back to a default
        built here. The fallback was ``get_default_device_policy()``, which
        resolves a device on its own -- exactly the second device resolver
        non-negotiable 9b forbids, reached without the caller ever asking. The
        one live caller always passes a policy
        (``generation_3d_orchestration_service.py:178`` does
        ``device_policy or create_auto_device_policy()``), so this changes no
        reachable behaviour; it just refuses to invent one (non-negotiable 3).
        """
        if self._device_policy is None:
            raise ValueError(
                "No device_policy was injected into this GenerationPipeline. "
                "Construct it through the composition root "
                "(domain.services.generation_3d_orchestration_service), or pass "
                "device_policy= explicitly. The model layer does not build "
                "device policies (non-negotiable 5)."
            )
        return self._device_policy

    def _ensure_cpu(self, data: Any) -> Any:
        """Move outputs to CPU for downstream consumers."""
        if self._cpu_policy is None:
            raise ValueError(
                "No cpu_policy was injected into this GenerationPipeline, so "
                "outputs cannot be moved to CPU. Pass cpu_policy= from the "
                "composition root."
            )
        return self._cpu_policy.move_to_device(data)

    @abstractmethod
    def generate(
        self,
        input_data: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        """Generate 3D output from input data."""

    @abstractmethod
    def get_pipeline_info(self) -> dict[str, Any]:
        """Get information about the pipeline configuration."""


class TrellisPipeline(GenerationPipeline):
    """Pipeline for TRELLIS-based 3D generation.

    Uses TRELLIS generators to create 3D assets from 2D images.
    """

    def __init__(
        self,
        generator_type: str = "trellis_gaussian",
        *,
        device_policy: SupportsDeviceMovement | None = None,
        cpu_policy: SupportsDeviceMovement | None = None,
    ) -> None:
        """Initialize TRELLIS pipeline.

        Args:
            generator_type: Type of TRELLIS generator to use

        """
        super().__init__(device_policy=device_policy, cpu_policy=cpu_policy)
        self.factory = get_model_factory()
        self.generator_type = generator_type
        self.generator: IGenerator | None = None
        self.logger = logging.getLogger(__name__)
        self.renderer = self._build_renderer(generator_type)
        self.last_artifacts: dict[str, Any] | None = None

    def _ensure_generator(self, in_channels: int = 1, out_channels: int = 1):
        """Ensure generator is loaded."""
        if self.generator is None:
            try:
                generator = self.factory.create_generator(
                    self.generator_type,
                    in_channels=in_channels,
                    out_channels=out_channels,
                )
                generator = self.device_policy.move_to_device(generator)
                generator.eval()
                self.generator = generator
            except Exception as e:
                self.logger.warning(
                    "Failed to create generator '%s': %s",
                    self.generator_type,
                    e,
                )
                self.logger.info(
                    "Using minimal fallback implementation for TRELLIS",
                )
                # Don't raise - use fallback implementation instead

    def _build_renderer(self, generator_type: str):
        """Instantiate a renderer that matches the generator type."""

        if (
            generator_type == "trellis_gaussian"
            or generator_type == GeneratorTypes.TRELLIS_3D.value
        ):
            return TrellisGaussianRenderer(GaussianSLATConfig())
        if generator_type == "trellis_mesh":
            return TrellisMeshRenderer(MeshSLATConfig())
        return None

    def _build_artifacts(
        self,
        raw_output: Any,
        source_tensor: torch.Tensor,
        *,
        renderer_options: Any,
    ) -> dict[str, Any]:
        """Wrap raw generator output together with rendered representations."""

        artifacts: dict[str, Any] = {"raw": raw_output}

        if self.renderer is not None:
            try:
                render_kwargs = renderer_options or {}
                render_output = self.renderer.render(
                    raw_output,
                    source=source_tensor,
                    **render_kwargs,
                )
                artifacts["representations"] = {
                    render_output.format: render_output.items,
                }
                artifacts["representation_metadata"] = {
                    render_output.format: render_output.metadata,
                }
            except Exception as exc:  # pragma: no cover - defensive guard
                self.logger.warning(
                    "Renderer failed (%s); returning raw output only",
                    exc,
                )

        return artifacts

    def generate(
        self,
        input_data: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        """Generate 3D output using TRELLIS.

        Args:
            input_data: Input 2D image (B, C, H, W)
            **kwargs: Additional generation parameters

        Returns:
            Generated 3D output

        """
        # Infer parameters from input data
        in_channels = input_data.shape[1] if len(input_data.shape) > 1 else 1
        out_channels = int(kwargs.get("out_channels", 1))

        generator_kwargs = dict(kwargs)
        renderer_options = generator_kwargs.pop("renderer_options", {})
        requested_type = generator_kwargs.pop("generator_type", None)

        if requested_type and requested_type != self.generator_type:
            self.generator_type = requested_type
            self.generator = None
            self.renderer = self._build_renderer(requested_type)

        self._ensure_generator(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        input_tensor = input_data
        with torch.no_grad():
            device_tensor = self.device_policy.move_to_device(input_tensor)

            if self.generator is None:
                output = self._fallback_generate(
                    device_tensor,
                    **generator_kwargs,
                )
            else:
                output = self.generator.generate(
                    input_tensor,
                    **generator_kwargs,
                )

        output_cpu = self._cpu_policy.move_to_device(output)
        source_cpu = self._cpu_policy.move_to_device(input_tensor)
        artifacts = self._build_artifacts(
            output_cpu,
            source_tensor=source_cpu,
            renderer_options=renderer_options,
        )
        self.last_artifacts = artifacts

        return artifacts.get("raw", output_cpu)

    def _fallback_generate(
        self,
        input_data: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Fallback Gaussian parameter synthesis from 2D inputs."""

        # Default depth of generated volume when no generator is available
        depth = int(kwargs.get("depth", 16))
        depth = max(1, depth)

        # Interpolate slices along depth axis to mimic volumetric output
        slices = []
        for idx in range(depth):
            blend = 0.0 if depth == 1 else idx / float(depth - 1)
            noise = torch.randn_like(input_data) * 0.05
            slice_tensor = torch.clamp(input_data + blend * noise, -1.0, 1.0)
            slices.append(slice_tensor)

        volume = torch.stack(slices, dim=2)
        return volume

    def get_pipeline_info(self) -> dict[str, Any]:
        """Get TRELLIS pipeline information."""
        return {
            "pipeline_type": "trellis",
            "generator_type": self.generator_type,
            "description": "TRELLIS-based 3D asset generation from 2D images",
            "capabilities": ["gaussian_splatting", "mesh_generation"],
            "representation_formats": self._describe_representation_formats(),
        }

    def _describe_representation_formats(self) -> list[str]:
        """_describe_representation_formats.

        Returns:
            list[str]: Description.
        """
        if isinstance(self.renderer, TrellisGaussianRenderer):
            return ["gaussian"]
        if isinstance(self.renderer, TrellisMeshRenderer):
            return ["mesh"]
        return []


class XDiffusionPipeline(GenerationPipeline):
    """Pipeline for X-Diffusion-based 3D generation.

    Uses X-Diffusion latent generator to create 3D volumes from 2D slices.
    """

    def __init__(
        self,
        generator_type: str = "x_diffusion_latent",
        *,
        device_policy: SupportsDeviceMovement | None = None,
        cpu_policy: SupportsDeviceMovement | None = None,
    ) -> None:
        """Initialize X-Diffusion pipeline.

        Args:
            generator_type: Type of X-Diffusion generator to use

        """
        super().__init__(device_policy=device_policy, cpu_policy=cpu_policy)
        self.factory = get_model_factory()
        self.generator_type = generator_type
        self.generator: IGenerator | None = None
        self.logger = logging.getLogger(__name__)
        self.renderer = VolumeRenderer()
        self.last_artifacts: dict[str, Any] | None = None

    def _ensure_generator(self, in_channels: int = 1, out_channels: int = 1):
        """Ensure generator is loaded."""
        if self.generator is None:
            try:
                generator = self.factory.create_generator(
                    self.generator_type,
                    in_channels=in_channels,
                    out_channels=out_channels,
                )
                generator = self.device_policy.move_to_device(generator)
                generator.eval()
                self.generator = generator
            except Exception as e:
                self.logger.warning(
                    "Failed to create generator '%s': %s",
                    self.generator_type,
                    e,
                )
                self.logger.info(
                    "Using minimal fallback implementation for X-Diffusion",
                )
                # Don't raise - use fallback implementation instead

    def generate(
        self,
        input_data: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        """Generate 3D output using X-Diffusion.

        Args:
            input_data: Input 2D slice (B, C, H, W)
            **kwargs: Additional generation parameters

        Returns:
            Generated 3D volume (B, C, D, H, W)

        """
        # Infer parameters from input data
        in_channels = input_data.shape[1] if len(input_data.shape) > 1 else 1
        out_channels = int(kwargs.get("out_channels", 1))

        generator_kwargs = dict(kwargs)
        requested_type = generator_kwargs.pop("generator_type", None)
        renderer_options = generator_kwargs.pop("renderer_options", {})

        if requested_type and requested_type != self.generator_type:
            self.generator_type = requested_type
            self.generator = None

        self._ensure_generator(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        input_tensor = input_data
        with torch.no_grad():
            device_tensor = self.device_policy.move_to_device(input_tensor)

            if self.generator is None:
                output = self._fallback_generate(
                    device_tensor,
                    **generator_kwargs,
                )
            else:
                output = self.generator.generate(
                    input_tensor,
                    **generator_kwargs,
                )

        output_cpu = self._cpu_policy.move_to_device(output)
        artifacts = self._build_artifacts(
            output_cpu,
            renderer_options=renderer_options,
        )
        self.last_artifacts = artifacts

        return artifacts.get("raw", output_cpu)

    def _build_artifacts(
        self,
        raw_output: torch.Tensor,
        *,
        renderer_options: Any,
    ) -> dict[str, Any]:
        """Wrap X-Diffusion outputs with volume representations."""

        artifacts: dict[str, Any] = {"raw": raw_output}

        if self.renderer is not None:
            try:
                render_kwargs = renderer_options or {}
                render_output = self.renderer.render(
                    raw_output,
                    **render_kwargs,
                )
                artifacts["representations"] = {
                    render_output.format: render_output.items,
                }
                artifacts["representation_metadata"] = {
                    render_output.format: render_output.metadata,
                }
            except Exception as exc:  # pragma: no cover - defensive guard
                self.logger.warning(
                    "Volume renderer failed (%s); returning raw output only",
                    exc,
                )

        return artifacts

    def _fallback_generate(
        self,
        input_data: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Fallback 3D generation using simple diffusion-like process.

        Args:
            input_data: Input 2D slice (B, C, H, W)
            **kwargs: Additional generation parameters

        Returns:
            Generated 3D volume (B, C, D, H, W)

        """
        # Simple fallback: create 3D volume by stacking modified versions
        # of the input slice with progressive noise addition
        depth = int(kwargs.get("depth", 16))

        # Start with the input slice
        current_slice = input_data

        # Create volume by stacking slices with increasing noise
        volume_slices: list[torch.Tensor] = []

        for step in tqdm(range(depth), desc="Generating 3D Fallback"):
            # Add progressive noise (simulating diffusion process)
            noise_level = float(step) / max(depth, 1)
            noise = torch.randn_like(current_slice) * noise_level * 0.5
            noisy_slice = torch.clamp(current_slice + noise, -1.0, 1.0)
            volume_slices.append(noisy_slice)

        # Stack slices into 3D volume
        output_3d = torch.stack(volume_slices, dim=2)  # (B, C, D, H, W)

        return output_3d

    def get_pipeline_info(self) -> dict[str, Any]:
        """Get X-Diffusion pipeline information."""
        return {
            "pipeline_type": "x_diffusion",
            "generator_type": self.generator_type,
            "description": ("X-Diffusion-based 3D volume generation from 2D slices"),
            "capabilities": ["latent_diffusion", "cross_sectional_generation"],
            "representation_formats": ["volume"],
        }


class Hybrid3DPipeline(GenerationPipeline):
    """Hybrid pipeline combining TRELLIS and X-Diffusion approaches.

    Allows switching between different 3D generation strategies based
    on the input type and desired output format.
    """

    def __init__(
        self,
        *,
        device_policy: SupportsDeviceMovement | None = None,
        cpu_policy: SupportsDeviceMovement | None = None,
    ) -> None:
        """Initialize hybrid pipeline."""
        super().__init__(device_policy=device_policy, cpu_policy=cpu_policy)
        shared_policy = self.device_policy
        self.trellis_pipeline = TrellisPipeline(device_policy=shared_policy, cpu_policy=cpu_policy)
        self.x_diffusion_pipeline = XDiffusionPipeline(
            device_policy=shared_policy,
            cpu_policy=cpu_policy,
        )
        self.logger = logging.getLogger(__name__)

    def generate(
        self,
        input_data: torch.Tensor,
        pipeline_type: str = "auto",
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate 3D output using appropriate pipeline.

        Args:
            input_data: Input data (2D image/slice)
            pipeline_type: Pipeline to use ("trellis", "x_diffusion", "auto")
            **kwargs: Additional generation parameters

        Returns:
            Generated 3D output

        """
        if pipeline_type == "auto":
            # Auto-select based on input characteristics
            pipeline_type = self._auto_select_pipeline(input_data)

        if pipeline_type == GeneratorTypes.TRELLIS_3D.value:
            return self.trellis_pipeline.generate(input_data, **kwargs)
        if pipeline_type == GeneratorTypes.DIFFUSION.value:
            return self.x_diffusion_pipeline.generate(input_data, **kwargs)
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    def _auto_select_pipeline(self, input_data: torch.Tensor) -> str:
        """Auto-select pipeline based on input characteristics."""
        # Simple heuristic: use X-Diffusion for medical imaging
        # (single channel) and TRELLIS for general images (3 channels)
        # and TRELLIS for general images (3 channels)
        if input_data.shape[1] == 1:
            return GeneratorTypes.DIFFUSION.value  # Likely medical MRI
        return GeneratorTypes.TRELLIS_3D.value  # Likely general images

    def get_available_pipelines(self) -> dict[str, dict[str, Any]]:
        """Get information about available pipelines."""
        return {
            GeneratorTypes.TRELLIS_3D.value: self.trellis_pipeline.get_pipeline_info(),
            GeneratorTypes.DIFFUSION.value: self.x_diffusion_pipeline.get_pipeline_info(),
        }

    def get_pipeline_info(self) -> dict[str, Any]:
        """Get hybrid pipeline information."""
        return {
            "pipeline_type": "hybrid",
            "description": ("Hybrid pipeline combining TRELLIS and X-Diffusion approaches"),
            "capabilities": [
                "gaussian_splatting",
                "mesh_generation",
                "latent_diffusion",
                "cross_sectional_generation",
            ],
        }


def create_3d_generation_pipeline(
    pipeline_type: str = "hybrid",
    *,
    device_policy: SupportsDeviceMovement | None = None,
    cpu_policy: SupportsDeviceMovement | None = None,
) -> GenerationPipeline:
    """Factory function for creating 3D generation pipelines.

    Args:
        pipeline_type: Type of pipeline to create

    Returns:
        Configured pipeline instance

    """
    if pipeline_type == "trellis":
        return TrellisPipeline(device_policy=device_policy, cpu_policy=cpu_policy)
    if pipeline_type == "x_diffusion":
        return XDiffusionPipeline(device_policy=device_policy, cpu_policy=cpu_policy)
    if pipeline_type == "hybrid":
        return Hybrid3DPipeline(device_policy=device_policy, cpu_policy=cpu_policy)
    raise ValueError(f"Unknown pipeline type: {pipeline_type}")


# Factory function for getting model factory (for testing)
def get_model_factory():
    """Get model factory instance."""
    from spectramr.models.factories.model_factory import get_model_factory

    return get_model_factory()
