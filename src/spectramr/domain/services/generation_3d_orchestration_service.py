"""3D Generation Orchestration Service

Domain service for orchestrating 3D asset generation using TRELLIS
and X-Diffusion pipelines following SOLID principles.

.. warning::

   **DORMANT (2026-06-12).** This service is not wired into any live
   execution path. Its pipeline backend,
   :mod:`spectramr.models.pipelines.generation_pipeline`, is broken dead code
   (it raises ``ImportError`` on import — see that module's banner), and the
   import here is guarded by ``try/except ImportError`` so
   ``generation_pipeline_module`` is always ``None`` in practice — meaning
   :meth:`Generation3DOrchestrationService._create_default_pipelines` cannot
   build its defaults. Retained pending a keep-or-remove decision; see
   ``TODO/backlog_dead_code_pipelines_audit_2026_06_12.md``.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4

import torch

from spectramr.domain.entities.entities_3d import SLAT, Gaussian3D, Mesh, MRIVolume

try:  # pragma: no cover - optional dependency
    from spectramr.models.pipelines import generation_pipeline as generation_pipeline_module
except ImportError:  # pragma: no cover - optional dependency
    generation_pipeline_module = None

from spectramr.domain.interfaces.infrastructure_interfaces import (
    IDevicePolicy,
    IPipelineExecutionService,
)

# We keep factory import for the default argument in __init__ (pragmatic exception)
# or we move default logic to the factory function at bottom.
from spectramr.infrastructure.services.device_policy import (
    create_auto_device_policy,
    create_cpu_device_policy,
)
from spectramr.infrastructure.services.pipeline_execution_service import (
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,  # Enums might need their own common place
)

# Note: create_auto_device_policy and ExecutionContext/Result/Status implementations
# are still needed for instantiation helpers or enum references if strictly required,
# but we aim to depend on interfaces in the class body.
# For strictly breaking the import, we should move factories to DI layer.
# However, for this remediation, we primarily switch Type Hints.


class GenerationMode(Enum):
    """Modes for 3D generation."""

    TRELLIS_GAUSSIAN = "trellis_gaussian"
    TRELLIS_MESH = "trellis_mesh"
    X_DIFFUSION_LATENT = "x_diffusion_latent"
    HYBRID = "hybrid"


@dataclass
class GenerationRequest:
    """Request for 3D generation."""

    input_data: torch.Tensor
    mode: GenerationMode
    parameters: dict[str, Any]
    metadata: dict[str, Any] | None = None


@dataclass
class GenerationResult:
    """Result of 3D generation."""

    output_data: Any
    mode_used: GenerationMode
    parameters_used: dict[str, Any]
    metadata: dict[str, Any]
    success: bool
    error_message: str | None = None


class I3DGenerationPipeline(Protocol):
    """Protocol for 3D generation pipelines."""

    def generate(self, input_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate 3D output from input data."""
        ...

    def get_pipeline_info(self) -> dict[str, Any]:
        """Get pipeline information."""
        ...


class I3DGenerationService(ABC):
    """Interface for 3D generation services.

    Defines the contract for services that can generate 3D assets
    from various input modalities.
    """

    @abstractmethod
    def generate_3d(self, request: GenerationRequest) -> GenerationResult:
        """Generate 3D output from input data."""

    @abstractmethod
    def validate_request(self, request: GenerationRequest) -> bool:
        """Validate a generation request."""

    @abstractmethod
    def get_supported_modes(self) -> list[GenerationMode]:
        """Get list of supported generation modes."""

    @abstractmethod
    def get_mode_requirements(self, mode: GenerationMode) -> dict[str, Any]:
        """Get requirements for a specific generation mode."""


class Generation3DOrchestrationService(I3DGenerationService):
    """Domain service for orchestrating 3D generation.

    This service coordinates between different 3D generation pipelines
    and provides a unified interface for 3D asset generation.

    ### Orchestration Flow
    ```mermaid
    sequenceDiagram
        participant Client
        participant Orchestrator as Generation3DOrchestrationService
        participant Pipeline as I3DGenerationPipeline
        participant Executor as PipelineExecutionService

        Client->>Orchestrator: generate_3d(request)
        Orchestrator->>Orchestrator: validate_request()

        alt Invalid Request
            Orchestrator-->>Client: GenerationResult (Success=False)
        end

        Orchestrator->>Orchestrator: _prepare_input_for_execution()

        alt With Executor
            Orchestrator->>Executor: execute_pipeline(input, context)
            Executor->>Pipeline: generate(input, **kwargs)
            Pipeline-->>Executor: RawOutput
            Executor-->>Orchestrator: RawOutput, ExecutionResult
        else Direct Execution
            Orchestrator->>Pipeline: generate(input, **kwargs)
            Pipeline-->>Orchestrator: RawOutput
        end

        Orchestrator->>Orchestrator: _convert_pipeline_output(RawOutput)
        Orchestrator-->>Client: GenerationResult (Success=True)
    ```
    """

    def __init__(
        self,
        logger: logging.Logger,
        pipelines: dict[GenerationMode, I3DGenerationPipeline] | None = None,
        pipeline_executor: IPipelineExecutionService | None = None,
        device_policy: IDevicePolicy | None = None,
    ):
        """Initialize the 3D generation orchestration service.

        Args:
            logger: Logger instance for logging
            pipelines: Dictionary of generation pipelines by mode

        """
        self.logger = logger
        self._pipelines = pipelines or {}
        self._pipeline_executor = pipeline_executor
        self._device_policy = device_policy or create_auto_device_policy()
        # The model layer no longer builds this for itself: constructing an
        # infrastructure service from ``models/`` is the upward import
        # non-negotiable 5 forbids, so the composition root supplies it.
        # ``GenerationPipeline._ensure_cpu`` raises without it rather than
        # silently skipping the move to CPU.
        self._cpu_policy = create_cpu_device_policy()
        if not pipelines:
            self._initialize_pipelines()

    def _initialize_pipelines(self):
        """Initialize available generation pipelines."""
        if generation_pipeline_module is None:
            self.logger.warning(
                "Generation pipeline module unavailable; skipping initialization",
            )
            self.logger.info(
                "3D generation pipelines not available - generators may not be registered",
            )
            return

        try:
            # Initialize individual pipelines
            self._pipelines[GenerationMode.TRELLIS_GAUSSIAN] = (
                generation_pipeline_module.create_3d_generation_pipeline(
                    "trellis",
                    device_policy=self._device_policy,
                    cpu_policy=self._cpu_policy,
                )
            )
            self._pipelines[GenerationMode.TRELLIS_MESH] = (
                generation_pipeline_module.create_3d_generation_pipeline(
                    "trellis",
                    device_policy=self._device_policy,
                    cpu_policy=self._cpu_policy,
                )
            )
            self._pipelines[GenerationMode.X_DIFFUSION_LATENT] = (
                generation_pipeline_module.create_3d_generation_pipeline(
                    "x_diffusion",
                    device_policy=self._device_policy,
                    cpu_policy=self._cpu_policy,
                )
            )
            self._pipelines[GenerationMode.HYBRID] = (
                generation_pipeline_module.create_3d_generation_pipeline(
                    "hybrid",
                    device_policy=self._device_policy,
                    cpu_policy=self._cpu_policy,
                )
            )

            self.logger.info("Initialized 3D generation pipelines")
        except Exception as e:
            self.logger.error(f"Failed to initialize pipelines: {e}")
            self.logger.error(
                "3D generation service cannot operate without pipelines",
            )
            raise RuntimeError(
                "Failed to initialize 3D generation pipelines. "
                "Ensure that required generators are registered in "
                "the model factory.",
            ) from e

    def generate_3d(self, request: GenerationRequest) -> GenerationResult:
        """Generate 3D output from input data.

        Args:
            request: Generation request with input data and parameters

        Returns:
            Generation result with output data and metadata

        """
        try:
            # Validate request
            if not self.validate_request(request):
                return GenerationResult(
                    output_data=torch.empty(0),
                    mode_used=request.mode,
                    parameters_used=request.parameters,
                    metadata={"validation_failed": True},
                    success=False,
                    error_message="Request validation failed",
                )

            # Get appropriate pipeline
            pipeline = self._pipelines.get(request.mode)
            if pipeline is None:
                error_msg = (
                    f"No pipeline available for mode: {request.mode}. "
                    "3D generation pipelines may not be properly configured. "
                    "Check that required generators are registered in "
                    "the model factory."
                )
                return GenerationResult(
                    output_data=torch.empty(0),
                    mode_used=request.mode,
                    parameters_used=request.parameters,
                    metadata={"pipeline_not_found": True},
                    success=False,
                    error_message=error_msg,
                )

            # Generate 3D output
            self.logger.info(
                "Starting 3D generation with mode: %s",
                request.mode,
            )

            input_tensor = self._prepare_input_for_execution(request.input_data)
            output_data, exec_result = self._execute_pipeline(
                pipeline,
                input_tensor,
                request,
            )

            (
                processed_output,
                conversion_metadata,
            ) = self._convert_pipeline_output(output_data)

            # Create result metadata
            metadata: dict[str, Any] = {
                "input_shape": list(request.input_data.shape),
                "output_type": type(processed_output).__name__,
                "generation_mode": request.mode.value,
                "pipeline_info": pipeline.get_pipeline_info(),
                **(request.metadata or {}),
            }
            metadata.update(conversion_metadata)
            if exec_result and exec_result.metrics:
                metadata["execution_metrics"] = asdict(exec_result.metrics)

            self.logger.info(
                "Successfully generated 3D output with shape: %s",
                metadata.get("raw_output_shape") or metadata.get("representation_counts"),
            )

            return GenerationResult(
                output_data=processed_output,
                mode_used=request.mode,
                parameters_used=request.parameters,
                metadata=metadata,
                success=True,
            )

        except Exception as e:
            self.logger.error(f"3D generation failed: {e}")
            return GenerationResult(
                output_data=torch.empty(0),
                mode_used=request.mode,
                parameters_used=request.parameters,
                metadata={"error": str(e)},
                success=False,
                error_message=str(e),
            )

    def _prepare_input_for_execution(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move input tensor to the device specified by the device policy."""

        if self._device_policy is None:
            return tensor

        try:
            moved = self._device_policy.move_to_device(tensor)
        except Exception:  # pragma: no cover - defensive fallback
            return tensor
        return moved

    def _execute_pipeline(
        self,
        pipeline: I3DGenerationPipeline,
        input_tensor: torch.Tensor,
        request: GenerationRequest,
    ) -> tuple[torch.Tensor, ExecutionResult | None]:
        """Execute the pipeline via the executor if available."""

        generator_kwargs = dict(request.parameters)
        if request.mode in [
            GenerationMode.TRELLIS_GAUSSIAN,
            GenerationMode.TRELLIS_MESH,
        ]:
            generator_kwargs.setdefault(
                "generator_type",
                (
                    "trellis_gaussian"
                    if request.mode == GenerationMode.TRELLIS_GAUSSIAN
                    else "trellis_mesh"
                ),
            )

        if self._pipeline_executor is None:
            output = pipeline.generate(input_tensor, **generator_kwargs)
            return output, None

        context = ExecutionContext(
            execution_id=str(uuid4()),
            device=self._device_policy.get_device(),
            enable_profiling=False,
        )

        exec_result = self._pipeline_executor.execute_pipeline(
            pipeline,
            input_tensor,
            context,
            **generator_kwargs,
        )

        if exec_result.status == ExecutionStatus.COMPLETED and exec_result.output_data is not None:
            return exec_result.output_data, exec_result

        # Fallback to direct execution if executor failed
        self.logger.warning(
            "Pipeline executor failed or returned no output; "
            "falling back to direct pipeline execution",
        )
        output = pipeline.generate(input_tensor, **generator_kwargs)
        return output, exec_result

    def validate_request(self, request: GenerationRequest) -> bool:
        """Validate a generation request.

        Args:
            request: Request to validate

        Returns:
            True if request is valid

        """
        # Check if mode is supported
        if request.mode not in self.get_supported_modes():
            self.logger.warning(f"Unsupported generation mode: {request.mode}")
            return False

        # Check input data
        if not isinstance(request.input_data, torch.Tensor):
            self.logger.warning("Input data must be a torch.Tensor")
            return False

        if request.input_data.dim() != 4:  # Expected: (B, C, H, W)
            self.logger.warning(
                "Input data must be 4D tensor, got %sD",
                request.input_data.dim(),
            )
            return False

        # Check parameters based on mode requirements
        requirements = self.get_mode_requirements(request.mode)
        for param, param_type in requirements.items():
            if param in request.parameters:
                if not isinstance(request.parameters[param], param_type):
                    # param_type may be a tuple of types (e.g. (int, float)),
                    # which has no __name__ — render it tuple-safely.
                    type_name = (
                        param_type.__name__
                        if isinstance(param_type, type)
                        else " or ".join(t.__name__ for t in param_type)
                    )
                    self.logger.warning(
                        "Parameter %s must be of type %s",
                        param,
                        type_name,
                    )
                    return False

        return True

    def get_supported_modes(self) -> list[GenerationMode]:
        """Get list of supported generation modes."""
        return list(self._pipelines.keys())

    def get_mode_requirements(self, mode: GenerationMode) -> dict[str, Any]:
        """Get requirements for a specific generation mode.

        Args:
            mode: Generation mode

        Returns:
            Dictionary of parameter requirements

        """
        base_requirements = {
            "guidance_scale": (int, float),
            "num_inference_steps": int,
        }

        mode_specific = {
            GenerationMode.TRELLIS_GAUSSIAN: {
                "num_points": int,
                "resolution": int,
            },
            GenerationMode.TRELLIS_MESH: {
                "num_faces": int,
                "resolution": int,
                "simplify_mesh": bool,
            },
            GenerationMode.X_DIFFUSION_LATENT: {
                "latent_dim": int,
                "num_slices": int,
                "timesteps": int,
                "beta_schedule": str,
            },
            GenerationMode.HYBRID: {},  # Auto-determined
        }

        requirements = base_requirements.copy()
        requirements.update(mode_specific.get(mode, {}))

        return requirements

    def _convert_pipeline_output(
        self,
        pipeline_output: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Convert pipeline output into domain entities with metadata."""

        metadata: dict[str, Any] = {}

        if not isinstance(pipeline_output, dict):
            metadata.update(self._summarise_raw_output(pipeline_output))
            return pipeline_output, metadata

        processed: dict[str, Any] = {}

        raw_component = pipeline_output.get("raw")
        if raw_component is not None:
            processed["raw"] = raw_component
            metadata.update(self._summarise_raw_output(raw_component))

        representation_metadata = pipeline_output.get(
            "representation_metadata",
        )
        if representation_metadata:
            metadata["representation_metadata"] = representation_metadata

        representations = pipeline_output.get("representations") or {}
        domain_representations: dict[str, list[Any]] = {}

        for format_type, records in representations.items():
            domain_representations[format_type] = [
                self._build_domain_entity(format_type, record) for record in records
            ]

        if domain_representations:
            processed["representations"] = domain_representations
            metadata["representation_formats"] = sorted(
                domain_representations.keys(),
            )
            metadata["representation_counts"] = {
                key: len(items) for key, items in domain_representations.items()
            }

        slat_payload = pipeline_output.get("slat")
        if slat_payload is not None:
            slat_entity = self._build_slat_entity(slat_payload)
            if slat_entity is not None:
                processed.setdefault("representations", {}).setdefault(
                    "slat",
                    [],
                ).append(slat_entity)
                formats = metadata.setdefault("representation_formats", [])
                if "slat" not in formats:
                    formats.append("slat")
                counts = metadata.setdefault("representation_counts", {})
                counts["slat"] = len(processed["representations"]["slat"])

        return processed or pipeline_output, metadata

    def _build_domain_entity(self, format_type: str, payload: Any) -> Any:
        """Build domain entities from renderer payloads."""

        if format_type == "gaussian" and isinstance(payload, dict):
            return Gaussian3D(
                positions=payload["positions"],
                rotations=payload["rotations"],
                scales=payload["scales"],
                opacities=payload["opacities"],
                colors=payload["colors"],
                sh_degree=int(payload.get("sh_degree", 0)),
            )

        if format_type == "mesh" and isinstance(payload, dict):
            return Mesh(
                vertices=payload["vertices"],
                faces=payload["faces"],
                normals=payload.get("normals"),
                colors=payload.get("colors"),
            )

        if format_type == "volume" and isinstance(payload, dict):
            direction = payload.get("direction")
            if not isinstance(direction, torch.Tensor):
                direction = torch.as_tensor(direction, dtype=torch.float32)
            return MRIVolume(
                volume=payload["volume"],
                spacing=tuple(payload.get("spacing", (1.0, 1.0, 1.0))),
                origin=tuple(payload.get("origin", (0.0, 0.0, 0.0))),
                direction=direction,
                metadata=payload.get("metadata", {}),
            )

        return payload

    def _build_slat_entity(self, payload: Any) -> SLAT | None:
        """Construct a SLAT entity when payload data is available."""

        if not isinstance(payload, dict):
            return None

        sparse_structure = payload.get("sparse_structure")
        latent = payload.get("slat_latent") or payload.get("latent")
        resolution = payload.get("resolution")
        if sparse_structure is None or latent is None or resolution is None:
            return None

        return SLAT(
            sparse_structure=sparse_structure,
            slat_latent=latent,
            resolution=tuple(resolution),
            metadata=payload.get("metadata", {}),
        )

    @staticmethod
    def _summarise_raw_output(raw_output: Any) -> dict[str, Any]:
        """Create a lightweight summary for raw pipeline outputs."""

        summary: dict[str, Any] = {
            "raw_output_type": type(raw_output).__name__,
        }

        if isinstance(raw_output, torch.Tensor):
            summary["raw_output_shape"] = list(raw_output.shape)
            summary["raw_output_dtype"] = str(raw_output.dtype)
        elif isinstance(raw_output, dict):
            summary["raw_output_keys"] = list(raw_output.keys())
        elif isinstance(raw_output, (list, tuple)):
            summary["raw_output_length"] = len(raw_output)

        return summary

    def get_service_status(self) -> dict[str, Any]:
        """Get the current status of the service."""
        status = {
            "service_type": "3d_generation_orchestration",
            "supported_modes": [mode.value for mode in self.get_supported_modes()],
            "pipelines_initialized": len(self._pipelines),
            "healthy": len(self._pipelines) > 0,
        }

        if len(self._pipelines) == 0:
            status["status_message"] = (
                "No pipelines available. 3D generation requires generator "
                "registration in model factory."
            )
            status["setup_required"] = True
        else:
            status["status_message"] = "Service ready for 3D generation"
            status["setup_required"] = False

        return status


def create_generation_3d_orchestration_service(
    logger: logging.Logger,
    pipelines: dict[GenerationMode, I3DGenerationPipeline] | None = None,
) -> I3DGenerationService:
    """Factory function for creating Generation3DOrchestrationService.

    Args:
        logger: Logger instance for the service
        pipelines: Optional dictionary of generation pipelines

    Returns:
        Configured I3DGenerationService instance

    """
    return Generation3DOrchestrationService(logger=logger, pipelines=pipelines)
