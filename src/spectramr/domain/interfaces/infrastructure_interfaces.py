from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class IDevicePolicy(Protocol):
    """Protocol for device usage policies."""

    def get_device(self) -> torch.device:
        """Get the primary computation device."""
        ...

    def move_to_device(self, data: Any) -> Any:
        """Move tensor or collection to the target device."""
        ...


@runtime_checkable
class IExecutionContext(Protocol):
    """Protocol for execution context."""

    execution_id: str
    device: torch.device
    enable_profiling: bool
    metadata: dict[str, Any]


@runtime_checkable
class IExecutionResult(Protocol):
    """Protocol for execution results."""

    execution_id: str
    status: Any
    output_data: Any
    error: Any
    metrics: Any


@runtime_checkable
class IPipelineExecutionService(Protocol):
    """Protocol for pipeline execution service.

    ### Infrastructure Execution Hierarchy
    ```mermaid
    classDiagram
        class IDevicePolicy {
            <<protocol>>
            +get_device()
            +move_to_device(data)
        }
        class IPipelineExecutionService {
            <<protocol>>
            +execute_pipeline(pipeline, data, context)
        }
        class IExecutionContext {
            <<protocol>>
            +str execution_id
            +torch.device device
            +bool enable_profiling
        }
        class IExecutionResult {
            <<protocol>>
            +str execution_id
            +Any status
            +Any output_data
        }
        IPipelineExecutionService ..> IExecutionContext : uses
        IPipelineExecutionService ..> IExecutionResult : returns
    ```
    """

    def execute_pipeline(
        self, pipeline: Any, data: Any, context: IExecutionContext, **kwargs
    ) -> IExecutionResult:
        """Execute a pipeline with monitoring and error handling."""
        ...


@runtime_checkable
class IPhysicsOperator(Protocol):
    """
    Protocol for physics operators (FFT, NUFFT, data consistency).

    All physics operators must implement this interface.
    """

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply the physics operator.

        Args:
            x: Input tensor
            **kwargs: Operator-specific parameters (e.g., mask, sensitivity maps)

        Returns:
            Transformed tensor
        """
        ...

    def adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply the adjoint (inverse) of the physics operator.

        Args:
            y: Input tensor in transformed domain
            **kwargs: Operator-specific parameters

        Returns:
            Tensor in original domain
        """
        ...


@runtime_checkable
class ITransform(Protocol):
    """
    Protocol for data transforms (augmentation, preprocessing).

    Wraps external transforms (TorchIO, MONAI) behind a consistent interface.
    """

    def __call__(self, data: Any) -> Any:
        """
        Apply the transform.

        Args:
            data: Input data (tensor, dict, or TorchIO Subject)

        Returns:
            Transformed data
        """
        ...

    @property
    def is_invertible(self) -> bool:
        """Whether this transform can be inverted."""
        ...


@runtime_checkable
class ILoss(Protocol):
    """
    Protocol for loss functions.

    All loss functions must return a dict with at least 'loss' key.
    """

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, **context
    ) -> dict[str, torch.Tensor]:
        """
        Compute the loss.

        Args:
            pred: Predicted tensor
            target: Target tensor
            **context: Additional context (e.g., mask, weight)

        Returns:
            Dict with 'loss' key and optional auxiliary losses
        """
        ...
