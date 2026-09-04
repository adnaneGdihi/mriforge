"""3D Generation Pipeline Execution Service

Infrastructure service for executing 3D generation pipelines with
device management, memory optimization, and performance monitoring.
"""

import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from spectramr.infrastructure.services.device_policy import (
    DevicePolicy,
    get_default_device_policy,
)


class ExecutionStatus(Enum):
    """Status of pipeline execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionContext:
    """Context for pipeline execution."""

    execution_id: str
    device: torch.device
    memory_limit: int | None = None
    timeout_seconds: int | None = None
    enable_profiling: bool = False
    checkpoint_interval: int | None = None


@dataclass
class ExecutionMetrics:
    """Metrics collected during pipeline execution."""

    execution_time: float
    peak_memory_usage: int
    gpu_utilization: float | None = None
    throughput: float | None = None
    error_count: int = 0
    warnings: list[str] | None = None

    def __post_init__(self):
        """__post_init__.

        Returns:
            Any: Description.
        """
        if self.warnings is None:
            self.warnings = []


@dataclass
class ExecutionResult:
    """Result of pipeline execution."""

    execution_id: str
    status: ExecutionStatus
    output_data: torch.Tensor | None = None
    metrics: ExecutionMetrics | None = None
    error_message: str | None = None
    checkpoints: list[dict[str, Any]] | None = None


class IPipelineExecutionService(ABC):
    """Interface for pipeline execution services.

    Defines the contract for services that execute 3D generation
    pipelines with infrastructure concerns.
    """

    @abstractmethod
    def execute_pipeline(
        self,
        pipeline: Any,
        input_data: torch.Tensor,
        context: ExecutionContext,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a 3D generation pipeline."""

    @abstractmethod
    def get_available_devices(self) -> list[torch.device]:
        """Get list of available compute devices."""

    @abstractmethod
    def optimize_memory_usage(
        self,
        model: Any,
        target_device: torch.device,
    ) -> dict[str, Any]:
        """Optimize memory usage for model execution."""

    @abstractmethod
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running pipeline execution."""


class PipelineExecutionService(IPipelineExecutionService):
    """Infrastructure service for executing 3D generation pipelines.

    Handles device management, memory optimization, performance monitoring,
    and fault-tolerant execution of 3D generation pipelines.
    """

    def __init__(
        self,
        *,
        device_policy: DevicePolicy | None = None,
        memory_optimizer: Any | None = None,
        profiling_service: Any | None = None,
    ):
        """Initialize the pipeline execution service."""
        self.logger = logging.getLogger(__name__)
        self._active_executions: dict[str, dict[str, Any]] = {}
        self._device_policy = device_policy or get_default_device_policy()
        self._memory_optimizer = (
            memory_optimizer
            if memory_optimizer is not None
            else self._initialize_memory_optimizer()
        )
        self._profiling_service = profiling_service

    def _initialize_memory_optimizer(self) -> Any:
        """Initialize memory optimization component."""
        try:
            # Correctly import the concrete implementation
            from spectramr.infrastructure.services.memory_optimization_service import (
                NoOpMemoryOptimizationService,
            )

            return NoOpMemoryOptimizationService()
        except ImportError:
            self.logger.warning(
                "NoOpMemoryOptimizationService not available, using basic optimization",
            )
            return BasicMemoryOptimizer()

    def execute_pipeline(
        self,
        pipeline: Any,
        input_data: torch.Tensor,
        context: ExecutionContext,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a 3D generation pipeline with infrastructure management.

        Args:
            pipeline: The pipeline to execute
            input_data: Input data for the pipeline
            context: Execution context with configuration
            **kwargs: Additional pipeline parameters

        Returns:
            Execution result with output data and metrics

        """
        execution_id = context.execution_id
        self._active_executions[execution_id] = {
            "status": ExecutionStatus.RUNNING,
            "start_time": time.time(),
            "context": context,
        }

        try:
            # Setup execution environment
            with self._setup_execution_environment(context):
                # Move input data to target device
                input_data = self._move_to_device(input_data, context.device)

                # Optimize memory if needed
                if context.memory_limit:
                    self._memory_optimizer.optimize_for_limit(
                        context.memory_limit,
                        context.device,
                    )

                # Execute pipeline with monitoring
                start_time = time.time()
                metrics = ExecutionMetrics(
                    execution_time=0,
                    peak_memory_usage=0,
                )

                with self._monitor_execution(metrics, context):
                    output_data = pipeline.generate(input_data, **kwargs)

                # Calculate final metrics
                metrics.execution_time = time.time() - start_time
                metrics.throughput = self._calculate_throughput(
                    input_data,
                    output_data,
                    metrics.execution_time,
                )

                # Mark as completed
                self._active_executions[execution_id]["status"] = ExecutionStatus.COMPLETED

                return ExecutionResult(
                    execution_id=execution_id,
                    status=ExecutionStatus.COMPLETED,
                    output_data=output_data,
                    metrics=metrics,
                )

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            self._active_executions[execution_id]["status"] = ExecutionStatus.FAILED

            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                error_message=str(e),
            )

        finally:
            # Cleanup
            if execution_id in self._active_executions:
                del self._active_executions[execution_id]

    @contextmanager
    def _setup_execution_environment(self, context: ExecutionContext):
        """Setup execution environment (device, memory, etc.)."""
        original_device = torch.cuda.current_device() if torch.cuda.is_available() else None

        try:
            # Set device
            if context.device.type == "cuda":
                torch.cuda.set_device(context.device)

            # Set memory limits if specified
            if context.memory_limit and torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(
                    context.memory_limit
                    / torch.cuda.get_device_properties(context.device).total_memory,
                )

            yield

        finally:
            # Restore original device
            if original_device is not None and torch.cuda.is_available():
                torch.cuda.set_device(original_device)

    def _move_to_device(
        self,
        data: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Move tensor to specified device using the configured device policy."""
        if data is None:
            return data

        if self._device_policy is not None:
            try:
                moved = self._device_policy.move_to_device(data)
            except Exception:  # pragma: no cover - defensive fallback
                moved = data.to(device)
            else:
                if isinstance(moved, torch.Tensor) and moved.device != device:
                    return moved.to(device)
                return moved

        return data.to(device)

    @contextmanager
    def _monitor_execution(
        self,
        metrics: ExecutionMetrics,
        context: ExecutionContext,
    ):
        """Monitor execution and collect metrics."""
        if not context.enable_profiling:
            yield
            return

        # Basic memory monitoring
        if torch.cuda.is_available() and context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)

        yield

        # Collect memory metrics
        if torch.cuda.is_available() and context.device.type == "cuda":
            metrics.peak_memory_usage = torch.cuda.max_memory_allocated(
                context.device,
            )

    def _calculate_throughput(
        self,
        input_data: torch.Tensor,
        output_data: torch.Tensor,
        execution_time: float,
    ) -> float:
        """Calculate throughput (samples per second)."""
        batch_size = input_data.shape[0]
        return batch_size / execution_time if execution_time > 0 else 0.0

    def get_available_devices(self) -> list[torch.device]:
        """Get list of available compute devices."""
        devices = [torch.device("cpu")]

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(torch.device(f"cuda:{i}"))

        return devices

    def optimize_memory_usage(
        self,
        model: Any,
        target_device: torch.device,
    ) -> dict[str, Any]:
        """Optimize memory usage for model execution."""
        return self._memory_optimizer.optimize_model(model, target_device)

    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running pipeline execution."""
        if execution_id in self._active_executions:
            self._active_executions[execution_id]["status"] = ExecutionStatus.CANCELLED
            self.logger.info(f"Cancelled execution: {execution_id}")
            return True

        return False

    def get_execution_status(
        self,
        execution_id: str,
    ) -> ExecutionStatus | None:
        """Get status of a pipeline execution."""
        execution = self._active_executions.get(execution_id)
        return execution["status"] if execution else None

    def get_active_executions(self) -> list[str]:
        """Get list of active execution IDs."""
        return list(self._active_executions.keys())


class BasicDeviceManager:
    """Basic device management fallback."""

    def get_available_devices(self) -> list[torch.device]:
        """Get basic list of devices."""
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda:0"))
        return devices


class BasicMemoryOptimizer:
    """Basic memory optimization fallback."""

    def optimize_for_limit(self, memory_limit: int, device: torch.device):
        """Basic memory limit setting."""
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(
                memory_limit / torch.cuda.get_device_properties(device).total_memory,
            )

    def optimize_model(
        self,
        model: Any,
        device: torch.device,
    ) -> dict[str, Any]:
        """Basic model optimization."""
        return {"optimization": "basic", "device": str(device)}
