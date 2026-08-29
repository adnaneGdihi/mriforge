"""Memory optimization service implementation."""

import gc
import logging
import os
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any

import psutil
import torch
import torch.nn as nn

from mriforge.domain.interfaces.service_interfaces import IMemoryOptimizationService

# NOTE: ``PerceptualLoss`` (→ ``torchvision``) is imported lazily inside
# ``PerceptualLossSafe.__init__`` rather than at module scope. Otherwise
# merely importing this service (which ``bootstrap`` does) would drag in
# torchvision before any loss config is even read.

logger = logging.getLogger(__name__)


def handle_cuda_oom(
    max_retries: int = 3,
    fallback_batch_size: int | None = None,
    memory_service: IMemoryOptimizationService | None = None,
):
    """Decorator to handle CUDA out of memory errors with automatic retry.

    Args:
        max_retries: Maximum number of retries before giving up
        fallback_batch_size: Optional batch size to try if OOM occurs

    ### Recovery Flow
    ```mermaid
    graph TD
        A[Execute Function] --> B{Caught OOM?}
        B -- No --> C[Success]
        B -- Yes --> D{Retries < Max?}
        D -- No --> E[Raise Exception]
        D -- Yes --> F[Force GC/Cache Clear]
        F --> G{Fallback Batch Size?}
        G -- Yes --> H[Halve Batch Size]
        G -- No --> I[Retry with Same Batch]
        H --> A
        I --> A
    ```
    """

    def decorator(func: Callable) -> Callable:
        """decorator.

        Args:
            func (Callable): Description.
        Returns:
            Callable: Description.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            """wrapper.

            Returns:
                Any: Description.
            """
            # Use injected service or resolve from DI container
            if memory_service is None:
                from mriforge.infrastructure.di.di_container import resolve_service

                svc = resolve_service(IMemoryOptimizationService)
            else:
                svc = memory_service
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    with svc.memory_efficient_context():
                        return func(*args, **kwargs)
                except RuntimeError as e:
                    last_exception = e
                    if "out of memory" in str(e).lower():
                        msg = f"CUDA OOM on attempt {attempt + 1}/{max_retries + 1}: {e}"
                        logger.warning(msg)

                        if attempt < max_retries:
                            # Clean up memory aggressively
                            svc.force_cleanup()

                            # Try reducing batch size if specified
                            if fallback_batch_size and "batch_size" in kwargs:
                                original_batch_size = kwargs["batch_size"]
                                new_batch = max(1, original_batch_size // 2)
                                kwargs["batch_size"] = new_batch
                                msg = (
                                    f"Reducing batch size from "
                                    f"{original_batch_size} to "
                                    f"{kwargs['batch_size']}"
                                )
                                logger.info(msg)

                            continue
                        logger.error(
                            f"Failed after {max_retries + 1} attempts: {e}",
                        )
                        raise e
                    # Not a CUDA OOM error, re-raise immediately
                    raise e

            # This should never be reached, but just in case
            raise last_exception

        return wrapper

    return decorator


class PerceptualLossSafe:
    """Safe wrapper for PerceptualLoss that handles CUDA OOM gracefully."""

    def __init__(
        self,
        *args,
        memory_service: IMemoryOptimizationService | None = None,
        **kwargs,
    ):
        """__init__.

        Args:
            memory_service (Optional[IMemoryOptimizationService]): Description.
        """
        if memory_service is None:
            from mriforge.infrastructure.di.di_container import resolve_service

            self.memory_service = resolve_service(IMemoryOptimizationService)
        else:
            self.memory_service = memory_service
        self.loss_fn = None
        self.initialized = False

        # SSOT PerceptualLoss is always available now. Imported here
        # (not at module scope) so torchvision is only pulled when a
        # perceptual loss is actually constructed.
        from mriforge.models.losses.perceptual_loss import PerceptualLoss

        try:
            self.loss_fn = PerceptualLoss(*args, **kwargs)
            self.initialized = True
            logger.info("PerceptualLoss initialized successfully")
        except Exception as e:
            logger.warning(f"PerceptualLoss initialization failed: {e}")
            self.initialized = False

    def __call__(self, *args, **kwargs):
        """__call__.

        Returns:
            Any: Description.
        """
        if not self.initialized:
            # Return zero loss if perceptual loss is not available
            if len(args) >= 2:
                # Try to infer device from first argument
                device = args[0].device if hasattr(args[0], "device") else torch.device("cpu")
                return torch.tensor(0.0, device=device, requires_grad=True)
            return torch.tensor(0.0, requires_grad=True)

        try:
            with self.memory_service.memory_efficient_context():
                return self.loss_fn(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(
                    "PerceptualLoss forward pass failed due to CUDA OOM, returning zero loss",
                )
                # Return zero loss to allow training to continue
                if len(args) >= 2:
                    device = args[0].device if hasattr(args[0], "device") else torch.device("cpu")
                    return torch.tensor(0.0, device=device, requires_grad=True)
                return torch.tensor(0.0, requires_grad=True)
            raise e


class NoOpMemoryOptimizationService(IMemoryOptimizationService):
    """No-operation memory optimization service that does nothing.

    This service is used when memory optimization is disabled.
    """

    def optimize_memory_usage(self) -> None:
        """No-op implementation."""
        pass

    def clear_cache(self) -> None:
        """No-op implementation."""
        pass

    def clear_cache_and_gc(self) -> None:
        """No-op implementation."""
        pass

    def get_memory_stats(self) -> dict[str, float]:
        """Return empty memory stats."""
        return {}

    def setup_memory_optimization(self) -> None:
        """No-op implementation."""
        pass

    def optimize_batch_size(
        self,
        models: list[nn.Module],
        device: torch.device,
        current_batch_size: int,
        safety_margin: float = 0.8,
    ) -> int:
        """Return current batch size (no optimization)."""
        return current_batch_size

    def monitor_memory_usage(self) -> dict[str, Any]:
        """Return empty monitoring stats."""
        return {}

    def cleanup_memory(self) -> None:
        """No-op implementation."""
        pass


class MemoryOptimizationService(IMemoryOptimizationService):
    """Memory optimization service for managing GPU/CPU memory usage.

    Provides functionality to:
    - Monitor memory usage
    - Optimize batch sizes
    - Handle out-of-memory situations
    - Clean up memory resources
    """

    def __init__(self):
        """__init__."""
        self.memory_stats = {}
        self.optimization_settings = {
            "gradient_checkpointing": False,
            "mixed_precision": False,
            "batch_size_optimization": True,
            "memory_cleanup_interval": 100,
        }

    def optimize_memory_usage(self) -> None:
        """Optimize memory usage."""
        # NOTE: torch.cuda.empty_cache() removed per performance guidelines
        # PyTorch's caching allocator manages memory efficiently
        # Manual cache clearing causes GPU stalls and memory fragmentation

        # Setup garbage collection
        gc.set_threshold(700, 10, 10)  # More aggressive garbage collection

    def clear_cache(self) -> None:
        """Clear memory cache."""
        # NOTE: torch.cuda.empty_cache() removed per performance guidelines
        # Use gradient checkpointing or reduce batch size for OOM instead

        gc.collect()

    def clear_cache_and_gc(self) -> None:
        """Clear memory cache and run garbage collection."""
        self.clear_cache()

    def get_memory_stats(self) -> dict[str, float]:
        """Get memory statistics."""
        stats = {}

        if torch.cuda.is_available():
            stats["cuda_allocated"] = torch.cuda.memory_allocated() / 1024**2  # MB
            stats["cuda_reserved"] = torch.cuda.memory_reserved() / 1024**2  # MB
        else:
            stats["cuda_available"] = False

        return stats

    # Additional methods for enhanced functionality
    def setup_memory_optimization(self) -> None:
        """Setup memory optimization settings."""
        # NOTE: torch.cuda.empty_cache() removed per performance guidelines
        # PyTorch manages CUDA memory automatically and efficiently
        if torch.cuda.is_available():
            # Enable CUDA memory management

            # Set memory fraction if needed
            if hasattr(torch.cuda, "set_per_process_memory_fraction"):
                try:
                    torch.cuda.set_per_process_memory_fraction(0.9)
                except (RuntimeError, AttributeError) as _exc:
                    logger.debug(
                        "Suppressed exception: %s", _exc
                    )  # Silently ignore if not available

            # Set environment variables for memory management (from MemoryManager)
            #
            # TRITON_CACHE_DIR is deliberately ABSENT here. This block used to
            # makedirs("/tmp/triton_cache") and point Triton at it. `/tmp` is
            # sticky and shared, so a bare `/tmp/<name>` with no `$USER` term
            # belongs to whoever created it first -- on a shared cluster node
            # every later user gets EACCES out of a directory they cannot list.
            # `env_resolver` names exactly this hazard on its `_EXAMPLE_ROOT`.
            # It also made this service a SECOND writer to a variable
            # `configure_cache_environment()` already owns; both used
            # setdefault semantics, so the winner was whichever ran first, and
            # this one runs only once a training run reaches the service.
            # `main.py` calls that resolver at import, above `import torch`, on
            # every entry path. Do not re-add a writer here -- steer the root
            # with `MRIFORGE_CACHE_ROOT` (or `TMPDIR`) in `.env`.
            env_vars = {
                "PYTORCH_CUDA_ALLOC_CONF": ("expandable_segments:True,max_split_size_mb:512"),
                # [FIX] Enable caching (0 disables, non-zero enables with size limit)
                "TRITON_CACHE_SIZE": "10737418240",  # 10 GB
                "TORCHINDUCTOR_CACHE_SIZE": "10737418240",  # 10 GB
                "TORCH_DYNAMO_VERBOSE": "0",
                "CUDA_LAUNCH_BLOCKING": "0",
            }

            for var, value in env_vars.items():
                if var not in os.environ:
                    os.environ[var] = value
                    logger.info(f"Set {var}={value}")

        # Setup garbage collection
        gc.set_threshold(700, 10, 10)  # More aggressive garbage collection

    @contextmanager
    def memory_efficient_context(self):
        """Context manager for memory-efficient operations."""
        try:
            yield
        finally:
            # Do not force cleanup on exit, let allocator manage memory
            pass

    def optimize_batch_size(
        self,
        models: list[nn.Module],
        device: torch.device,
        current_batch_size: int,
        safety_margin: float = 0.8,
    ) -> int:
        """Optimize batch size based on available memory and model requirements.

        Args:
            models: List of PyTorch models to test
            device: Target device for memory estimation
            current_batch_size: Current batch size to start from
            safety_margin: Fraction of available memory to use (default 0.8)

        Returns:
            Optimal batch size

        """
        if not torch.cuda.is_available() or device.type != "cuda":
            return min(current_batch_size, 4)  # Conservative for CPU

        # Get total model memory requirements
        total_model_memory_gb = sum(self._estimate_model_memory(model) for model in models)

        # Get available memory
        device_props = torch.cuda.get_device_properties(device)
        total_memory_gb = device_props.total_memory / (1024**3)
        available_memory_gb = total_memory_gb * safety_margin

        # Reserve memory for model parameters and gradients
        usable_memory_gb = available_memory_gb - total_model_memory_gb

        if usable_memory_gb <= 0:
            # Not enough memory even for the model, return minimum batch size
            return 1

        # Estimate memory per sample (rough heuristic based on input size)
        # For k-space data (complex 2-channel), estimate ~8MB per sample at batch size 1
        # This is a rough estimate and may need tuning
        memory_per_sample_mb = 8.0  # MB per sample estimate

        # Calculate maximum batch size that fits in available memory
        max_batch_size = int((usable_memory_gb * 1024) / memory_per_sample_mb)

        # Clamp to reasonable range
        optimal_batch_size = max(1, min(max_batch_size, current_batch_size * 2))

        return optimal_batch_size

    def _estimate_model_memory(self, model: nn.Module) -> float:
        """Estimate memory usage of model in GB."""
        total_params = sum(p.numel() for p in model.parameters())

        # Rough estimate: 4 bytes per parameter (float32) + overhead
        memory_gb = (total_params * 4) / (1024**3)
        memory_gb *= 2  # Double for gradients and optimizer states

        return memory_gb

    def monitor_memory_usage(self) -> dict[str, Any]:
        """Monitor current memory usage."""
        stats = {}

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            try:
                stats["cuda_allocated"] = torch.cuda.memory_allocated() / (1024**3)
                stats["cuda_reserved"] = torch.cuda.memory_reserved() / (1024**3)
                total_mem = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated()
                stats["cuda_free"] = (total_mem - allocated) / (1024**3)
            except (AssertionError, RuntimeError):
                # CUDA is available but device access failed
                stats["cuda_available"] = False
        else:
            stats["cuda_available"] = False

        # CPU memory (rough estimate)
        try:
            process = psutil.Process()
            stats["cpu_memory"] = process.memory_info().rss / (1024**3)
        except (ImportError, AttributeError):
            stats["cpu_memory"] = 0.0

        self.memory_stats.update(stats)
        return stats

    def cleanup_memory(self) -> None:
        """Clean up memory resources (Passive).

        This method is called frequently and should NOT perform expensive operations
        like gc.collect() or torch.cuda.empty_cache() unless absolutely necessary.
        """
        # Passive cleanup: let the allocator handle it.
        pass

    def force_cleanup(self) -> None:
        """Force aggressive memory cleanup.

        Use this ONLY when recovering from OOM errors.
        NOTE: torch.cuda.empty_cache() removed - use gradient checkpointing,
        AMP, or reduce batch size instead for better performance.
        """
        # PyTorch's allocator handles memory reclamation automatically
        gc.collect()
