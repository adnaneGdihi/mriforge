"""
Advanced Profiling Module
========================

Comprehensive performance profiling for spectraMR research including:
- PyTorch profiler integration with NVTX markers
- FLOP counting and parameter analysis
- Memory usage tracking and optimization
- Gradient checkpointing utilities
- Sliding window inference for large volumes
- Performance bottleneck identification

Author: spectraMR Research Project
Date: August 2025
"""

import logging

logger = logging.getLogger(__name__)

import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

try:
    import GPUtil
except ImportError:
    GPUtil = None
import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.checkpoint import checkpoint

# Initialize issue tracker
# init_issue_tracker(log_dir="logs", csv_filename="profiling_issues.csv")


@dataclass
class ProfilingConfig:
    """Configuration for profiling operations."""

    enable_profiler: bool = True
    enable_nvtx: bool = True
    track_memory: bool = True
    track_flops: bool = True
    profile_activities: list[ProfilerActivity] = field(
        default_factory=lambda: [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    )
    schedule_config: dict[str, Any] = field(
        default_factory=lambda: {"wait": 1, "warmup": 1, "active": 3, "repeat": 1}
    )
    log_dir: str = "profiling_results"
    max_memory_samples: int = 1000


from spectramr.core.metrics.performance import PerformanceMetrics


class NVTXProfiler:
    """NVTX (NVIDIA Tools Extension) profiler for GPU tracing."""

    def __init__(self, enabled: bool = True, config: ProfilingConfig | None = None):
        """__init__.

        Args:
            enabled (bool): Description.
            config (Optional[ProfilingConfig]): Description.
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.config = config or ProfilingConfig()

        if self.enabled:
            try:
                import nvtx

                self.nvtx = nvtx
            except ImportError:
                # Only log warning if profiling is explicitly enabled
                if self.config.enable_profiler:
                    logger.warning("NVTX not available, GPU profiling disabled")
                self.enabled = False
                self.nvtx = None
        else:
            self.nvtx = None

    @contextmanager
    def profile_range(self, name: str, color: str = "blue"):
        """Context manager for profiling a range of operations."""
        if self.enabled and self.nvtx:
            with self.nvtx.annotate(name, color=color):
                yield
        else:
            yield

    def mark(self, name: str):
        """Add a point marker."""
        if self.enabled and self.nvtx:
            self.nvtx.mark(name)


class MemoryTracker:
    """Advanced memory tracking and analysis."""

    def __init__(self, config: ProfilingConfig):
        """__init__.

        Args:
            config (ProfilingConfig): Description.
        """
        self.config = config
        self.memory_history: list[dict[str, float]] = []
        self.peak_memory = 0.0

    def get_memory_stats(self) -> dict[str, float]:
        """Get current memory statistics."""
        stats = {}

        # GPU memory
        if torch.cuda.is_available():
            stats.update(
                {
                    "gpu_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                    "gpu_reserved": torch.cuda.memory_reserved() / 1024**3,
                    "gpu_max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
                }
            )

        # CPU memory
        process = psutil.Process()
        memory_info = process.memory_info()
        stats.update(
            {
                "cpu_rss": memory_info.rss / 1024**3,  # GB
                "cpu_vms": memory_info.vms / 1024**3,
            }
        )

        # GPU utilization
        try:
            if GPUtil:
                gpus = GPUtil.getGPUs()
                if gpus:
                    stats["gpu_utilization"] = gpus[0].load * 100
        except BaseException:
            stats["gpu_utilization"] = 0.0

        # CPU utilization
        stats["cpu_utilization"] = psutil.cpu_percent(interval=None)

        return stats

    def record_memory_sample(self):
        """Record current memory sample."""
        if len(self.memory_history) >= self.config.max_memory_samples:
            self.memory_history.pop(0)  # Remove oldest sample

        stats = self.get_memory_stats()
        self.memory_history.append(stats)

        # Update peak memory
        current_allocated = stats.get("gpu_allocated", 0)
        self.peak_memory = max(self.peak_memory, current_allocated)

    def get_memory_report(self) -> dict[str, Any]:
        """Generate memory usage report."""
        if not self.memory_history:
            return {"error": "No memory samples recorded"}

        # Calculate statistics
        gpu_allocated = [s.get("gpu_allocated", 0) for s in self.memory_history]
        gpu_reserved = [s.get("gpu_reserved", 0) for s in self.memory_history]
        cpu_rss = [s.get("cpu_rss", 0) for s in self.memory_history]

        return {
            "peak_gpu_allocated": max(gpu_allocated) if gpu_allocated else 0,
            "avg_gpu_allocated": np.mean(gpu_allocated) if gpu_allocated else 0,
            "peak_gpu_reserved": max(gpu_reserved) if gpu_reserved else 0,
            "avg_gpu_reserved": np.mean(gpu_reserved) if gpu_reserved else 0,
            "peak_cpu_rss": max(cpu_rss) if cpu_rss else 0,
            "avg_cpu_rss": np.mean(cpu_rss) if cpu_rss else 0,
            "total_samples": len(self.memory_history),
            "overall_peak": self.peak_memory,
        }


class FLOPProfiler:
    """FLOP counting and analysis for models."""

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count total parameters in model."""
        return sum(p.numel() for p in model.parameters())

    @staticmethod
    def count_trainable_parameters(model: nn.Module) -> int:
        """Count trainable parameters in model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def estimate_flops(model: nn.Module, input_shape: tuple[int, ...]) -> dict[str, Any]:
        """
        Estimate FLOPs for a model (simplified estimation).

        Note: This is a rough estimation. For accurate FLOP counting,
        consider using libraries like ptflops or fvcore.
        """
        try:
            # Create dummy input
            dummy_input = torch.randn(1, *input_shape)

            # Count operations (simplified) # IMPL
            total_flops = 0
            layer_info = []

            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    # Conv2D FLOPs: 2 * H * W * C_out * (C_in * K * K)
                    h, w = dummy_input.shape[-2:]
                    flops = (
                        2
                        * h
                        * w
                        * module.out_channels
                        * (module.in_channels * module.kernel_size[0] * module.kernel_size[1])
                    )
                    total_flops += flops
                    layer_info.append(
                        {
                            "name": name,
                            "type": "Conv2D",
                            "flops": flops,
                            "params": module.weight.numel()
                            + (module.bias.numel() if module.bias is not None else 0),
                        }
                    )

                elif isinstance(module, nn.Linear):
                    # Linear FLOPs: 2 * in_features * out_features
                    flops = 2 * module.in_features * module.out_features
                    total_flops += flops
                    layer_info.append(
                        {
                            "name": name,
                            "type": "Linear",
                            "flops": flops,
                            "params": module.weight.numel()
                            + (module.bias.numel() if module.bias is not None else 0),
                        }
                    )

            return {
                "total_flops": total_flops,
                "flops_per_sample": total_flops,
                "layer_info": layer_info,
                "input_shape": input_shape,
            }

        except Exception as e:
            logger.error(f"Failed to estimate FLOPs: {e}")
            return {"error": str(e)}


def _make_checkpointed_forward(orig_forward: Callable) -> Callable:
    """Build a checkpointed ``forward`` that wraps ``orig_forward``.

    Defined at module scope (not as an inline lambda inside the patch loop) so
    each layer's own ``forward`` is bound once, by value, at factory-call time
    — sidestepping Python's closure late-binding over the loop variable.

    Uses the non-reentrant checkpoint variant: with reentrant checkpointing a
    layer whose input tensor does not itself require grad (e.g. a stem conv fed
    straight from the dataloader) silently produces ``None`` gradients, so the
    layer never trains. ``use_reentrant=False`` keeps gradients flowing.
    """

    def checkpointed_forward(x: "torch.Tensor") -> "torch.Tensor":
        def custom_forward(*inputs: "torch.Tensor") -> "torch.Tensor":
            return orig_forward(*inputs)

        return checkpoint(custom_forward, x, use_reentrant=False)

    return checkpointed_forward


class GradientCheckpointing:
    """Utilities for gradient checkpointing to save memory."""

    @staticmethod
    def apply_checkpointing(model: nn.Module, checkpoint_ratio: float = 0.5):
        """
        Apply gradient checkpointing to model layers.

        Args:
            model: PyTorch model
            checkpoint_ratio: Ratio of layers to checkpoint (0.0 to 1.0)
        """
        if not 0 <= checkpoint_ratio <= 1:
            raise ValueError("checkpoint_ratio must be between 0 and 1")

        # Get all modules that support checkpointing
        checkpointable_modules = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                checkpointable_modules.append((name, module))

        # Apply checkpointing to a subset
        n_to_checkpoint = int(len(checkpointable_modules) * checkpoint_ratio)
        modules_to_checkpoint = checkpointable_modules[:n_to_checkpoint]

        logger.info(f"Applying gradient checkpointing to {len(modules_to_checkpoint)} layers")

        for name, module in modules_to_checkpoint:
            # Replace forward with a checkpointed wrapper. The factory binds
            # ``module.forward`` per layer; assigning the lambda inline inside
            # the loop instead would capture the loop variable by reference
            # (Python closure late-binding) and collapse EVERY patched layer's
            # forward onto the last module's — a network whose stem is
            # ``Conv2d(1, C)`` then crashes with "expected input to have C
            # channels, but got 1" the instant the stem runs the head's conv.
            module.forward = _make_checkpointed_forward(module.forward)

    @staticmethod
    def create_checkpointed_function(func: Callable, *args, **kwargs) -> Callable:
        """Create a checkpointed version of a function."""

        @wraps(func)
        def checkpointed_func(*inputs):
            """checkpointed_func.

            Returns:
                Any: Description.
            """
            return checkpoint(func, *inputs, **kwargs)

        return checkpointed_func


class SlidingWindowInference:
    """Sliding window inference for large volume processing."""

    def __init__(
        self,
        window_size: tuple[int, ...],
        stride: tuple[int, ...] | None = None,
        overlap: float = 0.5,
    ):
        """
        Initialize sliding window inference.

        Args:
            window_size: Size of sliding window
            stride: Stride for sliding window (if None, computed from overlap)
            overlap: Overlap ratio between windows (0.0 to 1.0)
        """
        self.window_size = window_size
        if stride is None:
            self.stride = tuple(int(s * (1 - overlap)) for s in window_size)
        else:
            self.stride = stride
        self.overlap = overlap

    def __call__(self, model: nn.Module, volume: torch.Tensor) -> torch.Tensor:
        """
        Apply sliding window inference to a volume.

        Args:
            model: Neural network model
            volume: Input volume tensor

        Returns:
            Processed volume with sliding window inference
        """
        return self.infer_sliding_window(model, volume)

    def infer_sliding_window(self, model: nn.Module, volume: torch.Tensor) -> torch.Tensor:
        """Perform sliding window inference."""
        if volume.dim() != 4:  # Expected: [B, C, H, W]
            raise ValueError("Volume must be 4D tensor [B, C, H, W]")

        batch_size, channels, height, width = volume.shape
        window_h, window_w = self.window_size[-2:]
        stride_h, stride_w = self.stride[-2:]

        # Calculate output dimensions
        out_h = (height - window_h) // stride_h + 1
        out_w = (width - window_w) // stride_w + 1

        # Initialize output volume
        output_volume = torch.zeros_like(volume)

        # Weight matrix for averaging overlapping regions
        weight_matrix = torch.zeros_like(volume)

        # Slide window over volume
        for i in range(out_h):
            for j in range(out_w):
                # Calculate window coordinates
                h_start = i * stride_h
                h_end = h_start + window_h
                w_start = j * stride_w
                w_end = w_start + window_w

                # Extract window
                window = volume[:, :, h_start:h_end, w_start:w_end]

                # Process window through model
                with torch.no_grad():
                    window_output = model(window)

                # Store result
                output_volume[:, :, h_start:h_end, w_start:w_end] += window_output

                # Update weight matrix
                weight_matrix[:, :, h_start:h_end, w_start:w_end] += 1

        # Average overlapping regions
        output_volume = output_volume / weight_matrix.clamp(min=1)

        return output_volume

    def get_optimal_window_params(
        self, volume_shape: tuple[int, ...], memory_limit_gb: float = 8.0
    ) -> dict[str, Any]:
        """
        Get optimal window parameters based on memory constraints.

        Args:
            volume_shape: Shape of input volume
            memory_limit_gb: Memory limit in GB

        Returns:
            Dictionary with optimal parameters
        """
        channels, height, width = volume_shape

        # Estimate memory per window (rough approximation)
        # Assume float32, add some overhead
        bytes_per_voxel = 4 * 2  # input + output
        memory_per_window_gb = (
            self.window_size[0] * self.window_size[1] * channels * bytes_per_voxel
        ) / (1024**3)

        # Calculate maximum windows that fit in memory
        max_windows = int(memory_limit_gb / memory_per_window_gb) if memory_per_window_gb > 0 else 0

        # Calculate stride to fit within memory
        total_windows_needed = (height // self.stride[0]) * (width // self.stride[1])

        if total_windows_needed > max_windows and max_windows > 0:
            # Need to increase stride
            scale_factor = (total_windows_needed / max_windows) ** 0.5
            new_stride_h = int(self.stride[0] * scale_factor)
            new_stride_w = int(self.stride[1] * scale_factor)

            return {
                "recommended_stride": (new_stride_h, new_stride_w),
                "memory_efficient": True,
                "estimated_memory": memory_per_window_gb * max_windows,
            }

        return {
            "current_stride": self.stride,
            "memory_efficient": True,
            "estimated_memory": memory_per_window_gb * total_windows_needed,
        }


class PerformanceProfiler:
    """Main performance profiling class."""

    def __init__(self, config: ProfilingConfig):
        """__init__.

        Args:
            config (ProfilingConfig): Description.
        """
        self.config = config
        self.nvtx_profiler = NVTXProfiler(config.enable_nvtx)
        self.memory_tracker = MemoryTracker(config) if config.track_memory else None
        self.profiler = None
        self.metrics = PerformanceMetrics()

    def start_profiling(self, model: nn.Module | None = None):
        """Start performance profiling."""
        if not self.config.enable_profiler:
            return

        logger.info("Starting performance profiling...")

        # Start PyTorch profiler
        self.profiler = profile(
            activities=self.config.profile_activities,
            schedule=torch.profiler.schedule(**self.config.schedule_config),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.config.log_dir),
            record_shapes=True,
            profile_memory=self.config.track_memory,
            with_flops=self.config.track_flops,
        )

        self.profiler.start()

        # Record initial memory state
        if self.memory_tracker:
            self.memory_tracker.record_memory_sample()

    def stop_profiling(self) -> dict[str, Any]:
        """Stop profiling and return results."""
        if not self.profiler:
            return {"error": "Profiler not started"}

        self.profiler.stop()

        # Get final memory state
        if self.memory_tracker:
            self.memory_tracker.record_memory_sample()

        # Generate report
        report = self._generate_report()
        return report

    def profile_function(self, func: Callable, *args, **kwargs) -> tuple[Any, dict[str, Any]]:
        """Profile a function execution."""
        start_time = time.time()

        # Get function name, handling both functions and callable objects
        func_name = getattr(func, "__name__", None)
        if func_name is None:
            func_name = f"{func.__class__.__name__}"

        with self.nvtx_profiler.profile_range(f"Profile: {func_name}"):
            if self.profiler:
                with record_function(func_name):
                    result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

        end_time = time.time()

        # Record memory
        if self.memory_tracker:
            self.memory_tracker.record_memory_sample()

        execution_time = end_time - start_time

        metrics = {
            "execution_time": execution_time,
            "function_name": func_name,
        }

        if self.memory_tracker:
            metrics.update(self.memory_tracker.get_memory_report())

        return result, metrics

    def profile_model(self, model: nn.Module, input_tensor: torch.Tensor) -> dict[str, Any]:
        """Profile model performance."""
        logger.info(f"Profiling model: {model.__class__.__name__}")

        # Count parameters
        total_params = FLOPProfiler.count_parameters(model)
        trainable_params = FLOPProfiler.count_trainable_parameters(model)

        # Estimate FLOPs
        flops_info = {}
        if self.config.track_flops:
            flops_info = FLOPProfiler.estimate_flops(model, input_tensor.shape[1:])

        # Profile forward pass
        model.eval()
        with torch.no_grad():
            _, forward_metrics = self.profile_function(model, input_tensor)

        # Profile forward + backward if training
        model.train()
        input_tensor.requires_grad_(True)
        output = model(input_tensor)
        loss = output.mean()

        def backward_fn():
            """backward_fn.

            Returns:
                Any: Description.
            """
            loss.backward()

        _, backward_metrics = self.profile_function(backward_fn)

        # Combine metrics
        model_metrics = {
            "model_name": model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_shape": tuple(input_tensor.shape),
            "output_shape": tuple(output.shape),
            "forward_time": forward_metrics.get("execution_time", 0),
            "backward_time": backward_metrics.get("execution_time", 0),
            "total_time": (
                forward_metrics.get("execution_time", 0) + backward_metrics.get("execution_time", 0)
            ),
        }

        model_metrics.update(flops_info)

        # Update global metrics
        self.metrics.forward_time = model_metrics["forward_time"]
        self.metrics.backward_time = model_metrics["backward_time"]
        self.metrics.total_time = model_metrics["total_time"]
        self.metrics.parameters = total_params
        self.metrics.flops = flops_info.get("total_flops", 0)

        if self.memory_tracker:
            memory_report = self.memory_tracker.get_memory_report()
            self.metrics.memory_peak = memory_report.get("peak_gpu_allocated", 0)
            self.metrics.memory_allocated = memory_report.get("avg_gpu_allocated", 0)
            model_metrics.update(memory_report)

        return model_metrics

    def _generate_report(self) -> dict[str, Any]:
        """Generate comprehensive profiling report."""
        report = {
            "profiling_config": {
                "enable_profiler": self.config.enable_profiler,
                "enable_nvtx": self.config.enable_nvtx,
                "track_memory": self.config.track_memory,
                "track_flops": self.config.track_flops,
            },
            "performance_metrics": {
                "forward_time": self.metrics.forward_time,
                "backward_time": self.metrics.backward_time,
                "total_time": self.metrics.total_time,
                "memory_peak": self.metrics.memory_peak,
                "memory_allocated": self.metrics.memory_allocated,
                "flops": self.metrics.flops,
                "parameters": self.metrics.parameters,
            },
        }

        if self.memory_tracker:
            report["memory_analysis"] = self.memory_tracker.get_memory_report()

        return report

    def get_bottleneck_analysis(self) -> dict[str, Any]:
        """Analyze performance bottlenecks."""
        analysis = {
            "memory_bound": False,
            "compute_bound": False,
            "recommendations": [],
        }

        if self.memory_tracker:
            memory_report = self.memory_tracker.get_memory_report()
            peak_memory = memory_report.get("peak_gpu_allocated", 0)

            # Check if memory usage is close to GPU limit
            if torch.cuda.is_available():
                gpu_props = torch.cuda.get_device_properties(0)
                total_memory = gpu_props.total_memory / 1024**3
                memory_utilization = peak_memory / total_memory

                if memory_utilization > 0.9:
                    analysis["memory_bound"] = True
                    analysis["recommendations"].append(
                        "High memory utilization detected. Consider gradient checkpointing."
                    )
                    analysis["recommendations"].append(
                        "Try reducing batch size or using mixed precision training."
                    )

        # Time analysis
        if self.metrics.backward_time > self.metrics.forward_time * 2:
            analysis["compute_bound"] = True
            analysis["recommendations"].append(
                "Backward pass is significantly slower. Consider gradient accumulation."
            )

        return analysis


# Convenience functions
def profile_model_performance(
    model: nn.Module,
    input_shape: tuple[int, ...],
    config: ProfilingConfig | None = None,
) -> dict[str, Any]:
    """Convenience function to profile model performance."""
    if config is None:
        config = ProfilingConfig()

    profiler = PerformanceProfiler(config)

    # Create dummy input
    dummy_input = torch.randn(1, *input_shape)

    # Profile model
    profiler.start_profiling(model)
    metrics = profiler.profile_model(model, dummy_input)
    report = profiler.stop_profiling()

    return {**metrics, **report}


def apply_memory_optimizations(
    model: nn.Module, checkpoint_ratio: float = 0.3, enable_cudnn_benchmark: bool = True
):
    """Apply memory optimizations to a model."""
    logger.info("Applying memory optimizations...")

    # Enable cuDNN benchmark for faster convolutions
    if enable_cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN benchmark")

    # Apply gradient checkpointing
    if checkpoint_ratio > 0:
        GradientCheckpointing.apply_checkpointing(model, checkpoint_ratio)
        logger.info(f"Applied gradient checkpointing with ratio {checkpoint_ratio}")

    return model


def create_sliding_window_inference(
    window_size: tuple[int, ...], overlap: float = 0.5
) -> SlidingWindowInference:
    """Create sliding window inference utility."""
    return SlidingWindowInference(window_size=window_size, overlap=overlap)


if __name__ == "__main__":
    # Example usage
    config = ProfilingConfig()
    profiler = PerformanceProfiler(config)

    # Create a simple model for testing
    model = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 3, padding=1))

    # Profile the model
    metrics = profile_model_performance(model, (1, 64, 64), config)
    print("Profiling Results:", metrics)
