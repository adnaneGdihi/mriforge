"""Performance Optimizations for Inference

CUDA graph optimizations and memory-efficient sampling techniques
for improved inference performance.
"""

import logging
import time
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


class CUDAGraphWrapper:
    """Wrapper for CUDA graph optimization of inference steps.

    Implements LRU caching with automatic eviction to prevent unbounded GPU memory growth.
    """

    def __init__(self, device: torch.device, max_cached_graphs: int = 16):
        """Initialize CUDA graph wrapper.

        Args:
            device: CUDA device for graph capture
            max_cached_graphs: Maximum number of graphs to cache (default: 16)
        """
        self.device = device
        self.max_cached = max_cached_graphs
        self.graphs = {}  # {graph_key: graph}
        self.static_inputs_dict = {}  # {graph_key: static_inputs}
        self.static_outputs_dict = {}  # {graph_key: static_outputs}
        self.timestamps = {}  # {graph_key: last_access_time}
        self.memory_usage = {}  # {graph_key: memory_bytes}

    def capture_graph(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        conditioning_input: torch.Tensor | None = None,
        timestep: int = 0,
    ) -> str:
        """Capture a CUDA graph for the inference step with LRU eviction.

        Args:
            model: The model to capture
            sample_input: Sample input tensor for graph capture
            conditioning_input: Optional conditioning input
            timestep: Timestep for capture

        Returns:
            Graph key for later retrieval
        """
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return None  # Skip graph capture on non-CUDA devices

        # Generate graph key
        graph_key = f"batch_{sample_input.shape[0]}_t_{timestep}"

        # Check if already cached
        if graph_key in self.graphs:
            self.timestamps[graph_key] = time.time()
            return graph_key

        # Evict LRU if cache full
        self._evict_lru_if_needed()

        # Set model to eval mode
        model.eval()

        # Create static inputs
        static_inputs = {}
        static_inputs["x_t"] = sample_input.clone().detach().requires_grad_(False)
        static_inputs["t"] = torch.tensor([timestep], device=self.device).detach()

        if conditioning_input is not None:
            static_inputs["conditioning"] = (
                conditioning_input.clone().detach().requires_grad_(False)
            )

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                if conditioning_input is not None:
                    model_input = torch.cat(
                        [static_inputs["x_t"], static_inputs["conditioning"]], dim=1
                    )
                else:
                    model_input = static_inputs["x_t"]
                _ = model(model_input, static_inputs["t"])

        # Capture graph
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(graph):
            if conditioning_input is not None:
                model_input = torch.cat(
                    [static_inputs["x_t"], static_inputs["conditioning"]], dim=1
                )
            else:
                model_input = static_inputs["x_t"]
            static_outputs = model(model_input, static_inputs["t"])

        torch.cuda.synchronize()

        # Store graph and inputs/outputs
        self.graphs[graph_key] = graph
        self.static_inputs_dict[graph_key] = static_inputs
        self.static_outputs_dict[graph_key] = static_outputs
        self.timestamps[graph_key] = time.time()

        # Estimate memory usage
        self.memory_usage[graph_key] = self._estimate_memory_usage(static_inputs, static_outputs)

        logger.debug(
            f"Captured CUDA graph {graph_key} ({self.memory_usage[graph_key] / 1024 / 1024:.2f} MB)"
        )

        return graph_key

    def _evict_lru_if_needed(self):
        """Evict least recently used graph if cache is full."""
        if len(self.graphs) >= self.max_cached:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            del self.graphs[oldest_key]
            del self.static_inputs_dict[oldest_key]
            del self.static_outputs_dict[oldest_key]
            del self.timestamps[oldest_key]
            memory_freed = self.memory_usage.pop(oldest_key, 0)
            logger.debug(
                f"Evicted CUDA graph {oldest_key} (freed {memory_freed / 1024 / 1024:.2f} MB)"
            )

    def _estimate_memory_usage(self, inputs: dict, outputs: Any) -> int:
        """Estimate memory usage of graph inputs and outputs."""
        total = 0
        for tensor in inputs.values():
            if isinstance(tensor, torch.Tensor):
                total += tensor.nelement() * tensor.element_size()

        if isinstance(outputs, torch.Tensor):
            total += outputs.nelement() * outputs.element_size()
        elif isinstance(outputs, dict):
            for tensor in outputs.values():
                if isinstance(tensor, torch.Tensor):
                    total += tensor.nelement() * tensor.element_size()

        return total

    def replay_graph(
        self,
        x_t: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        timestep: int = 0,
    ) -> torch.Tensor:
        """Replay the captured CUDA graph.

        Args:
            x_t: Current sample
            conditioning: Conditioning tensor
            timestep: Current timestep

        Returns:
            Model output from graph replay
        """
        graph_key = f"batch_{x_t.shape[0]}_t_{timestep}"

        if graph_key not in self.graphs:
            raise RuntimeError(
                f"CUDA graph not cached. Call capture_graph() first for {graph_key}."
            )

        # Update timestamp
        self.timestamps[graph_key] = time.time()

        # Copy inputs to static tensors
        self.static_inputs_dict[graph_key]["x_t"].copy_(x_t)
        self.static_inputs_dict[graph_key]["t"].fill_(timestep)

        if conditioning is not None and "conditioning" in self.static_inputs_dict[graph_key]:
            self.static_inputs_dict[graph_key]["conditioning"].copy_(conditioning)

        # Replay graph
        self.graphs[graph_key].replay()

        return self.static_outputs_dict[graph_key].clone()


class MemoryEfficientSampler:
    """Memory-efficient sampling for diffusion models."""

    def __init__(self, device: torch.device, config: dict[str, Any]):
        """Initialize memory-efficient sampler.

        Args:
            device: Device for sampling
            config: Configuration dictionary
        """
        self.device = device
        self.config = config

        # Memory optimization settings
        self.use_gradient_checkpointing = config.get("use_gradient_checkpointing", False)
        self.enable_attention_slicing = config.get("enable_attention_slicing", False)
        self.use_vae_slicing = config.get("use_vae_slicing", False)
        self.use_tiled_vae = config.get("use_tiled_vae", False)

        # CUDA graph settings
        self.use_cuda_graphs = config.get("use_cuda_graphs", False)
        self.cuda_graph_cache = {}

    def optimize_model_for_inference(self, model: nn.Module):
        """Apply memory optimizations to the model.

        Args:
            model: Model to optimize
        """
        if self.use_gradient_checkpointing:
            # Enable gradient checkpointing for memory efficiency
            model.gradient_checkpointing_enable()

        if self.enable_attention_slicing:
            # Enable attention slicing to reduce memory usage
            if hasattr(model, "enable_attention_slicing"):
                model.enable_attention_slicing()

        if self.use_vae_slicing:
            # Enable VAE slicing for decoder
            if hasattr(model, "enable_vae_slicing"):
                model.enable_vae_slicing()

        if self.use_tiled_vae:
            # Enable tiled VAE for large images
            if hasattr(model, "enable_tiled_vae"):
                model.enable_tiled_vae()

    def prepare_cuda_graphs(
        self,
        model: nn.Module,
        batch_size: int = 1,
        channels: int = 1,
        height: int = 256,
        width: int = 256,
        conditioning_channels: int | None = None,
    ):
        """Prepare CUDA graphs for different timesteps.

        Args:
            model: The diffusion model
            batch_size: Batch size for graphs
            channels: Number of input channels
            height: Image height
            width: Image width
            conditioning_channels: Number of conditioning channels
        """
        if not self.use_cuda_graphs or not torch.cuda.is_available():
            return

        # Create sample inputs
        sample_x_t = torch.randn(batch_size, channels, height, width, device=self.device)
        sample_conditioning = None

        if conditioning_channels is not None:
            sample_conditioning = torch.randn(
                batch_size, conditioning_channels, height, width, device=self.device
            )

        # Capture graphs for different timesteps (sample a few key ones)
        key_timesteps = [0, 100, 500, 900, 999]  # Sample timesteps for graph capture

        for t in key_timesteps:
            graph_key = f"batch_{batch_size}_t_{t}"
            graph_wrapper = CUDAGraphWrapper(self.device)
            graph_wrapper.capture_graph(model, sample_x_t, sample_conditioning, t)
            self.cuda_graph_cache[graph_key] = graph_wrapper

    def get_cuda_graph(self, batch_size: int, timestep: int) -> CUDAGraphWrapper | None:
        """Get cached CUDA graph for given parameters.

        Args:
            batch_size: Current batch size
            timestep: Current timestep

        Returns:
            CUDA graph wrapper if available, None otherwise
        """
        if not self.use_cuda_graphs:
            return None

        # Find closest cached timestep
        cached_timesteps = [
            int(key.split("_t_")[1])
            for key in self.cuda_graph_cache.keys()
            if key.startswith(f"batch_{batch_size}_t_")
        ]

        if not cached_timesteps:
            return None

        # Use closest timestep
        closest_t = min(cached_timesteps, key=lambda x: abs(x - timestep))
        graph_key = f"batch_{batch_size}_t_{closest_t}"

        return self.cuda_graph_cache.get(graph_key)

    def efficient_predict_noise(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: int,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict noise using memory-efficient techniques.

        Args:
            model: The diffusion model
            x_t: Current sample
            t: Current timestep
            conditioning: Conditioning tensor

        Returns:
            Predicted noise
        """
        # Try CUDA graph first
        batch_size = x_t.shape[0]
        cuda_graph = self.get_cuda_graph(batch_size, t)

        if cuda_graph is not None:
            return cuda_graph.replay_graph(x_t, conditioning, t)

        # Fallback to regular inference
        t_tensor = torch.tensor([t], device=self.device).repeat(x_t.shape[0])

        with torch.no_grad():
            if conditioning is not None:
                model_input = torch.cat([x_t, conditioning], dim=1)
                noise_pred = model(model_input, t_tensor)
            else:
                noise_pred = model(x_t, t_tensor)

        # Handle different model output formats
        if isinstance(noise_pred, dict):
            noise_pred = noise_pred.get("noise_pred", noise_pred.get("output", noise_pred))

        return noise_pred


class InferencePerformanceOptimizer:
    """Comprehensive performance optimizer for inference."""

    def __init__(
        self,
        device: torch.device,
        config: dict[str, Any],
        max_metric_history: int = 1000,
    ):
        """Initialize performance optimizer.

        Args:
            device: Device for inference
            config: Performance configuration
            max_metric_history: Maximum number of metric samples to keep (prevents unbounded RAM)
        """
        from collections import deque

        self.device = device
        self.config = config
        self.max_metric_history = max_metric_history

        # Initialize components
        self.memory_efficient_sampler = MemoryEfficientSampler(device, config)
        self.cuda_graphs_enabled = config.get("use_cuda_graphs", False)

        # Autocast defaults — set conservatively until optimize_model() is called
        # so run_optimized_inference() never hits an AttributeError when invoked
        # before optimize_model() (e.g. CUDA-graphs-only path).
        self.enable_mixed_precision: bool = config.get("use_autocast", True)
        self.autocast_enabled: bool = self.enable_mixed_precision and torch.cuda.is_available()
        self.autocast_dtype: torch.dtype | None = torch.float16 if self.autocast_enabled else None

        # Performance metrics with bounded history (using deque for O(1) popleft)
        self.inference_times = deque(maxlen=max_metric_history)
        self.memory_usage = deque(maxlen=max_metric_history)

    def optimize_model(self, model: nn.Module):
        """Apply all performance optimizations to the model with consistent autocast settings.

        Args:
            model: Model to optimize
        """
        self.memory_efficient_sampler.optimize_model_for_inference(model)

        # Additional optimizations
        model.eval()

        # Enable autocast for mixed precision if available - validate consistency
        self.enable_mixed_precision = self.config.get("use_autocast", True)

        # Only enable autocast if using mixed precision AND CUDA is available
        self.autocast_enabled = self.enable_mixed_precision and torch.cuda.is_available()

        if self.autocast_enabled:
            self.autocast_dtype = torch.float16  # Default dtype for autocast
        else:
            self.autocast_dtype = None

    def prepare_for_inference(
        self,
        model: nn.Module,
        batch_size: int = 1,
        channels: int = 1,
        height: int = 256,
        width: int = 256,
        conditioning_channels: int | None = None,
    ):
        """Prepare model and caches for inference.

        Args:
            model: The model to prepare
            batch_size: Expected batch size
            channels: Number of input channels
            height: Image height
            width: Image width
            conditioning_channels: Number of conditioning channels
        """
        # Prepare CUDA graphs
        if self.cuda_graphs_enabled:
            self.memory_efficient_sampler.prepare_cuda_graphs(
                model, batch_size, channels, height, width, conditioning_channels
            )

    def run_optimized_inference(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: int,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run inference with performance optimizations and consistent autocast.

        Args:
            model: The model
            x_t: Current sample
            t: Current timestep
            conditioning: Conditioning tensor

        Returns:
            Model output
        """
        import time

        start_time = time.time()

        # Use memory-efficient prediction with consistent autocast
        if self.autocast_enabled and torch.cuda.is_available():
            with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
                output = self.memory_efficient_sampler.efficient_predict_noise(
                    model, x_t, t, conditioning
                )
        else:
            # No autocast or not on CUDA
            with torch.no_grad():
                output = self.memory_efficient_sampler.efficient_predict_noise(
                    model, x_t, t, conditioning
                )

        # Track performance
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            self.memory_usage.append(memory_used)

        return output

    def get_performance_stats(self) -> dict[str, float]:
        """Get performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {}

        return {
            "avg_inference_time": sum(self.inference_times) / len(self.inference_times),
            "max_inference_time": max(self.inference_times),
            "min_inference_time": min(self.inference_times),
            "total_inference_time": sum(self.inference_times),
            "avg_memory_usage_gb": (
                sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
            ),
            "max_memory_usage_gb": max(self.memory_usage) if self.memory_usage else 0,
        }
