"""Training Utilities Module

This module contains utility classes and functions for the training pipeline,
including data loading, profiling, gradient scaling, and model calling.
"""

# Enhanced logging for performance
import logging
import os
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from torch.cuda.amp import GradScaler as TorchGradScaler

    # Use a dummy class for type checking if torch is not available
    class GradScaler(TorchGradScaler):
        """GradScaler class."""

        pass

else:
    try:
        # PyTorch 2.4+ (consolidated amp)
        from torch.amp import GradScaler
    except ImportError:
        try:
            # PyTorch 1.6 - 2.3
            from torch.cuda.amp import GradScaler
        except ImportError:
            # Fallback for CPU-only or very old PyTorch
            class GradScaler:  # type: ignore[no-redef]
                """GradScaler class."""

                def __init__(self, enabled: bool = False):
                    """__init__.

                    Args:
                        enabled (bool): Description.
                    """
                    self._enabled = enabled

                def scale(self, loss: torch.Tensor) -> torch.Tensor:
                    """scale.

                    Args:
                        loss (torch.Tensor): Description.
                    Returns:
                        torch.Tensor: Description.
                    """
                    return loss

                def step(self, optimizer: Any) -> None:
                    """step.

                    Args:
                        optimizer (Any): Description.
                    """
                    optimizer.step()

                def update(self) -> None:
                    """update."""
                    pass

                def unscale_(self, optimizer: Any) -> None:
                    """unscale_.

                    Args:
                        optimizer (Any): Description.
                    """
                    pass

                def is_enabled(self) -> bool:
                    """is_enabled.

                    Returns:
                        bool: Description.
                    """
                    return self._enabled

                def get_scale(self) -> float:
                    """get_scale.

                    Returns:
                        float: Description.
                    """
                    return 1.0


from mriforge.config.schemas.enums import TrainingMode
from mriforge.core.compute_device import (
    AcceleratorRequiredError,
    cpu_opt_in_from_env,
    resolve_torch_device,
)
from mriforge.infrastructure.logging import MetricsTracker  # noqa: F401

# Default: issue tracker may not be available when running unit tests
ISSUE_TRACKER_AVAILABLE = False

# NOTE: the project default device is resolved lazily via
# ``get_default_device()`` (defined at the bottom of this module) — there is
# deliberately no module-scope ``device`` global, so importing this module no
# longer initialises CUDA / caps GPU memory as a side effect.

# Global flags for gradient detection (kept for compatibility)
_vanishing_gradient_detected = False
_exploding_gradient_detected = False
_exploding_gradient_critical = False


def clamp_to_range(
    tensor: torch.Tensor,
    min_val: float = -1.0,
    max_val: float = 1.0,
    *,
    enable: bool = True,
    telemetry: bool = False,
    logger: logging.Logger | None = None,
) -> torch.Tensor:
    """Clamp tensor to a given range with optional telemetry.

    This can be used to enforce the project-wide convention that generator
    outputs lie in [-1, 1]. Set `enable=False` to bypass. When `telemetry` is
    True, logs basic stats about clamping.
    """
    if not enable:
        return tensor

    if telemetry:
        below = (tensor < min_val).sum().item()
        above = (tensor > max_val).sum().item()
        if (below + above) > 0:
            msg = f"Clamping output: {below} values below {min_val}, {above} above {max_val}"
            (logger or logging.getLogger(__name__)).debug(msg)

    return torch.clamp(tensor, min=min_val, max=max_val)


def sample_diffusion_timesteps(
    batch_size: int,
    num_timesteps: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample random timesteps for diffusion training.

    Canonical utility function for sampling diffusion timesteps, used by both
    diffusion models and training strategies.

    Args:
        batch_size: Number of timesteps to sample
        num_timesteps: Total number of timesteps in the schedule (e.g., 1000)
        device: Device for tensor allocation. Defaults to CPU if not specified.

    Returns:
        Tensor of shape [batch_size] with sampled timestep indices

    Example:
        >>> timesteps = sample_diffusion_timesteps(batch_size=32, num_timesteps=1000)
        >>> timesteps.shape
        torch.Size([32])
    """
    if device is None:
        device = torch.device("cpu")

    return torch.randint(
        low=0,
        high=num_timesteps,
        size=(batch_size,),
        device=device,
        dtype=torch.long,
    )


class LimitedDataLoader:
    """Wraps a DataLoader to limit the number of batches, used for
    profiling.
    """

    def __init__(self, dataloader: DataLoader[Any], max_batches: int) -> None:
        """__init__.

        Args:
            dataloader (DataLoader[Any]): Description.
            max_batches (int): Description.
        """
        self.dataloader = dataloader
        self.max_batches = max_batches
        self.iterator: Iterator[Any] | None = None
        self.count = 0

    def __iter__(self) -> "LimitedDataLoader":
        """__iter__.

        Returns:
            'LimitedDataLoader': Description.
        """
        self.iterator = iter(self.dataloader)
        self.count = 0
        return self

    def __next__(self) -> Any:
        """__next__.

        Returns:
            Any: Description.
        """
        if self.count >= self.max_batches:
            raise StopIteration
        self.count += 1
        return next(self.iterator)

    def __len__(self) -> int:
        """__len__.

        Returns:
            int: Description.
        """
        return min(len(self.dataloader), self.max_batches)


# --- Torch-Fidelity Metrics Tracking ---

METRICS_TRACKER_AVAILABLE = True

# --- Profiler Import (delayed to avoid circular imports) ---
PROFILER_AVAILABLE = False
TrainingProfiler: Any = None


class ProfilingState:
    """Centralized profiling state management for consistent
    profiling behavior.
    """

    def __init__(self, profiler: Any = None, profiling_mode: bool = False) -> None:
        """__init__.

        Args:
            profiler (Any): Description.
            profiling_mode (bool): Description.
        """
        self.profiler = profiler
        self.profiling_mode = profiling_mode
        self._cached_profiling_enabled: bool | None = None
        self._cached_event_sets: list[Any] | None = None
        self._last_error: str | None = None

    def is_profiling_enabled(self) -> bool:
        """Check if profiling is enabled and available."""
        if self._cached_profiling_enabled is not None:
            return self._cached_profiling_enabled

        if not self.profiling_mode:
            self._cached_profiling_enabled = False
            return False

        if self.profiler is None:
            self._cached_profiling_enabled = False
            return False

        # Check if profiler is properly initialized
        if not hasattr(self.profiler, "papi_initialized"):
            self._cached_profiling_enabled = False
            return False

        if not self.profiler.papi_initialized:
            self._cached_profiling_enabled = False
            return False

        self._cached_profiling_enabled = True
        return True

    def should_profile_this_epoch(self, epoch: int | None = None) -> bool:
        """Determine if profiling should occur for this epoch."""
        if not self.is_profiling_enabled():
            return False

        if self.profiling_mode:
            # In profiling mode, profile every epoch
            return True

        # In normal mode, check if profiler should continue
        if self.profiler is not None and hasattr(
            self.profiler,
            "should_continue_profiling",
        ):
            return self.profiler.should_continue_profiling()  # type: ignore

        return False

    def get_event_sets(self) -> list[Any]:
        """Safely get event sets from profiler."""
        if not self.is_profiling_enabled():
            return []

        if self._cached_event_sets is not None:
            return self._cached_event_sets

        try:
            if self.profiler is not None and hasattr(self.profiler, "event_sets"):
                self._cached_event_sets = self.profiler.event_sets
                return self._cached_event_sets  # type: ignore
        except Exception as e:
            self._last_error = f"Error getting event sets: {e}"
            logging.warning(self._last_error)

        return []

    def safe_start(self, region_name: str, event_set: Any = None) -> Any:
        """Safely start profiling with error handling."""
        if not self.is_profiling_enabled():
            return False

        try:
            if self.profiler is not None:
                if event_set is not None:
                    return self.profiler.start(region_name, event_set=event_set)
                return self.profiler.start(region_name)
        except Exception as e:
            self._last_error = f"Error starting profiler for {region_name}: {e}"
            logging.warning(self._last_error)
            return False
        return False

    def safe_stop(self, region_name: str, event_set: Any = None) -> Any:
        """Safely stop profiling with error handling."""
        if not self.is_profiling_enabled():
            return False

        try:
            if self.profiler is not None:
                if event_set is not None:
                    return self.profiler.stop(region_name, event_set=event_set)
                return self.profiler.stop(region_name)
        except Exception as e:
            self._last_error = f"Error stopping profiler for {region_name}: {e}"
            logging.warning(self._last_error)
            return False
        return False

    def get_last_error(self) -> str | None:
        """Get the last error message."""
        return self._last_error

    def clear_cache(self) -> None:
        """Clear cached values (useful when profiler state changes)."""
        self._cached_profiling_enabled = None
        self._cached_event_sets = None
        self._last_error = None

    def print_status(self, epoch: int, total_epochs: int) -> None:
        """Print consistent profiling status messages."""
        if self.profiling_mode:
            if self.is_profiling_enabled():
                logging.info("=" * 80)
                logging.info(
                    f"PROFILING MODE - EPOCH {epoch}/{total_epochs} - PAPI PROFILING ENABLED",
                )
                logging.info("=" * 80)
                logging.debug(
                    "  - PAPI profiling active for comprehensive performance analysis",
                )

                event_sets = self.get_event_sets()
                if event_sets:
                    logging.debug(f"  - Event sets available: {len(event_sets)}")
                    if self.profiler is not None and hasattr(
                        self.profiler,
                        "current_event_set_idx",
                    ):
                        try:
                            current_idx = self.profiler.current_event_set_idx
                            logging.debug(
                                f"  - Current event set: {current_idx + 1}/{len(event_sets)}",
                            )
                        except Exception as e:
                            logging.debug(f"  - Current event set: {e}")
                else:
                    logging.debug("  - Event sets: Not available")

                logging.info("=" * 80)
            else:
                logging.warning("=" * 80)
                logging.warning(
                    f"PROFILING MODE - EPOCH {epoch}/{total_epochs} - PROFILER NOT AVAILABLE",
                )
                logging.warning("=" * 80)
                logging.warning("  - PAPI profiler not initialized or not available")
                logging.warning("  - Training will continue without profiling")
                if self._last_error:
                    logging.warning(f"  - Last error: {self._last_error}")
                logging.warning("=" * 80)
        elif self.should_profile_this_epoch(epoch):
            logging.info(
                f"\n=== Profiling enabled for epoch {epoch}. Profiling individual phases. ===",
            )
            if (
                self.profiler is not None
                and hasattr(self.profiler, "has_remaining_event_sets")
                and not self.profiler.has_remaining_event_sets()
            ):
                logging.info(
                    "Profiler has completed 30 epochs. No more profiling will occur.",
                )


# --- Profiler Import (delayed to avoid circular imports) ---
PROFILER_AVAILABLE = False
TrainingProfiler = None


def initialize_profiler() -> None:
    """Initialize profiler if available"""
    global PROFILER_AVAILABLE, TrainingProfiler, profile_training
    try:
        import mriforge.models.analysis.profile_training as profile_training  # type: ignore

        TrainingProfiler = profile_training.TrainingProfiler  # type: ignore
        if TrainingProfiler is not None:
            PROFILER_AVAILABLE = True
            logging.debug("Profiler available for performance monitoring")
        else:
            PROFILER_AVAILABLE = False
            logging.warning("Profiler not available.")
    except ImportError:
        PROFILER_AVAILABLE = False
        logging.warning("Profiler module not found.")


_DEFAULT_GPU_MEMORY_FRACTION = 0.85


def _resolve_gpu_memory_fraction() -> float:
    """Resolve the per-process GPU memory cap (pitfall #15: wired knob).

    The 85 % cap inside :func:`initialize_device` used to be hardcoded.
    ``MRIFORGE_GPU_MEMORY_FRACTION`` (registered in ``core/env.py``) now
    overrides it; an invalid value RAISES (pitfall #9) — a silently-ignored
    typo would leave the user running under a cap they never chose.
    """
    raw = os.environ.get("MRIFORGE_GPU_MEMORY_FRACTION")
    if raw is None:
        return _DEFAULT_GPU_MEMORY_FRACTION
    try:
        fraction = float(raw)
    except ValueError:
        raise ValueError(
            f"MRIFORGE_GPU_MEMORY_FRACTION must be a float in (0, 1], got {raw!r}"
        ) from None
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"MRIFORGE_GPU_MEMORY_FRACTION must be in (0, 1], got {fraction}")
    return fraction


def initialize_device() -> torch.device:
    """Initialize the training device under the accelerated-run contract.

    Raises ``AcceleratorRequiredError`` when no accelerator is available. This
    function previously wrapped its whole body in a bare ``except Exception``
    that logged "Falling back to CPU mode..." and returned ``torch.device("cpu")``
    — so a genuine CUDA driver fault, a bad ``CUDA_VISIBLE_DEVICES``, or an OOM
    during the probe allocation all produced a *successful* CPU training run at
    ~100x slowdown. That was the most expensive silent fallback in the codebase.

    CPU is still reachable, but only when the user dictates it:
    ``CUDA_VISIBLE_DEVICES=""`` or ``FORCE_CPU=true``.
    """
    logging.info("Initializing device...")

    # User-dictated CPU: the sanctioned opt-out (see core.compute_device).
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "" or cpu_opt_in_from_env():
        logging.warning(
            "[DEVICE] CPU mode forced by environment (CUDA_VISIBLE_DEVICES='' "
            "or FORCE_CPU). Training will be ~100x slower than on GPU."
        )
        return torch.device("cpu")

    # No accelerator + no opt-in → raise. Never degrade silently.
    decision = resolve_torch_device("auto", pipeline="train", source="initialize_device")
    if not decision.accelerated:
        return torch.device(decision.device)

    try:
        logging.info("Checking CUDA availability...")
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            logging.info("CUDA is available. Attempting to initialize...")

            # Conservative per-process memory cap to prevent OOM. Wired knob:
            # MRIFORGE_GPU_MEMORY_FRACTION (default 0.85), validated + logged.
            memory_fraction = _resolve_gpu_memory_fraction()
            torch.cuda.set_per_process_memory_fraction(memory_fraction)

            # Test GPU access without hanging
            device = torch.device("cuda")

            # Try to allocate a small tensor to test GPU
            test_tensor = torch.ones(1, device=device)
            logging.debug("GPU test successful")

            # Only get device info if GPU is working
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            logging.info(f"Using CUDA device: {device_name}")
            logging.info(f"Available GPUs: {device_count}")
            # Stamp the resolved knob value into the run log (pitfall #15c).
            logging.info(
                f"GPU Memory: {total_memory:.1f} GB "
                f"(cap {memory_fraction:.0%} = "
                f"{total_memory * memory_fraction:.1f} GB, "
                f"via MRIFORGE_GPU_MEMORY_FRACTION)",
            )

            logging.info(f"PyTorch version: {torch.__version__}")

            # Deliberately does NOT touch torch.backends here. It used to set
            # `cudnn.benchmark = True` and `matmul.allow_tf32 = True`
            # unconditionally, which made this a SECOND writer to globals
            # `accelerator.seed_everything` already owns -- and the two
            # disagreed. `seed_everything` sets `benchmark = not deterministic`
            # precisely because the autotuner re-introduces the run-to-run
            # variance the determinism flag exists to remove; this function set
            # it back to True regardless. It is reached lazily via
            # `_DEFAULT_DEVICE`, so whichever ran last won, and a run declaring
            # determinism could be silently un-determinized by an unrelated
            # first touch of the default device.
            #
            # TF32 has one owner for the same reason: `seed_everything` sets it
            # from `allow_tf32`, next to the determinism decision it trades
            # against, and logs the resolved value.

            # Clean up test tensor
            del test_tensor

            return device

        # Unreachable: resolve_torch_device above already raised if CUDA is
        # absent and the user did not opt into CPU. Kept as a hard assertion so
        # a future edit cannot quietly re-introduce a CPU path here.
        raise AcceleratorRequiredError(
            "initialize_device reached the no-CUDA branch even though the "
            "accelerated-run contract resolved an accelerated device — CUDA "
            "disappeared mid-initialization."
        )

    except AcceleratorRequiredError:
        raise
    except Exception as e:
        # A real CUDA fault (driver mismatch, ECC error, OOM on the probe
        # allocation) must NOT become a silent CPU run: the job would burn its
        # full wall-clock allocation ~100x slower while reporting success.
        logging.exception(f"Error during device initialization: {e}")
        raise AcceleratorRequiredError(
            f"CUDA device initialization failed: {e}. Refusing to fall back to "
            "CPU — a silent CPU run wastes the GPU allocation. Fix the "
            "environment, or opt into CPU explicitly with FORCE_CPU=true."
        ) from e


def call_generator_model(
    model: torch.nn.Module,
    lr_input: torch.Tensor,
    model_type: str,
    model_info: dict[str, Any] | None,
    z_dim: int,
    device: torch.device,
    *,
    _allow_chunking: bool = True,
) -> torch.Tensor:
    """Dispatches generator calls based on model type with OOM-safe chunking and AMP.

    This refactor keeps the original behavior but centralizes handlers for clarity.
    """

    def _use_amp_context() -> bool:
        """_use_amp_context.

        Returns:
            bool: Description.
        """
        grad_enabled = torch.is_grad_enabled()
        # Enable mixed precision for GAN models, disable for diffusion models
        gan_models = ["unet", "vit", "swin", "stylegan", "standard", "kan"]
        if model_type == TrainingMode.DIFFUSION.value:
            return False
        if model_type in gan_models:
            return (not grad_enabled) and (device.type == "cuda")
        # Default behavior for other model types
        return (not grad_enabled) and (device.type == "cuda")

    def _forward_with_amp(fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        """_forward_with_amp.

        Args:
            fn (Callable[[], torch.Tensor]): Description.
        Returns:
            torch.Tensor: Description.
        """
        if _use_amp_context():
            from torch.amp.autocast_mode import autocast as _autocast

            with _autocast(device_type=device.type, enabled=True):
                return fn()
        else:
            return fn()

    def _ensure_dtype(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.dtype]:
        """_ensure_dtype.

        Args:
            input_tensor (torch.Tensor): Description.
        Returns:
            tuple[torch.Tensor, torch.dtype]: Description.
        """
        try:
            gen_param = next(model.parameters())
            gen_dtype = gen_param.dtype
            gen_device = gen_param.device
        except (StopIteration, AttributeError):
            gen_dtype = input_tensor.dtype
            gen_device = device

        # Always ensure tensor is on the same device as the model
        if input_tensor.device != gen_device:
            logging.debug(
                f"Moving input from {input_tensor.device} to model device {gen_device}",
            )
            input_tensor = input_tensor.to(device=gen_device)

        # Ensure tensor has the same dtype as the model
        if input_tensor.dtype != gen_dtype:
            logging.debug(
                f"Converting input from {input_tensor.dtype} to model dtype {gen_dtype}",
            )
            input_tensor = input_tensor.to(dtype=gen_dtype)

        return input_tensor, gen_dtype

    def _validate_model_device_consistency(input_tensor: torch.Tensor) -> torch.Tensor:
        """Validate that model and input are on the same device"""
        try:
            # Check if model has parameters and supports iteration
            params_callable = callable(getattr(model, "parameters", None))
            if hasattr(model, "parameters") and params_callable:
                try:
                    model_device = next(model.parameters()).device
                    if input_tensor.device != model_device:
                        logging.warning(
                            f"Device mismatch: input on {input_tensor.device}, "
                            f"model on {model_device}",
                        )
                        # Move input to match model device
                        input_tensor = input_tensor.to(model_device)
                        logging.info(f"Moved input to {model_device}")
                except (StopIteration, TypeError) as e:
                    # Model has no parameters or iteration failed
                    logging.debug(
                        f"Parameter validation failed ({type(e).__name__}), "
                        "skipping device validation",
                    )
            else:
                logging.debug(
                    "Model doesn't support parameter iteration, skipping validation",
                )
        except Exception as e:
            logging.warning(f"Could not validate device consistency: {e}")
        return input_tensor

    def _chunked_call(inputs: torch.Tensor) -> torch.Tensor | None:
        """_chunked_call.

        Args:
            inputs (torch.Tensor): Description.
        Returns:
            Optional[torch.Tensor]: Description.
        """
        if not _allow_chunking:
            return None
        batch = inputs.size(0)
        chunk_size = max(1, batch // 2)
        while chunk_size >= 1:
            try:
                outputs = []
                for i in range(0, batch, chunk_size):
                    sub = inputs[i : i + chunk_size]
                    out = call_generator_model(
                        model,
                        sub,
                        model_type,
                        model_info,
                        z_dim,
                        device,
                        _allow_chunking=False,
                    )
                    outputs.append(out)
                return torch.cat(outputs, dim=0)
            except torch.cuda.OutOfMemoryError:
                # PyTorch allocator handles memory reclamation
                chunk_size = chunk_size // 2
            except (RuntimeError, ValueError, TypeError):
                break
        return None

    try:
        # Align input dtype with model parameters
        lr_input, gen_dtype = _ensure_dtype(lr_input)

        # Validate device consistency before forward pass
        lr_input = _validate_model_device_consistency(lr_input)

        # Handler: Diffusion models expect timesteps/noise schedule
        if model_info and model_info.get("type") == "Diffusion":
            batch_size = lr_input.size(0)

            if hasattr(model, "sample_timesteps"):
                timesteps = model.sample_timesteps(batch_size).to(device, non_blocking=True)
            else:
                timesteps = torch.randint(
                    low=0,
                    high=1000,
                    size=(batch_size,),
                    device=device,
                    dtype=torch.long,
                )

            def _call_diffusion() -> torch.Tensor:
                """_call_diffusion.

                Returns:
                    torch.Tensor: Description.
                """
                return model(lr_input, timesteps)

            try:
                output = _forward_with_amp(_call_diffusion)
            except torch.cuda.OutOfMemoryError:
                # Fallback to chunked processing
                output = _chunked_call(lr_input)
                if output is None:
                    raise

        # Handler: StyleGAN-like models take pure noise z
        elif model_info and "stylegan" in model_info.get("name", "").lower():
            batch_size = lr_input.size(0)
            stylegan_z_dim = getattr(model, "z_dim", 512)

            def _call_stylegan() -> torch.Tensor:
                """_call_stylegan.

                Returns:
                    torch.Tensor: Description.
                """
                z = torch.randn(
                    batch_size,
                    stylegan_z_dim,
                    device=device,
                    dtype=gen_dtype,
                )
                return model(z)

            try:
                output = _forward_with_amp(_call_stylegan)
            except torch.cuda.OutOfMemoryError:
                # Fallback to chunked processing
                output = _chunked_call(lr_input)
                if output is None:
                    raise

        # Handler: models that require noise vector (flagged in model_info)
        elif model_info and model_info.get("requires_noise", False):
            batch_size = lr_input.size(0)

            def _call_noise() -> torch.Tensor:
                """_call_noise.

                Returns:
                    torch.Tensor: Description.
                """
                z = torch.randn(batch_size, z_dim, device=device, dtype=gen_dtype)
                return model(z)

            try:
                output = _forward_with_amp(_call_noise)
            except torch.cuda.OutOfMemoryError:
                # Fallback to chunked processing
                output = _chunked_call(lr_input)
                if output is None:
                    raise

        # Handler: Vision Transformer / Swin - ensure input size
        elif model_info and "transformer" in model_info.get("type", "").lower():
            # OPTIMIZATION: Removed redundant resizing. The MRIDataset already resizes all
            # images to a standard 240x240, so this on-the-fly interpolation is not needed
            # and adds computational overhead to every batch.
            try:
                # The resizing logic was here. It has been removed.

                def _call_transformer() -> torch.Tensor:
                    """_call_transformer.

                    Returns:
                        torch.Tensor: Description.
                    """
                    return model(lr_input)

                try:
                    output = _forward_with_amp(_call_transformer)
                except torch.cuda.OutOfMemoryError:
                    # Fallback to chunked processing
                    output = _chunked_call(lr_input)
                    if output is None:
                        raise
            except (RuntimeError, ValueError, TypeError):
                # Fallback: ensure size and call
                if lr_input.shape[2:] != (240, 240):
                    lr_input = torch.nn.functional.interpolate(
                        lr_input,
                        size=(240, 240),
                        mode="bilinear",
                        align_corners=False,
                    )

                def _call_fallback() -> torch.Tensor:
                    """_call_fallback.

                    Returns:
                        torch.Tensor: Description.
                    """
                    return model(lr_input)

                try:
                    output = _forward_with_amp(_call_fallback)
                except torch.cuda.OutOfMemoryError:
                    # Fallback to chunked processing
                    output = _chunked_call(lr_input)
                    if output is None:
                        raise

        else:
            # Standard forward
            def _call_std() -> torch.Tensor:
                """_call_std.

                Returns:
                    torch.Tensor: Description.
                """
                return model(lr_input)

            try:
                output = _forward_with_amp(_call_std)
            except torch.cuda.OutOfMemoryError:
                # Fallback to chunked processing
                output = _chunked_call(lr_input)
                if output is None:
                    raise

        # Safety checks on output
        if output is not None:
            if torch.isnan(output).any() or torch.isinf(output).any():
                output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            out_min, out_max = output.min().item(), output.max().item()
            if out_max > 5.0 or out_min < -5.0:
                output = torch.clamp(output, min=-3.0, max=3.0)
            if output.dtype != gen_dtype:
                output = output.to(dtype=gen_dtype)

        # Fallback to zero tensor if output is None (should not happen if logic is correct)
        if output is None:
            output = torch.zeros_like(lr_input)

        return output

    except Exception as e:
        logging.exception(f"Error in call_generator_model: {e}")
        logging.exception(
            f"Model type: {model_type}, Input shape: {getattr(lr_input, 'shape', 'unknown')}",
        )
        raise


class SafeGradScaler:
    """Wrapper for GradScaler that handles state tracking, error recovery, and gradient stability"""

    def __init__(self, enabled: bool = True, stability_manager: Any = None) -> None:
        """__init__.

        Args:
            enabled (bool): Description.
            stability_manager (Any): Description.
        """
        self.scaler = GradScaler(enabled=enabled)
        self.stability_manager = stability_manager
        self.step_count = 0
        self.scale_calls = 0
        self.last_scale_value = 1.0
        self.enabled = enabled  # Store the enabled state for safety

    def is_enabled(self) -> bool:
        """is_enabled.

        Returns:
            bool: Description.
        """
        try:
            return self.scaler.is_enabled()
        except (RuntimeError, ValueError, TypeError):
            # Fallback to stored state if scaler fails
            return self.enabled

    def get_scale(self) -> float:
        """get_scale.

        Returns:
            float: Description.
        """
        try:
            return self.scaler.get_scale()
        except (RuntimeError, ValueError, TypeError):
            return self.last_scale_value

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale a loss tensor"""
        if self.is_enabled():
            try:
                scaled_loss = self.scaler.scale(loss)
                self.scale_calls += 1
                self.last_scale_value = self.get_scale()
                return scaled_loss
            except (RuntimeError, ValueError, TypeError) as e:
                logging.warning(f"GradScaler.scale() failed: {e}")
                self.scale_calls += 1
                return loss
        else:
            return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients for the given optimizer"""
        if self.is_enabled():
            try:
                self.scaler.unscale_(optimizer)
            except (RuntimeError, ValueError, TypeError) as e:
                logging.warning(f"GradScaler.unscale_() failed: {e}")
        # If not enabled, no unscaling needed

    def step(
        self,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module | None = None,
        model_name: str = "model",
        epoch: int = 0,
        model_type: str = "gan",
        noise_schedule_factor: float = 1.0,
    ) -> dict[str, Any]:
        """Safely perform optimizer step with gradient stability measures"""
        gradient_stats: dict[str, Any] = {}

        try:
            # Apply gradient stability measures using the stability manager
            if self.stability_manager is not None and model is not None:
                # Use specialized diffusion optimization if applicable
                is_diffusion = str(model_type or "").lower() in [
                    "diffusion",
                    "chi_square_diffusion",
                    "stable_diffusion",
                    "gaussian_diffusion",
                ]

                if is_diffusion:
                    gradient_stats.update(
                        self.stability_manager.step_diffusion_optimization(
                            model,
                            optimizer,
                            epoch=epoch,
                            noise_schedule_factor=noise_schedule_factor,
                            grad_scaler=self,
                        ),
                    )
                    # If stability manager handled diffusion optimization, it already called
                    # grad_scaler.step() and grad_scaler.update(), so we should not call
                    # self.scaler.step() again to avoid the "unscale_() after step()"
                    # error
                    self.step_count += 1
                    return gradient_stats
                gradient_stats.update(
                    self.stability_manager.step_optimization(
                        model,
                        optimizer,
                        grad_scaler=self,
                        model_name=model_name,
                        epoch=epoch,
                        model_type=model_type,
                    ),
                )
                # If stability manager handled optimization, it already called
                # grad_scaler.step() and grad_scaler.update(), so we should not call
                # self.scaler.step() again to avoid the "unscale_() after step()"
                # error
                self.step_count += 1
                return gradient_stats

            # Emergency checks for non-finite gradients
            param_count = 0
            for group in optimizer.param_groups:
                for p in group.get("params", []):
                    if p is None or p.grad is None:
                        continue
                    g = p.grad.data
                    if torch.any(torch.isnan(g)) or torch.any(torch.isinf(g)):
                        logging.warning(
                            f"Non-finite gradients detected for {model_name}; resetting optimizer state",
                        )
                        optimizer.zero_grad(set_to_none=True)
                        gradient_stats["emergency_state_reset"] = True
                        return gradient_stats
                    param_count += 1

            # Use the scaler's step method, which handles skipping on non-finite gradients
            # The scaler returns the scale value for the next iteration, or None if
            # the step was skipped
            self.scaler.step(optimizer)
            self.step_count += 1
            return gradient_stats

        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logging.exception(f"Error in GradScaler step for {model_name}: {e}")
            try:
                optimizer.step()
            except (RuntimeError, ValueError, TypeError) as step_error:
                logging.exception(
                    f"Emergency: Failed to step optimizer for {model_name}: {step_error}",
                )
                optimizer.zero_grad(set_to_none=True)
        return gradient_stats

    def update(self, model_name: str = "model") -> None:
        """Safely perform scaler update with proper state management"""
        try:
            # Only update if scaler is enabled AND scaling was actually used
            if self.is_enabled() and self.scale_calls > 0:
                self.scaler.update()
                # Reset counters after successful update
                self.scale_calls = 0
                self.step_count = 0
            else:
                # For disabled scaler or when no scaling was done, just reset
                # counters
                self.scale_calls = 0
                self.step_count = 0
        except AssertionError as e:
            if "No inf checks were recorded" in str(e) or "_scale is None" in str(e):
                logging.warning(
                    f"GradScaler update assertion error for {model_name}, resetting state",
                )
                # Reset state and try to recover
                self.scale_calls = 0
                self.step_count = 0
            else:
                raise e
        except (RuntimeError, ValueError, TypeError) as e:
            logging.exception(f"Error in GradScaler update for {model_name}: {e}")
            # Reset state on any error
            self.scale_calls = 0
            self.step_count = 0

    def get_gradient_stats(self) -> dict[str, Any]:
        """Get gradient stability statistics if available"""
        if self.stability_manager is not None:
            return self.stability_manager.get_comprehensive_stats()
        return {}


def feature_matching_loss_fn(
    real_features_list: list[torch.Tensor], fake_features_list: list[torch.Tensor]
) -> torch.Tensor:
    """feature_matching_loss_fn.

    Args:
        real_features_list (list[torch.Tensor]): Description.
        fake_features_list (list[torch.Tensor]): Description.
    Returns:
        torch.Tensor: Description.
    """
    # Place the accumulator on the same device as the features rather than a
    # module-global (the inputs are the source of truth for device placement).
    if real_features_list:
        device = real_features_list[0].device
    elif fake_features_list:
        device = fake_features_list[0].device
    else:
        device = get_default_device()
    loss = torch.tensor(0.0, device=device)
    if (
        real_features_list
        and fake_features_list
        and len(real_features_list) == len(fake_features_list)
    ):
        # SSOT: Use LossRegistry via wrapper
        from mriforge.models.losses import create_loss

        criterion_l1 = create_loss("l1").to(device, non_blocking=True)
        for real_feat, fake_feat in zip(real_features_list, fake_features_list, strict=False):
            loss += criterion_l1(fake_feat, real_feat.detach())
    return loss


def handle_training_error(error: Exception, batch_idx: int, epoch: int, model_type: str) -> str:
    """Handle training errors gracefully"""
    error_msg = str(error)

    # Common error types and their fixes
    if "Input type" in error_msg and "weight type" in error_msg:
        logging.error(f"Device mismatch error in batch {batch_idx}: {error_msg}")
        return "device_mismatch"
    if "criterion_gan_loss" in error_msg:
        logging.error(
            f"Loss function error in batch {batch_idx}: Fixed in newer version",
        )
        return "loss_function_error"
    if "invalid syntax" in error_msg:
        logging.error(f"Syntax error in batch {batch_idx}: Profiling code issue")
        return "syntax_error"
    logging.error(f"Unknown error in batch {batch_idx}: {error_msg}")
    return "unknown_error"


_DEFAULT_DEVICE: "torch.device | None" = None


def get_default_device() -> "torch.device":
    """Lazily resolve (and cache) the project default device.

    Replaces the old module-scope ``device = initialize_device()`` which ran
    at *import* time — initialising CUDA, capping the process at 85 % of GPU
    memory, and flipping ``cudnn.benchmark = True`` as a side effect for every
    importer (CLI, ``mriforge audit``, the test suite), despite having zero
    consumers of the resulting global. Callers that genuinely need the device
    call this; everyone else pays nothing at import.
    """
    global _DEFAULT_DEVICE
    if _DEFAULT_DEVICE is None:
        _DEFAULT_DEVICE = initialize_device()
    return _DEFAULT_DEVICE


# Force a safe default dtype across the project to avoid unintended half casts
try:
    torch.set_default_dtype(torch.float32)
except (AttributeError, RuntimeError) as _exc:
    logging.getLogger(__name__).debug("Suppressed exception: %s", _exc)
