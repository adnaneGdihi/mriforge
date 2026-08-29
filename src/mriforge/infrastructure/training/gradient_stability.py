"""Advanced Gradient Stability and Optimization Module
===================================================

Provides a comprehensive suite of tools to ensure stable and efficient training
for deep learning models, especially GANs and Diffusion models.

Features:
- GradientStabilityManager: A central coordinator for all stability features.
- Adaptive Gradient Clipping (AGC): Clips gradients based on parameter norms.
- Lookahead Optimizer: Improves optimizer stability and convergence.
- EnhancedLRScheduler: Factory for advanced learning rate schedulers like OneCycleLR.
- GradientMonitor: Detailed monitoring of gradient statistics.
- SpectralNormalizationManager: Applies and removes spectral normalization.
- WeightInitializer: A collection of standard weight initialization schemes.
- Gradient Noise Injection: Adds noise to gradients for regularization.

Author: MRIForge Research Project (with Gemini Code Assist)
Date: August 2025
"""

import logging
from collections import defaultdict
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    OneCycleLR,
    StepLR,
)
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


# --- 1. Lookahead Optimizer Wrapper ---
# ``Lookahead`` moved to ``optimizers/lookahead.py`` — the canonical home for
# optimizer classes, beside the registry that dispatches them. Re-exported here
# because this was its historical import path and its tests reference it.
from mriforge.infrastructure.training.optimizers.lookahead import (  # noqa: E402
    Lookahead,
)


def is_diffusion_model(model_type: str) -> bool:
    """Lightweight detection of diffusion-like model types.

    Returns True if the string contains 'diffusion' or common aliases.
    """
    if not model_type:
        return False
    mt = str(model_type).lower()
    if "diffusion" in mt or "ddpm" in mt:
        return True
    # Common explicit variants used across the codebase/tests
    prefixes = ("stable", "score_based", "chi_square", "denoising", "laplace", "rician")
    return any(mt.startswith(p) and "diffusion" in mt for p in prefixes)


def apply_diffusion_gradient_fixes(
    model: nn.Module,
    total_grad_norm: float,
    model_type: str,
) -> None:
    """Apply conservative gradient fixes for diffusion models.

    - Removes NaN/Inf gradients
    - Applies adaptive gradient clipping with a mild factor
    - Slightly stronger clipping for KAN diffusion variants
    """
    if model is None:
        return

    # 1) Sanitize NaN/Inf gradients
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad
        if torch.isnan(g).any() or torch.isinf(g).any():
            g.data = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    # 2) Adaptive clipping (mild)
    mt = (model_type or "").lower()
    clip_factor = 0.01
    if "kan" in mt:
        clip_factor = 0.008  # slightly stronger for KAN diffusion
    adaptive_gradient_clip_(model.parameters(), clip_factor=clip_factor, eps=1e-3)

    # 3) Preferentially cap certain heads if present
    critical_names = ("time", "noise", "score", "adaptive")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if any(k in name.lower() for k in critical_names):
                p.grad.clamp_(min=-1.0, max=1.0)


# --- 2. Adaptive Gradient Clipping (AGC) ---
def unitwise_norm(x: torch.Tensor, norm_type: float = 2.0):
    """Computes norms of each output unit separately."""
    if x.ndim <= 1:
        return x.norm(norm_type)
    # norms of matrices
    return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)


def adaptive_gradient_clip_(parameters, clip_factor: float = 0.01, eps: float = 1e-3):
    """Clips gradients based on the ratio of gradient norms to parameter norms.
    Reference: https://arxiv.org/abs/2102.06171
    """
    for p in parameters:
        if p.grad is None:
            continue

        p_data = p.detach()
        g_data = p.grad.detach()

        # AGC (Brock et al. 2021): eps floors the *parameter* norm so a
        # zero-init param yields a tiny but non-zero clipping ceiling instead
        # of annihilating the gradient. The grad-norm divisor only needs a
        # tiny floor to avoid a 0/0 when the gradient itself is zero.
        param_norm = torch.maximum(unitwise_norm(p_data), torch.tensor(eps, device=g_data.device))
        max_norm = param_norm * clip_factor
        grad_norm = unitwise_norm(g_data)

        clipped_grad = g_data * (
            max_norm / torch.maximum(grad_norm, torch.tensor(1e-12, device=g_data.device))
        )

        # Only clip where grad_norm > max_norm
        p.grad.detach().copy_(torch.where(grad_norm > max_norm, clipped_grad, g_data))


# --- 3. Gradient Monitor ---
class GradientMonitor:
    """Monitors gradient statistics during training."""

    def __init__(self, log_frequency: int = 100):
        """__init__.

        Args:
            log_frequency (int): Description.
        """
        self.log_frequency = log_frequency
        self.step_count = 0

    def monitor(self, model: nn.Module, model_name: str):
        """Monitor gradient statistics with sparse GPU→CPU syncing.

        ✅ PERFORMANCE FIX: Only sync tensors to CPU every log_frequency steps.
        This prevents blocking the GPU on every training iteration.
        See: SKILL_PERFORMANCE.md - Priority 0 (GPU Synchronization)
        """
        self.step_count += 1
        if self.step_count % self.log_frequency != 0:
            return {}

        total_norm = torch.tensor(0.0, device=next(model.parameters()).device)
        nan_grads = 0
        zero_grads = 0
        total_params = 0

        for _name, p in model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm**2
                nan_grads += torch.isnan(p.grad).sum()
                zero_grads += (p.grad == 0).sum()
                total_params += p.grad.numel()

        # ✅ PERF FIX: Convert to Python scalars only at log_frequency intervals
        # This defers GPU→CPU sync to reduce iteration overhead (10-20% speedup).
        # On non-reporting steps, the method returns {} above (zero overhead).
        total_norm = total_norm.sqrt().item()
        nan_grads = int(nan_grads.item())
        zero_grads = int(zero_grads.item())
        stats = {
            f"{model_name}_total_grad_norm": total_norm,
            f"{model_name}_nan_grad_count": nan_grads,
            f"{model_name}_zero_grad_ratio": (zero_grads / total_params if total_params > 0 else 0),
        }
        logger.debug(
            f"Grad Stats ({model_name}): Norm={total_norm:.4f}, Zeros={(stats[f'{model_name}_zero_grad_ratio']):.2%}",
        )
        return stats


# --- 4. Spectral Normalization Manager ---
class SpectralNormalizationManager:
    """Applies and removes spectral normalization from specified layers."""

    def apply_spectral_norm(self, model: nn.Module, apply_to: list[str] = None):
        """apply_spectral_norm.

        Args:
            model (nn.Module): Description.
            apply_to (list[str]): Description.
        Returns:
            Any: Description.
        """
        if apply_to is None:
            apply_to = ["Conv2d", "Linear"]

        applied_count = 0
        for name, module in model.named_modules():
            if module.__class__.__name__ in apply_to:
                # Check if spectral norm is already applied to avoid
                # double registration
                if not hasattr(module, "weight_u"):
                    try:
                        nn.utils.spectral_norm(module)
                        applied_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to apply spectral norm to {name}: {e}")
                else:
                    logger.debug(f"Spectral norm already applied to {name}, skipping")

        logger.debug(
            f"Applied spectral normalization to {applied_count} layers: {apply_to}",
        )

    def remove_spectral_norm(self, model: nn.Module, apply_to: list[str] = None):
        """remove_spectral_norm.

        Args:
            model (nn.Module): Description.
            apply_to (list[str]): Description.
        Returns:
            Any: Description.
        """
        if apply_to is None:
            apply_to = ["Conv2d", "Linear"]
        for _name, module in model.named_modules():
            if module.__class__.__name__ in apply_to and hasattr(module, "weight_g"):
                nn.utils.remove_spectral_norm(module)
        logger.debug(f"Removed spectral normalization from layers: {apply_to}")


# --- 5. Weight Initializer ---
class WeightInitializer:
    """Provides various weight initialization schemes."""

    def __init__(self, init_method: str = "he_normal"):
        """__init__.

        Args:
            init_method (str): Description.
        """
        self.init_method = init_method

    def initialize(self, model: nn.Module):
        """initialize.

        Args:
            model (nn.Module): Description.
        Returns:
            Any: Description.
        """
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if self.init_method == "he_normal":
                    nn.init.kaiming_normal_(
                        m.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )
                elif self.init_method == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)


# --- 6. Enhanced LR Scheduler Factory ---
class EnhancedLRScheduler:
    """Factory for creating advanced learning rate schedulers."""

    def __init__(
        self,
        strategy: str = "cosine_annealing",
        total_epochs: int = 100,
        **kwargs,
    ):
        """__init__.

        Args:
            strategy (str): Description.
            total_epochs (int): Description.
        """
        self.strategy = strategy
        self.total_epochs = total_epochs
        self.kwargs = kwargs

    def create(self, optimizer: Optimizer, steps_per_epoch: int):
        """create.

        Args:
            optimizer (Optimizer): Description.
            steps_per_epoch (int): Description.
        Returns:
            Any: Description.
        """
        if self.strategy == "one_cycle":
            return OneCycleLR(
                optimizer,
                max_lr=optimizer.param_groups[0]["lr"],
                steps_per_epoch=steps_per_epoch,
                epochs=self.total_epochs,
                **self.kwargs,
            )
        if self.strategy == "cosine_annealing":
            return CosineAnnealingLR(optimizer, T_max=self.total_epochs, **self.kwargs)
        if self.strategy == "exponential":
            return ExponentialLR(optimizer, gamma=self.kwargs.get("gamma", 0.95))
        if self.strategy == "step":
            return StepLR(
                optimizer,
                step_size=self.kwargs.get("step_size", 30),
                gamma=self.kwargs.get("gamma", 0.1),
            )
        raise ValueError(
            f"Unknown scheduler strategy {self.strategy!r}. "
            "Known: 'one_cycle', 'cosine_annealing', 'exponential', 'step'."
        )


# --- 7. Gradient Stability Manager ---
class GradientStabilityManager:
    """Central coordinator for gradient stability features.
    This class delegates tasks to specialized components.
    """

    def __init__(
        self,
        clip_gradients: bool = True,
        clip_method: str = "norm",
        clip_value: float = 1.0,
        agc_clip_factor: float = 0.01,
        use_spectral_norm: bool = False,
        spectral_norm_layers: list[str] | None = None,
        monitor_gradients: bool = True,
        monitor_log_frequency: int = 100,
        init_method: str = "he_normal",
        lr_strategy: str = "cosine_annealing",
        total_epochs: int = 100,
        add_gradient_noise: float = 0.0,
    ):
        """__init__.

        Args:
            clip_gradients (bool): Description.
            clip_method (str): Description.
            clip_value (float): Description.
            agc_clip_factor (float): Description.
            use_spectral_norm (bool): Description.
            spectral_norm_layers (Optional[list[str]]): Description.
            monitor_gradients (bool): Description.
            monitor_log_frequency (int): Description.
            init_method (str): Description.
            lr_strategy (str): Description.
            total_epochs (int): Description.
            add_gradient_noise (float): Description.
        """
        self.clip_gradients = clip_gradients
        self.clip_method = clip_method
        self.clip_value = clip_value
        self.agc_clip_factor = agc_clip_factor
        self.add_gradient_noise = add_gradient_noise

        self.monitor = GradientMonitor(monitor_log_frequency) if monitor_gradients else None
        self.spectral_manager = SpectralNormalizationManager() if use_spectral_norm else None
        self.initializer = WeightInitializer(init_method)
        self.scheduler_factory = EnhancedLRScheduler(lr_strategy, total_epochs)

        if use_spectral_norm and spectral_norm_layers is None:
            self.spectral_norm_layers = ["Conv2d", "Linear"]
        else:
            self.spectral_norm_layers = spectral_norm_layers

    def initialize_weights(self, model: nn.Module):
        """Initialize model weights."""
        self.initializer.initialize(model)

    def apply_spectral_norm(self, model: nn.Module):
        """Apply spectral normalization to the model."""
        if self.spectral_manager:
            self.spectral_manager.apply_spectral_norm(model, self.spectral_norm_layers)

    def create_scheduler(self, optimizer: Optimizer, steps_per_epoch: int):
        """Create a learning rate scheduler."""
        return self.scheduler_factory.create(optimizer, steps_per_epoch)

    def create_optimizer(
        self,
        params,
        optimizer_class=torch.optim.Adam,
        use_lookahead=False,
        **kwargs,
    ):
        """Create an optimizer, optionally wrapped with Lookahead."""
        # [LOGGING] Trace creation
        logger.info(f"Creating Optimizer: {optimizer_class.__name__} | Params: {kwargs}")

        optimizer = optimizer_class(params, **kwargs)
        if use_lookahead:
            optimizer = Lookahead(optimizer)
            logger.debug("Optimizer wrapped with Lookahead.")
        return optimizer

    def _perform_gradient_clipping(self, model: nn.Module):
        """Internal method to apply the configured gradient clipping."""
        if not self.clip_gradients:
            return

        params = [p for p in model.parameters() if p.grad is not None]
        if not params:
            return

        if self.clip_method == "norm":
            nn.utils.clip_grad_norm_(params, max_norm=self.clip_value)
        elif self.clip_method == "value":
            nn.utils.clip_grad_value_(params, clip_value=self.clip_value)
        elif self.clip_method == "agc":
            adaptive_gradient_clip_(params, clip_factor=self.agc_clip_factor)
        else:
            raise ValueError(
                f"Unknown clip_method {self.clip_method!r}. Supported: 'norm', 'value', 'agc'."
            )

    def _inject_gradient_noise(self, model: nn.Module):
        """Internal method to inject noise into gradients."""
        if self.add_gradient_noise <= 0.0:
            return

        for p in model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * self.add_gradient_noise
                p.grad.add_(noise)

    def step_optimization(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        grad_scaler: torch.amp.GradScaler | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Performs a full, stabilized optimization step.

        This includes unscaling gradients (if using AMP), clipping, monitoring,
        and stepping the optimizer.

        Returns:
            A dictionary of gradient statistics.

        """
        stats = {}

        # Unscale gradients before clipping if using GradScaler
        if grad_scaler is not None and grad_scaler.is_enabled():
            grad_scaler.unscale_(optimizer)

        # 1. Perform gradient clipping
        self._perform_gradient_clipping(model)

        # 2. Inject gradient noise
        self._inject_gradient_noise(model)

        # 3. Step the optimizer (with GradScaler if provided)
        if grad_scaler is not None and grad_scaler.is_enabled():
            # Route through grad_scaler.step() (NOT a bare optimizer.step()):
            # we already called unscale_() above for clipping, and scaler.step()
            # detects that, runs the inf/NaN check, and SKIPS the step on an
            # overflow instead of corrupting the weights.
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()

        # 4. Monitor gradients after the step
        if self.monitor:
            stats.update(self.monitor.monitor(model, kwargs.get("model_name", "model")))

        return stats

    def step_diffusion_optimization(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int = 0,
        noise_schedule_factor: float = 1.0,
        grad_scaler: torch.amp.GradScaler | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Performs an optimization step optimized for diffusion models.

        For diffusion models, we apply gradient stability measures to the generator
        but handle the discriminator (diffusion process) appropriately.

        Returns:
            A dictionary of gradient statistics.

        """
        stats = {}

        # Unscale gradients before clipping if using GradScaler
        if grad_scaler is not None and grad_scaler.is_enabled():
            grad_scaler.unscale_(optimizer)

        # 1. Perform gradient clipping (only for the generator model)
        self._perform_gradient_clipping(model)

        # 2. Inject gradient noise
        self._inject_gradient_noise(model)

        # 3. Step the optimizer (with GradScaler if provided)
        if grad_scaler is not None and grad_scaler.is_enabled():
            # Route through grad_scaler.step() (NOT a bare optimizer.step()):
            # we already called unscale_() above for clipping, and scaler.step()
            # detects that, runs the inf/NaN check, and SKIPS the step on an
            # overflow instead of corrupting the weights.
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()

        # 4. Monitor gradients after the step
        if self.monitor:
            stats.update(
                self.monitor.monitor(
                    model,
                    kwargs.get("model_name", "diffusion_generator"),
                ),
            )

        # Add diffusion-specific stats
        stats["epoch"] = epoch
        stats["noise_schedule_factor"] = noise_schedule_factor

        return stats

    def handle_gradient_issues(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        grad_scaler: torch.amp.GradScaler | None = None,
        auto_adjust: bool = True,
        model_name: str = "model",
    ) -> dict[str, Any]:
        """Detects and handles gradient issues like NaN, Inf, or explosion.

        Args:
            auto_adjust (bool): If True, automatically adjusts clipping/LR.

        Returns:
            A dictionary of actions taken.

        """
        actions = defaultdict(bool)
        is_unstable = False

        for p in model.parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                is_unstable = True
                break

        if is_unstable:
            actions["nan_inf_detected"] = True
            optimizer.zero_grad(set_to_none=True)
            logger.warning(
                f"NaN/Inf gradients detected in {model_name}. Optimizer state zeroed.",
            )

            if grad_scaler is not None and grad_scaler.is_enabled():
                # Reduce scaler scale to recover
                grad_scaler.update(new_scale=grad_scaler.get_scale() / 2.0)
                actions["grad_scaler_reduced"] = True
                logger.warning(f"GradScaler scale reduced to: {grad_scaler.get_scale()}")

            if auto_adjust:
                # Make clipping more aggressive
                if self.clip_method == "norm":
                    self.clip_value = max(0.1, self.clip_value * 0.8)
                    actions["clip_value_reduced"] = self.clip_value
                elif self.clip_method == "agc":
                    self.agc_clip_factor = max(0.001, self.agc_clip_factor * 0.8)
                    actions["agc_factor_reduced"] = self.agc_clip_factor

                # Reduce learning rate
                for g in optimizer.param_groups:
                    g["lr"] *= 0.9
                actions["lr_reduced"] = optimizer.param_groups[0]["lr"]
                logger.warning(
                    f"Auto-adjusted stability params for {model_name}: clip_value={self.clip_value}, lr={actions['lr_reduced']}",
                )

        return dict(actions)

    def get_comprehensive_stats(self) -> dict[str, Any]:
        """Returns a comprehensive dictionary of all tracked statistics."""
        stats = {
            "gradient_clipping": {
                "method": self.clip_method,
                "value": self.clip_value,
                "agc_factor": self.agc_clip_factor,
            },
            "gradient_monitoring": {
                "enabled": self.monitor is not None,
                "log_frequency": self.monitor.log_frequency if self.monitor else None,
            },
            "spectral_normalization": {
                "enabled": self.spectral_manager is not None,
                "layers": self.spectral_norm_layers if self.spectral_manager else [],
            },
            "weight_initialization": {
                "method": self.initializer.init_method,
            },
            "lr_scheduler": {
                "strategy": self.scheduler_factory.strategy,
            },
        }
        return stats
