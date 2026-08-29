#!/usr/bin/env python3
"""GAN Balance Manager
===================

This module provides comprehensive discriminator/generator balance management
to prevent discriminator overpowering and exploding gradients in GAN training.

Key Features:
- Dynamic discriminator training frequency control
- Adaptive learning rate adjustment
- Discriminator regularization and noise injection
- Model-specific gradient clipping optimization
- Emergency balance recovery mechanisms

Author: MRIForge Research Project
Date: August 2025
"""

import logging
from typing import Any

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


class GANBalanceManager:
    """Comprehensive GAN balance manager to prevent discriminator overpowering
    and maintain stable training dynamics.
    """

    def __init__(
        self,
        model_type: str,
        device: torch.device,
        window_size: int = 20,
        balance_threshold: float = 0.1,
        emergency_threshold: float = 0.05,
        min_d_loss: float = 0.1,
        max_g_d_ratio: float = 10.0,
    ):
        """__init__.

        Args:
            model_type (str): Description.
            device (torch.device): Description.
            window_size (int): Description.
            balance_threshold (float): Description.
            emergency_threshold (float): Description.
            min_d_loss (float): Description.
            max_g_d_ratio (float): Description.
        """
        self.model_type = model_type.lower()
        self.device = device
        self.window_size = window_size
        self.balance_threshold = balance_threshold
        self.emergency_threshold = emergency_threshold
        self.min_d_loss = min_d_loss
        self.max_g_d_ratio = max_g_d_ratio

        # Loss tracking (Tensor Ring Buffers)
        self.d_loss_history = torch.zeros(window_size, device=device)
        self.g_loss_history = torch.zeros(window_size, device=device)
        self.history_ptr = 0
        self.history_size = 0

        # Balance state
        self.d_skip_count = 0
        self.total_steps = 0
        self.emergency_mode = False
        self.d_lr_reduction_factor = 1.0
        self.consecutive_d_wins = 0

        # Model-specific configuration
        self.configure_for_model_type()

        logger.info(f"🎯 GANBalanceManager initialized for {model_type}")
        logger.info(f"   Balance threshold: {balance_threshold}")
        logger.info(f"   Emergency threshold: {emergency_threshold}")
        logger.info(f"   Gradient clip: {self.gradient_clip_norm}")

    def configure_for_model_type(self):
        """Configure balance parameters based on model type."""
        if "kan" in self.model_type:
            # KAN models - standard clipping with careful tuning
            self.gradient_clip_norm = 1.0  # Standard clipping
            self.d_handicap_factor = 0.5  # Moderate D learning rate reduction
            self.noise_injection_strength = 0.02
            self.balance_threshold = 0.1  # Reasonable balance threshold
            self.emergency_threshold = 0.05

        elif any(x in self.model_type for x in ["vit", "swin", "transformer"]):
            # Transformer models with standard gradient control
            self.gradient_clip_norm = 1.0  # Standard clipping
            self.d_handicap_factor = 0.7
            self.noise_injection_strength = 0.01
            self.balance_threshold = 0.1
            self.emergency_threshold = 0.05

        elif "unet" in self.model_type:
            # U-Net models with standard settings
            self.gradient_clip_norm = 1.0  # Standard clipping
            self.d_handicap_factor = 0.8
            self.noise_injection_strength = 0.01
            self.balance_threshold = 0.1
            self.emergency_threshold = 0.05

        elif "diffusion" in self.model_type:
            # Diffusion models have different dynamics
            self.gradient_clip_norm = 0.1
            self.d_handicap_factor = 0.8
            self.noise_injection_strength = 0.01

        else:
            # Standard models - still more aggressive than before
            self.gradient_clip_norm = 0.2
            self.d_handicap_factor = 0.7
            self.noise_injection_strength = 0.02

    def update_losses(
        self,
        d_loss: float,
        g_loss: float,
        gradient_norm: float | None = None,
    ):
        """Update loss history and balance state (Optimized).

        Optimized: Uses GPU-resident tensor ring buffers to avoid CPU sync.
        """
        # Convert to tensor if float, or detach if tensor
        if isinstance(d_loss, (float, int)):
            d_val = torch.tensor(abs(d_loss), device=self.device)
        else:
            d_val = d_loss.detach().abs()

        if isinstance(g_loss, (float, int)):
            g_val = torch.tensor(abs(g_loss), device=self.device)
        else:
            g_val = g_loss.detach().abs()

        # Update ring buffers
        self.d_loss_history[self.history_ptr] = d_val
        self.g_loss_history[self.history_ptr] = g_val

        self.history_ptr = (self.history_ptr + 1) % self.window_size
        self.history_size = min(self.history_size + 1, self.window_size)

        self.total_steps += 1

        # Update consecutive discriminator wins
        # We need the value for logic, but we can check it on GPU or sync just this scalar
        # Syncing one scalar is fine, syncing the whole history is bad.
        d_loss_val = d_val.item()

        if d_loss_val < self.balance_threshold:
            self.consecutive_d_wins += 1
        else:
            self.consecutive_d_wins = 0

        # Check for emergency mode
        if d_loss_val < self.emergency_threshold or self.consecutive_d_wins > 10:
            if not self.emergency_mode:
                logger.warning(
                    f"🚨 EMERGENCY MODE ACTIVATED: D_loss={d_loss_val:.6f}, consecutive_wins={self.consecutive_d_wins}",
                )
            self.emergency_mode = True
        elif self.emergency_mode and d_loss_val > self.balance_threshold * 2:
            logger.info("[OK] Emergency mode deactivated - balance improving")
            self.emergency_mode = False
            self.consecutive_d_wins = 0

    def should_skip_discriminator_update(self) -> bool:
        """Determine if discriminator update should be skipped."""
        if self.history_size < 5:
            return False

        # Calculate mean of last 5 entries on GPU
        start_idx = (self.history_ptr - 5) % self.window_size
        # Handling wrap-around for slice is tricky with ring buffer.
        # Easier to just gather indices or use a mask, but for small window=5,
        # we can just take the last 5 added.
        # Actually, since we just need the mean, we can maintain a running sum or just
        # iterate. But for speed, let's just grab the last 5 logical elements.

        # Simpler approach:
        # indices = [(self.history_ptr - i - 1) % self.window_size for i in range(5)]
        # recent_d_loss = self.d_loss_history[indices].mean().item()

        # Even simpler: just use the last 5 values if we don't care about order (mean is order-independent)
        # But ring buffer is not sorted by time in memory.

        indices = (
            torch.arange(self.history_ptr - 5, self.history_ptr, device=self.device)
            % self.window_size
        )
        recent_d_loss = self.d_loss_history[indices].mean().item()

        # Skip if discriminator is winning too much
        if recent_d_loss < self.balance_threshold:
            skip_prob = min(
                0.8,
                float(
                    (self.balance_threshold - recent_d_loss) / self.balance_threshold,
                ),
            )

            if self.emergency_mode:
                skip_prob = min(
                    0.9,
                    skip_prob * 2,
                )  # More aggressive skipping in emergency

            if np.random.random() < skip_prob:
                self.d_skip_count += 1
                logger.debug(
                    f"⏸️ Skipping D update (prob={skip_prob:.2f}, d_loss={recent_d_loss:.4f})",
                )
                return True

        return False

    def get_discriminator_lr_factor(self) -> float:
        """Get learning rate reduction factor for discriminator."""
        if self.history_size < 5:
            return 1.0

        indices = (
            torch.arange(self.history_ptr - 5, self.history_ptr, device=self.device)
            % self.window_size
        )
        recent_d_loss = self.d_loss_history[indices].mean().item()

        # Reduce discriminator learning rate if it's winning
        if recent_d_loss < self.balance_threshold:
            reduction = 1.0 - min(
                0.8,
                float(
                    (self.balance_threshold - recent_d_loss) / self.balance_threshold,
                ),
            )

            if self.emergency_mode:
                reduction *= 0.5  # Even more reduction in emergency

            self.d_lr_reduction_factor = reduction
            return reduction

        # Gradually restore learning rate when balance improves
        if self.d_lr_reduction_factor < 1.0:
            self.d_lr_reduction_factor = min(1.0, self.d_lr_reduction_factor + 0.05)

        return self.d_lr_reduction_factor

    def apply_discriminator_regularization(
        self,
        discriminator: nn.Module,
        real_samples: torch.Tensor,
        lambda_reg: float = 0.1,
    ) -> torch.Tensor:
        """Apply R1 regularization to discriminator."""
        if self.history_size < 3:
            return torch.tensor(0.0, device=self.device)

        indices = (
            torch.arange(self.history_ptr - 3, self.history_ptr, device=self.device)
            % self.window_size
        )
        recent_d_loss = self.d_loss_history[indices].mean().item()

        # Only apply when discriminator is winning
        if recent_d_loss > self.balance_threshold:
            return torch.tensor(0.0, device=self.device)

        # Increase regularization strength when discriminator is overpowering
        reg_strength = lambda_reg * (
            1.0 + (self.balance_threshold - recent_d_loss) / self.balance_threshold
        )

        if self.emergency_mode:
            reg_strength *= 2.0  # Stronger regularization in emergency

        # R1 penalty
        real_samples.requires_grad_(True)
        real_pred = discriminator(real_samples)

        # Handle different discriminator output shapes
        if real_pred.dim() > 1:
            real_pred = real_pred.mean()

        gradients = torch.autograd.grad(
            outputs=real_pred,
            inputs=real_samples,
            grad_outputs=torch.ones_like(real_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = gradients.view(gradients.size(0), -1).norm(2, dim=1).pow(2).mean()

        return torch.tensor(float(reg_strength), device=self.device) * gradient_penalty

    def inject_discriminator_noise(
        self,
        discriminator_output: torch.Tensor,
    ) -> torch.Tensor:
        """Inject noise to discriminator output when it's too confident."""
        if self.history_size < 3:
            return discriminator_output

        indices = (
            torch.arange(self.history_ptr - 3, self.history_ptr, device=self.device)
            % self.window_size
        )
        recent_d_loss = self.d_loss_history[indices].mean().item()

        # Only inject noise when discriminator is winning
        if recent_d_loss > self.balance_threshold:
            return discriminator_output

        # Calculate noise strength based on how much discriminator is winning
        noise_strength = (
            self.noise_injection_strength
            * (self.balance_threshold - recent_d_loss)
            / self.balance_threshold
        )

        if self.emergency_mode:
            noise_strength *= 2.0

        noise = torch.randn_like(discriminator_output) * float(noise_strength)

        logger.debug(f"💧 Injecting D noise: strength={noise_strength:.4f}")

        return discriminator_output + noise

    def get_adaptive_gradient_clip_norm(self, current_gradient_norm: float) -> float:
        """Get adaptive gradient clipping norm based on current gradients."""
        base_norm = self.gradient_clip_norm

        # Standard adaptive clipping - only adjust for very large gradients
        if current_gradient_norm > 20.0:
            # Major explosion - moderate adjustment
            adaptive_norm = base_norm * 0.5  # 50% reduction
        elif current_gradient_norm > 10.0:
            # Moderate explosion - minor adjustment
            adaptive_norm = base_norm * 0.7  # 30% reduction
        else:
            # Normal range - use base norm
            adaptive_norm = base_norm

        # Ensure minimum clipping value for stability
        adaptive_norm = max(adaptive_norm, 0.1)

        if current_gradient_norm > 10.0:
            logger.info(
                f" Adaptive clipping: grad_norm={current_gradient_norm:.3f} "
                f"→ clip_norm={adaptive_norm:.3f}",
            )

        return adaptive_norm

    def check_gradient_health(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        current_grad_norm: float,
    ) -> bool:
        """Check gradient health and apply simple recovery if needed."""
        if current_grad_norm > 50.0:
            logger.warning(f"⚠️  Large gradient norm detected: {current_grad_norm:.1f}")

            # Simple recovery: zero gradients and continue
            optimizer.zero_grad(set_to_none=True)

            # Log but don't apply emergency measures
            logger.info("🔄 Gradients zeroed, continuing training normally")
            return True

        return False

    def get_balance_status(self) -> dict[str, Any]:
        """Get current balance status and statistics."""
        if self.history_size == 0:
            return {"balance_status": "initializing"}

        limit = min(5, self.history_size)
        indices = (
            torch.arange(self.history_ptr - limit, self.history_ptr, device=self.device)
            % self.window_size
        )

        recent_d_loss = self.d_loss_history[indices].mean().item()
        recent_g_loss = self.g_loss_history[indices].mean().item()

        g_d_ratio = float(recent_g_loss) / max(float(recent_d_loss), 1e-8)

        status = {
            "d_loss_avg": recent_d_loss,
            "g_loss_avg": recent_g_loss,
            "g_d_ratio": g_d_ratio,
            "d_skip_rate": self.d_skip_count / max(self.total_steps, 1),
            "emergency_mode": self.emergency_mode,
            "consecutive_d_wins": self.consecutive_d_wins,
            "d_lr_factor": self.d_lr_reduction_factor,
            "gradient_clip_norm": self.gradient_clip_norm,
        }

        # Determine balance status
        if self.emergency_mode:
            status["balance_status"] = "EMERGENCY"
        elif recent_d_loss < self.balance_threshold:
            status["balance_status"] = "D_WINNING"
        elif g_d_ratio > self.max_g_d_ratio:
            status["balance_status"] = "G_STRUGGLING"
        else:
            status["balance_status"] = "BALANCED"

        return status

    def emergency_reset(self, discriminator_optimizer: torch.optim.Optimizer):
        """Emergency reset when balance is severely compromised."""
        logger.warning("🚨 EMERGENCY BALANCE RESET")

        # Reset discriminator learning rate to very low value
        for param_group in discriminator_optimizer.param_groups:
            param_group["lr"] *= 0.1

        # Reset ring buffer tracking (zero-fill and reset pointers)
        self.d_loss_history.zero_()
        self.g_loss_history.zero_()
        self.history_ptr = 0
        self.history_size = 0

        # Reset counters
        self.consecutive_d_wins = 0
        self.d_skip_count = 0
        self.d_lr_reduction_factor = 0.1

        logger.info("[OK] Emergency reset complete")


def create_balance_manager(model_type: str, device: torch.device) -> GANBalanceManager:
    """Factory function to create a properly configured balance manager."""
    return GANBalanceManager(
        model_type=model_type,
        device=device,
        window_size=20,
        balance_threshold=0.1,
        emergency_threshold=0.05,
    )
