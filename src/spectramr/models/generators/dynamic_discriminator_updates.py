"""Dynamic Discriminator Updates for Adaptive GAN Training

This module implements adaptive discriminator updates based on R1
gradient penalty statistics to modulate the D:G update ratio for
more stable GAN training.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class GradientPenaltyStats:
    """Statistics for gradient penalty monitoring."""

    r1_penalty: float
    gradient_norm: float
    discriminator_loss: float
    generator_loss: float
    step: int


@dataclass
class UpdateRatioConfig:
    """Configuration for dynamic update ratio modulation."""

    base_d_updates: int = 1
    base_g_updates: int = 1
    max_d_updates: int = 5
    min_d_updates: int = 1
    adaptation_window: int = 100
    r1_threshold_high: float = 10.0
    r1_threshold_low: float = 1.0
    smoothing_factor: float = 0.1


class AdaptiveUpdateScheduler:
    """Adaptive scheduler for discriminator-to-generator update ratios.

    Uses R1 gradient penalty statistics to dynamically adjust the number of
    discriminator updates per generator update for stable GAN training.
    """

    def __init__(self, config: UpdateRatioConfig):
        """__init__.

        Args:
            config (UpdateRatioConfig): Description.
        """
        self.config = config
        self.stats_history = deque(maxlen=config.adaptation_window)
        self.current_d_updates = config.base_d_updates
        self.smoothed_r1 = None

    def update_stats(self, stats: GradientPenaltyStats) -> None:
        """Update statistics history with new gradient penalty data."""
        self.stats_history.append(stats)

        # Update smoothed R1 penalty
        if self.smoothed_r1 is None:
            self.smoothed_r1 = stats.r1_penalty
        else:
            self.smoothed_r1 = (
                self.config.smoothing_factor * stats.r1_penalty
                + (1 - self.config.smoothing_factor) * self.smoothed_r1
            )

    def get_update_ratio(self) -> tuple[int, int]:
        """Calculate adaptive D:G update ratio based on recent statistics.

        Returns:
            Tuple of (discriminator_updates, generator_updates)

        """
        if len(self.stats_history) < 10:  # Need minimum history
            return self.config.base_d_updates, self.config.base_g_updates

        # Adaptive discriminator updates based on R1 penalty
        if self.smoothed_r1 > self.config.r1_threshold_high:
            # High R1 penalty indicates discriminator is too strong
            d_updates = max(self.config.min_d_updates, self.current_d_updates - 1)
        elif self.smoothed_r1 < self.config.r1_threshold_low:
            # Low R1 penalty indicates discriminator is too weak
            d_updates = min(self.config.max_d_updates, self.current_d_updates + 1)
        else:
            # R1 penalty in acceptable range
            d_updates = self.config.base_d_updates

        self.current_d_updates = d_updates
        return d_updates, self.config.base_g_updates

    def get_training_stats(self) -> dict[str, Any]:
        """Get current training statistics for monitoring."""
        if not self.stats_history:
            return {}

        recent_stats = list(self.stats_history)[-10:]  # Last 10 steps

        return {
            "current_d_updates": self.current_d_updates,
            "smoothed_r1_penalty": self.smoothed_r1,
            "avg_r1_penalty": (sum(s.r1_penalty for s in recent_stats) / len(recent_stats)),
            "avg_gradient_norm": (sum(s.gradient_norm for s in recent_stats) / len(recent_stats)),
            "avg_d_loss": (sum(s.discriminator_loss for s in recent_stats) / len(recent_stats)),
            "avg_g_loss": (sum(s.generator_loss for s in recent_stats) / len(recent_stats)),
        }


class DynamicDiscriminatorTrainer:
    """GAN trainer with dynamic discriminator updates.

    Implements adaptive discriminator training based on gradient penalty
    statistics to maintain stable G/D balance.
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        g_optimizer: torch.optim.Optimizer,
        d_optimizer: torch.optim.Optimizer,
        config: UpdateRatioConfig,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        ),
    ):
        """__init__.

        Args:
            generator (nn.Module): Description.
            discriminator (nn.Module): Description.
            g_optimizer (torch.optim.Optimizer): Description.
            d_optimizer (torch.optim.Optimizer): Description.
            config (UpdateRatioConfig): Description.
            device (torch.device): Description.
        """
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.device = device

        self.scheduler = AdaptiveUpdateScheduler(config)
        self.step_count = 0

        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.r1_gamma = 10.0  # R1 regularization strength

    def compute_gradient_penalty(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> tuple[float, float]:
        """Compute R1 gradient penalty and gradient norm.

        Args:
            real_images: Real images from dataset
            fake_images: Generated fake images

        Returns:
            Tuple of (r1_penalty, gradient_norm)

        """
        real_images.requires_grad_(True)

        # Get discriminator output for real images
        real_logits = self.discriminator(real_images)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=real_logits.sum(),
            inputs=real_images,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # R1 regularization
        if len(gradients.shape) == 4:  # Image tensors
            r1_penalty = gradients.pow(2).sum(dim=[1, 2, 3]).mean()
            gradient_norm = gradients.norm(dim=[1, 2, 3]).mean().item()
        else:  # Feature vectors or other tensors
            r1_penalty = gradients.pow(2).sum(dim=[1]).mean()
            gradient_norm = gradients.norm(dim=[1]).mean().item()

        return r1_penalty.item(), gradient_norm

    def train_step(
        self,
        real_images: torch.Tensor,
        conditions: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Perform one training step with adaptive discriminator updates.

        Args:
            real_images: Batch of real images
            conditions: Optional conditioning information

        Returns:
            Dictionary of training metrics

        """
        self.step_count += 1

        if real_images is None:
            raise ValueError("real_images cannot be None")

        real_images = real_images.to(self.device)

        # Generate fake images
        with torch.no_grad():
            if conditions is not None:
                conditions = conditions.to(self.device)
                fake_images = self.generator(conditions)
            else:
                fake_images = self.generator(torch.randn_like(real_images))

        # Compute gradient penalty statistics
        r1_penalty, gradient_norm = self.compute_gradient_penalty(
            real_images,
            fake_images,
        )

        # Get adaptive update ratio
        d_updates, g_updates = self.scheduler.get_update_ratio()

        # Train discriminator multiple times if needed
        d_loss_accum = 0.0
        for _ in range(d_updates):
            self.d_optimizer.zero_grad()

            # Real images
            real_logits = self.discriminator(real_images)
            real_loss = self.adversarial_loss(real_logits, torch.ones_like(real_logits))

            # Fake images
            fake_logits = self.discriminator(fake_images.detach())
            fake_loss = self.adversarial_loss(
                fake_logits,
                torch.zeros_like(fake_logits),
            )

            # Total discriminator loss with R1 regularization
            d_loss = (real_loss + fake_loss) / 2 + self.r1_gamma * r1_penalty / 2
            d_loss.backward()
            self.d_optimizer.step()

            d_loss_accum += d_loss.item()

        d_loss_avg = d_loss_accum / d_updates

        # Train generator
        self.g_optimizer.zero_grad()

        fake_logits = self.discriminator(fake_images)
        g_loss = self.adversarial_loss(fake_logits, torch.ones_like(fake_logits))

        g_loss.backward()
        self.g_optimizer.step()

        # Update statistics
        stats = GradientPenaltyStats(
            r1_penalty=r1_penalty,
            gradient_norm=gradient_norm,
            discriminator_loss=d_loss_avg,
            generator_loss=g_loss.item(),
            step=self.step_count,
        )
        self.scheduler.update_stats(stats)

        return {
            "d_loss": d_loss_avg,
            "g_loss": g_loss.item(),
            "r1_penalty": r1_penalty,
            "gradient_norm": gradient_norm,
            "d_updates": d_updates,
            "g_updates": g_updates,
        }

    def get_training_stats(self) -> dict[str, Any]:
        """Get comprehensive training statistics."""
        base_stats = self.scheduler.get_training_stats()
        base_stats.update(
            {
                "total_steps": self.step_count,
                "device": str(self.device),
            },
        )
        return base_stats


def create_adaptive_trainer(
    generator: nn.Module,
    discriminator: nn.Module,
    lr: float = 2e-4,
    config: UpdateRatioConfig | None = None,
) -> DynamicDiscriminatorTrainer:
    """Factory function to create adaptive discriminator trainer.

    Args:
        generator: Generator network
        discriminator: Discriminator network
        lr: Learning rate for both optimizers
        config: Configuration for adaptive updates

    Returns:
        Configured DynamicDiscriminatorTrainer instance

    """
    if config is None:
        config = UpdateRatioConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move models to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    from spectramr.infrastructure.training.builders import OptimizationBuilder

    # Create optimizers
    g_optimizer = OptimizationBuilder.create_single_optimizer(
        generator.parameters(), learning_rate=lr, betas=(0.5, 0.999)
    )
    d_optimizer = OptimizationBuilder.create_single_optimizer(
        discriminator.parameters(),
        learning_rate=lr,
        betas=(0.5, 0.999),
    )

    return DynamicDiscriminatorTrainer(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        config=config,
        device=device,
    )


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = UpdateRatioConfig(
        base_d_updates=1,
        max_d_updates=3,
        adaptation_window=50,
        r1_threshold_high=5.0,
        r1_threshold_low=0.5,
    )

    logger.debug("Dynamic Discriminator Updates Configuration:")
    logger.debug(f"Base D updates: {config.base_d_updates}")
    logger.debug(f"Max D updates: {config.max_d_updates}")
    logger.debug(f"Adaptation window: {config.adaptation_window}")
    logger.debug(f"R1 threshold high: {config.r1_threshold_high}")
    logger.debug(f"R1 threshold low: {config.r1_threshold_low}")

    # Create scheduler
    scheduler = AdaptiveUpdateScheduler(config)

    # Simulate training statistics
    logger.debug("\nSimulating adaptive updates:")
    for step in range(20):
        # Simulate varying R1 penalty
        r1_penalty = 2.0 + 3.0 * torch.sin(torch.tensor(step * 0.5)).item()

        stats = GradientPenaltyStats(
            r1_penalty=r1_penalty,
            gradient_norm=1.0,
            discriminator_loss=0.5,
            generator_loss=1.0,
            step=step,
        )

        scheduler.update_stats(stats)
        d_updates, g_updates = scheduler.get_update_ratio()

        logger.debug(".1f")
