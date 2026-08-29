#!/usr/bin/env python3
"""Latent Model Training Strategies Module
=======================================

This module implements training strategies specifically designed for latent
space models including VAE, VQ-VAE, latent GAN, and latent diffusion models.
"""

from typing import Any

import torch
from torch import nn

from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy
from mriforge.models.losses.latent_losses import (
    LatentDiffusionLoss,
    VQLoss,
    create_beta_kl_loss,
)


class VAETrainingStrategy(BaseTrainingStrategy):
    """Training strategy for Variational Autoencoder (VAE) models.

    This strategy handles the training of VAE models with KL divergence
    regularization and reconstruction loss.

    Args:
        model: VAE model (encoder + decoder)
        optimizer: Optimizer for model parameters
        device: Device to run training on
        beta: Weight for KL divergence loss
        reconstruction_loss: Loss function for reconstruction

    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        beta: float = 1.0,
        reconstruction_loss: nn.Module | None = None,
    ):
        """__init__.

        Args:
            model (nn.Module): Description.
            optimizer (torch.optim.Optimizer): Description.
            device (torch.device): Description.
            beta (float): Description.
            reconstruction_loss (Optional[nn.Module]): Description.
        """
        super().__init__(model, optimizer, device)
        self.beta = beta
        self.kl_loss = create_beta_kl_loss(beta=beta)
        self.reconstruction_loss = reconstruction_loss or nn.MSELoss()

    def train_step(self, batch: torch.Tensor, **kwargs) -> dict[str, float]:
        """Perform a single training step for VAE.

        Args:
            batch: Input batch [B, C, H, W]
            **kwargs: Additional arguments

        Returns:
            Dictionary of loss components

        """
        self.model.train()
        batch = batch.to(self.device)

        # Forward pass
        reconstructed, mu, logvar = self.model(batch)

        # Compute losses
        reconstruction_loss = self.reconstruction_loss(reconstructed, batch)
        kl_loss = self.kl_loss(mu, logvar)
        total_loss = reconstruction_loss + kl_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "kl_loss": kl_loss.item(),
            "beta": self.beta,
        }

    @torch.no_grad()
    def validation_step(self, batch: torch.Tensor, **kwargs) -> dict[str, float]:
        """Perform a validation step for VAE.

        Args:
            batch: Input batch [B, C, H, W]
            **kwargs: Additional arguments

        Returns:
            Dictionary of validation metrics

        """
        self.model.eval()
        batch = batch.to(self.device)

        with torch.no_grad():
            # Forward pass
            reconstructed, mu, logvar = self.model(batch)

            # Compute losses
            reconstruction_loss = self.reconstruction_loss(reconstructed, batch)
            kl_loss = self.kl_loss(mu, logvar)
            total_loss = reconstruction_loss + kl_loss

        return {
            "val_total_loss": total_loss.item(),
            "val_reconstruction_loss": reconstruction_loss.item(),
            "val_kl_loss": kl_loss.item(),
        }


class VQVAETrainingStrategy(BaseTrainingStrategy):
    """Training strategy for Vector Quantized VAE (VQ-VAE) models.

    This strategy handles the training of VQ-VAE models with vector
    quantization and commitment losses.

    Args:
        model: VQ-VAE model
        optimizer: Optimizer for model parameters
        device: Device to run training on
        commitment_weight: Weight for commitment loss
        codebook_weight: Weight for codebook loss

    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        commitment_weight: float = 0.25,
        codebook_weight: float = 1.0,
    ):
        """__init__.

        Args:
            model (nn.Module): Description.
            optimizer (torch.optim.Optimizer): Description.
            device (torch.device): Description.
            commitment_weight (float): Description.
            codebook_weight (float): Description.
        """
        super().__init__(model, optimizer, device)
        self.vq_loss = VQLoss(
            commitment_weight=commitment_weight,
            codebook_weight=codebook_weight,
        )

    def train_step(self, batch: torch.Tensor, **kwargs) -> dict[str, float]:
        """Perform a single training step for VQ-VAE.

        Args:
            batch: Input batch [B, C, H, W]
            **kwargs: Additional arguments

        Returns:
            Dictionary of loss components

        """
        self.model.train()
        batch = batch.to(self.device)

        # Forward pass
        reconstructed, quantized, encodings, embedded = self.model(batch)

        # Compute VQ loss
        total_loss, reconstruction_loss, commitment_loss, codebook_loss = self.vq_loss(
            reconstructed,
            batch,
            quantized,
            encodings,
            embedded,
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "commitment_loss": commitment_loss.item(),
            "codebook_loss": codebook_loss.item(),
        }

    @torch.no_grad()
    def validation_step(self, batch: torch.Tensor, **kwargs) -> dict[str, float]:
        """Perform a validation step for VQ-VAE.

        Args:
            batch: Input batch [B, C, H, W]
            **kwargs: Additional arguments

        Returns:
            Dictionary of validation metrics

        """
        self.model.eval()
        batch = batch.to(self.device)

        with torch.no_grad():
            # Forward pass
            reconstructed, quantized, encodings, embedded = self.model(batch)

            # Compute VQ loss
            (
                total_loss,
                reconstruction_loss,
                commitment_loss,
                codebook_loss,
            ) = self.vq_loss(reconstructed, batch, quantized, encodings, embedded)

        return {
            "val_total_loss": total_loss.item(),
            "val_reconstruction_loss": reconstruction_loss.item(),
            "val_commitment_loss": commitment_loss.item(),
            "val_codebook_loss": codebook_loss.item(),
        }


class LatentGANTrainingStrategy(BaseTrainingStrategy):
    """Enhanced training strategy for latent GAN models with dual discriminators.

    This strategy handles the training of GAN models that operate in
    latent space with both pixel and latent discriminators, R1 penalty,
    and adjustable discriminator-to-generator update ratios.

    Args:
        encoder: Encoder model (LR -> latent)
        generator: Generator model (latent -> HR)
        pixel_discriminator: Discriminator for pixel space (optional)
        latent_discriminator: Discriminator for latent space
        g_optimizer: Optimizer for generator
        d_optimizer: Optimizer for discriminators
        device: Device to run training on
        gan_loss: GAN loss function
        r1_penalty_weight: Weight for R1 gradient penalty
        disc_updates_per_gen: Number of discriminator updates per generator update

    """

    def __init__(
        self,
        encoder: nn.Module,
        generator: nn.Module,
        pixel_discriminator: nn.Module | None,
        latent_discriminator: nn.Module,
        g_optimizer: torch.optim.Optimizer,
        d_optimizer: torch.optim.Optimizer,
        device: torch.device,
        gan_loss: nn.Module | None = None,
        r1_penalty_weight: float = 10.0,
        disc_updates_per_gen: int = 1,
    ):
        # Initialize with generator as the main model for BaseTrainingStrategy
        """__init__.

        Args:
            encoder (nn.Module): Description.
            generator (nn.Module): Description.
            pixel_discriminator (Optional[nn.Module]): Description.
            latent_discriminator (nn.Module): Description.
            g_optimizer (torch.optim.Optimizer): Description.
            d_optimizer (torch.optim.Optimizer): Description.
            device (torch.device): Description.
            gan_loss (Optional[nn.Module]): Description.
            r1_penalty_weight (float): Description.
            disc_updates_per_gen (int): Description.
        """
        super().__init__(generator, g_optimizer, device)
        self.encoder = encoder
        self.generator = generator  # Keep reference for clarity
        self.pixel_discriminator = pixel_discriminator
        self.latent_discriminator = latent_discriminator
        self.d_optimizer = d_optimizer
        self.gan_loss = gan_loss or nn.BCEWithLogitsLoss()
        self.r1_penalty_weight = r1_penalty_weight
        self.disc_updates_per_gen = disc_updates_per_gen

    def _compute_r1_penalty(
        self,
        discriminator: nn.Module,
        real_data: torch.Tensor,
    ) -> torch.Tensor:
        """Compute R1 gradient penalty for discriminator regularization.

        Args:
            discriminator: Discriminator model
            real_data: Real data batch

        Returns:
            R1 penalty loss

        """
        real_data.requires_grad_(True)
        real_pred = discriminator(real_data)
        gradients = torch.autograd.grad(
            outputs=real_pred.sum(),
            inputs=real_data,
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        r1_penalty = torch.mean(torch.sum(gradients**2, dim=1))
        return r1_penalty

    def train_step(
        self,
        lr_batch: torch.Tensor,
        hr_batch: torch.Tensor,
        **kwargs,
    ) -> dict[str, float]:
        """Perform a single training step for latent GAN with dual discriminators.

        Args:
            lr_batch: Low-resolution input batch [B, C, H, W]
            hr_batch: High-resolution target batch [B, C, H, W]
            **kwargs: Additional arguments

        Returns:
            Dictionary of loss components

        """
        self.encoder.train()
        self.generator.train()
        if self.pixel_discriminator:
            self.pixel_discriminator.train()
        self.latent_discriminator.train()

        lr_batch = lr_batch.to(self.device)
        hr_batch = hr_batch.to(self.device)
        batch_size = lr_batch.size(0)

        # Create labels
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        losses = {}

        # ===== Train Discriminators =====
        for _ in range(self.disc_updates_per_gen):
            self.d_optimizer.zero_grad()

            # Encode real LR to latent
            with torch.no_grad():
                real_latent = self.encoder(lr_batch)

            # Generate fake HR from real latent
            fake_hr = self.generator(real_latent)

            d_loss = 0.0

            # Pixel discriminator loss (if available)
            if self.pixel_discriminator:
                # Real pixel data
                pixel_d_real = self.pixel_discriminator(hr_batch)
                d_loss_real_pixel = self.gan_loss(pixel_d_real, real_labels)

                # Fake pixel data
                pixel_d_fake = self.pixel_discriminator(fake_hr.detach())
                d_loss_fake_pixel = self.gan_loss(pixel_d_fake, fake_labels)

                # R1 penalty for pixel discriminator
                r1_pixel = self._compute_r1_penalty(self.pixel_discriminator, hr_batch)

                d_loss += (
                    d_loss_real_pixel + d_loss_fake_pixel + self.r1_penalty_weight * r1_pixel
                ).item()

                losses.update(
                    {
                        "d_loss_real_pixel": d_loss_real_pixel.item(),
                        "d_loss_fake_pixel": d_loss_fake_pixel.item(),
                        "r1_penalty_pixel": r1_pixel.item(),
                    },
                )

            # Latent discriminator loss
            latent_d_real = self.latent_discriminator(real_latent)
            d_loss_real_latent = self.gan_loss(latent_d_real, real_labels)

            # Generate fake latent from noise (for latent discriminator)
            noise = torch.randn(batch_size, real_latent.size(1), device=self.device)
            latent_d_fake = self.latent_discriminator(noise)
            d_loss_fake_latent = self.gan_loss(latent_d_fake, fake_labels)

            # R1 penalty for latent discriminator
            r1_latent = self._compute_r1_penalty(self.latent_discriminator, real_latent)

            d_loss += (
                d_loss_real_latent + d_loss_fake_latent + self.r1_penalty_weight * r1_latent
            ).item()

            losses.update(
                {
                    "d_loss_real_latent": d_loss_real_latent.item(),
                    "d_loss_fake_latent": d_loss_fake_latent.item(),
                    "r1_penalty_latent": r1_latent.item(),
                    "d_loss_total": d_loss.item(),
                },
            )

            d_loss.backward()
            self.d_optimizer.step()

        # ===== Train Generator =====
        self.optimizer.zero_grad()

        # Encode LR to latent
        real_latent = self.encoder(lr_batch)

        # Generate HR from latent
        fake_hr = self.generator(real_latent)

        g_loss = 0.0

        # Pixel discriminator loss (if available)
        if self.pixel_discriminator:
            pixel_d_fake = self.pixel_discriminator(fake_hr)
            g_loss_pixel = self.gan_loss(pixel_d_fake, real_labels)
            g_loss += g_loss_pixel.item()
            losses["g_loss_pixel"] = g_loss_pixel.item()

        # Latent discriminator loss
        latent_d_fake = self.latent_discriminator(real_latent)
        g_loss_latent = self.gan_loss(latent_d_fake, real_labels)
        g_loss += g_loss_latent.item()
        losses["g_loss_latent"] = g_loss_latent.item()

        # Reconstruction loss (optional)
        if "reconstruction_loss" in kwargs:
            recon_loss = kwargs["reconstruction_loss"](fake_hr, hr_batch)
            g_loss += recon_loss.item()
            losses["g_loss_recon"] = recon_loss.item()

        losses["g_loss_total"] = g_loss.item()

        g_loss.backward()
        self.optimizer.step()

        return losses

    @torch.no_grad()
    def validation_step(
        self,
        lr_batch: torch.Tensor,
        hr_batch: torch.Tensor,
        **kwargs,
    ) -> dict[str, float]:
        """Perform a validation step for latent GAN with dual discriminators.

        Args:
            lr_batch: Low-resolution input batch [B, C, H, W]
            hr_batch: High-resolution target batch [B, C, H, W]
            **kwargs: Additional arguments

        Returns:
            Dictionary of validation metrics

        """
        self.encoder.eval()
        self.generator.eval()
        if self.pixel_discriminator:
            self.pixel_discriminator.eval()
        self.latent_discriminator.eval()

        lr_batch = lr_batch.to(self.device)
        hr_batch = hr_batch.to(self.device)
        batch_size = lr_batch.size(0)

        with torch.no_grad():
            # Encode LR to latent
            real_latent = self.encoder(lr_batch)

            # Generate HR from latent
            fake_hr = self.generator(real_latent)

            val_losses = {}

            # Pixel discriminator validation (if available)
            if self.pixel_discriminator:
                pixel_d_real = self.pixel_discriminator(hr_batch)
                pixel_d_fake = self.pixel_discriminator(fake_hr)

                val_losses.update(
                    {
                        "val_pixel_d_real": pixel_d_real.mean().item(),
                        "val_pixel_d_fake": pixel_d_fake.mean().item(),
                    },
                )

            # Latent discriminator validation
            latent_dim = real_latent.size(1)
            noise = torch.randn(batch_size, latent_dim, device=self.device)
            latent_d_real = self.latent_discriminator(real_latent)
            latent_d_fake = self.latent_discriminator(noise)

            val_losses.update(
                {
                    "val_latent_d_real": latent_d_real.mean().item(),
                    "val_latent_d_fake": latent_d_fake.mean().item(),
                },
            )

        return val_losses


class LatentDiffusionTrainingStrategy(BaseTrainingStrategy):
    """Training strategy for latent diffusion models.

    This strategy handles the training of diffusion models that operate
    in latent space with denoising objectives.

    Args:
        model: Latent diffusion model
        optimizer: Optimizer for model parameters
        device: Device to run training on
        noise_scheduler: Noise scheduler for diffusion process
        prediction_type: Type of prediction ('epsilon', 'x0', 'v')

    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        noise_scheduler: Any,
        prediction_type: str = "epsilon",
    ):
        """__init__.

        Args:
            model (nn.Module): Description.
            optimizer (torch.optim.Optimizer): Description.
            device (torch.device): Description.
            noise_scheduler (Any): Description.
            prediction_type (str): Description.
        """
        super().__init__(model, optimizer, device)
        self.noise_scheduler = noise_scheduler
        self.prediction_type = prediction_type
        self.diffusion_loss = LatentDiffusionLoss(prediction_type=prediction_type)

    def train_step(self, batch: torch.Tensor, **kwargs) -> dict[str, float]:
        """Perform a single training step for latent diffusion.

        Args:
            batch: Input batch [B, C, H, W]
            **kwargs: Additional arguments

        Returns:
            Dictionary of loss components

        """
        self.model.train()
        batch = batch.to(self.device)

        # Sample noise
        noise = torch.randn_like(batch)
        batch_size = batch.size(0)

        # Sample timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()

        # Add noise to batch
        noisy_batch = self.noise_scheduler.add_noise(batch, noise, timesteps)

        # Predict
        predicted = self.model(noisy_batch, timesteps)

        # Compute target based on prediction type
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "x0":
            target = batch
        elif self.prediction_type == "v":
            # Velocity prediction (flow-matching)
            alpha_t = self.noise_scheduler.alphas_cumprod[timesteps]
            target = alpha_t.sqrt() * noise - (1 - alpha_t).sqrt() * batch
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Compute loss
        total_loss, denoising_loss = self.diffusion_loss(predicted, target)

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "denoising_loss": denoising_loss.item(),
        }

    @torch.no_grad()
    def validation_step(self, batch: torch.Tensor, **kwargs) -> dict[str, float]:
        """Perform a validation step for latent diffusion.

        Args:
            batch: Input batch [B, C, H, W]
            **kwargs: Additional arguments

        Returns:
            Dictionary of validation metrics

        """
        self.model.eval()
        batch = batch.to(self.device)

        with torch.no_grad():
            # Sample noise
            noise = torch.randn_like(batch)
            batch_size = batch.size(0)

            # Sample timesteps
            timesteps = torch.randint(
                0,
                self.noise_scheduler.num_train_timesteps,
                (batch_size,),
                device=self.device,
            ).long()

            # Add noise to batch
            noisy_batch = self.noise_scheduler.add_noise(batch, noise, timesteps)

            # Predict
            predicted = self.model(noisy_batch, timesteps)

            # Compute target
            if self.prediction_type == "epsilon":
                target = noise
            elif self.prediction_type == "x0":
                target = batch
            elif self.prediction_type == "v":
                alpha_t = self.noise_scheduler.alphas_cumprod[timesteps]
                target = alpha_t.sqrt() * noise - (1 - alpha_t).sqrt() * batch

            # Compute loss
            total_loss, denoising_loss = self.diffusion_loss(predicted, target)

        return {
            "val_total_loss": total_loss.item(),
            "val_denoising_loss": denoising_loss.item(),
        }


# Convenience functions for creating training strategies
def create_vae_training_strategy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    **kwargs,
) -> VAETrainingStrategy:
    """Create VAE training strategy.

    Args:
        model: VAE model
        optimizer: Optimizer
        device: Training device
        **kwargs: Additional arguments

    Returns:
        VAETrainingStrategy instance

    """
    return VAETrainingStrategy(model, optimizer, device, **kwargs)


def create_vqvae_training_strategy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    **kwargs,
) -> VQVAETrainingStrategy:
    """Create VQ-VAE training strategy.

    Args:
        model: VQ-VAE model
        optimizer: Optimizer
        device: Training device
        **kwargs: Additional arguments

    Returns:
        VQVAETrainingStrategy instance

    """
    return VQVAETrainingStrategy(model, optimizer, device, **kwargs)


def create_latent_gan_training_strategy(
    encoder: nn.Module,
    generator: nn.Module,
    pixel_discriminator: nn.Module | None,
    latent_discriminator: nn.Module,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    device: torch.device,
    gan_loss: nn.Module | None = None,
    r1_penalty_weight: float = 10.0,
    disc_updates_per_gen: int = 1,
) -> LatentGANTrainingStrategy:
    """Create latent GAN training strategy with dual discriminators.

    Args:
        encoder: Encoder model (LR -> latent)
        generator: Generator model (latent -> HR)
        pixel_discriminator: Discriminator for pixel space (optional)
        latent_discriminator: Discriminator for latent space
        g_optimizer: Optimizer for generator
        d_optimizer: Optimizer for discriminators
        device: Device to run training on
        gan_loss: GAN loss function
        r1_penalty_weight: Weight for R1 gradient penalty
        disc_updates_per_gen: Number of discriminator updates
            per generator update

    Returns:
        LatentGANTrainingStrategy instance

    """
    return LatentGANTrainingStrategy(
        encoder=encoder,
        generator=generator,
        pixel_discriminator=pixel_discriminator,
        latent_discriminator=latent_discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        device=device,
        gan_loss=gan_loss,
        r1_penalty_weight=r1_penalty_weight,
        disc_updates_per_gen=disc_updates_per_gen,
    )


def create_latent_diffusion_training_strategy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    noise_scheduler: Any,
    **kwargs,
) -> LatentDiffusionTrainingStrategy:
    """Create latent diffusion training strategy.

    Args:
        model: Latent diffusion model
        optimizer: Optimizer
        device: Training device
        noise_scheduler: Noise scheduler
        **kwargs: Additional arguments

    Returns:
        LatentDiffusionTrainingStrategy instance

    """
    return LatentDiffusionTrainingStrategy(
        model,
        optimizer,
        device,
        noise_scheduler,
        **kwargs,
    )
