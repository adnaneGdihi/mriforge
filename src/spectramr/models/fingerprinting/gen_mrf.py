"""Generative MR Fingerprinting (Gen-MRF).

Innovation XV from SOTA Roadmap: Continuous tissue parameter mapping.

Key Insight:
    Traditional MRF uses dictionary matching, limited to discrete parameter values.
    Gen-MRF learns a generative model of signal evolution, enabling continuous
    parameter estimation without dictionary size constraints.

Mathematical Foundation:
    Signal model: S(t) = f(T1, T2, ρ; Sequence)

    Generative approach:
        - Encoder: z = E(S)  (signal → latent)
        - Decoder: S' = D(z) (latent → signal)
        - Mapper: (T1, T2) = M(z) (latent → parameters)

    Latent search for parameter estimation:
        z* = argmin_z ||D(z) - S_measured||²
        (T1*, T2*) = M(z*)

Benefits:
    - Continuous parameter space (no discretization)
    - Faster than dictionary matching
    - Handles noise and artifacts

References:
    - [29] MR Fingerprinting
    - [56] Deep learning for quantitative MRI

Author: Physics-AI MRI Framework
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.registry import register_model


class SignalEncoder(nn.Module):
    """Encode MRF signal evolution to latent space.

    Args:
        signal_length: Number of time points in signal
        latent_dim: Latent space dimension
        hidden_dims: Hidden layer dimensions
    """

    def __init__(
        self,
        signal_length: int = 1000,
        latent_dim: int = 32,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
    ) -> None:
        """__init__.

        Args:
            signal_length (int): Description.
            latent_dim (int): Description.
            hidden_dims (tuple[int, ...]): Description.
        """
        super().__init__()

        layers = []
        in_dim = signal_length * 2  # Real and imaginary

        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.ReLU(),
                ]
            )
            in_dim = h_dim

        self.encoder = nn.Sequential(*layers)

        # VAE-style: output mean and log-variance
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

    def forward(
        self,
        signal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode signal to latent.

        Args:
            signal: MRF signal (batch, signal_length, 2)

        Returns:
            (z, mu, logvar)

        forward method for SignalEncoder.

        Executes PyTorch tensor operations.

        Args:
            signal (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Flatten signal
        x = signal.reshape(signal.shape[0], -1)

        # Encode
        h = self.encoder(x)

        # Latent distribution parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar


class SignalDecoder(nn.Module):
    """Decode latent to MRF signal.

    Args:
        latent_dim: Latent space dimension
        signal_length: Output signal length
        hidden_dims: Hidden layer dimensions
    """

    def __init__(
        self,
        latent_dim: int = 32,
        signal_length: int = 1000,
        hidden_dims: tuple[int, ...] = (64, 128, 256),
    ) -> None:
        """__init__.

        Args:
            latent_dim (int): Description.
            signal_length (int): Description.
            hidden_dims (tuple[int, ...]): Description.
        """
        super().__init__()
        self.signal_length = signal_length

        layers = []
        in_dim = latent_dim

        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.ReLU(),
                ]
            )
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, signal_length * 2))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to signal.

        Args:
            z: Latent vector (batch, latent_dim)

        Returns:
            Signal (batch, signal_length, 2)

        forward method for SignalDecoder.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.decoder(z)
        return x.reshape(x.shape[0], self.signal_length, 2)


class ParameterMapper(nn.Module):
    """Map latent to tissue parameters.

    Args:
        latent_dim: Latent dimension
        num_params: Number of output parameters (e.g., 2 for T1, T2)
        hidden_dims: Hidden dimensions
    """

    def __init__(
        self,
        latent_dim: int = 32,
        num_params: int = 2,
        hidden_dims: tuple[int, ...] = (64, 32),
    ) -> None:
        """__init__.

        Args:
            latent_dim (int): Description.
            num_params (int): Description.
            hidden_dims (tuple[int, ...]): Description.
        """
        super().__init__()

        layers = []
        in_dim = latent_dim

        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(),
                ]
            )
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, num_params))

        self.mapper = nn.Sequential(*layers)

        # Softplus ensures positive outputs for T1, T2
        self.softplus = nn.Softplus()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent to parameters.

        Args:
            z: Latent vector (batch, latent_dim)

        Returns:
            Parameters (batch, num_params) in ms

        forward method for ParameterMapper.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        params = self.mapper(z)
        # Ensure positive and scale appropriately
        # T1 typically 100-3000ms, T2 typically 10-500ms
        params = self.softplus(params) * 100 + 10
        return params


@register_model(name="gen_mrf", training_mode="reconstruction")
class GenMRF(nn.Module):
    """Generative MR Fingerprinting model.

    VAE-based architecture for continuous tissue parameter mapping.

    Args:
        signal_length: MRF signal length
        latent_dim: Latent dimension
        num_params: Number of tissue parameters
    """

    def __init__(
        self,
        signal_length: int = 1000,
        latent_dim: int = 32,
        num_params: int = 2,
    ) -> None:
        """__init__.

        Args:
            signal_length (int): Description.
            latent_dim (int): Description.
            num_params (int): Description.
        """
        super().__init__()

        self.encoder = SignalEncoder(signal_length, latent_dim)
        self.decoder = SignalDecoder(latent_dim, signal_length)
        self.mapper = ParameterMapper(latent_dim, num_params)

    def forward(
        self,
        signal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            signal: Input MRF signal (batch, signal_length, 2)

        Returns:
            (params, recon_signal, mu, logvar)

        forward method for GenMRF.

        Executes PyTorch tensor operations.

        Args:
            signal (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Encode
        z, mu, logvar = self.encoder(signal)

        # Decode
        recon = self.decoder(z)

        # Map to parameters
        params = self.mapper(z)

        return params, recon, mu, logvar

    def compute_loss(
        self,
        signal: torch.Tensor,
        target_params: torch.Tensor | None = None,
        kl_weight: float = 0.001,
        param_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        """Compute VAE + parameter loss.

        Args:
            signal: Input signal
            target_params: Ground truth T1, T2 (optional)
            kl_weight: KL divergence weight
            param_weight: Parameter loss weight

        Returns:
            (total_loss, loss_dict)
        """
        params, recon, mu, logvar = self(signal)

        losses = {}

        # Reconstruction loss
        loss_recon = F.mse_loss(recon, signal)
        losses["recon"] = loss_recon.item()

        # KL divergence
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_kl = loss_kl / signal.shape[0]
        losses["kl"] = loss_kl.item()

        # Parameter loss (if supervision available)
        if target_params is not None:
            loss_param = F.mse_loss(params, target_params)
            losses["param"] = loss_param.item()
        else:
            loss_param = torch.tensor(0.0, device=signal.device)

        total = loss_recon + kl_weight * loss_kl + param_weight * loss_param
        losses["total"] = total.item()

        return total, losses

    def estimate_parameters(
        self,
        signal: torch.Tensor,
        num_iterations: int = 100,
        lr: float = 0.01,
    ) -> torch.Tensor:
        """Estimate parameters via latent optimization.

        Args:
            signal: Measured signal (batch, signal_length, 2)
            num_iterations: Optimization iterations
            lr: Learning rate

        Returns:
            Estimated parameters (batch, num_params)
        """
        # Initialize from encoder
        with torch.no_grad():
            z, _, _ = self.encoder(signal)

        z = z.detach().requires_grad_(True)
        from spectramr.infrastructure.training.builders import OptimizationBuilder

        optimizer = OptimizationBuilder.create_single_optimizer([z], learning_rate=lr)

        for _ in range(num_iterations):
            optimizer.zero_grad()

            recon = self.decoder(z)
            loss = F.mse_loss(recon, signal)

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            params = self.mapper(z)

        return params


def create_gen_mrf(
    signal_length: int = 1000,
    latent_dim: int = 32,
) -> GenMRF:
    """Factory function for Gen-MRF.

    Args:
        signal_length: MRF signal length
        latent_dim: Latent space dimension

    Returns:
        Configured Gen-MRF model
    """
    return GenMRF(
        signal_length=signal_length,
        latent_dim=latent_dim,
    )


__all__ = [
    "GenMRF",
    "ParameterMapper",
    "SignalDecoder",
    "SignalEncoder",
    "create_gen_mrf",
]
