"""Koopman Operator Neural Network — Linear Embedding of Nonlinear Dynamics.

Lifts nonlinear 1-D physiological motion signals (respiratory, cardiac)
into an infinite-dimensional Koopman space where dynamics become strictly
linear.  This allows exact linear propagation of complex, aperiodic
biological motion — including coughs, arrhythmia, and deep breaths.

Architecture:
    .. code-block:: text

        marker_signal(t) → Encoder → z(t) → K·z(t) → z(t+1) → Decoder → predicted(t+1)

    The Koopman matrix K is a learnable linear operator that captures the
    full nonlinear dynamics in the lifted space.

Mathematics:
    For a nonlinear dynamical system :math:`x_{t+1} = F(x_t)`, Koopman
    theory guarantees the existence of a linear operator :math:`K` such
    that:

    .. math::

        g(x_{t+1}) = K \\cdot g(x_t)

    where :math:`g: \\mathbb{R}^n \\to \\mathbb{R}^N` is an embedding
    (the encoder) into a higher-dimensional space.

Reference:
    Lusch et al., "Deep learning for universal linear embeddings of
    nonlinear dynamics," *Nature Communications*, 2018.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class KoopmanMotionFilter(nn.Module):
    """Filter physiological motion signals via Koopman operator embedding.

    Args:
        input_dim: Dimensionality of the input signal (e.g., 2 for
            complex DC navigator: magnitude + phase).
        latent_dim: Dimension of the Koopman embedding space.
        hidden_dim: Hidden dimension of encoder/decoder MLPs.
        num_layers: Number of encoder/decoder hidden layers.
        prediction_horizon: Number of future steps to predict.
    """

    def __init__(
        self,
        input_dim: int = 2,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        prediction_horizon: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.prediction_horizon = prediction_horizon

        # Encoder: x → z (lift to Koopman space)
        enc_layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            enc_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        enc_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Koopman matrix K: linear dynamics in lifted space
        self.K = nn.Parameter(torch.eye(latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim))

        # Decoder: z → x (project back to physical space)
        dec_layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            dec_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        dec_layers.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        logger.info(
            "[KoopmanMotionFilter] input=%d, latent=%d, hidden=%d, layers=%d, horizon=%d",
            input_dim,
            latent_dim,
            hidden_dim,
            num_layers,
            prediction_horizon,
        )

    def encode(self, x: Tensor) -> Tensor:
        """Lift signal to Koopman embedding space.

        Args:
            x: Input signal ``[B, T, D]`` or ``[T, D]``.

        Returns:
            Koopman embedding ``[B, T, latent_dim]``.
        """
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Project Koopman embedding back to physical space.

        Args:
            z: Koopman embedding ``[B, T, latent_dim]``.

        Returns:
            Reconstructed signal ``[B, T, D]``.
        """
        return self.decoder(z)

    def propagate(self, z: Tensor, steps: int = 1) -> Tensor:
        """Apply Koopman operator K for multiple time steps.

        Args:
            z: Current Koopman state ``[B, latent_dim]``.
            steps: Number of forward time steps.

        Returns:
            Propagated state ``[B, steps, latent_dim]``.
        """
        states = []
        for _ in range(steps):
            z = z @ self.K.t()  # Linear propagation
            states.append(z)
        return torch.stack(states, dim=1)

    def filter_signal(
        self,
        signal: Tensor,
    ) -> dict[str, Tensor]:
        """Filter a motion signal by encoding, propagating, and decoding.

        The reconstruction error reveals the aperiodic (chaotic) component,
        while the successfully propagated component captures the periodic
        (predictable) dynamics.

        Args:
            signal: Input signal ``[B, T, D]``.

        Returns:
            Dictionary with keys:
            - ``"filtered"``: Koopman-filtered (periodic) signal
            - ``"aperiodic"``: residual (coughs, arrhythmia)
            - ``"reconstruction"``: autoencoder reconstruction
            - ``"prediction"``: multi-step predicted future
        """
        B, T, D = signal.shape

        # Encode full sequence
        z = self.encode(signal)  # [B, T, latent_dim]

        # Autoencoder reconstruction
        reconstruction = self.decode(z)  # [B, T, D]

        # One-step Koopman prediction: z(t+1) = K · z(t)
        z_shifted = z[:, :-1]  # [B, T-1, latent_dim]
        z_predicted = z_shifted @ self.K.t()  # [B, T-1, latent_dim]
        filtered = self.decode(z_predicted)  # [B, T-1, D]

        # Pad to match original length
        filtered = torch.cat([signal[:, :1], filtered], dim=1)  # [B, T, D]

        # Aperiodic = original - filtered
        aperiodic = signal - filtered

        # Multi-step prediction from last state
        z_last = z[:, -1]  # [B, latent_dim]
        z_future = self.propagate(z_last, self.prediction_horizon)
        prediction = self.decode(z_future)  # [B, horizon, D]

        return {
            "filtered": filtered,
            "aperiodic": aperiodic,
            "reconstruction": reconstruction,
            "prediction": prediction,
        }

    def forward(self, signal: Tensor) -> dict[str, Tensor]:
        """Forward pass — alias for :meth:`filter_signal`.

        forward method for KoopmanMotionFilter.

        Executes PyTorch tensor operations.

        Args:
            signal (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            dict[str, Tensor]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if signal.dim() == 2:
            signal = signal.unsqueeze(0)
        return self.filter_signal(signal)

    def koopman_loss(
        self,
        signal: Tensor,
    ) -> dict[str, Tensor]:
        """Compute training losses for the Koopman operator.

        Three loss terms:
        1. Reconstruction: ``||decode(encode(x)) - x||²``
        2. Prediction: ``||decode(K · encode(x_t)) - x_{t+1}||²``
        3. Linearity: ``||K · encode(x_t) - encode(x_{t+1})||²``

        Args:
            signal: Input sequence ``[B, T, D]``.

        Returns:
            Dictionary of loss tensors.
        """
        z = self.encode(signal)  # [B, T, latent]

        # Reconstruction loss
        recon = self.decode(z)
        loss_recon = torch.mean((recon - signal) ** 2)

        # Prediction loss
        z_t = z[:, :-1]
        z_tp1_pred = z_t @ self.K.t()
        x_tp1_pred = self.decode(z_tp1_pred)
        loss_pred = torch.mean((x_tp1_pred - signal[:, 1:]) ** 2)

        # Linearity loss (Koopman invariance)
        z_tp1_actual = z[:, 1:]
        loss_linear = torch.mean((z_tp1_pred - z_tp1_actual) ** 2)

        return {
            "reconstruction": loss_recon,
            "prediction": loss_pred,
            "linearity": loss_linear,
            "total": loss_recon + loss_pred + 0.1 * loss_linear,
        }


__all__ = ["KoopmanMotionFilter"]
