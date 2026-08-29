"""Activation normalization (ActNorm) for normalizing flows.

ActNorm applies an affine transform ``z = s * x + b`` with per-channel
scale ``s`` and bias ``b``. Parameters are initialised data-dependently
on the first minibatch so that the post-transform activations have zero
mean and unit variance per channel. The log-Jacobian-determinant of the
transform is ``H * W * sum_c log|s_c|`` (the spatial factor arises
because the same affine is applied at every spatial location).

Reference: D. P. Kingma and P. Dhariwal, "Glow: Generative flow with
invertible 1x1 convolutions," NeurIPS 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActNorm2d(nn.Module):
    """Per-channel affine normalization with data-dependent init.

    Args:
        num_channels: Number of feature channels ``C`` in the ``[B, C, H, W]``
            input tensor.
        eps: Numerical floor used when standardising on the first batch.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        # log-scale is stored so |s| = exp(logs) > 0 by construction.
        self.logs = nn.Parameter(torch.zeros(1, self.num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, self.num_channels, 1, 1))
        # Buffer so the init flag survives state_dict round-trips.
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))
        # Python-side memo of the ``initialized`` buffer. The buffer stays the
        # serialization SSOT (it must survive state_dict round-trips, or a
        # resumed model would re-run the data-dependent init and overwrite the
        # trained ``logs``/``bias``); this flag only spares the hot path the
        # ``.item()`` device sync that reading it costs on every forward.
        self._initialized = False

    @torch.no_grad()
    def _initialize(self, x: torch.Tensor) -> None:
        # Mean/std over batch + spatial dims, per channel.
        dims = (0, 2, 3)
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, keepdim=True)
        self.bias.data.copy_(-mean / (std + self.eps))
        self.logs.data.copy_(-torch.log(std + self.eps))
        self.initialized.fill_(True)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the affine transform.

        Args:
            x: ``[B, C, H, W]`` input.
            reverse: If ``True`` compute the inverse ``x = (z - b) / s``.

        Returns:
            ``(y, log_det)`` where ``log_det`` is a per-sample
            ``[B]`` tensor (negated in the reverse direction).
        """
        if not self._initialized and not reverse:
            # One device sync on the first forward of this process, none after.
            if not bool(self.initialized.item()):
                self._initialize(x)
            self._initialized = True

        b, _, h, w = x.shape
        # Per-sample log-det: H * W * sum_c logs_c.
        log_det = self.logs.sum() * float(h * w)
        log_det = log_det.expand(b)

        if reverse:
            y = (x - self.bias) * torch.exp(-self.logs)
            return y, -log_det
        y = x * torch.exp(self.logs) + self.bias
        return y, log_det
