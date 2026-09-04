"""Continuous-time selective state-space scan along a space-filling curve.

This is the SFC-Mamba branch of CS-MNO. The discrete update per token
is

.. math::

    \\bar A_t = \\exp(\\Delta_t \\cdot A(v_t)),\\qquad
    \\bar B_t = \\Delta_t \\cdot B(v_t)

    h_t = \\bar A_t \\cdot h_{t-1} + \\bar B_t \\cdot v_t,\\qquad
    y_t = C(v_t) \\cdot h_t + D \\cdot v_t

where :math:`\\Delta_t` is the **physical-arc** distance between
adjacent samples on the SFC,
:math:`\\Delta_t = \\| H(s_t) - H(s_{t-1}) \\|_\\Omega`. This is the
mechanism behind Theorem A of ``TODO/Mamba_NO.md``: the discrete scan
becomes an Euler quadrature of the continuous-time SSM along the
curve, which converges as :math:`N \\to \\infty` independent of the
discretisation resolution.

If ``use_physical_arc=False``, the block falls back to vanilla Mamba's
learnt scalar :math:`\\Delta` parameter — the critical ablation that
disables the resolution-invariance guarantee.

Implementation note. The forward loop is intentionally pure PyTorch
for correctness and clarity. Production training of the volumetric
variant should use the Triton parallel-scan kernel from CS-MNO Phase 2
(see the integration plan); the kernel is a drop-in replacement for
this loop with identical numerics.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ContinuousSFCMamba"]


class ContinuousSFCMamba(nn.Module):
    """Selective state-space scan with optional per-token physical Δt.

    Args:
        dim:            Channel dimension of the input sequence.
        d_state:        Hidden state dimension of the SSM. Standard
                        Mamba uses 16; larger values trade memory for
                        capacity.
        use_physical_arc: If True, multiply the SSM time-step by the
                        physical arc length of the SFC (passed in as
                        ``delta_phys`` at forward time). If False,
                        ignore ``delta_phys`` and use a learnt scalar
                        Δ that is shared across tokens (vanilla
                        Mamba behaviour). The critical CS-MNO ablation.
        kernel_backend: ``'python'`` (always), ``'compile'`` (torch.
                        compile), ``'triton'`` (CUDA + triton), or
                        ``'auto'`` (Triton on GPU when available,
                        else compile, else Python). All four backends
                        produce numerically equivalent outputs; the
                        non-default backends are pure speedups.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        use_physical_arc: bool = True,
        kernel_backend: str = "python",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.use_physical_arc = use_physical_arc
        if kernel_backend not in ("python", "compile", "triton", "auto", "mamba_ssm"):
            raise ValueError(
                f"ContinuousSFCMamba: kernel_backend must be one of "
                f"{{'python', 'compile', 'triton', 'auto', 'mamba_ssm'}}, "
                f"got {kernel_backend!r}"
            )
        self.kernel_backend = kernel_backend

        # SSM parameters. A is parameterised in log-space and forced
        # negative-real (`-exp(...)`) for stability — standard Mamba.
        self.A_log = nn.Parameter(torch.randn(dim, d_state))
        self.B_proj = nn.Linear(dim, d_state)
        self.C_proj = nn.Linear(d_state, dim)
        self.D = nn.Parameter(torch.zeros(dim))

        # Vanilla-Mamba fallback: learnt scalar Δ shared across tokens.
        # Used only when use_physical_arc=False.
        if not use_physical_arc:
            self.delta_log = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x_seq: torch.Tensor,
        delta_phys: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the per-token scan.

        Args:
            x_seq:      ``[B, N, C]`` token sequence in SFC order.
            delta_phys: ``[N]`` per-token physical Δt. Required when
                        ``use_physical_arc=True``; ignored otherwise.

        Returns:
            ``[B, N, C]`` output sequence.
        """
        B, N, C = x_seq.shape
        if self.dim != C:
            raise ValueError(f"ContinuousSFCMamba: input dim {C} != configured dim {self.dim}")

        # SSM per-channel state matrix A_diag (dim, d_state).
        A_diag = -torch.exp(self.A_log)

        # Effective per-token Δ.
        if self.use_physical_arc:
            if delta_phys is None:
                raise ValueError("use_physical_arc=True requires delta_phys at forward time")
            if delta_phys.shape[0] != N:
                raise ValueError(f"delta_phys length {delta_phys.shape[0]} != sequence length {N}")
            dt = delta_phys.to(x_seq.device, x_seq.dtype)  # (N,)
        else:
            dt = torch.exp(self.delta_log).expand(N).to(x_seq.device, x_seq.dtype)

        # Compute the per-token B projection once (cheap, fully parallel)
        # then hand the recurrence to the chosen backend.
        B_proj = self.B_proj(x_seq)  # (B, N, d_state)

        if self.kernel_backend == "python":
            # Inline the canonical loop for backwards-compat with tests
            # that introspect intermediate state.
            h = x_seq.new_zeros(B, self.d_state)
            outputs_h = x_seq.new_empty(B, N, self.d_state)
            for t in range(N):
                A_t_diag = torch.exp(dt[t] * A_diag).mean(dim=0)  # (d_state,)
                B_t = B_proj[:, t] * dt[t]  # (B, d_state)
                h = h * A_t_diag + B_t
                outputs_h[:, t] = h
            h_seq = outputs_h
        else:
            # Use the accelerated dispatcher.
            from spectramr.models.blocks.triton_scan import selective_scan

            h_seq = selective_scan(A_diag, B_proj, dt, backend=self.kernel_backend)

        # Output projection: y_t = C(v_t) @ h_t + D * v_t  — fully parallel.
        y_seq = self.C_proj(h_seq) + self.D * x_seq  # (B, N, C)
        return y_seq
