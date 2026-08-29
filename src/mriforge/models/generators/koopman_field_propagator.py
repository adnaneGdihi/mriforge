"""Koopman linear field-propagator (MICCAI MRIxFields2026, B-3.10).

Cross-field translation as a CONTINUOUS Koopman semigroup along the physical field
coordinate ``tau = log10(B0)``. An encoder lifts the magnitude image to a latent observable
``z = phi(x)``; a single learned infinitesimal generator ``A`` propagates it LINEARLY across
the field gap ``dtau = tau_t - tau_s`` via the matrix exponential ``z_t = exp(dtau*A) z_s``
(:func:`~mriforge.infrastructure.physics.koopman_operator.koopman_continuous_propagate`); a
decoder renders ``psi(z_t)``.

The distinctive, falsifiable structure is the one-parameter group: ``exp((a+b)A) =
exp(aA)exp(bA)`` holds EXACTLY, so ONE operator handles ANY field gap (any-to-any), composition
is exact (s->m->t == s->t), and the generator's eigenvalue spectrum is an interpretable physics
witness (spectral abscissa < 0 => contractive propagation). ``A`` is zero-initialised so the
propagation delta is 0 at init (the bare autoencoder).

One-knob ablation ``use_koopman_semigroup``: ``False`` replaces the linear operator with a
field-conditioned NONLINEAR latent transform (per-pixel MLP on ``[z_s, tau_s, tau_t]``) — no
linear operator, no semigroup. BOTH arms are zero-delta at init (the linear arm's ``A`` is
zero-init; the nonlinear arm's final MLP conv is zero-init), so they start as the SAME bare
autoencoder — the knob isolates linear-vs-nonlinear structure, not initialisation (#17).
Magnitude-only (1->1); both ``field_strength`` and
``field_strength_target`` are REQUIRED forward kwargs (no default) so the Tier-2 probe injects
them. Distinct from the TEMPORAL Koopman code (fMRI k-space-time / motion in
``models/temporal/`` + ``infrastructure/physics/koopman_operator.py`` EDMD) — here the axis is
the B0 FIELD, the operator is learned end-to-end, and the roll-out is continuous.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mriforge.infrastructure.physics.koopman_operator import (
    koopman_continuous_propagate,
    koopman_spectral_abscissa,
)
from mriforge.models.registry import register_model


def _log10(b: torch.Tensor) -> torch.Tensor:
    return torch.log10(b.float().clamp_min(1.0e-6))


@register_model(name="koopman_field_propagator", training_mode="koopman_field")
class KoopmanFieldPropagator(nn.Module):
    """Encoder -> continuous Koopman semigroup propagation -> decoder (B-3.10)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 48,
        koopman_dim: int = 32,
        n_blocks: int = 2,
        use_koopman_semigroup: bool = True,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                f"KoopmanFieldPropagator is magnitude-only (1->1); got in={in_channels}, "
                f"out={out_channels}."
            )
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1; got {n_blocks}.")
        if koopman_dim < 2:
            raise ValueError(f"koopman_dim must be >= 2; got {koopman_dim}.")
        if not isinstance(use_koopman_semigroup, bool):
            raise ValueError(
                f"use_koopman_semigroup must be a bool; got {type(use_koopman_semigroup).__name__}."
            )
        self.width = int(width)
        self.koopman_dim = int(koopman_dim)
        self.n_blocks = int(n_blocks)
        self.use_koopman_semigroup = bool(use_koopman_semigroup)
        pad = kernel_size // 2

        def _conv(cin: int, cout: int, k: int = kernel_size, p: int = pad) -> nn.Conv2d:
            return nn.Conv2d(cin, cout, k, padding=p)

        # Encoder phi: x -> latent observable z (keeps spatial dims).
        enc: list[nn.Module] = [_conv(1, width), nn.SiLU()]
        for _ in range(n_blocks):
            enc += [_conv(width, width), nn.GroupNorm(min(8, width), width), nn.SiLU()]
        enc += [_conv(width, self.koopman_dim, k=1, p=0)]
        self.encoder = nn.Sequential(*enc)

        # Decoder psi: latent -> image.
        dec: list[nn.Module] = [_conv(self.koopman_dim, width, k=1, p=0), nn.SiLU()]
        for _ in range(n_blocks):
            dec += [_conv(width, width), nn.GroupNorm(min(8, width), width), nn.SiLU()]
        dec += [_conv(width, 1, k=1, p=0)]
        self.decoder = nn.Sequential(*dec)

        if self.use_koopman_semigroup:
            # Shared infinitesimal generator A (zero-init -> exp(dtau*A)=I -> delta 0 at init).
            self.generator = nn.Parameter(torch.zeros(self.koopman_dim, self.koopman_dim))
            self.field_mlp = None
        else:
            # NONLINEAR control: a per-pixel MLP (1x1 convs) on [z_s, tau_s, tau_t] -> latent
            # delta. No linear operator, no semigroup (the one-knob ablation).
            self.generator = None
            self.field_mlp = nn.Sequential(
                _conv(self.koopman_dim + 2, self.koopman_dim, k=1, p=0),
                nn.SiLU(),
                _conv(self.koopman_dim, self.koopman_dim, k=1, p=0),
            )
            # Zero-init the final conv so the latent delta is 0 at init -> z_t = z_s, MATCHING the
            # semigroup arm's zero-init A (exp(0)=I). Without this the nonlinear arm would be
            # field-dependent at init while the linear arm is field-blind — an init-asymmetry
            # confound (#17). Now BOTH arms start as the same bare autoencoder.
            nn.init.zeros_(self.field_mlp[-1].weight)
            nn.init.zeros_(self.field_mlp[-1].bias)

    def spectral_abscissa(self) -> float | None:
        """Continuous-time spectral abscissa max Re(lambda(A)) — None for the nonlinear arm."""
        if self.generator is None:
            return None
        return koopman_spectral_abscissa(self.generator.detach())

    def forward(
        self,
        x: torch.Tensor,
        *,
        field_strength: torch.Tensor,
        field_strength_target: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        if torch.is_complex(x):
            raise ValueError("KoopmanFieldPropagator expects magnitude (real) input.")
        tau_s = _log10(field_strength).reshape(-1)
        tau_t = _log10(field_strength_target).reshape(-1)
        if tau_s.shape[0] != x.shape[0] or tau_t.shape[0] != x.shape[0]:
            # No silent size-1 broadcast of one field across the batch (pitfall #9).
            raise ValueError(
                f"field batch ({tau_s.shape[0]},{tau_t.shape[0]}) != input batch {x.shape[0]}; "
                "pass one (field_strength, field_strength_target) pair per sample."
            )
        dtau = tau_t - tau_s  # [B] continuous field gap
        z_s = self.encoder(x)
        if self.use_koopman_semigroup:
            assert self.generator is not None
            z_t = koopman_continuous_propagate(z_s, self.generator, dtau)
        else:
            assert self.field_mlp is not None
            b, _, h, w = z_s.shape
            tau_map = torch.stack([tau_s, tau_t], dim=1)[:, :, None, None].expand(b, 2, h, w)
            z_t = z_s + self.field_mlp(torch.cat([z_s, tau_map.to(z_s)], dim=1))
        return self.decoder(z_t)
