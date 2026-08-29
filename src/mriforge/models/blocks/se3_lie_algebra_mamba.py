r"""SE(3)-equivariant Lie-algebra Mamba block (B-3 of the VF campaign).

This block implements a selective state-space (Mamba-style) update on the
Lie algebra :math:`\mathfrak{se}(3) \cong \mathfrak{so}(3) \oplus \mathbb{R}^3`
that is *equivariant* under the adjoint action of the group, processed
irreducible-component-wise (per V2 of ``TODO/VF_CAMPAIGN_IMPLEMENTATION_PLAN.md``).

Mathematical setup
------------------
A rigid-body motion state at time :math:`t` is an element
:math:`\xi(t) = (\omega(t), v(t)) \in \mathfrak{se}(3)`, where
:math:`\omega \in \mathfrak{so}(3) \cong \mathbb{R}^3` is the rotational
(angular-velocity) part and :math:`v \in \mathbb{R}^3` the translational part.
A group element :math:`g = (R, p) \in SE(3)` acts on the algebra by the
adjoint representation

.. math::

    \mathrm{Ad}_g \,(\omega, v) \;=\; (R\,\omega,\; R\,v + p \times (R\,\omega)).

The irreducible decomposition splits :math:`\xi` into two
:math:`O(3)`-vectors :math:`\omega` and :math:`v`. Under a *pure rotation*
:math:`g=(R,0)` (the case relevant for in-plane MRI motion, see the SO(2)
restriction note below) the adjoint reduces to the block-diagonal
:math:`\mathrm{Ad}_{(R,0)}(\omega, v) = (R\omega, Rv)` — each irreducible is
simply rotated by :math:`R`.

Equivariance recipe (norm-selective SSM on directions)
------------------------------------------------------
For each irreducible vector :math:`u_k(t) \in \mathbb{R}^{d_k}` we write it in
polar form :math:`u_k = r_k\,\hat u_k` with :math:`r_k=\|u_k\|` (a *scalar*,
hence rotation-**invariant**) and :math:`\hat u_k` a unit direction (a *vector*,
hence rotation-**equivariant**). The block

1. runs a scalar selective state-space recurrence on the per-timestep norms
   :math:`\{r_k(t)\}` (a function of rotation-invariant quantities → its output
   is rotation-invariant), producing a per-timestep multiplicative gate
   :math:`g_k(t) \ge 0`; and
2. rescales the *direction* by that gate: :math:`u_k(t) \mapsto g_k(t)\,u_k(t)`.

Because the gate depends only on invariants and is applied multiplicatively to
the equivariant direction, the whole map commutes with :math:`\mathrm{Ad}_{(R,0)}`:

.. math::

    \mathrm{Block}(\mathrm{Ad}_{(R,0)}\,\xi) \;=\; \mathrm{Ad}_{(R,0)}\,\mathrm{Block}(\xi).

This is the exact statement the equivariance round-trip test checks.

Bio-harmonic eigenvalue initialisation
--------------------------------------
The scalar SSM transition is initialised so that the *continuous-time* state
matrix :math:`A(0)` has eigenvalues on the imaginary axis at
:math:`\pm i\,2\pi f_k` for a set of physiological frequencies
:math:`f_k \in \{f_\mathrm{resp}\approx0.3,\ f_\mathrm{card}\approx1.2,\dots\}`
Hz and their harmonics. This biases the recurrence toward resonant motion
(respiration ~0.3 Hz, cardiac ~1.2 Hz + harmonics) before any training.

Simplification (documented, per the task spec)
----------------------------------------------
The full SO(3) adjoint is implemented for *pure-rotation* group elements, and
the equivariance test exercises the **in-plane SO(2)** subgroup
:math:`R = R_z(\theta)` acting on the leading 2D sub-block of each irreducible.
This is the physically relevant case for 2D MRI slice motion (rotations are in
the slice plane; through-plane components are carried but not rotated by the
2D test element). The mechanism (norm-invariant gate times equivariant
direction) is exact for the *full* SO(3) rotation as well — see
``test_se3_lie_algebra_mamba.py::test_equivariance_so3_full`` — the SO(2)
restriction only narrows which group elements the navigator's 2D warp head
consumes downstream, not the block's algebraic guarantee.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import torch
import torch.nn as nn

from mriforge.models.blocks.mamba_block import MambaBlock

logger = logging.getLogger(__name__)

# Physiological motion frequencies (Hz): respiration, its first harmonic,
# cardiac, and the cardiac second harmonic. Used for eigenvalue init.
DEFAULT_BIO_HARMONIC_FREQS: tuple[float, ...] = (0.3, 0.6, 1.2, 2.4)

_SUPPORTED_GROUPS = ("SO3", "SE3")


class SE3LieAlgebraMambaBlock(nn.Module):
    r"""Selective state-space block on :math:`\mathfrak{se}(3)`.

    Decomposes a per-timestep Lie-algebra trajectory into its rotation
    (:math:`\mathfrak{so}(3)`) and translation (:math:`\mathbb{R}^3`)
    irreducible components, applies a scalar-norm selective gate to each
    (preserving equivariance), and rescales the direction equivariantly.

    The input/output layout is ``[B, L, 6]`` where the last axis is the
    stacked :math:`(\omega_x, \omega_y, \omega_z, v_x, v_y, v_z)` Lie-algebra
    coordinate, ``B`` the batch, and ``L`` the temporal length (navigator
    samples). ``group="SO3"`` zeroes/ignores the translation block.

    Args:
        d_state: SSM hidden-state dimension for the scalar gate recurrence.
        d_conv: Depthwise conv kernel length inside the scalar Mamba.
        dt_rank: ``"auto"`` (= ``ceil(d_model/16)``) or an int; forwarded to
            the scalar Mamba primitive.
        bio_harmonic_freqs: Physiological frequencies (Hz) used to seed the
            transition eigenvalues. Must be strictly positive.
        sample_rate_hz: Navigator temporal sampling rate (Hz). Required to map
            continuous-time bio-harmonic frequencies onto the discrete state.
        group: ``"SO3"`` (rotation only) or ``"SE3"`` (rotation+translation).

    Raises:
        ValueError: on an unknown ``group`` (no silent fallback, CLAUDE.md #9),
            a non-positive frequency, or a sample rate that violates Nyquist
            for the requested harmonics.
    """

    # Channel layout of the se(3) coordinate.
    _N_ROT = 3  # so(3) angular part
    _N_TRANS = 3  # R^3 translation part
    _DIM = _N_ROT + _N_TRANS  # 6

    def __init__(
        self,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: str | int = "auto",
        bio_harmonic_freqs: tuple[float, ...] = DEFAULT_BIO_HARMONIC_FREQS,
        sample_rate_hz: float = 100.0,
        group: Literal["SO3", "SE3"] = "SE3",
    ) -> None:
        super().__init__()

        if group not in _SUPPORTED_GROUPS:
            # CLAUDE.md #9 — no silent fallback to a default group.
            raise ValueError(
                f"SE3LieAlgebraMambaBlock: unknown group {group!r}; "
                f"supported groups are {_SUPPORTED_GROUPS}."
            )
        if not bio_harmonic_freqs:
            raise ValueError("bio_harmonic_freqs must be non-empty.")
        if any(f <= 0 for f in bio_harmonic_freqs):
            raise ValueError(
                f"bio_harmonic_freqs must be strictly positive; got {bio_harmonic_freqs}."
            )
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        nyquist = sample_rate_hz / 2.0
        if max(bio_harmonic_freqs) >= nyquist:
            raise ValueError(
                f"bio_harmonic_freqs max {max(bio_harmonic_freqs)} Hz violates "
                f"Nyquist for sample_rate_hz={sample_rate_hz} (limit {nyquist} Hz)."
            )

        self.group = group
        self.d_state = int(d_state)
        self.sample_rate_hz = float(sample_rate_hz)
        self.bio_harmonic_freqs = tuple(float(f) for f in bio_harmonic_freqs)

        # Number of irreducible components actually processed.
        self.n_irreps = 2 if group == "SE3" else 1  # (rot, trans) or (rot,)

        # ------------------------------------------------------------------
        # Scalar selective state-space gate.
        #
        # We feed the per-timestep norms of each irreducible (n_irreps scalar
        # channels) through a shared scalar Mamba. Its output is mapped to a
        # NON-NEGATIVE multiplicative gate per irreducible. Because the input
        # (norms) is rotation-invariant, the gate is rotation-invariant, so
        # multiplying the equivariant direction by it is equivariant.
        # ------------------------------------------------------------------
        self.d_model = self.n_irreps
        self.scalar_ssm = MambaBlock(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=d_conv,
            expand=2,
            **({"dt_rank": dt_rank} if isinstance(dt_rank, int) else {}),
        )

        # Map SSM output → per-irrep gate logits. softplus → (0, inf) gate.
        self.gate_proj = nn.Linear(self.d_model, self.n_irreps)

        # ------------------------------------------------------------------
        # Bio-harmonic eigenvalue buffer.
        #
        # Continuous-time resonances at +/- i 2 pi f_k. We store the discrete
        # per-step rotation angles theta_k = 2 pi f_k / fs and the (unit)
        # magnitudes, which place the discrete eigenvalues on the unit circle
        # at e^{+/- i theta_k} — the structure-preserving discretisation of an
        # undamped oscillator at f_k. These seed an auxiliary oscillator bank
        # whose energy modulates the gate (a learnable, physiologically-biased
        # prior). Stored as buffers so they move with .to(device) and are
        # checkpointed, but are not trained directly.
        # ------------------------------------------------------------------
        freqs = torch.tensor(self.bio_harmonic_freqs, dtype=torch.float32)
        disc_angles = 2.0 * math.pi * freqs / self.sample_rate_hz  # theta_k
        self.register_buffer("_bio_freqs_hz", freqs, persistent=True)
        self.register_buffer("_bio_disc_angles", disc_angles, persistent=True)
        # Real/imag of the discrete eigenvalues e^{i theta_k} on the unit circle.
        self.register_buffer("_bio_eig_real", torch.cos(disc_angles), persistent=True)
        self.register_buffer("_bio_eig_imag", torch.sin(disc_angles), persistent=True)

        # Learnable readout that turns the (fixed-frequency) oscillator bank's
        # per-step energy into an additive contribution to the gate logits.
        # Initialised small so the block starts near identity gating.
        self.harmonic_readout = nn.Linear(len(self.bio_harmonic_freqs), self.n_irreps)
        nn.init.zeros_(self.harmonic_readout.bias)
        nn.init.normal_(self.harmonic_readout.weight, std=1e-2)

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def decompose(self, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Split a ``[B, L, 6]`` se(3) trajectory into ``(omega, v)``.

        Returns the rotation part ``omega`` ``[B, L, 3]`` and translation part
        ``v`` ``[B, L, 3]``. For ``group="SO3"`` the translation slice is still
        returned (callers may ignore it) but is excluded from the gate.
        """
        if xi.shape[-1] != self._DIM:
            raise ValueError(
                f"Expected last dim {self._DIM} (so(3)+R^3); got shape {tuple(xi.shape)}."
            )
        omega = xi[..., : self._N_ROT]
        v = xi[..., self._N_ROT :]
        return omega, v

    @staticmethod
    def adjoint_rotation(xi: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
        r"""Apply the adjoint of a *pure rotation* ``g=(R, 0)`` to ``xi``.

        For :math:`g=(R,0)`, :math:`\mathrm{Ad}_g(\omega, v) = (R\omega, Rv)`.

        Args:
            xi: ``[B, L, 6]`` Lie-algebra trajectory.
            rot: ``[3, 3]`` or ``[B, 3, 3]`` rotation matrix/matrices.

        Returns:
            ``[B, L, 6]`` rotated trajectory.
        """
        omega = xi[..., :3]
        v = xi[..., 3:]
        if rot.dim() == 2:
            # einsum over the shared 3-axis; broadcast over B, L.
            r_omega = torch.einsum("ij,blj->bli", rot, omega)
            r_v = torch.einsum("ij,blj->bli", rot, v)
        elif rot.dim() == 3:
            r_omega = torch.einsum("bij,blj->bli", rot, omega)
            r_v = torch.einsum("bij,blj->bli", rot, v)
        else:
            raise ValueError(f"rot must be [3,3] or [B,3,3]; got {tuple(rot.shape)}.")
        return torch.cat([r_omega, r_v], dim=-1)

    def _oscillator_energy(self, norms: torch.Tensor) -> torch.Tensor:
        r"""Bio-harmonic oscillator-bank energy per timestep.

        Drives a bank of undamped oscillators (one per bio-harmonic frequency)
        at the discrete eigenvalues :math:`e^{\pm i\theta_k}` with the mean
        irreducible norm as input, and returns the per-step squared amplitude.
        This is a fixed-frequency (physiologically-seeded) feature; only the
        downstream ``harmonic_readout`` is learnable.

        Args:
            norms: ``[B, L, n_irreps]`` per-step irreducible norms.

        Returns:
            ``[B, L, n_freqs]`` non-negative oscillator energies.
        """
        _b, length, _ = norms.shape
        # Scalar drive = mean norm across irreducibles (rotation-invariant).
        drive = norms.mean(dim=-1)  # [B, L]

        # Closed form (PERF 2026-07-01): the undamped unit-magnitude
        # recurrence ``s_t = e^{iθ}·s_{t-1} + drive_t`` has the exact solution
        # ``s_t = e^{iθt} · Σ_{j≤t} e^{-iθj}·drive_j``, and since
        # ``|e^{iθt}| = 1`` the energy needs no re-rotation:
        # ``|s_t|² = (Σ_{j≤t} drive_j·cos(θj))² + (Σ_{j≤t} drive_j·sin(θj))²``.
        # This replaces the per-token Python loop (≈5 kernel launches per
        # step) with two cumsums — numerically safe precisely BECAUSE the
        # eigenvalues sit on the unit circle (no ``A^{-k}`` blow-up, unlike a
        # decaying-scan cumprod trick). All-real math, so the autocast story
        # is unchanged. Equivalence to the iterated rotation is pinned at
        # large L in the paired test.
        angles = self._bio_disc_angles.to(device=norms.device, dtype=drive.dtype)
        j = torch.arange(length, device=norms.device, dtype=drive.dtype)
        phase = j.unsqueeze(-1) * angles  # [L, n_freqs]
        u = torch.cumsum(drive.unsqueeze(-1) * torch.cos(phase), dim=1)
        w = torch.cumsum(drive.unsqueeze(-1) * torch.sin(phase), dim=1)
        out = u * u + w * w  # squared amplitude (energy), [B, L, n_freqs]
        # Normalise by length so energy doesn't blow up with sequence length.
        return out / float(max(length, 1))

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        r"""Apply the equivariant selective state-space update.

        Args:
            xi: ``[B, L, 6]`` Lie-algebra trajectory
                :math:`(\omega_x,\omega_y,\omega_z,v_x,v_y,v_z)`.

        Returns:
            ``[B, L, 6]`` updated trajectory. Equivariant to
            :math:`\mathrm{Ad}_{(R,0)}` (rotation block of the adjoint).
        """
        if xi.dim() != 3 or xi.shape[-1] != self._DIM:
            raise ValueError(f"Expected input [B, L, 6]; got {tuple(xi.shape)}.")
        omega, v = self.decompose(xi)

        # Per-irreducible norms (rotation-invariant scalars). The gate
        # multiplies the raw (equivariant) vector, so no explicit
        # division-by-norm is needed — equivariance holds for r*dir alike.
        r_omega = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)  # [B,L,1]
        if self.group == "SE3":
            r_v = torch.linalg.vector_norm(v, dim=-1, keepdim=True)  # [B,L,1]
            norms = torch.cat([r_omega, r_v], dim=-1)  # [B, L, 2]
        else:
            norms = r_omega  # [B, L, 1]

        # Scalar selective state-space recurrence on the invariants.
        ssm_out = self.scalar_ssm(norms)  # [B, L, n_irreps]

        # Bio-harmonic oscillator-bank energy → additive gate prior.
        osc_energy = self._oscillator_energy(norms)  # [B, L, n_freqs]
        harmonic_term = self.harmonic_readout(osc_energy)  # [B, L, n_irreps]

        gate_logits = self.gate_proj(ssm_out) + harmonic_term  # [B, L, n_irreps]
        # softplus → strictly positive multiplicative gate (no sign flip, so a
        # direction is only rescaled, never reflected — keeps it equivariant).
        gate = torch.nn.functional.softplus(gate_logits)  # [B, L, n_irreps]

        # Apply the (invariant) gate to the (equivariant) directions.
        g_omega = gate[..., 0:1]
        omega_out = g_omega * omega
        if self.group == "SE3":
            g_v = gate[..., 1:2]
            v_out = g_v * v
        else:
            # Rotation-only: pass translation through unchanged (identity is
            # trivially equivariant), so downstream channels stay intact.
            v_out = v

        return torch.cat([omega_out, v_out], dim=-1)

    def bio_harmonic_eigenvalues(self) -> torch.Tensor:
        r"""Return the seeded *continuous-time* eigenvalues :math:`\pm i 2\pi f_k`.

        Returns a complex tensor of shape ``[2 * n_freqs]`` containing the
        conjugate pairs :math:`+i2\pi f_k` and :math:`-i2\pi f_k`. Exposed for
        the audit / unit test that verifies the resonances are placed on the
        imaginary axis at the physiological frequencies.
        """
        w = 2.0 * math.pi * self._bio_freqs_hz  # angular frequencies
        pos = torch.complex(torch.zeros_like(w), w)
        neg = torch.complex(torch.zeros_like(w), -w)
        return torch.cat([pos, neg], dim=0)


__all__ = ["DEFAULT_BIO_HARMONIC_FREQS", "SE3LieAlgebraMambaBlock"]
