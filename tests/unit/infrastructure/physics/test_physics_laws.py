"""Cardinal physics laws — locked-in property tests.

This is the consolidation point for the four physics invariants any
reconstruction code in this repo relies on. Each invariant is tested
across multiple seeds and shapes to catch regressions that escape the
per-module tests.

Locks in ``TODO/backlog_test_suite_completion_roadmap.md`` PR-T1.

The invariants:

1. **FFT round-trip identity.** :math:`\\mathcal{F}^{-1}\\mathcal{F}x = x`
   for centered, ortho-normed 2D FFT (``fft2c`` / ``ifft2c``) on any
   complex tensor with shape ``[B, H, W]`` or ``[B, C, H, W]``.
2. **Parseval's theorem.** Energy preservation:
   :math:`\\|x\\|_2 = \\|\\mathcal{F}x\\|_2`. Required for AMP-safe
   loss scaling.
3. **SENSE adjoint inner-product.** For Hermitian adjoint operator
   pair :math:`(A, A^*)`:
   :math:`\\langle A x, y\\rangle = \\langle x, A^* y\\rangle`. This
   is the *defining* property of an adjoint and breaks if any
   centering / norm convention drifts.
4. **DC idempotency.** ``DC ∘ DC = DC`` for hard data consistency
   (projection). Soft DC is contractive but not idempotent — tested
   separately for boundedness.

These are deliberately written without ``hypothesis`` so the suite
runs under the default ``pyproject.toml`` ``pytest`` config (which
does not depend on the ``hypothesis`` plug-in).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

# Each test runs across these shapes to catch shape-conditional bugs.
SHAPES_2D: tuple[tuple[int, int, int], ...] = (
    (1, 8, 8),
    (1, 16, 16),
    (2, 32, 32),
    (1, 16, 32),  # non-square
    (1, 17, 15),  # non-power-of-two
)

SEEDS: tuple[int, ...] = (0, 1, 42, 1337)


# ----------------------------------------------------------------------------
# Invariant 1 — FFT round-trip identity
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES_2D)
@pytest.mark.parametrize("seed", SEEDS)
def test_fft_round_trip_identity_complex(shape: tuple[int, int, int], seed: int) -> None:
    """``ifft2c(fft2c(x)) == x`` for complex tensors."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, dtype=torch.complex64, generator=gen)
    recon = ifft2c(fft2c(x))
    assert torch.allclose(recon, x, rtol=1e-5, atol=1e-6), (
        f"FFT round-trip failed for shape={shape} seed={seed}: "
        f"max_err={torch.abs(recon - x).max().item():.2e}"
    )


@pytest.mark.parametrize("shape", SHAPES_2D)
@pytest.mark.parametrize("seed", SEEDS)
def test_fft_round_trip_identity_real_via_complex_coercion(
    shape: tuple[int, int, int], seed: int
) -> None:
    """Real-valued input coerced to complex still round-trips cleanly."""
    gen = torch.Generator().manual_seed(seed)
    x_real = torch.randn(*shape, generator=gen)
    x_complex = torch.complex(x_real, torch.zeros_like(x_real))
    recon = ifft2c(fft2c(x_complex))
    assert torch.allclose(recon, x_complex, rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------------
# Invariant 2 — Parseval (energy preservation)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES_2D)
@pytest.mark.parametrize("seed", SEEDS)
def test_parseval_energy_preservation(shape: tuple[int, int, int], seed: int) -> None:
    """``||x||_2 == ||fft2c(x)||_2`` for ortho-normed FFT."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, dtype=torch.complex64, generator=gen)
    e_spatial = torch.sum(torch.abs(x) ** 2)
    e_freq = torch.sum(torch.abs(fft2c(x)) ** 2)
    rel_err = torch.abs(e_spatial - e_freq) / e_spatial
    assert rel_err.item() < 1e-5, (
        f"Parseval violation: spatial={e_spatial.item():.4e}, "
        f"freq={e_freq.item():.4e}, rel_err={rel_err.item():.2e}"
    )


# ----------------------------------------------------------------------------
# Invariant 3 — Adjoint inner-product (Hermitian symmetry of FFT pair)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES_2D)
@pytest.mark.parametrize("seed", SEEDS)
def test_fft_pair_is_self_adjoint_under_ortho_norm(
    shape: tuple[int, int, int], seed: int
) -> None:
    """``<fft2c(x), y> == <x, ifft2c(y)>`` (the adjoint test).

    Because ``fft2c`` uses ``norm="ortho"``, the unitary FFT's adjoint is
    its inverse — this identity is what justifies using ``ifft2c`` as
    the adjoint operator in SENSE / DC.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, dtype=torch.complex64, generator=gen)
    y = torch.randn(*shape, dtype=torch.complex64, generator=gen)
    lhs = (fft2c(x).conj() * y).sum()
    rhs = (x.conj() * ifft2c(y)).sum()
    diff = torch.abs(lhs - rhs)
    norm = max(torch.abs(lhs).item(), torch.abs(rhs).item(), 1e-12)
    assert (diff.item() / norm) < 1e-5, (
        f"Adjoint identity broken: lhs={lhs.item()}, rhs={rhs.item()}, "
        f"rel_err={diff.item() / norm:.2e}"
    )


# ----------------------------------------------------------------------------
# Invariant 4 — Hard DC idempotency
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_data_consistency_idempotent(seed: int) -> None:
    """``DC(DC(x; y, m); y, m) == DC(x; y, m)`` for hard DC (projection).

    Hard DC replaces predicted k-space at sampled locations with the
    measured k-space. Applying it twice must equal applying it once.
    """
    from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

    gen = torch.Generator().manual_seed(seed)
    shape = (1, 16, 16)
    x_pred = torch.randn(*shape, dtype=torch.complex64, generator=gen)
    y_meas = torch.randn(*shape, dtype=torch.complex64, generator=gen)
    mask = torch.zeros(*shape)
    # Sample center lines + every 4th line — typical pattern
    mask[:, ::4, :] = 1.0
    mask[:, 6:10, :] = 1.0  # ACS

    def hard_dc(image: torch.Tensor) -> torch.Tensor:
        k_pred = fft2c(image)
        k_combined = torch.where(mask.to(torch.bool), y_meas, k_pred)
        return ifft2c(k_combined)

    once = hard_dc(x_pred)
    twice = hard_dc(once)
    assert torch.allclose(once, twice, rtol=1e-4, atol=1e-5), (
        f"Hard DC not idempotent at seed={seed}: "
        f"max_err={torch.abs(twice - once).max().item():.2e}"
    )


# ----------------------------------------------------------------------------
# Invariant 5 — Mask coverage invariants
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("acceleration", (2.0, 4.0, 8.0))
@pytest.mark.parametrize("seed", SEEDS)
def test_uniform_cartesian_mask_acceleration_within_tolerance(
    acceleration: float, seed: int
) -> None:
    """Cartesian-line mask sampling rate matches requested acceleration ±10%.

    Acceleration R means roughly 1/R lines are kept. Some slack is
    needed because ACS center lines are added on top of the under-
    sampling pattern, raising the sampled fraction slightly.
    """
    from spectramr.infrastructure.physics.sampling import MaskType, create_mask_generator

    gen = create_mask_generator(seed=seed)
    shape = (256, 256)
    mask = gen.generate_mask(
        mask_type=MaskType.UNIFORM_CARTESIAN,
        shape=shape,
        acceleration_factor=acceleration,
    )
    # Sampled fraction along the phase-encode (height) direction.
    sampled_lines = (mask.sum(dim=-1) > 0).float().mean().item()
    expected = 1.0 / acceleration
    # ACS center fraction is small but non-negligible; allow ±0.10 absolute slack.
    assert abs(sampled_lines - expected) < 0.10, (
        f"Cartesian mask R={acceleration}: sampled fraction "
        f"{sampled_lines:.3f}, expected ~{expected:.3f}"
    )


def test_zero_mask_implies_zero_observed_kspace() -> None:
    """``fft2c_masked(x, mask=0)`` returns zero (sanity boundary)."""
    from spectramr.infrastructure.physics.fft_ops import fft2c_masked

    x = torch.randn(2, 16, 16, dtype=torch.complex64)
    out = fft2c_masked(x, torch.zeros(2, 16, 16))
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)
