"""Unit tests for src/spectramr/infrastructure/physics/sense.py.

Covers:
  - SENSESubspaceProjector (complex and real-stacked inputs)
  - compute_sense_loss

Note: The fft_ops sense_forward / sense_adjoint pair already has a full
dot-product adjoint test in tests/physics/adjoint/test_sense_adjoint.py.
This file tests sense.py's OWN classes/functions (SENSESubspaceProjector,
compute_sense_loss) without duplicating that coverage.
"""

from __future__ import annotations

import pytest
import torch

from tests.utils.phantoms import random_complex
from tests.utils.shape_matrices import SHAPE_MATRIX_KSPACE, shape_id
from tests.utils.tolerances import tol_for


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_complex_kspace(B: int, C: int, H: int, W: int, seed: int = 0) -> torch.Tensor:
    """Random complex k-space [B, C, H, W]."""
    g = torch.Generator().manual_seed(seed)
    return random_complex((B, C, H, W), dtype=torch.complex64, generator=g)


def _make_complex_smaps(B: int, C: int, H: int, W: int, seed: int = 42) -> torch.Tensor:
    """Unit-normalised complex sensitivity maps [B, C, H, W]."""
    g = torch.Generator().manual_seed(seed)
    s = random_complex((B, C, H, W), dtype=torch.complex64, generator=g)
    # Normalise per-coil magnitude so RSS ≈ 1 at every pixel
    norm = s.abs().norm(dim=1, keepdim=True).clamp_min(1e-8)
    return s / norm


def _real_stack(t: torch.Tensor) -> torch.Tensor:
    """Convert [B, C, H, W] complex64 → [B, 2C, H, W] real-stacked."""
    B, C, H, W = t.shape
    # view_as_real gives [B, C, H, W, 2]; permute to [B, C, 2, H, W] then reshape
    return (
        torch.view_as_real(t)  # [B, C, H, W, 2]
        .permute(0, 1, 4, 2, 3)  # [B, C, 2, H, W]
        .reshape(B, C * 2, H, W)
    )


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_canary_sense_imports() -> None:
    """Module-level imports succeed without error."""
    from spectramr.infrastructure.physics.sense import (
        SENSESubspaceProjector,
        compute_sense_loss,
    )

    assert SENSESubspaceProjector is not None
    assert compute_sense_loss is not None


@pytest.mark.physics
def test_canary_projector_complex_tiny() -> None:
    """SENSESubspaceProjector runs forward on the smallest valid complex input."""
    from spectramr.infrastructure.physics.sense import SENSESubspaceProjector

    proj = SENSESubspaceProjector()
    B, C, H, W = 1, 2, 8, 8
    kspace = _make_complex_kspace(B, C, H, W)
    smaps = _make_complex_smaps(B, C, H, W)
    out = proj(kspace, smaps)

    assert out.shape == kspace.shape, f"output shape {out.shape} != input {kspace.shape}"
    assert out.dtype == torch.complex64
    assert not torch.isnan(out).any(), "NaN in projector output"
    assert not torch.isinf(out).any(), "Inf in projector output"


@pytest.mark.physics
def test_canary_compute_sense_loss_tiny() -> None:
    """compute_sense_loss returns a scalar on minimal inputs."""
    from spectramr.infrastructure.physics.sense import compute_sense_loss

    B, C, H, W = 1, 2, 8, 8
    kspace_pred = _make_complex_kspace(B, C, H, W, seed=0)
    kspace_target = _make_complex_kspace(B, C, H, W, seed=1)
    smaps = _make_complex_smaps(B, C, H, W)

    loss = compute_sense_loss(kspace_pred, kspace_target, smaps)
    assert loss.ndim == 0, "loss must be a scalar"
    assert loss.item() >= 0.0, "L1 loss must be non-negative"
    assert not torch.isnan(loss), "NaN loss"


# ---------------------------------------------------------------------------
# Parametrised: realistic shapes / coil counts
# ---------------------------------------------------------------------------

_COIL_SHAPES: list[tuple[int, int, int, int]] = [
    (1, 2, 16, 16),
    (1, 4, 32, 32),
    (2, 8, 32, 32),
    (1, 16, 64, 64),
]


@pytest.mark.physics
@pytest.mark.parametrize(
    "shape",
    _COIL_SHAPES,
    ids=[shape_id(s) for s in _COIL_SHAPES],
)
def test_projector_shape_preserved_complex(shape: tuple[int, int, int, int]) -> None:
    """SENSESubspaceProjector output has same shape as input (complex path)."""
    from spectramr.infrastructure.physics.sense import SENSESubspaceProjector

    B, C, H, W = shape
    proj = SENSESubspaceProjector()
    kspace = _make_complex_kspace(B, C, H, W)
    smaps = _make_complex_smaps(B, C, H, W)
    out = proj(kspace, smaps)

    assert out.shape == kspace.shape
    assert not torch.isnan(out).any()


@pytest.mark.physics
@pytest.mark.parametrize(
    "shape",
    _COIL_SHAPES,
    ids=[shape_id(s) for s in _COIL_SHAPES],
)
def test_projector_shape_preserved_real_stacked(shape: tuple[int, int, int, int]) -> None:
    """SENSESubspaceProjector accepts real-stacked [B, 2C, H, W] inputs."""
    from spectramr.infrastructure.physics.sense import SENSESubspaceProjector

    B, C, H, W = shape
    proj = SENSESubspaceProjector()
    kspace_c = _make_complex_kspace(B, C, H, W)
    smaps_c = _make_complex_smaps(B, C, H, W)

    kspace_r = _real_stack(kspace_c)  # [B, 2C, H, W]
    smaps_r = _real_stack(smaps_c)  # [B, 2C, H, W]

    out = proj(kspace_r, smaps_r)
    assert out.shape == kspace_r.shape, f"real-stacked output {out.shape} != input {kspace_r.shape}"
    assert out.dtype == torch.float32


@pytest.mark.physics
@pytest.mark.parametrize(
    "shape",
    _COIL_SHAPES,
    ids=[shape_id(s) for s in _COIL_SHAPES],
)
def test_sense_loss_nonneg(shape: tuple[int, int, int, int]) -> None:
    """compute_sense_loss is non-negative for all tested shapes."""
    from spectramr.infrastructure.physics.sense import compute_sense_loss

    B, C, H, W = shape
    kspace_pred = _make_complex_kspace(B, C, H, W, seed=7)
    kspace_target = _make_complex_kspace(B, C, H, W, seed=8)
    smaps = _make_complex_smaps(B, C, H, W)

    loss = compute_sense_loss(kspace_pred, kspace_target, smaps)
    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# Sanity-shape test (in/out contract via shape_matrices)
# ---------------------------------------------------------------------------


@pytest.mark.physics
@pytest.mark.sanity_shape
@pytest.mark.parametrize(
    "shape",
    SHAPE_MATRIX_KSPACE,
    ids=[shape_id(s) for s in SHAPE_MATRIX_KSPACE],
)
def test_projector_sanity_shape_kspace(shape: tuple[int, int, int, int]) -> None:
    """SENSESubspaceProjector output shape matches k-space shape matrix."""
    from spectramr.infrastructure.physics.sense import SENSESubspaceProjector

    B, C, H, W = shape
    proj = SENSESubspaceProjector()
    kspace = _make_complex_kspace(B, C, H, W)
    smaps = _make_complex_smaps(B, C, H, W)
    out = proj(kspace, smaps)
    assert out.shape == (B, C, H, W)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_projector_edge_single_coil() -> None:
    """SENSESubspaceProjector handles a single coil (C=1) without error."""
    from spectramr.infrastructure.physics.sense import SENSESubspaceProjector

    proj = SENSESubspaceProjector()
    B, C, H, W = 1, 1, 16, 16
    kspace = _make_complex_kspace(B, C, H, W)
    smaps = _make_complex_smaps(B, C, H, W)
    out = proj(kspace, smaps)
    assert out.shape == (B, C, H, W)


@pytest.mark.physics
def test_projector_edge_batch_1() -> None:
    """SENSESubspaceProjector handles B=1 (smallest batch)."""
    from spectramr.infrastructure.physics.sense import SENSESubspaceProjector

    proj = SENSESubspaceProjector()
    kspace = _make_complex_kspace(1, 4, 16, 16)
    smaps = _make_complex_smaps(1, 4, 16, 16)
    out = proj(kspace, smaps)
    assert out.shape == kspace.shape


@pytest.mark.physics
def test_projector_edge_zero_kspace() -> None:
    """Projecting zero k-space should return zero k-space."""
    from spectramr.infrastructure.physics.sense import SENSESubspaceProjector

    proj = SENSESubspaceProjector()
    B, C, H, W = 1, 4, 16, 16
    kspace = torch.zeros(B, C, H, W, dtype=torch.complex64)
    smaps = _make_complex_smaps(B, C, H, W)
    out = proj(kspace, smaps)
    tol = tol_for(torch.complex64)
    assert torch.allclose(out.abs(), torch.zeros_like(out.abs()), atol=tol * 10), (
        f"Expected near-zero output for zero input, got max={out.abs().max():.3e}"
    )


@pytest.mark.physics
def test_sense_loss_edge_identical_inputs() -> None:
    """compute_sense_loss returns 0.0 when pred == target."""
    from spectramr.infrastructure.physics.sense import compute_sense_loss

    B, C, H, W = 1, 4, 16, 16
    kspace = _make_complex_kspace(B, C, H, W)
    smaps = _make_complex_smaps(B, C, H, W)
    loss = compute_sense_loss(kspace, kspace, smaps)
    assert loss.item() < tol_for(torch.float32), (
        f"Loss should be ~0 for identical inputs, got {loss.item():.3e}"
    )


@pytest.mark.physics
def test_sense_loss_edge_real_stacked_smaps() -> None:
    """compute_sense_loss handles real-stacked smaps (non-complex path)."""
    from spectramr.infrastructure.physics.sense import compute_sense_loss

    B, C, H, W = 1, 4, 16, 16
    kspace_pred = _make_complex_kspace(B, C, H, W, seed=3)
    kspace_target = _make_complex_kspace(B, C, H, W, seed=4)
    smaps_c = _make_complex_smaps(B, C, H, W)
    smaps_r = _real_stack(smaps_c)

    loss_c = compute_sense_loss(kspace_pred, kspace_target, smaps_c)
    loss_r = compute_sense_loss(kspace_pred, kspace_target, smaps_r)
    # Both code paths should produce a valid scalar
    assert loss_r.ndim == 0
    assert not torch.isnan(loss_r)
    # Sanity: values should be finite and non-negative
    assert loss_r.item() >= 0.0


# ---------------------------------------------------------------------------
# raises: documented preconditions
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_projector_raises_odd_channel_real_stacked() -> None:
    """SENSESubspaceProjector raises when real-stacked input has odd channels.

    Real-stacked format requires C_real = 2 * C_complex; an odd channel
    count is ill-formed and should surface an error (not silently truncate).
    """
    from spectramr.infrastructure.physics.sense import SENSESubspaceProjector

    proj = SENSESubspaceProjector()
    B, C_odd, H, W = 1, 3, 16, 16  # odd C → can't be real-stacked
    # Build a real (non-complex) tensor with odd channels
    kspace_r = torch.randn(B, C_odd, H, W, dtype=torch.float32)
    smaps_r = torch.randn(B, C_odd, H, W, dtype=torch.float32)
    with pytest.raises(Exception):
        proj(kspace_r, smaps_r)


# ---------------------------------------------------------------------------
# SENSE subspace projection is idempotent (projection property)
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_projector_idempotent() -> None:
    """Applying SENSESubspaceProjector twice equals applying it once.

    The SENSE subspace is an orthogonal projection; P(P(x)) = P(x).
    """
    from spectramr.infrastructure.physics.sense import SENSESubspaceProjector

    proj = SENSESubspaceProjector()
    B, C, H, W = 1, 4, 32, 32
    kspace = _make_complex_kspace(B, C, H, W)
    smaps = _make_complex_smaps(B, C, H, W)

    once = proj(kspace, smaps)
    twice = proj(once, smaps)

    tol = tol_for(torch.complex64) * 100  # 100× for accumulated numerical error
    assert torch.allclose(once, twice, atol=tol, rtol=tol), (
        f"Projection is not idempotent: max diff = {(once - twice).abs().max().item():.3e}"
    )


@pytest.mark.physics
def test_sense_loss_is_componentwise_not_modulus_l1() -> None:
    """compute_sense_loss is L1 over stacked (real, imag) channels.

    Doc-contract regression: the loss penalizes the real and imaginary parts
    independently via ``torch.view_as_real`` (component-wise ``|dRe| + |dImag|``),
    NOT the complex-modulus L1 ``mean(|z_pred - z_target|)``. The two differ
    (``|a| + |b|`` vs ``sqrt(a^2 + b^2)``); this test pins the documented
    component-wise contract so a future swap to modulus-L1 is a deliberate change.
    """
    import torch.nn.functional as F

    from spectramr.infrastructure.physics.fft_ops import ifft2c
    from spectramr.infrastructure.physics.sense import compute_sense_loss

    B, C, H, W = 1, 4, 16, 16
    pred = _make_complex_kspace(B, C, H, W, seed=1)
    target = _make_complex_kspace(B, C, H, W, seed=2)
    s = _make_complex_smaps(B, C, H, W, seed=3)

    loss = compute_sense_loss(pred, target, s)

    # Reconstruct the SENSE-combined images to build both candidate metrics.
    rss = torch.sqrt((s.abs() ** 2).sum(1, keepdim=True) + 1e-8)
    sn = s / rss
    ps = (ifft2c(pred) * sn.conj()).sum(1, keepdim=True)
    ts = (ifft2c(target) * sn.conj()).sum(1, keepdim=True)
    componentwise = F.l1_loss(torch.view_as_real(ps), torch.view_as_real(ts))
    modulus = (ps - ts).abs().mean()

    assert torch.allclose(loss, componentwise, atol=1e-6)  # matches implementation
    assert not torch.allclose(loss, modulus, atol=1e-4)  # and is NOT modulus-L1
