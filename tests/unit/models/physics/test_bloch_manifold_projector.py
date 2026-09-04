"""Unit tests for :class:`BlochManifoldProjector` (Idea 2 / B-2).

Direct-import tests (no factory / registry resolution) covering:
- registry membership + declared capabilities,
- on-manifold residual ~ 0,
- off-manifold residual non-increasing across GN inner iterations,
- approximate projection idempotency,
- parameter-box enforcement (T2 <= T1 barrier),
- multi-coil / complex shape preservation + phase retention,
- no-NaN and deterministic seeding,
- the unsupported-sequence raise (no silent fallback).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.physics.bloch_manifold_projector import BlochManifoldProjector
from spectramr.models.registry import MODEL_REGISTRY


def _on_manifold_image(
    proj: BlochManifoldProjector,
    m0: float = 1.2,
    t1: float = 900.0,
    t2: float = 70.0,
    b: int = 1,
    hw: int = 4,
) -> torch.Tensor:
    """Build a [b, K, hw, hw] image that lies exactly on the Bloch manifold."""
    k = proj.flip_angles_rad.numel()
    n = b * hw * hw
    m0t = torch.full((n, 1), m0)
    t1t = torch.full((n, 1), t1)
    t2t = torch.full((n, 1), t2)
    bank = proj._bloch_signal_bank(m0t, t1t, t2t)  # [n, K]
    return bank.reshape(b, hw, hw, k).permute(0, 3, 1, 2).contiguous()


def test_registered_with_capabilities():
    assert "bloch_manifold_projector" in MODEL_REGISTRY
    entry = MODEL_REGISTRY["bloch_manifold_projector"]
    assert entry["mode"] == "utility"
    caps = entry["capabilities"]
    assert caps.accepts_complex is True
    assert caps.input_domain == "image"
    assert caps.output_domain == "image"
    assert 2 in caps.spatial_dims and 3 in caps.spatial_dims


def test_on_manifold_residual_near_zero():
    proj = BlochManifoldProjector(n_inner_iterations=25)
    x = _on_manifold_image(proj)
    res = proj.residual(x)
    # On-manifold signal: residual is tiny relative to the signal scale.
    assert res.shape == (1, 1, 4, 4)
    assert float(res.mean()) < 1e-2
    assert float(res.mean()) < 0.2 * float(x.abs().mean())


def test_off_manifold_residual_non_increasing():
    """Manifold residual must be non-increasing as inner iterations grow."""
    k = BlochManifoldProjector().flip_angles_rad.numel()
    torch.manual_seed(7)
    x_off = torch.rand(1, k, 4, 4) * 0.05 + 0.01
    residuals = []
    for n_inner in (1, 2, 4, 8, 16):
        proj = BlochManifoldProjector(n_inner_iterations=n_inner)
        residuals.append(float(proj.residual(x_off).mean()))
    for i in range(len(residuals) - 1):
        assert residuals[i + 1] <= residuals[i] + 1e-6, residuals
    # And it should make genuine progress over the schedule.
    assert residuals[-1] < residuals[0]


def test_projection_approximately_idempotent():
    proj = BlochManifoldProjector(n_inner_iterations=25)
    x = _on_manifold_image(proj, b=2)
    px = proj(x)
    ppx = proj(px)
    rel = (ppx - px).abs().max() / (px.abs().max() + 1e-9)
    assert float(rel) < 5e-2
    # proj(x) lies on the manifold => its own residual is ~0.
    assert float(proj.residual(px).mean()) < 1e-2


def test_parameter_box_and_t1_t2_barrier_enforced():
    bounds = {"M0": (0.5, 1.5), "T1": (200.0, 1200.0), "T2": (20.0, 300.0)}
    proj = BlochManifoldProjector(n_inner_iterations=10, param_bounds=bounds)
    k = proj.flip_angles_rad.numel()
    torch.manual_seed(0)
    x = torch.rand(1, k, 5, 5) * 0.1
    bank, *_ = proj._to_bank(x.to(torch.float32))
    m0, t1, t2 = proj.fit_parameters(bank)
    assert float(m0.min()) >= bounds["M0"][0] - 1e-4
    assert float(m0.max()) <= bounds["M0"][1] + 1e-4
    assert float(t1.min()) >= bounds["T1"][0] - 1e-4
    assert float(t1.max()) <= bounds["T1"][1] + 1e-4
    # The T2 <= T1 barrier must hold everywhere.
    assert torch.all(t2 <= t1 + 1e-4)


def test_multicoil_complex_shape_and_phase_preserved():
    proj = BlochManifoldProjector(n_inner_iterations=6)
    k = proj.flip_angles_rad.numel()
    torch.manual_seed(3)
    x = torch.complex(
        torch.rand(2, k, 8, 8) * 0.05 + 0.01,
        torch.rand(2, k, 8, 8) * 0.05,
    )
    out = proj(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    # Complex projector re-attaches the original per-channel phase.
    assert torch.allclose(torch.angle(out), torch.angle(x), atol=1e-4)


def test_arbitrary_channel_count_is_shape_preserving():
    proj = BlochManifoldProjector(n_inner_iterations=4)
    for c in (1, 2, 4, 8):
        x = torch.rand(1, c, 6, 6) * 0.1
        out = proj(x)
        assert out.shape == x.shape


def test_no_nan_and_finite():
    proj = BlochManifoldProjector(n_inner_iterations=10)
    k = proj.flip_angles_rad.numel()
    torch.manual_seed(11)
    x = torch.rand(2, k, 8, 8) * 0.2
    out = proj(x)
    assert torch.isfinite(out).all()
    assert torch.isfinite(proj.residual(x)).all()


def test_deterministic_seeding():
    k = BlochManifoldProjector().flip_angles_rad.numel()
    torch.manual_seed(5)
    x = torch.rand(1, k, 6, 6) * 0.1
    p1 = BlochManifoldProjector(n_inner_iterations=8)
    p2 = BlochManifoldProjector(n_inner_iterations=8)
    out1 = p1(x.clone())
    out2 = p2(x.clone())
    assert torch.allclose(out1, out2, atol=1e-7)


def test_5d_depth_singleton_roundtrip():
    proj = BlochManifoldProjector(n_inner_iterations=4)
    k = proj.flip_angles_rad.numel()
    x = (torch.rand(1, k, 5, 5) * 0.1).unsqueeze(-1)  # [1,K,5,5,1]
    out = proj(x)
    assert out.shape == x.shape


def test_unsupported_sequence_raises():
    with pytest.raises(ValueError, match="not"):
        BlochManifoldProjector(pulse_sequence="epi")


def test_invalid_inner_iterations_raises():
    with pytest.raises(ValueError):
        BlochManifoldProjector(n_inner_iterations=0)


def test_residual_wrong_ndim_raises():
    proj = BlochManifoldProjector(n_inner_iterations=2)
    with pytest.raises(ValueError):
        proj(torch.rand(4, 4))  # 2D not allowed
