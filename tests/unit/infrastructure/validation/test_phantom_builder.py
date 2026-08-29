"""Unit tests for the Tier-2 probe's Shepp-Logan phantom helper.

Verifies that:
* The phantom respects the requested shape for both 4-D and 5-D ranks.
* It is **not** white noise — i.e. has visible structure (variance,
  bounded range, distinct foreground / background).
* Complex output with ``add_phase=True`` produces a non-trivial imag
  channel.
* Invalid ranks raise ``ValueError``.
* Image-save degrades gracefully when matplotlib is missing or the
  tensor shape is unsupported.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mriforge.infrastructure.validation.phantom_builder import (  # noqa: E402
    save_probe_images,
    synthetic_phantom,
)

# ── synthetic_phantom shape contract ───────────────────────────────────


def test_phantom_4d_shape_contract() -> None:
    p = synthetic_phantom((2, 3, 32, 32))
    assert p.shape == (2, 3, 32, 32)
    assert p.dtype == torch.float32


def test_phantom_5d_shape_contract() -> None:
    p = synthetic_phantom((1, 1, 8, 16, 16))
    assert p.shape == (1, 1, 8, 16, 16)


def test_phantom_invalid_rank_raises() -> None:
    """F-PHANTOM-RANK3 / 2026-05-20 — rank-3 is now valid (1-D models like
    HRF). Invalid rank is now rank-2 or below / rank-6 or above.
    """
    with pytest.raises(ValueError, match="rank"):
        synthetic_phantom((1, 2))  # rank-2


def test_phantom_3d_shape_contract() -> None:
    """F-PHANTOM-RANK3 / 2026-05-20 — synthetic_phantom must support
    rank-3 ``[B, C, L]`` for 1-D models (HRF time courses, tangent-score
    networks). Smoke audit 20260518 surfaced the rank-3 failure as
    ``synthetic_phantom failed (shape must be 4-D ...) ; falling back to
    randn`` which lost the deterministic spectral content.
    """
    p = synthetic_phantom((2, 1, 32))
    assert p.shape == (2, 1, 32)
    # Values lie within the Shepp-Logan magnitude range (real, [0, 1])
    # since we sliced a row of the canonical 2-D phantom.
    assert p.min() >= 0.0
    assert p.max() <= 1.0 + 1e-6


def test_phantom_complex_dtype_with_phase() -> None:
    p = synthetic_phantom((1, 1, 16, 16), dtype="complex64", add_phase=True)
    assert p.is_complex()
    # Magnitude should still resemble the phantom.
    mag = p.abs()
    assert mag.min() >= 0.0
    assert mag.max() <= 1.0 + 1e-6
    # Imag part must be non-trivial when add_phase=True.
    assert p.imag.abs().mean() > 0.0


def test_phantom_3d_complex_with_phase_has_nonzero_imag() -> None:
    """Regression (2026-07-14): add_phase was ignored on the rank-3 path.

    The 3-D branch returned before the phase ramp, so ``add_phase=True`` handed
    back a zero-imag "complex" tensor. Every other rank honoured the knob, so
    the 1-D probe silently exercised a purely real signal — an unwired knob
    (pitfall #15) on the one input layout MRF/complex 1-D models actually use.
    Both ranks now share ``_apply_dtype_and_phase``.
    """
    p = synthetic_phantom((2, 1, 32), dtype="complex64", add_phase=True)
    assert p.is_complex()
    assert p.imag.abs().mean() > 0.0
    assert p.abs().max() <= 1.0 + 1e-6


def test_phantom_complex_without_phase_has_zero_imag() -> None:
    p = synthetic_phantom((1, 1, 16, 16), dtype="complex64", add_phase=False)
    assert torch.allclose(p.imag, torch.zeros_like(p.imag))


# ── It's actually a phantom, not white noise ──────────────────────────


def test_phantom_is_not_white_noise() -> None:
    """Two key signatures: bounded range AND a discrete intensity histogram."""
    p = synthetic_phantom((1, 1, 64, 64))
    # White noise would not be in [0, 1].
    assert p.min().item() >= 0.0
    assert p.max().item() <= 1.0 + 1e-6
    # Phantom has only ~10 distinct grey levels (10 ellipses summed).
    n_unique = torch.unique(p).numel()
    assert n_unique < 50, (
        f"Expected a small number of distinct intensities (< 50) for a "
        f"Shepp-Logan phantom, got {n_unique} — looks like noise."
    )


def test_phantom_has_non_trivial_foreground_background() -> None:
    """Centre (brain) must be brighter than corner (background)."""
    p = synthetic_phantom((1, 1, 64, 64)).squeeze()
    centre = float(p[32, 32])
    corner = float(p[2, 2])
    assert centre > corner + 0.05, (
        f"Centre intensity {centre} should be at least 0.05 brighter than "
        f"corner {corner} for a Shepp-Logan phantom."
    )


def test_phantom_is_deterministic() -> None:
    """Same shape should produce the same tensor (no random seed)."""
    a = synthetic_phantom((1, 1, 32, 32))
    b = synthetic_phantom((1, 1, 32, 32))
    assert torch.equal(a, b)


def test_phantom_5d_slices_all_identical() -> None:
    """Volume slices are tiles of the same 2D phantom (intentional)."""
    p = synthetic_phantom((1, 1, 4, 16, 16))
    for d in range(1, 4):
        assert torch.equal(p[0, 0, 0], p[0, 0, d])


# ── Non-square patches stay recognisable ──────────────────────────────


def test_phantom_non_square_centre_crop() -> None:
    p = synthetic_phantom((1, 1, 32, 16))
    assert p.shape == (1, 1, 32, 16)
    # Should still have visible structure: more than 5 unique levels.
    assert torch.unique(p).numel() > 5


# ── save_probe_images is best-effort ──────────────────────────────────


def test_save_probe_images_returns_empty_when_matplotlib_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If matplotlib import fails, the helper just returns []."""
    import builtins

    real_import = builtins.__import__

    def _no_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_matplotlib)
    written = save_probe_images(
        str(tmp_path),
        "arm",
        input_tensor=synthetic_phantom((1, 1, 16, 16)),
    )
    assert written == []


def test_save_probe_images_writes_pngs_when_matplotlib_present(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    inp = synthetic_phantom((1, 1, 16, 16))
    out = synthetic_phantom((1, 1, 16, 16))
    written = save_probe_images(
        str(tmp_path),
        "arm_x",
        input_tensor=inp,
        output_tensor=out,
        target_tensor=inp,
    )
    assert len(written) == 3
    for path in written:
        p = Path(path)
        assert p.exists()
        assert p.stat().st_size > 0
        assert p.suffix == ".png"
        # Filename schema: <arm>_<role>.png
        assert p.stem.startswith("arm_x_")


def test_save_probe_images_handles_complex_tensor(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    cplx = synthetic_phantom((1, 1, 16, 16), dtype="complex64", add_phase=True)
    written = save_probe_images(str(tmp_path), "cplx", input_tensor=cplx)
    # Complex path emits at least one PNG (mag and phase tiled).
    assert len(written) >= 1
