"""Unit tests for the Tier-2 probe's Shepp-Logan phantom helper.

Targets :mod:`mriforge.infrastructure.validation.phantom_builder`:

* :func:`synthetic_phantom` — shape contract for 4-D / 5-D, complex
  output, determinism, edge-case ranks.
* :func:`save_probe_images` — best-effort PNG dump, complex tensors,
  graceful matplotlib-missing degradation.

The module under test is a Tier-2 audit dependency added on the ``dev``
branch. When running on a tree that does not yet ship the module, the
top-level ``pytest.importorskip`` cleanly skips every test rather than
failing collection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
phantom_builder = pytest.importorskip(
    "mriforge.infrastructure.validation.phantom_builder"
)

synthetic_phantom = phantom_builder.synthetic_phantom
save_probe_images = phantom_builder.save_probe_images


# ── synthetic_phantom shape + dtype contract ───────────────────────────


class TestPhantomShapeContract:
    """Shape / dtype matches what the caller declared."""

    def test_3d_shape_matches(self) -> None:
        """Rank-3 (B, C, L) is SUPPORTED, not an error (F-PHANTOM-RANK3).

        Replaces the old ``TestPhantomInvalidInput::test_rank_3_shape_raises``,
        which asserted the opposite and was red on ``dev`` (issue #286). 1-D
        models (HRF time courses, MRF fingerprints, tangent-score nets) probe
        with this layout; before support landed they fell back to ``randn`` and
        lost the phantom's deterministic spectral content.
        """
        p = synthetic_phantom((1, 2, 3))
        assert p.shape == (1, 2, 3)
        assert p.dtype == torch.float32

    def test_3d_complex_with_phase_has_nonzero_imag(self) -> None:
        """Regression: ``add_phase`` was a silent no-op on the rank-3 path.

        The 3-D branch returned early, before the phase ramp, so a complex 1-D
        phantom came back with a **zero imaginary part** — a real signal wearing
        a complex dtype, which is precisely what this module exists to avoid
        (see the module docstring). MRF fingerprints are complex, so the probe
        needs a genuine phase here or it cannot exercise the complex path.
        """
        p = synthetic_phantom((1, 1, 32), dtype="complex64", add_phase=True)
        assert p.is_complex()
        assert p.shape == (1, 1, 32)
        assert p.imag.abs().mean().item() > 0.0
        # Magnitude is still the phantom row: bounded in [0, 1].
        assert p.abs().max().item() <= 1.0 + 1e-6

    def test_3d_complex_without_phase_has_zero_imag(self) -> None:
        p = synthetic_phantom((1, 1, 32), dtype="complex64", add_phase=False)
        assert p.is_complex()
        assert torch.allclose(p.imag, torch.zeros_like(p.imag))

    def test_4d_shape_matches(self) -> None:
        p = synthetic_phantom((2, 3, 32, 32))
        assert p.shape == (2, 3, 32, 32)
        assert p.dtype == torch.float32

    def test_5d_shape_matches(self) -> None:
        p = synthetic_phantom((1, 1, 8, 16, 16))
        assert p.shape == (1, 1, 8, 16, 16)
        assert p.dtype == torch.float32

    def test_complex64_dtype(self) -> None:
        p = synthetic_phantom((1, 1, 16, 16), dtype="complex64")
        assert p.is_complex()
        assert p.dtype == torch.complex64
        assert p.shape == (1, 1, 16, 16)

    def test_complex_with_phase_has_nonzero_imag(self) -> None:
        p = synthetic_phantom((1, 1, 16, 16), dtype="complex64", add_phase=True)
        assert p.is_complex()
        assert p.imag.abs().mean().item() > 0.0
        # Magnitude still resembles the phantom: bounded in [0, 1].
        mag = p.abs()
        assert mag.min().item() >= 0.0
        assert mag.max().item() <= 1.0 + 1e-6

    def test_complex_without_phase_has_zero_imag(self) -> None:
        p = synthetic_phantom((1, 1, 16, 16), dtype="complex64", add_phase=False)
        assert torch.allclose(p.imag, torch.zeros_like(p.imag))

    def test_non_square_4d_shape(self) -> None:
        p = synthetic_phantom((1, 1, 32, 16))
        assert p.shape == (1, 1, 32, 16)
        # Should still resemble a phantom (multiple intensity levels).
        assert torch.unique(p).numel() > 5


# ── synthetic_phantom invalid input → ValueError ───────────────────────


class TestPhantomInvalidInput:
    """Rank / shape errors must be loud, never silent.

    Rank-3 is deliberately absent from this class: ``(B, C, L)`` is a
    *supported* layout (F-PHANTOM-RANK3 / 2026-05-20), pinned positively by
    ``TestPhantomShapeContract.test_3d_shape_matches`` below. The old
    ``test_rank_3_shape_raises`` here contradicted that contract and had been
    red on ``dev`` ever since 1-D support landed (issue #286).
    """

    def test_rank_2_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="rank"):
            synthetic_phantom((32, 32))

    def test_rank_6_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="rank"):
            synthetic_phantom((1, 1, 1, 1, 32, 32))

    def test_empty_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="rank"):
            synthetic_phantom(())


# ── synthetic_phantom: structure-not-noise contract ───────────────────


class TestPhantomStructure:
    """A phantom must look like a phantom — bounded range + few levels."""

    def test_values_bounded_in_unit_interval(self) -> None:
        p = synthetic_phantom((1, 1, 64, 64))
        assert p.min().item() >= 0.0
        assert p.max().item() <= 1.0 + 1e-6

    def test_few_discrete_intensities(self) -> None:
        """Shepp-Logan is ~10 ellipses summed → low cardinality histogram."""
        p = synthetic_phantom((1, 1, 64, 64))
        n_unique = torch.unique(p).numel()
        assert n_unique < 50, (
            f"Expected a discrete-level phantom (<50 unique values); "
            f"got {n_unique} — output looks like noise, not Shepp-Logan."
        )

    def test_centre_brighter_than_corner(self) -> None:
        """The skull / brain interior must exceed the air background."""
        p = synthetic_phantom((1, 1, 64, 64)).squeeze()
        centre = float(p[32, 32])
        corner = float(p[2, 2])
        assert centre > corner + 0.05, (
            f"Centre intensity {centre} must exceed corner intensity {corner} "
            f"by > 0.05 for a Shepp-Logan phantom."
        )


# ── Determinism contract ──────────────────────────────────────────────


class TestPhantomDeterminism:
    """Same input → byte-identical output, no hidden RNG."""

    def test_same_shape_same_bytes(self) -> None:
        a = synthetic_phantom((1, 1, 32, 32))
        b = synthetic_phantom((1, 1, 32, 32))
        assert torch.equal(a, b)

    def test_seed_argument_is_ignored_for_determinism(self) -> None:
        """The ``seed`` arg is reserved; outputs must match regardless."""
        a = synthetic_phantom((1, 1, 32, 32), seed=0)
        b = synthetic_phantom((1, 1, 32, 32), seed=12345)
        assert torch.equal(a, b)

    def test_volume_slices_are_tiled(self) -> None:
        """5-D output tiles the same 2D phantom along depth — slice 0 == slice d."""
        p = synthetic_phantom((1, 1, 4, 16, 16))
        for d in range(1, 4):
            assert torch.equal(p[0, 0, 0], p[0, 0, d])


# ── Dtype fallback (CLAUDE.md #9 trade-off) ───────────────────────────


class TestPhantomDtypeFallback:
    """Unrecognised dtypes warn-and-fall-back rather than raise.

    The phantom builder is allowed to fall back (see its docstring),
    because the probe wrapper itself swallows phantom failures (falls
    back to ``randn``). What we lock in: a clearly-bogus dtype must not
    explode, and must yield a tensor of the requested shape.
    """

    def test_unknown_dtype_string_falls_back_to_float32(self) -> None:
        out = synthetic_phantom((1, 1, 16, 16), dtype="not_a_real_dtype")
        assert out.dtype == torch.float32
        assert out.shape == (1, 1, 16, 16)

    def test_float64_dtype_is_cast(self) -> None:
        """``float64`` is a real torch dtype — best-effort cast must succeed."""
        out = synthetic_phantom((1, 1, 16, 16), dtype="float64")
        assert out.dtype == torch.float64
        assert out.shape == (1, 1, 16, 16)


# ── save_probe_images: best-effort PNG dump ───────────────────────────


class TestSaveProbeImages:
    """The image-save helper must NEVER raise — it's a diagnostic aid."""

    def test_returns_empty_when_matplotlib_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

    def test_writes_three_pngs_when_all_roles_supplied(self, tmp_path: Path) -> None:
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
            assert p.stem.startswith("arm_x_")

    def test_writes_only_supplied_roles(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        written = save_probe_images(
            str(tmp_path),
            "arm_y",
            input_tensor=synthetic_phantom((1, 1, 16, 16)),
        )
        assert len(written) == 1
        assert Path(written[0]).stem == "arm_y_input"

    def test_handles_complex_tensor_without_raising(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        cplx = synthetic_phantom((1, 1, 16, 16), dtype="complex64", add_phase=True)
        written = save_probe_images(str(tmp_path), "cplx", input_tensor=cplx)
        # Complex path emits at least one PNG (magnitude / phase tiled).
        assert len(written) >= 1

    def test_unsupported_rank_silently_skipped(self, tmp_path: Path) -> None:
        """A 1-D tensor cannot be plotted — helper must return empty list,
        not raise."""
        pytest.importorskip("matplotlib")
        weird = torch.zeros(8)
        written = save_probe_images(str(tmp_path), "arm_z", input_tensor=weird)
        assert written == []

    def test_non_tensor_input_silently_skipped(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        written = save_probe_images(
            str(tmp_path), "arm_w", input_tensor="not a tensor"  # type: ignore[arg-type]
        )
        assert written == []

    def test_creates_output_directory_if_missing(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        nested = tmp_path / "deep" / "nested" / "out"
        written = save_probe_images(
            str(nested),
            "arm",
            input_tensor=synthetic_phantom((1, 1, 16, 16)),
        )
        assert len(written) == 1
        assert nested.is_dir()
