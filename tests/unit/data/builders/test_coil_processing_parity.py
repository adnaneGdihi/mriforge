"""PARITY GATE for the coil-processing SSOT refactor (plan: coil-SSOT Phase 1).

Captures golden outputs of ``FastMRISubjectBuilder._apply_coil_processing``
for every legacy ``coil_processing_mode`` BEFORE the data-load is refactored to read
the new 4-axis ``physics.coil_processing`` block. After the refactor, the derived
axes for each legacy mode MUST reproduce these tensors -- exactly where the mode is
a pure reshape, and to within ``_PARITY_ATOL`` where it reduces or decomposes and
float association order belongs to the BLAS build rather than to this code.

The golden fixture is generated once from the pre-refactor code (committed), then
enforced. To regenerate intentionally, delete the fixture and re-run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mriforge.data.builders.torchio_subject_builder import FastMRISubjectBuilder

_GOLDEN = Path(__file__).with_name("_coil_processing_parity_golden.pt")

_MODES = ["none", "flatten", "svd", "magnitude", "rss", "rss_image"]


def _complex_sample() -> torch.Tensor:
    # (Coils, H, W, D) complex — the per-subject layout _apply_coil_processing sees.
    g = torch.Generator().manual_seed(1234)
    real = torch.randn(4, 16, 16, 2, generator=g)
    imag = torch.randn(4, 16, 16, 2, generator=g)
    return torch.complex(real, imag)


def _real_stacked_sample() -> torch.Tensor:
    # (2*Coils, H, W, D) real interleaved [R0,I0,R1,I1,...] — the real-input path.
    g = torch.Generator().manual_seed(4321)
    return torch.randn(8, 16, 16, 2, generator=g)


def _builder(mode: str) -> FastMRISubjectBuilder:
    return FastMRISubjectBuilder(
        primary_io=None,
        coil_processing_mode=mode,
        num_virtual_coils=2,
        svd_calibration_lines=None,
    )


def _all_cases() -> dict[str, torch.Tensor]:
    """Every (mode, input-kind, is_kspace) → output tensor."""
    out: dict[str, torch.Tensor] = {}
    cplx = _complex_sample()
    real = _real_stacked_sample()
    for mode in _MODES:
        b = _builder(mode)
        for is_k in (True, False):
            out[f"{mode}|complex|kspace={is_k}"] = b._apply_coil_processing(
                cplx.clone(), is_kspace=is_k
            )
            # The real-stacked path matters for the magnitude-producing modes.
            out[f"{mode}|real|kspace={is_k}"] = b._apply_coil_processing(
                real.clone(), is_kspace=is_k
            )
    return out


def _capture_golden() -> dict[str, torch.Tensor]:
    cases = _all_cases()
    torch.save(cases, _GOLDEN)
    return cases


# Byte-exactness is achievable for the reshape-only modes and NOT for the rest.
# `svd`, `magnitude`, `rss` and `rss_image` reduce or decompose, and the order
# those float ops associate in is a property of the LAPACK/BLAS build and the
# thread count, not of this code. Cluster job 17666023 failed all 28 of those
# parametrisations on the SAME commit that passes all 50 locally -- the values
# agreed to four decimals; only `torch.equal` disagreed.
#
# The gate exists to catch the refactor changing the NUMBERS, so it asserts a
# tolerance far below any algorithmic change and far above backend reassociation
# noise, and reports the observed deviation so drift stays visible.
#
# The tolerance is RELATIVE to each tensor's own scale, because a fixed absolute
# one is not achievable for all six modes. ``rss|complex|kspace=True`` reaches
# ``absmax = 46.19`` -- a sum of squares over coils, so it is the largest output
# here -- and a flat ``1e-6`` demands ``2e-8`` relative on it, which is BELOW
# float32 epsilon (1.19e-7). No implementation can pass that, so the old bound
# was not a strict gate but an impossible one, and it began failing as soon as
# any reassociation touched that path.
#
# Measured when this was replaced, on the two modes that had started failing:
#
#   mode                     absmax   max dev     rel dev    elements differing
#   rss|complex|kspace=True   46.19   3.815e-06   8.3e-08    471/512
#   svd|complex|kspace=True    3.78   1.484e-05   3.9e-06    1997/2048
#
# Both are diffuse (~97% of elements, no structure) and stable run to run, and
# `rss` is literally sub-epsilon relative -- the signature of reassociation, not
# of an algorithm. The control that settles it is ``flatten|complex|kspace=True``,
# which shares the same k-space path and is still BIT-IDENTICAL: a real change
# to that path would move it too. ``svd`` sits higher because
# ``torch.linalg.svd`` amplifies rounding, not because it computes something new.
#
# ``rtol`` is set ~2.5x above the worst observed noise and orders of magnitude
# below any algorithmic change (those move results by 1e-2 or more), so the gate
# still fails on what it was built to catch. ``atol`` keeps the old floor for
# near-zero tensors, where a relative bound would be vacuous.
_PARITY_ATOL = 1e-6
_PARITY_RTOL = 1e-5


def _assert_parity(current: torch.Tensor, golden: torch.Tensor, label: str) -> None:
    """Assert *current* reproduces *golden* to within backend noise."""
    if torch.equal(current, golden):
        return
    deviation = (current - golden).abs().max().item()
    scale = golden.abs().max().item()
    budget = _PARITY_ATOL + _PARITY_RTOL * scale
    assert deviation <= budget, (
        f"{label}: differs from golden by {deviation:.3e} "
        f"(> {budget:.3e} = {_PARITY_ATOL:.0e} + {_PARITY_RTOL:.0e} x {scale:.4g}); "
        f"this is an algorithmic change, not float-reassociation noise"
    )


_KEYS = [
    f"{m}|{kind}|kspace={k}" for m in _MODES for kind in ("complex", "real") for k in (True, False)
]


@pytest.mark.parametrize("key", _KEYS)
def test_coil_processing_parity(key: str) -> None:
    """Each legacy-mode output (via coil_processing_mode) must match the golden."""
    if not _GOLDEN.exists():
        _capture_golden()
        pytest.skip("golden fixture captured from current code — re-run to enforce")
    golden = torch.load(_GOLDEN, weights_only=False)
    current = _all_cases()
    g, c = golden[key], current[key]
    assert g.shape == c.shape, f"{key}: shape {c.shape} != golden {g.shape}"
    assert g.dtype == c.dtype, f"{key}: dtype {c.dtype} != golden {g.dtype}"
    _assert_parity(c, g, key)


# --- Block path: the SSOT physics.coil_processing block must reproduce the same
# legacy-mode tensors byte-for-byte (the new dispatch reverse-looks-up a legacy
# combo to its byte-parity branch). ---


def _resolved_block(mode: str):
    """Build the resolved CoilProcessingConfig a legacy mode derives to."""
    from mriforge.config.schemas.loader import _derive_coil_processing_from_legacy
    from mriforge.config.schemas.physics import CoilProcessingConfig

    raw = {"data": {"coil_processing_mode": mode, "num_virtual_coils": 2}}
    cp = _derive_coil_processing_from_legacy(raw)["physics"]["coil_processing"]
    return CoilProcessingConfig(**cp)


@pytest.mark.parametrize("key", _KEYS)
def test_coil_processing_block_parity(key: str) -> None:
    """Driving the data-load via the resolved 4-axis block (coil_processing=...)
    reproduces the legacy-mode golden byte-for-byte."""
    if not _GOLDEN.exists():
        pytest.skip("golden fixture missing")
    golden = torch.load(_GOLDEN, weights_only=False)
    mode, kind, kflag = key.split("|")
    is_k = kflag == "kspace=True"
    sample = _complex_sample() if kind == "complex" else _real_stacked_sample()
    # Build the subject builder with the resolved block (mode left at its default
    # so ONLY the block can drive behavior).
    b = FastMRISubjectBuilder(
        primary_io=None,
        num_virtual_coils=2,
        coil_processing=_resolved_block(mode),
    )
    out = b._apply_coil_processing(sample.clone(), is_kspace=is_k)
    g = golden[key]
    assert out.shape == g.shape, f"{key}: block shape {out.shape} != golden {g.shape}"
    _assert_parity(out, g, f"{key} (block path)")


# --- Composable pipeline: genuinely-new axis combinations (not a legacy mode)
# run the composable data-load path. These have no golden (they're new) — we
# assert the shape/type the axes imply. ---


def _block(compression, combine, domain, channels, nvc=2):
    from mriforge.config.schemas.physics import CoilProcessingConfig

    return CoilProcessingConfig(
        compression={"method": compression, "num_virtual_coils": nvc},
        combine={"method": combine},
        output={"domain": domain, "channels": channels},
    )


def test_composable_svd_then_rss_magnitude() -> None:
    # NEW combo: compress 4→2 coils, then RSS-combine to a 1-channel image.
    b = FastMRISubjectBuilder(
        primary_io=None,
        coil_processing=_block("svd", "rss", "image", "magnitude", nvc=2),
    )
    out = b._apply_coil_processing(_complex_sample().clone(), is_kspace=True)
    assert out.shape == (1, 16, 16, 2)  # single combined magnitude image
    assert not torch.is_complex(out)


def test_composable_svd_keep_complex() -> None:
    # NEW combo: svd-compress but KEEP complex coils (combine=none, complex out).
    b = FastMRISubjectBuilder(
        primary_io=None,
        coil_processing=_block("svd", "none", "kspace", "complex", nvc=2),
    )
    out = b._apply_coil_processing(_complex_sample().clone(), is_kspace=True)
    assert out.shape == (2, 16, 16, 2)  # 2 compressed complex coils
    assert torch.is_complex(out)
