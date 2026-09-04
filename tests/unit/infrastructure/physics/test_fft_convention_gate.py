"""Every public ``fft_ops`` transform puts DC where its NAME says it does.

``fft_ops`` is the FFT SSOT non-negotiable 2 points every caller at, and until
#1350 it exported an *uncentered* family under names (``fft_volume_spatial``,
``FFTTransformer.fft2``) that read as canonical and sat as peers of the centred
``fft2c``. The uncentered pair round-trips perfectly, so the skew is silent: it
only surfaces where an uncentered spectrum meets a *centred* artifact -- a mask
from ``MaskGenerator`` (centred for 9 of its 10 mask types), a measured k-space,
or a model trained on centred data. A contiguous ACS block loses 90 % of its
intended energy that way.

A naming convention nothing checks is a comment. This is the check: for a
constant input the spectrum is a delta at DC, so its ``argmax`` position *is*
the convention, and the assertion is against the name rather than against a
golden array.

Per non-negotiable 15 the detector ships with planted violations that turn it
red -- in BOTH directions (a centred-named transform missing its shifts, and an
uncentered-named one that has them), and at every spatial rank the family
covers (1-D along the partition axis, 2-D in-plane, 3-D volumetric). A gate is
only a gate for the violation shape you have watched it fail on.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import (
    FFTTransformer,
    fft2c,
    fft_volume_full3d_uncentered,
    fft_volume_slice_dimension_uncentered,
    fft_volume_spatial_uncentered,
    fftnc,
    ifft2c,
    ifft_volume_full3d_uncentered,
    ifft_volume_slice_dimension_uncentered,
    ifft_volume_spatial_uncentered,
    ifftnc,
)

#: ``[B, C, D, H, W]``. Odd sizes on purpose: ``N // 2`` and ``0`` coincide for
#: ``N == 1`` and the shift is its own inverse for even ``N``, so an even-only
#: shape would let a half-wrong implementation pass.
SHAPE_5D = (2, 1, 5, 6, 7)
SHAPE_4D = (2, 1, 6, 7)


def _dc_index(spectrum: torch.Tensor, axes: tuple[int, ...]) -> tuple[int, ...]:
    """Position of the peak of ``|spectrum|`` along ``axes``.

    All other axes are reduced away first, so the result is the DC coordinate
    in the transformed subspace alone.
    """
    mag = spectrum.abs()
    keep = set(axes)
    for ax in sorted((a for a in range(mag.ndim) if a not in keep), reverse=True):
        mag = mag.amax(dim=ax)
    flat = int(mag.reshape(-1).argmax().item())
    out: list[int] = []
    for size in reversed([mag.shape[i] for i in range(mag.ndim)]):
        out.append(flat % size)
        flat //= size
    return tuple(reversed(out))


def assert_forward_dc_matches_name(
    fn, *, centred: bool, shape: tuple[int, ...], axes: tuple[int, ...]
) -> None:
    """A constant input must transform to a delta at the NAME's DC location."""
    x = torch.ones(*shape)
    got = _dc_index(fn(x), axes)
    want = tuple(shape[a] // 2 if centred else 0 for a in axes)
    label = "centred" if centred else "uncentered"
    assert got == want, (
        f"{getattr(fn, '__name__', fn)} is named {label}, so DC belongs at "
        f"{want}; the transform put it at {got}."
    )


def assert_inverse_dc_matches_name(
    fn, *, centred: bool, shape: tuple[int, ...], axes: tuple[int, ...]
) -> None:
    """A delta at the NAME's DC location must invert to a flat field.

    The forward check does not constrain the inverse: a pair can be mutually
    consistent and both wrong. Feeding the delta the *name* implies is what ties
    the inverse to the convention rather than to its own partner.
    """
    k = torch.zeros(*shape, dtype=torch.complex64)
    idx: list[object] = [slice(None)] * len(shape)
    for a in axes:
        idx[a] = shape[a] // 2 if centred else 0
    k[tuple(idx)] = 1.0
    out = fn(k)
    # MAGNITUDE alone cannot see this. The inverse of a delta is a complex
    # exponential whose modulus is constant *wherever the delta sits* -- moving
    # it only tilts the phase. Both planted inverse violations passed a
    # magnitude-spread assertion, which is why they are in the file. Compare the
    # signed components: a delta at true DC inverts to a real positive constant,
    # a misplaced one to a ramp.
    parts = (out.real, out.imag) if torch.is_complex(out) else (out,)
    spread = max((p.amax() - p.amin()).item() for p in parts)
    label = "centred" if centred else "uncentered"
    assert spread < 1e-5, (
        f"{getattr(fn, '__name__', fn)} is named {label}, so a delta at its "
        f"DC location must invert to a constant field; component spread was "
        f"{spread:.3e} (a ramp, i.e. the delta was not at this transform's DC)."
    )


# --------------------------------------------------------------------------
# The real transforms
# --------------------------------------------------------------------------

FORWARD_CASES = [
    ("fft2c", fft2c, True, SHAPE_4D, (2, 3)),
    ("fftnc/2d", fftnc, True, SHAPE_4D, (2, 3)),
    ("fftnc/3d", fftnc, True, SHAPE_5D, (2, 3, 4)),
    ("fft_volume_spatial_uncentered", fft_volume_spatial_uncentered, False, SHAPE_5D, (3, 4)),
    ("fft_volume_full3d_uncentered", fft_volume_full3d_uncentered, False, SHAPE_5D, (2, 3, 4)),
    (
        "fft_volume_slice_dimension_uncentered",
        fft_volume_slice_dimension_uncentered,
        False,
        SHAPE_5D,
        (2,),
    ),
]

INVERSE_CASES = [
    ("ifft2c", ifft2c, True, SHAPE_4D, (2, 3)),
    ("ifftnc/2d", ifftnc, True, SHAPE_4D, (2, 3)),
    ("ifftnc/3d", ifftnc, True, SHAPE_5D, (2, 3, 4)),
    ("ifft_volume_spatial_uncentered", ifft_volume_spatial_uncentered, False, SHAPE_5D, (3, 4)),
    ("ifft_volume_full3d_uncentered", ifft_volume_full3d_uncentered, False, SHAPE_5D, (2, 3, 4)),
    (
        "ifft_volume_slice_dimension_uncentered",
        ifft_volume_slice_dimension_uncentered,
        False,
        SHAPE_5D,
        (2,),
    ),
]


@pytest.mark.parametrize(
    "name,fn,centred,shape,axes", FORWARD_CASES, ids=[c[0] for c in FORWARD_CASES]
)
def test_forward_transform_dc_matches_its_name(name, fn, centred, shape, axes) -> None:
    assert_forward_dc_matches_name(fn, centred=centred, shape=shape, axes=axes)


@pytest.mark.parametrize(
    "name,fn,centred,shape,axes", INVERSE_CASES, ids=[c[0] for c in INVERSE_CASES]
)
def test_inverse_transform_dc_matches_its_name(name, fn, centred, shape, axes) -> None:
    assert_inverse_dc_matches_name(fn, centred=centred, shape=shape, axes=axes)


def test_dims_override_is_centred_and_preserves_the_channel_axis() -> None:
    """``fftnc(dims=...)`` is how a uncentered call is corrected without also
    changing the channel layout.

    ``fft2c`` is *not* a drop-in replacement for an uncentered 2-D call: it runs
    ``_to_complex``, which reinterprets an even channel count as interleaved
    real/imag and halves it. At the migrated sites the channel axis is load
    bearing -- ``processing_strategies`` reshapes the spectrum with ``C`` read
    off the *input*, and the DC paths multiply the spectrum by a mask -- so a
    silent 2 -> 1 would be a second, unrelated defect smuggled into a
    centering fix.
    """
    x = torch.randn(2, 2, 16, 16)
    assert fftnc(x, dims=(-2, -1)).shape == x.shape
    assert fft2c(x).shape == (2, 1, 16, 16), "fft2c is expected to halve even C"

    # in-plane transform of a VOLUME: the default would be 3-D instead
    vol = torch.randn(2, 8, 5, 6, 7)
    inplane = fftnc(vol, dims=(-2, -1))
    assert inplane.shape == vol.shape
    assert_forward_dc_matches_name(
        lambda t: fftnc(t, dims=(-2, -1)), centred=True, shape=SHAPE_5D, axes=(3, 4)
    )
    # and it really is only a shift away from the uncentered transform
    assert torch.allclose(
        torch.fft.ifftshift(inplane, dim=(-2, -1)).abs(),
        fft_volume_spatial_uncentered(vol).abs(),
        atol=1e-5,
    )


def test_dims_override_round_trips() -> None:
    vol = torch.randn(2, 3, 5, 6, 7)
    back = ifftnc(fftnc(vol, dims=(-2, -1)), dims=(-2, -1)).real
    assert torch.allclose(back, vol, atol=1e-5)


TRANSFORMER_FORWARD = [
    ("fft2c", True, SHAPE_4D, (2, 3)),
    ("fftnc", True, SHAPE_5D, (2, 3, 4)),
    ("fft2_uncentered", False, SHAPE_5D, (3, 4)),
    ("fftn_uncentered", False, SHAPE_5D, (2, 3, 4)),
]


@pytest.mark.parametrize(
    "method,centred,shape,axes", TRANSFORMER_FORWARD, ids=[c[0] for c in TRANSFORMER_FORWARD]
)
def test_fft_transformer_method_dc_matches_its_name(method, centred, shape, axes) -> None:
    """The peer-exposure that made #1350 reachable: one object, both families.

    ``FFTTransformer`` may keep both, but a caller must not be able to pick the
    uncentered one by writing the shorter name.
    """
    fn = getattr(FFTTransformer(), method)
    assert_forward_dc_matches_name(fn, centred=centred, shape=shape, axes=axes)


def test_transformer_exposes_no_convention_silent_alias() -> None:
    """No public transform method may omit the convention from its name.

    This is the rule the rename encodes: ``fft2``/``ifft2``/``fftn``/``ifftn``
    were uncentered while reading as canonical.
    """
    banned = {"fft2", "ifft2", "fftn", "ifftn"}
    present = {m for m in banned if hasattr(FFTTransformer, m)}
    assert not present, (
        f"FFTTransformer re-exposes convention-silent transform name(s) {sorted(present)}. "
        "An alias re-creates exactly the trap #1350 removed: name the convention "
        "(*_uncentered) or use the centred entry point."
    )


# --------------------------------------------------------------------------
# Planted violations (non-negotiable 15) -- the gate must go red on each
# --------------------------------------------------------------------------


def _unshifted(x: torch.Tensor) -> torch.Tensor:
    """A transform claiming to be centred that forgot both shifts."""
    return torch.fft.fftn(x.to(torch.complex64), dim=(2, 3), norm="ortho")


def _shifted(x: torch.Tensor) -> torch.Tensor:
    """A transform claiming to be uncentered that shifts anyway."""
    dims = (3, 4)
    z = torch.fft.ifftshift(x.to(torch.complex64), dim=dims)
    return torch.fft.fftshift(torch.fft.fftn(z, dim=dims, norm="ortho"), dim=dims)


def _inv_unshifted(k: torch.Tensor) -> torch.Tensor:
    """An inverse claiming to be centred that forgot both shifts."""
    return torch.fft.ifftn(k, dim=(2, 3), norm="ortho")


def _inv_shifted(k: torch.Tensor) -> torch.Tensor:
    """An inverse claiming to be uncentered that shifts anyway."""
    dims = (3, 4)
    z = torch.fft.ifftshift(k, dim=dims)
    return torch.fft.fftshift(torch.fft.ifftn(z, dim=dims, norm="ortho"), dim=dims)


def _slice_dim_shifted(x: torch.Tensor) -> torch.Tensor:
    """1-D partition-axis transform claiming uncentered, shifted anyway."""
    z = torch.fft.ifftshift(x.to(torch.complex64), dim=(2,))
    return torch.fft.fftshift(torch.fft.fft(z, dim=2, norm="ortho"), dim=(2,))


@pytest.mark.parametrize(
    "fn,centred,shape,axes",
    [
        pytest.param(_unshifted, True, SHAPE_4D, (2, 3), id="centred-name-missing-shifts-2d"),
        pytest.param(_shifted, False, SHAPE_5D, (3, 4), id="uncentered-name-has-shifts-2d"),
        pytest.param(_slice_dim_shifted, False, SHAPE_5D, (2,), id="uncentered-name-has-shifts-1d"),
    ],
)
def test_planted_forward_violation_turns_the_gate_red(fn, centred, shape, axes) -> None:
    with pytest.raises(AssertionError, match=r"DC belongs at|put it at"):
        assert_forward_dc_matches_name(fn, centred=centred, shape=shape, axes=axes)


@pytest.mark.parametrize(
    "fn,centred,shape,axes",
    [
        pytest.param(_inv_unshifted, True, SHAPE_4D, (2, 3), id="centred-inverse-missing-shifts"),
        pytest.param(_inv_shifted, False, SHAPE_5D, (3, 4), id="uncentered-inverse-has-shifts"),
    ],
)
def test_planted_inverse_violation_turns_the_gate_red(fn, centred, shape, axes) -> None:
    with pytest.raises(AssertionError, match="constant field"):
        assert_inverse_dc_matches_name(fn, centred=centred, shape=shape, axes=axes)


def test_planted_alias_turns_the_alias_gate_red(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-adding a convention-silent alias must be caught, not merely reviewed."""
    monkeypatch.setattr(FFTTransformer, "fft2", FFTTransformer.fft2_uncentered, raising=False)
    with pytest.raises(AssertionError, match="convention-silent"):
        test_transformer_exposes_no_convention_silent_alias()
