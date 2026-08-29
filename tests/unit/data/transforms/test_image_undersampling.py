"""Unit tests for RetrospectiveImageUndersampling — the image→k-space bridge.

Pins the exp_c1 federated bridge (2026-06-21): a fully-sampled magnitude image
(MRIxFields NIfTI) is degraded into an aliased ``input`` while ``target`` stays
the full image, using the physics SSOT (fft2c/ifft2c + KSpaceMaskGenerator).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
tio = pytest.importorskip("torchio")

from mriforge.data.transforms.image_undersampling import (  # noqa: E402
    RetrospectiveImageUndersampling,
)


def _subject(h: int = 32, w: int = 32) -> tio.Subject:
    # mrixfields hands the transform a paired Subject of [C, H, W, D] = [1, H, W, 1].
    img = torch.rand(1, h, w, 1)
    return tio.Subject(
        input=tio.ScalarImage(tensor=img.clone()),
        target=tio.ScalarImage(tensor=img.clone()),
    )


def test_input_is_undersampled_target_is_preserved():
    subj = _subject()
    full = subj["input"].data.clone()
    out = RetrospectiveImageUndersampling(acceleration=4.0)(subj)
    # input changed (aliased), target untouched (still the full image)
    assert not torch.allclose(out["input"].data, full)
    assert torch.allclose(out["target"].data, full)
    # shape + dtype preserved, output real & finite
    assert out["input"].data.shape == full.shape
    assert out["input"].data.dtype == full.dtype
    assert torch.isfinite(out["input"].data).all()
    assert not out["input"].data.is_complex()


def test_acceleration_one_is_passthrough():
    subj = _subject()
    full = subj["input"].data.clone()
    out = RetrospectiveImageUndersampling(acceleration=1.0)(subj)
    assert torch.allclose(out["input"].data, full)


def test_deterministic_reproducible():
    # Equispaced Cartesian mask is deterministic → same input ⇒ same output.
    a = RetrospectiveImageUndersampling(acceleration=4.0)(_subject())
    b = RetrospectiveImageUndersampling(acceleration=4.0)(_subject())
    # both built from torch.rand → different content, but re-running on the SAME
    # tensor must be identical:
    subj = _subject()
    o1 = RetrospectiveImageUndersampling(acceleration=4.0)(
        tio.Subject(input=tio.ScalarImage(tensor=subj["input"].data.clone()))
    )
    o2 = RetrospectiveImageUndersampling(acceleration=4.0)(
        tio.Subject(input=tio.ScalarImage(tensor=subj["input"].data.clone()))
    )
    assert torch.allclose(o1["input"].data, o2["input"].data)
    assert a["input"].data.shape == b["input"].data.shape


def test_higher_acceleration_removes_more_energy():
    subj4 = _subject()
    base = subj4["input"].data.clone()
    subj8 = tio.Subject(input=tio.ScalarImage(tensor=base.clone()))
    subj4 = tio.Subject(input=tio.ScalarImage(tensor=base.clone()))
    r4 = RetrospectiveImageUndersampling(acceleration=2.0)(subj4)["input"].data
    r8 = RetrospectiveImageUndersampling(acceleration=8.0)(subj8)["input"].data
    # R=8 discards more k-space → larger deviation from the full image than R=2
    assert (r8 - base).abs().mean() > (r4 - base).abs().mean()


def test_mask_is_cached_per_shape():
    """Perf regression (2026-07-02): the deterministic mask must be memoized
    per ``(h, w)`` instead of regenerated in every ``apply_transform`` (worker
    hot path). Same shape → the exact same cached tensor object is reused; a
    new shape adds a distinct entry."""
    t = RetrospectiveImageUndersampling(acceleration=4.0)
    t(_subject(h=32, w=32))
    assert (32, 32) in t._mask_cache
    cached = t._mask_cache[(32, 32)]

    t(_subject(h=32, w=32))
    assert t._mask_cache[(32, 32)] is cached, "same shape must reuse the cache"

    t(_subject(h=16, w=24))
    assert (16, 24) in t._mask_cache and len(t._mask_cache) == 2


def test_cached_mask_matches_fresh_generate():
    """The cached mask is identical to a fresh ``MaskGenerator.generate_mask``
    — caching changes nothing numerically."""
    from mriforge.infrastructure.physics.sampling import MaskGenerator, MaskType

    t = RetrospectiveImageUndersampling(acceleration=4.0)
    t(_subject(h=32, w=32))
    fresh = MaskGenerator().generate_mask(
        MaskType.CARTESIAN_LINES, (32, 32),
        acceleration_factor=4.0, center_fraction=0.08,
    )
    assert torch.equal(t._mask_cache[(32, 32)], fresh)


def test_cached_transform_output_matches_uncached_reference():
    """End-to-end: the transform output is unchanged by the cache (compare to
    an independent instance's first-call output on the same input)."""
    base = torch.rand(1, 32, 32, 1)
    a = RetrospectiveImageUndersampling(acceleration=4.0)
    out1 = a(tio.Subject(input=tio.ScalarImage(tensor=base.clone())))["input"].data
    out2 = a(tio.Subject(input=tio.ScalarImage(tensor=base.clone())))["input"].data
    assert torch.equal(out1, out2)  # second call uses the cache


def test_missing_input_key_raises():
    subj = tio.Subject(target=tio.ScalarImage(tensor=torch.rand(1, 16, 16, 1)))
    with pytest.raises(KeyError, match="no 'input'"):
        RetrospectiveImageUndersampling()(subj)


@pytest.mark.parametrize(
    "kwargs",
    [{"acceleration": 0.5}, {"center_fraction": 0.0}, {"center_fraction": 1.5}],
)
def test_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        RetrospectiveImageUndersampling(**kwargs)
