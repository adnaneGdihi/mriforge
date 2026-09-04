"""Tests for the transversal DigitalTwinDegradation data transform.

Mirrors the accelerator (PhysicsInformedMasking) but applies the digital-twin
corruption in the data pipeline so any experiment can opt in. Pins: it
produces a corrupted ``input`` and a preserved clean ``target``, shape is
preserved, and the round-trip is k-space→image→k-space.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
tio = pytest.importorskip("torchio")

from spectramr.config.schemas.physics import DigitalTwinConfig  # noqa: E402
from spectramr.data.transforms.tio_physics import DigitalTwinDegradation  # noqa: E402
from spectramr.infrastructure.physics.digital_twin_simulator import (  # noqa: E402
    DigitalTwinSimulator,
)


def _subject(c: int = 2, w: int = 32, h: int = 32, d: int = 1) -> tio.Subject:
    torch.manual_seed(0)
    return tio.Subject(kspace=tio.ScalarImage(tensor=torch.randn(c, w, h, d)))


def _sim(**kw) -> DigitalTwinSimulator:
    cfg = DigitalTwinConfig(enable_b0=False, enable_b1=False, **kw)
    return DigitalTwinSimulator.from_config(cfg, (32, 32))


def test_produces_corrupted_input_and_clean_target() -> None:
    subj = _subject()
    raw = subj["kspace"].data.clone()
    out = DigitalTwinDegradation(
        _sim(enable_motion=True, motion_severity=3.0), degradation_only=True
    )(subj)

    assert {"input", "target"} <= set(out.keys())
    inp, tgt = out["input"].data, out["target"].data
    assert inp.shape == raw.shape  # shape preserved
    assert not torch.allclose(inp, tgt, atol=1e-4)  # corruption applied
    assert torch.allclose(tgt, raw, atol=1e-6)  # clean target preserved


def test_higher_severity_corrupts_more() -> None:
    """``enable_motion`` must be ON, or this asserts nothing about severity.

    Its default flipped to False on 2026-08-03 and these two tests kept passing
    ``motion_severity`` into a disabled branch. Measured: with motion off, severity
    0.0 and 5.0 produce BYTE-IDENTICAL output, so the only difference the assertion
    could see was the ambient noise draw between the two calls — a coin flip, and
    the reason this test was red on dev. With motion on it separates 6/6 with a wide
    margin (0.95 vs 0.15 mean absolute change).
    """
    raw = _subject()["kspace"].data
    mild = DigitalTwinDegradation(
        _sim(enable_motion=True, motion_severity=0.0), degradation_only=True
    )(tio.Subject(kspace=tio.ScalarImage(tensor=raw.clone())))["input"].data
    severe = DigitalTwinDegradation(
        _sim(enable_motion=True, motion_severity=5.0), degradation_only=True
    )(tio.Subject(kspace=tio.ScalarImage(tensor=raw.clone())))["input"].data
    assert (severe - raw).abs().mean() > (mild - raw).abs().mean()


def test_odd_channels_rejected() -> None:
    subj = tio.Subject(kspace=tio.ScalarImage(tensor=torch.randn(3, 16, 16, 1)))
    with pytest.raises(ValueError, match="even channels"):
        DigitalTwinDegradation(_sim(), degradation_only=True)(subj)


def test_missing_acquisition_raises_rather_than_passing_clean_data_through() -> None:
    """Silence here trained an arm on clean data while the log said otherwise.

    One step earlier the transform builder logs "[PHYSICS] Injecting transversal
    DigitalTwinDegradation" at INFO. This branch then returned the subject
    untouched at DEBUG, so at default verbosity the run REPORTED a degradation it
    never applied — pitfall #16 with a receipt. The message has to name what is
    missing and both ways out, because the arm author's mistake is a dataset_type
    that serves images, not a bug in the transform.
    """
    subj = tio.Subject(image=tio.ScalarImage(tensor=torch.randn(2, 16, 16, 1)))
    with pytest.raises(ValueError, match="no acquisition to corrupt"):
        DigitalTwinDegradation(_sim(), degradation_only=True)(subj)


def test_the_removed_source_domain_knob_cannot_come_back_silently() -> None:
    """``source_domain`` was accepted-but-broken and is gone.

    Its ``"image"`` value resolved its source from the k-space key anyway, so it
    meant "read k-space and pretend it is an image", and no caller ever passed it:
    the transform builder hardcoded the k-space route. What matters now is that
    passing it FAILS rather than being swallowed by ``**kwargs`` into an attribute
    nobody reads — which is how an unwired knob is born (pitfall #15).
    """
    with pytest.raises(TypeError, match="source_domain"):
        DigitalTwinDegradation(_sim(), source_domain="image")


def test_physics_masking_undersamples_width_constant_along_height() -> None:
    """Pins the documented axis convention (review 2026-07-01).

    ``PhysicsInformedMasking._generate_mask`` undersamples along WIDTH (this
    transform's phase-encode axis) and holds the pattern constant along HEIGHT
    (the fully-sampled/readout axis). Guards against a silent axis flip and
    documents the assumption the physics-SSOT consolidation must reconcile.
    """
    from spectramr.data.transforms.tio_physics import PhysicsInformedMasking

    torch.manual_seed(0)
    m = PhysicsInformedMasking(acceleration=4, center_fraction=0.25)
    width, height = 40, 24
    mask = m._generate_mask(width, height)

    assert mask.shape == (width, height)
    # Constant along height (dim=1): every column equals the first.
    assert torch.all(mask == mask[:, :1])
    # The center fraction is fully acquired along width (the ACS band).
    center = width // 2
    cf_w = int(width * 0.25)
    assert torch.all(mask[center - cf_w // 2 : center + cf_w // 2, 0] == 1)
    # Acceleration undersamples: not every width line is acquired.
    assert mask[:, 0].sum() < width


def test_physics_informed_mask_is_deterministic_and_cached() -> None:
    """Behavior + perf regression (2026-07-02): the mask used ``torch.randperm``
    UNSEEDED → a different mask per item, unlike every seeded generator in the
    physics SSOT. It is now seeded per width (reproducible) and cached per
    ``(width, height)`` so it is not recomputed in every ``apply_transform``."""
    from spectramr.data.transforms.tio_physics import PhysicsInformedMasking

    m = PhysicsInformedMasking(acceleration=4, center_fraction=0.1)
    a = m._generate_mask(48, 16)
    # Cache hit returns the same object (no recompute).
    b = m._generate_mask(48, 16)
    assert b is a
    assert (48, 16) in m._mask_cache

    # A fresh instance produces an identical mask (seeded, not global-RNG).
    m2 = PhysicsInformedMasking(acceleration=4, center_fraction=0.1)
    assert torch.equal(m2._generate_mask(48, 16), a)


def test_physics_informed_mask_seed_untouched_by_global_rng() -> None:
    """The seeded per-width generator must not depend on global torch RNG, so
    two instances agree regardless of intervening global draws."""
    from spectramr.data.transforms.tio_physics import PhysicsInformedMasking

    torch.manual_seed(1)
    a = PhysicsInformedMasking(acceleration=6)._generate_mask(64, 8)
    torch.manual_seed(999)
    _ = torch.rand(100)  # perturb global RNG
    b = PhysicsInformedMasking(acceleration=6)._generate_mask(64, 8)
    assert torch.equal(a, b)
