"""Tests for :mod:`spectramr.data.transforms.signature`.

The Phase-2 parity contract depends on a stable hash. These tests
pin the behavior that:

1. Identical transform chains hash identically (cross-process stable).
2. Augmentations are dropped before hashing (so train + infer match
   when their only difference is augmentation).
3. Order matters (reordering normalization vs resampling produces a
   different hash — which is correct).
4. ``diff_signatures`` produces actionable error messages.
"""
from __future__ import annotations

from spectramr.data.transforms.signature import (
    compute_transform_signature,
    diff_signatures,
)


# ── helpers: fake transforms (no TorchIO dependency required) ───────────────


def _make_transform(cls_name: str, **kwargs):
    """Create a synthetic transform object with the given class name
    and kwargs as attributes."""
    cls = type(cls_name, (), {})
    obj = cls()
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj


# ── identity & determinism ──────────────────────────────────────────────────


def test_empty_chain_has_stable_signature() -> None:
    sig1 = compute_transform_signature(None)
    sig2 = compute_transform_signature([])
    assert sig1 == sig2
    assert len(sig1) == 64  # sha256 hex
    # Repeated calls are identical (no clock/RNG dependence)
    assert sig1 == compute_transform_signature(None)


def test_same_chain_hashes_identically() -> None:
    a = [
        _make_transform("RescaleIntensity", out_min_max=(0, 1), percentiles=(0, 99)),
        _make_transform("ZNormalization"),
    ]
    b = [
        _make_transform("RescaleIntensity", out_min_max=(0, 1), percentiles=(0, 99)),
        _make_transform("ZNormalization"),
    ]
    assert compute_transform_signature(a) == compute_transform_signature(b)


def test_different_class_names_hash_differently() -> None:
    a = compute_transform_signature([_make_transform("RescaleIntensity")])
    b = compute_transform_signature([_make_transform("ZNormalization")])
    assert a != b


def test_different_load_bearing_kwargs_hash_differently() -> None:
    a = compute_transform_signature(
        [_make_transform("RescaleIntensity", out_min_max=(0, 1))]
    )
    b = compute_transform_signature(
        [_make_transform("RescaleIntensity", out_min_max=(-1, 1))]
    )
    assert a != b


def test_unknown_kwargs_do_not_affect_signature() -> None:
    """Kwargs NOT in ``_LOAD_BEARING_KWARGS`` are dropped — keeps the
    signature stable across TorchIO versions that add new optional
    parameters."""
    a = compute_transform_signature(
        [_make_transform("RescaleIntensity", out_min_max=(0, 1))]
    )
    b = compute_transform_signature(
        [
            _make_transform(
                "RescaleIntensity",
                out_min_max=(0, 1),
                some_new_kwarg="future_torchio_addition",
            )
        ]
    )
    assert a == b


# ── augmentation dropping ───────────────────────────────────────────────────


def test_augmentations_dropped_when_drop_augmentations_true() -> None:
    """Train chain with aug == infer chain without aug, modulo Random*."""
    train = [
        _make_transform("RescaleIntensity", out_min_max=(0, 1)),
        _make_transform("RandomAffine"),
        _make_transform("RandomFlip"),
        _make_transform("RandomNoise"),
    ]
    infer = [
        _make_transform("RescaleIntensity", out_min_max=(0, 1)),
    ]
    assert compute_transform_signature(train) == compute_transform_signature(infer)


def test_augmentations_preserved_when_drop_augmentations_false() -> None:
    """Opt-out mode: hash everything, including augs."""
    train = [
        _make_transform("RescaleIntensity", out_min_max=(0, 1)),
        _make_transform("RandomAffine"),
    ]
    infer = [
        _make_transform("RescaleIntensity", out_min_max=(0, 1)),
    ]
    # With drop_augmentations=False, train and infer should differ.
    sig_train = compute_transform_signature(train, drop_augmentations=False)
    sig_infer = compute_transform_signature(infer, drop_augmentations=False)
    assert sig_train != sig_infer


def test_repo_specific_augmentations_dropped() -> None:
    """spectraMR's own augmentation classes (PhaseAugmentation,
    RealisticDegradations, etc.) are also stripped from parity diffs."""
    train = [
        _make_transform("RescaleIntensity", out_min_max=(0, 1)),
        _make_transform("PhaseAugmentation"),
        _make_transform("RealisticDegradations"),
    ]
    infer = [
        _make_transform("RescaleIntensity", out_min_max=(0, 1)),
    ]
    assert compute_transform_signature(train) == compute_transform_signature(infer)


# ── order sensitivity ───────────────────────────────────────────────────────


def test_reordering_transforms_changes_signature() -> None:
    """Resampling before normalization ≠ normalization before resampling.
    Both behaviors are valid, but they're not equivalent — the signature
    must distinguish them."""
    a = compute_transform_signature(
        [
            _make_transform("Resample", target=1.0),
            _make_transform("ZNormalization"),
        ]
    )
    b = compute_transform_signature(
        [
            _make_transform("ZNormalization"),
            _make_transform("Resample", target=1.0),
        ]
    )
    assert a != b


# ── nested Compose handling ─────────────────────────────────────────────────


def test_nested_compose_is_flattened() -> None:
    """``tio.Compose([A, Compose([B, C])])`` should hash the same as
    ``Compose([A, B, C])`` because the wrapping is non-semantic."""

    class Compose:
        def __init__(self, *children):
            self.transforms = list(children)

    # Set __name__ on the class so _classname picks it up.
    Compose.__name__ = "Compose"

    flat = [
        _make_transform("A"),
        _make_transform("B"),
        _make_transform("C"),
    ]
    nested = [
        _make_transform("A"),
        Compose(_make_transform("B"), _make_transform("C")),
    ]
    assert compute_transform_signature(flat) == compute_transform_signature(nested)


# ── diff_signatures ────────────────────────────────────────────────────────


def test_diff_equal_signatures_returns_none() -> None:
    sig = compute_transform_signature([_make_transform("RescaleIntensity")])
    assert diff_signatures(sig, sig) is None


def test_diff_both_none_returns_none() -> None:
    """Two checkpoints without recorded signatures → no constraint."""
    assert diff_signatures(None, None) is None


def test_diff_train_missing_yields_message() -> None:
    """Pre-Phase-2 checkpoints don't carry a signature — surface that
    state cleanly rather than silently allowing divergence."""
    msg = diff_signatures(None, "abc123" * 10 + "abcd")
    assert msg is not None
    assert "pre-dates" in msg.lower() or "missing" in msg.lower()


def test_diff_infer_missing_yields_message() -> None:
    msg = diff_signatures("a" * 64, None)
    assert msg is not None
    assert "infer" in msg.lower() or "could not" in msg.lower()


def test_diff_unequal_signatures_names_the_divergence() -> None:
    sig_a = compute_transform_signature([_make_transform("RescaleIntensity")])
    sig_b = compute_transform_signature([_make_transform("ZNormalization")])
    msg = diff_signatures(sig_a, sig_b)
    assert msg is not None
    assert "parity broken" in msg.lower()
    # First 16 chars of each signature should appear in the message
    assert sig_a[:16] in msg
    assert sig_b[:16] in msg
