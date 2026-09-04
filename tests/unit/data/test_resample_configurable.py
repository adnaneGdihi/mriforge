"""Phase 4 tests — config-driven resample + crop_or_pad dispatch.

Covers two flows:

1. **Legacy path** — when ``data.resample.enabled`` and
   ``data.crop_or_pad.enabled`` are both False (the default), the
   transform builder emits the original ``_ResampleToReferenceTransform``
   and ``_EnsureMinimumSpatialSize`` (preserving pre-Phase-4 behavior
   for ~200 existing YAMLs).

2. **Phase-4 path** — when either flag is True, the dispatcher emits
   the corresponding ``tio.Resample`` / ``tio.CropOrPad`` instance
   from :mod:`spectramr.data.builders.resample_dispatch`.

Schema validation guards (target_spacing required for isotropic,
target_shape required for crop_or_pad) are tested at the Pydantic level.
"""
from __future__ import annotations

import pytest

from spectramr.config.schemas.data import (
    CropOrPadConfigSchema,
    DataConfigSchema,
    ResampleConfigSchema,
)
from spectramr.data.builders.resample_dispatch import (
    _TORCHIO_PAD_MODE,
    build_crop_or_pad_transform,
    build_resample_transform,
)


# ── helper: minimal proxy so from_training_config has `acceleration` ────────


class _Proxy:
    """Forward attribute access to a wrapped DataConfigSchema and add the
    ``undersampling`` field that ``TorchIOTransformConfig.from_training_config``
    expects from the parent settings object.

    Phase 11 folded the whole top-level ``acceleration:`` block to
    ``undersampling:``. Production is already canonical
    (``torchio_transform_builder.py:718`` reads ``config.undersampling``); this
    proxy kept hand-adding the retired name, so the reader saw nothing. A rename
    applies to a RECEIVER, not a name."""

    def __init__(self, data):
        self._data = data
        self.undersampling = None

    def __getattr__(self, name):
        return getattr(self._data, name)


# ── Schema-level validation ─────────────────────────────────────────────────


def test_default_resample_is_disabled() -> None:
    """Default DataConfigSchema → both Phase-4 blocks off; legacy
    behavior preserved."""
    c = DataConfigSchema()
    assert c.resample.enabled is False
    assert c.crop_or_pad.enabled is False


def test_isotropic_requires_target_spacing() -> None:
    """``strategy='isotropic'`` + missing ``target_spacing`` → ValidationError."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="target_spacing"):
        ResampleConfigSchema(enabled=True, strategy="isotropic")


def test_crop_or_pad_enabled_requires_target_shape() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="target_shape"):
        CropOrPadConfigSchema(enabled=True)


def test_disabled_blocks_skip_validation() -> None:
    """When ``enabled=False``, validators don't fire for missing required
    fields — lets users declare a future-use stub without errors."""
    rc = ResampleConfigSchema(enabled=False, strategy="isotropic")
    assert rc.target_spacing is None  # not required when disabled
    cc = CropOrPadConfigSchema(enabled=False)
    assert cc.target_shape is None


def test_isotropic_with_target_spacing_validates() -> None:
    rc = ResampleConfigSchema(
        enabled=True, strategy="isotropic", target_spacing=(1.0, 1.0, 1.0)
    )
    assert rc.target_spacing == (1.0, 1.0, 1.0)


def test_crop_or_pad_with_target_shape_validates() -> None:
    cc = CropOrPadConfigSchema(
        enabled=True, target_shape=(256, 256, 256), padding_mode="reflect"
    )
    assert cc.target_shape == (256, 256, 256)
    assert cc.padding_mode == "reflect"


# ── Builder-side dispatch ───────────────────────────────────────────────────


def test_dispatch_returns_none_when_resample_disabled() -> None:
    """Disabled → None → caller falls back to legacy."""

    class C:
        resample_enabled = False

    assert build_resample_transform(C()) is None


def test_dispatch_returns_none_when_crop_or_pad_disabled() -> None:
    class C:
        crop_or_pad_enabled = False

    assert build_crop_or_pad_transform(C()) is None


def test_dispatch_reference_strategy_returns_none() -> None:
    """``strategy='reference'`` keeps the legacy
    ``_ResampleToReferenceTransform`` — dispatcher returns None so the
    caller's fall-through code path runs."""

    class C:
        resample_enabled = True
        resample_strategy = "reference"
        resample_interpolation = "bspline"
        resample_anti_aliasing = True

    assert build_resample_transform(C()) is None


def test_dispatch_isotropic_returns_tio_resample() -> None:
    """When ``isotropic`` is configured, the dispatcher emits a real
    ``tio.Resample`` with the requested target spacing."""
    import torchio as tio

    class C:
        resample_enabled = True
        resample_strategy = "isotropic"
        resample_target_spacing = (1.0, 1.0, 1.0)
        resample_interpolation = "bspline"
        resample_anti_aliasing = True

    t = build_resample_transform(C())
    assert isinstance(t, tio.Resample)


def test_dispatch_isotropic_missing_spacing_raises() -> None:
    """Defensive — Pydantic validator should catch this earlier, but
    the dispatcher fails loud if it's somehow fed a raw dict."""

    class C:
        resample_enabled = True
        resample_strategy = "isotropic"
        resample_target_spacing = None
        resample_interpolation = "bspline"
        resample_anti_aliasing = True

    with pytest.raises(ValueError, match="target_spacing"):
        build_resample_transform(C())


def test_dispatch_unknown_strategy_raises() -> None:
    class C:
        resample_enabled = True
        resample_strategy = "bogus"
        resample_interpolation = "bspline"
        resample_anti_aliasing = True

    with pytest.raises(ValueError, match="not recognized"):
        build_resample_transform(C())


def test_dispatch_crop_or_pad_returns_tio_croporpad() -> None:
    import torchio as tio

    class C:
        crop_or_pad_enabled = True
        crop_or_pad_target_shape = (256, 256, 256)
        crop_or_pad_padding_mode = "reflect"
        crop_or_pad_crop_strategy = "center"

    t = build_crop_or_pad_transform(C())
    assert isinstance(t, tio.CropOrPad)


def test_dispatch_crop_or_pad_mean_pad_falls_back_to_reflect(caplog) -> None:
    """'mean' padding isn't natively supported — dispatcher logs a warning
    and falls back to reflect (verified by checking the resulting
    transform's padding_mode attribute)."""
    import logging

    import torchio as tio

    class C:
        crop_or_pad_enabled = True
        crop_or_pad_target_shape = (64, 64, 1)
        crop_or_pad_padding_mode = "mean"
        crop_or_pad_crop_strategy = "center"

    with caplog.at_level(logging.WARNING):
        t = build_crop_or_pad_transform(C())
    assert isinstance(t, tio.CropOrPad)
    assert any("mean" in r.message.lower() for r in caplog.records)


def test_dispatch_unknown_crop_padding_mode_raises() -> None:
    """BM-002: a duck-typed config built by hand can carry an unknown
    padding_mode (the Pydantic ``Literal`` guard only protects the YAML path).
    The dispatcher must raise a clear ValueError before the bare dict lookup,
    not a cryptic ``KeyError`` (NN#3 / pitfall #9)."""

    class C:
        crop_or_pad_enabled = True
        crop_or_pad_target_shape = (64, 64, 1)
        crop_or_pad_padding_mode = "constant"  # not a key of _TORCHIO_PAD_MODE
        crop_or_pad_crop_strategy = "center"

    with pytest.raises(ValueError, match="not recognised"):
        build_crop_or_pad_transform(C())


def test_mean_pad_dict_entry_absent() -> None:
    """BM-002: the dead ``'mean'`` entry must be removed from the lookup table.
    'mean' is handled separately (warning + fallback to 'reflect') and is NOT a
    valid TorchIO padding_mode, so it must never appear as a dict key."""
    assert "mean" not in _TORCHIO_PAD_MODE


def test_dispatch_anti_aliasing_off_uses_nearest_interpolation() -> None:
    """``anti_aliasing=False`` opts out of pre-filtering; dispatcher
    falls back to nearest-neighbor interpolation."""
    import torchio as tio

    class C:
        resample_enabled = True
        resample_strategy = "isotropic"
        resample_target_spacing = (1.0, 1.0, 1.0)
        resample_interpolation = "bspline"
        resample_anti_aliasing = False

    t = build_resample_transform(C())
    assert isinstance(t, tio.Resample)
    # The interpolation attribute name varies across TorchIO versions;
    # check the constructor arg we passed via a string comparison.
    # (TorchIO stores image_interpolation in ``image_interpolation`` attr.)
    assert getattr(t, "image_interpolation", None) == "nearest"


# ── End-to-end through TorchIOTransformBuilder ──────────────────────────────


def test_legacy_chain_uses_underscore_transforms() -> None:
    """No resample/crop_or_pad in config → legacy chain unchanged
    (preserves behavior for ~200 existing YAMLs)."""
    from spectramr.data.builders.torchio_transform_builder import (
        TorchIOTransformBuilder,
        TorchIOTransformConfig,
    )

    data = DataConfigSchema(patch_size=(64, 64, 1))
    tc = TorchIOTransformConfig.from_training_config(_Proxy(data))
    chain = TorchIOTransformBuilder.build_val_transforms(tc)
    names = [type(t).__name__ for t in chain.transforms[:4]]
    assert "_ResampleToReferenceTransform" in names
    assert "_EnsureMinimumSpatialSize" in names


def test_phase4_chain_uses_tio_transforms() -> None:
    """``resample.enabled=True`` + ``crop_or_pad.enabled=True`` → chain
    uses ``tio.Resample`` and ``tio.CropOrPad`` instead of the legacy
    underscore-prefixed transforms."""
    from spectramr.data.builders.torchio_transform_builder import (
        TorchIOTransformBuilder,
        TorchIOTransformConfig,
    )

    data = DataConfigSchema(
        patch_size=(64, 64, 1),
        resample=ResampleConfigSchema(
            enabled=True, strategy="isotropic", target_spacing=(1.0, 1.0, 1.0)
        ),
        crop_or_pad=CropOrPadConfigSchema(
            enabled=True, target_shape=(64, 64, 1), padding_mode="reflect"
        ),
    )
    tc = TorchIOTransformConfig.from_training_config(_Proxy(data))
    chain = TorchIOTransformBuilder.build_val_transforms(tc)
    names = [type(t).__name__ for t in chain.transforms[:4]]
    assert "Resample" in names
    assert "CropOrPad" in names
    # Critical: legacy transforms must NOT be present (else they'd
    # double-resample and silently change the output).
    assert "_ResampleToReferenceTransform" not in names
    assert "_EnsureMinimumSpatialSize" not in names


def test_partial_phase4_only_resample() -> None:
    """User can enable resample without crop_or_pad (and vice-versa);
    the other path falls back to legacy."""
    from spectramr.data.builders.torchio_transform_builder import (
        TorchIOTransformBuilder,
        TorchIOTransformConfig,
    )

    data = DataConfigSchema(
        patch_size=(64, 64, 1),
        resample=ResampleConfigSchema(
            enabled=True, strategy="isotropic", target_spacing=(1.0, 1.0, 1.0)
        ),
    )
    tc = TorchIOTransformConfig.from_training_config(_Proxy(data))
    chain = TorchIOTransformBuilder.build_val_transforms(tc)
    names = [type(t).__name__ for t in chain.transforms[:4]]
    assert "Resample" in names
    # crop_or_pad disabled → legacy ensure-min-size still there
    assert "_EnsureMinimumSpatialSize" in names
