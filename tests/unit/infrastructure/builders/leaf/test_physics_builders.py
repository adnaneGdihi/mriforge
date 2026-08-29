"""Paired tests for the BuilderContext migration of the (config)-only physics
builders (cached-cascade WS-D conventions freeze).

``FFTBuilder``, ``DataConsistencyBuilder`` and ``PhysicsBuilder`` were migrated
to the canonical ``def __init__(self, ctx: BuilderContext)`` convention behind
the Phase-0 ``accepts_builder_context`` back-compat shim. Each must therefore
construct identically from BOTH:

* the legacy ``Builder(config)`` positional shape, and
* the canonical ``Builder(BuilderContext(config=config))`` shape,

storing the same ``config`` SSOT on ``._config`` either way.

``MaskBuilder`` is intentionally NOT migrated (it takes ``(config, device)``) and
is not exercised here.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from mriforge.infrastructure.builders.context import BuilderContext
from mriforge.infrastructure.builders.leaf.physics_builders import (
    DataConsistencyBuilder,
    FFTBuilder,
    MaskBuilder,
    PhysicsBuilder,
)

# The migrated __init__ bodies only read ``ctx.config`` and stash it; a sentinel
# object is a sufficient, dependency-free stand-in for the TrainingSettings SSOT.
_STUB_CONFIG = object()

MIGRATED_BUILDERS = (FFTBuilder, DataConsistencyBuilder, PhysicsBuilder)


@pytest.mark.parametrize("builder_cls", MIGRATED_BUILDERS)
def test_legacy_positional_config_still_works(builder_cls) -> None:
    """Legacy ``Builder(config)`` call site keeps working via the shim."""
    builder = builder_cls(_STUB_CONFIG)
    assert builder._config is _STUB_CONFIG


@pytest.mark.parametrize("builder_cls", MIGRATED_BUILDERS)
def test_canonical_builder_context_works(builder_cls) -> None:
    """Canonical ``Builder(BuilderContext(config=config))`` call site works."""
    ctx = BuilderContext(config=_STUB_CONFIG)
    builder = builder_cls(ctx)
    assert builder._config is _STUB_CONFIG


@pytest.mark.parametrize("builder_cls", MIGRATED_BUILDERS)
def test_both_forms_produce_equivalent_state(builder_cls) -> None:
    """Both construction shapes converge on the same stored config SSOT."""
    legacy = builder_cls(_STUB_CONFIG)
    canonical = builder_cls(BuilderContext(config=_STUB_CONFIG))
    assert legacy._config is canonical._config is _STUB_CONFIG


# ── MaskBuilder correctness (pitfalls #15 silent no-op, #9 invalid default) ──


def test_maskbuilder_default_pattern_builds() -> None:
    """The default sampling pattern must be a real ``MaskType`` value: ``build()``
    calls ``MaskType(pattern)`` downstream, so a bogus default (``"cartesian"``,
    which is NOT a MaskType member) makes the very first shaped build crash with
    a swallowed ``ValueError`` → ``RuntimeError``."""
    mask = MaskBuilder(_STUB_CONFIG, device="cpu").with_shape((128, 128)).build()
    assert mask is not None
    # A real undersampling mask is not all-ones.
    assert 0.0 < float(mask.float().mean()) < 1.0


def test_maskbuilder_honors_acceleration() -> None:
    """``with_acceleration`` must reach ``generate_mask`` as ``acceleration_factor``;
    passing it under the wrong kwarg name lets it fall into ``**kwargs`` where it
    is ignored and the rate silently pins at the 2.0 default (pitfall #15)."""
    m4 = (
        MaskBuilder(_STUB_CONFIG, device="cpu")
        .with_shape((128, 128))
        .with_acceleration(4)
        .build()
    )
    m8 = (
        MaskBuilder(_STUB_CONFIG, device="cpu")
        .with_shape((128, 128))
        .with_acceleration(8)
        .build()
    )
    # Higher acceleration → strictly fewer sampled k-space points.
    assert float(m8.float().mean()) < float(m4.float().mean())


# ── FFTBuilder: inert norm/centering knobs must validate-and-raise (#15) ──


def test_fftbuilder_with_norm_rejects_unsupported() -> None:
    """``FFTTransformer`` is always ``norm='ortho'``; ``with_norm('')`` was
    accepted then silently ignored at build() (an inert knob). It must raise
    instead so an unsupported request fails loudly."""
    with pytest.raises(ValueError):
        FFTBuilder(_STUB_CONFIG).with_norm("")


def test_fftbuilder_with_centering_rejects_false() -> None:
    """k-space centering is mandatory (the physics SSOT); ``with_centering(False)``
    must raise rather than silently no-op.

    **This test is the sole owner of that invariant** (non-negotiable 17). A second
    copy lived in ``tests/unit/builders/test_phase1_leaf_builders.py`` and was
    deleted with #1462; do not re-add one. History worth keeping from it: that copy
    originally asserted ``builder._centering is False``, i.e. it *demanded the bug
    back* -- storing a value nothing reads is an unread knob (non-negotiable 8,
    pitfall #15) degrading silently (pitfall #9). The builder was fixed to raise and
    the assertion was inverted to match.

    ``match`` pins the **knob's API name**, not the sentence. The duplicate pinned
    ``"always centers"`` -- explanatory prose -- and PR #1442 reworded the message
    while touching neither test, so the duplicate went red for a rewording and the
    guard it claimed to protect never moved. A parameter rename is a breaking API
    change other tests catch; a reworded sentence is not.

    The pin is not decoration: of the nine ``raise ValueError`` sites in
    ``physics_builders``, exactly one message contains ``centering`` (the rest say
    ``norm``, ``Shape must be 2D``, ``Invalid pattern``/``acceleration``/``method``/
    ``weight``, ``center_fraction``, ``K-space shape must be specified``). Planting
    a wrong-``ValueError`` regression -- the guard raising ``Invalid
    center_fraction`` instead -- reddens this assertion and is silently **accepted**
    by the bare ``pytest.raises(ValueError)`` this replaced.
    """
    with pytest.raises(ValueError, match="centering"):
        FFTBuilder(_STUB_CONFIG).with_centering(False)


def test_fftbuilder_accepts_supported_values() -> None:
    """The supported (ortho, centered) values still chain + build."""
    fft = FFTBuilder(_STUB_CONFIG).with_norm("ortho").with_centering(True).build()
    assert fft is not None


@pytest.mark.parametrize("builder_cls", MIGRATED_BUILDERS)
def test_init_carries_accepts_builder_context_marker(builder_cls) -> None:
    """The decorator marks the migrated ``__init__`` for the fitness check."""
    assert getattr(builder_cls.__init__, "__accepts_builder_context__", False) is True


@pytest.mark.parametrize("builder_cls", MIGRATED_BUILDERS)
def test_legacy_path_is_silent_by_default(builder_cls) -> None:
    """Legacy ``(config)`` calls must not emit a DeprecationWarning (silent shim)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        builder = builder_cls(_STUB_CONFIG)
    assert builder._config is _STUB_CONFIG


def test_fft_builder_builds_from_both_forms() -> None:
    """End-to-end: a migrated builder produces an equivalent product either way."""
    legacy_fft = FFTBuilder(_STUB_CONFIG).with_norm("ortho").build()
    canonical_fft = FFTBuilder(BuilderContext(config=_STUB_CONFIG)).with_norm("ortho").build()
    assert type(legacy_fft) is type(canonical_fft)


def test_data_consistency_builder_builds_from_both_forms() -> None:
    """DataConsistencyBuilder builds an equivalent DC operator either way."""
    legacy_dc = DataConsistencyBuilder(_STUB_CONFIG).with_method("hard").with_device("cpu").build()
    canonical_dc = (
        DataConsistencyBuilder(BuilderContext(config=_STUB_CONFIG))
        .with_method("hard")
        .with_device("cpu")
        .build()
    )
    assert type(legacy_dc) is type(canonical_dc)
    assert legacy_dc.method == canonical_dc.method


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
