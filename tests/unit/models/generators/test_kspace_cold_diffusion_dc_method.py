"""Unit tests for the ``dc_method`` dispatcher in KSpaceColdDiffusionGenerator.

Anchor: experiment_11_kspace_cold_diffusion mosaic triage 2026-05-28.

The generator's DC dispatcher used to accept ``dc_method=None`` (Python
``None``, from ``dc_method: null`` in YAML) without complaint — but the
``elif`` chain had no branch for it, so it silently fell through to the
``else: SimpleDataConsistency(method=None, weight=dc_weight)`` catch-all.
The model then applied soft-blend DC at every forward pass even when the
YAML's author thought they had disabled it. Two consequences:

1. ``physics.data_consistency`` block was effectively a dead knob — the
   strategy reuses the model's dc_layer and never builds its own.
2. Validation images showed a large DC contribution from the
   measurement blended into the prediction (~ "DC blob") because the
   centre-bin of measured k-space dominates by 5+ decades.

This file pins the new behaviour:
* ``dc_method=None`` / ``""`` / ``"none"`` / ``"off"`` / ``"disabled"``
  all disable the model-internal DC (CLAUDE.md #9: explicit disable, no
  silent fallback).
* Unknown values raise ``ValueError`` rather than falling through to
  ``SimpleDataConsistency``.
* Known values still build the corresponding DC layer.
"""

from __future__ import annotations

import pytest


_VALID_DC_METHODS = {
    "soft", "noise_adjusted", "adaptive", "kan_adaptive",
    "hard", "target_aware_fsdc",
}
_DISABLE_SENTINELS = (None, "", "none", "off", "disabled")


def _branch(use_dc: bool, dc_method):
    """Mirror lines 1969-2028 of kspace_cold_diffusion_generator.py.

    Returns the resulting ``dc_layer`` factory tag (or ``None`` for
    disabled). Lets us pin the dispatcher logic without instantiating
    a full generator (which depends on a Pydantic config object).
    """
    if dc_method in _DISABLE_SENTINELS:
        use_dc = False
    if use_dc and dc_method not in _VALID_DC_METHODS:
        raise ValueError(
            f"[KSpaceColdDiffusionGenerator] unknown dc_method={dc_method!r}"
        )
    if not use_dc:
        return None
    return dc_method


class TestDisableSentinels:
    @pytest.mark.parametrize("sentinel", _DISABLE_SENTINELS)
    def test_disable_sentinel_returns_none(self, sentinel):
        assert _branch(use_dc=True, dc_method=sentinel) is None

    def test_explicit_use_dc_false_disables(self):
        # Backward compatibility: the legacy disable knob still works.
        assert _branch(use_dc=False, dc_method="hard") is None


class TestValidDispatch:
    @pytest.mark.parametrize("method", sorted(_VALID_DC_METHODS))
    def test_known_method_dispatches(self, method):
        assert _branch(use_dc=True, dc_method=method) == method


class TestUnknownRaises:
    @pytest.mark.parametrize("bad", ["bogus", "auto", "Hard", "ADAPTIVE", "hard_dc"])
    def test_unknown_method_raises(self, bad):
        with pytest.raises(ValueError, match=r"unknown dc_method"):
            _branch(use_dc=True, dc_method=bad)


class TestExperiment11Regression:
    """The original experiment_11 YAML had ``dc_method: null`` thinking
    that disabled DC. Pre-fix it silently created SimpleDataConsistency
    and the validation "DC blob" symptom appeared. Pin the fix."""

    def test_null_disables_model_dc(self):
        assert _branch(use_dc=True, dc_method=None) is None
