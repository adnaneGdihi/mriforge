"""Regression tests for ``strategies/mixins/utils.py``.

Covers two audit findings (2026-05-31 strategy audit):

* ``pick_present`` — the SSOT, tensor-safe replacement for the ``a or b``
  None-coalescing idiom. ``a or b`` evaluates ``bool(a)``, which raises
  ``RuntimeError: Boolean value of Tensor with more than one element is
  ambiguous`` for any multi-element tensor. ``pick_present`` selects purely
  on ``is None`` and never triggers that.
* ``_callable_accepts_kwarg`` cache keying — the cache must key on the
  underlying function, not ``id(bound_method)`` (which can be reused after
  GC and silently return a stale answer for an unrelated callable).
"""

from __future__ import annotations

import torch

from mriforge.infrastructure.training.strategies.mixins import utils as u
from mriforge.infrastructure.training.strategies.mixins.utils import (
    _callable_accepts_kwarg,
    pick_present,
)


class TestPickPresent:
    def test_returns_first_non_none(self):
        assert pick_present(None, 5, 7) == 5

    def test_all_none_returns_none(self):
        assert pick_present(None, None) is None

    def test_does_not_evaluate_tensor_truthiness(self):
        # The bug being fixed: `a or b` raises on a multi-element tensor.
        a = torch.zeros(2, 3, 4, 4)
        out = pick_present(a, None)
        assert out is a  # identity — selected without any bool() evaluation

    def test_skips_none_lhs_to_tensor_rhs(self):
        b = torch.ones(2, 3)
        assert pick_present(None, b) is b

    def test_single_arg_passthrough(self):
        t = torch.randn(4)
        assert pick_present(t) is t

    def test_zero_valued_tensor_is_still_selected(self):
        # `a or b` would treat an all-zero / 0-d tensor as falsy and skip it;
        # pick_present keys on None only, so a present-but-"falsy" tensor wins.
        z = torch.zeros(2, 2)
        assert pick_present(z, torch.ones(2, 2)) is z


class TestCallableAcceptsKwargCaching:
    def test_named_kwarg_detected(self):
        def f(x, timesteps=None):
            return x

        assert _callable_accepts_kwarg(f, "timesteps") is True

    def test_missing_kwarg_rejected(self):
        def g(x):
            return x

        assert _callable_accepts_kwarg(g, "timesteps") is False

    def test_var_keyword_accepts_any(self):
        def h(x, **kwargs):
            return x

        assert _callable_accepts_kwarg(h, "anything") is True

    def test_bound_method_shares_cache_across_instances(self):
        """Two bound-method objects of the same method (distinct ids) must
        collapse to ONE cache entry — proving the key is the underlying
        function, not the per-instance ``id()`` (the stale-after-GC bug)."""
        u._CALLABLE_KWARG_CACHE.clear()

        class B:
            def forward(self, x, kspace_measured=None):
                return x

        b1, b2 = B(), B()
        assert _callable_accepts_kwarg(b1.forward, "kspace_measured") is True
        assert _callable_accepts_kwarg(b2.forward, "kspace_measured") is True
        keys = [k for k in u._CALLABLE_KWARG_CACHE if k[1] == "kspace_measured"]
        assert len(keys) == 1


# ---------------------------------------------------------------------------
# _scoring_leaf — read the canonical spelling, not the retired fold
# ---------------------------------------------------------------------------
#
# `validation.domain` and `validation.output_transform` moved into a `scoring:`
# sub-block. RENAMES records both as FOLDS, so a YAML declaration in the old
# spelling still LOADS -- it just lands under `scoring`. Every consumer read them
# flat, got None on every real config, and no configured metric transform ever
# fired: 145 arms declare `output_transform` (120 of them `ifft_magnitude`) and
# not one had it applied.


class _Scoring:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Block:
    def __init__(self, scoring=None, **flat):
        if scoring is not None:
            self.scoring = scoring
        self.__dict__.update(flat)


def test_scoring_leaf_reads_the_canonical_sub_block():
    """The spelling a real Pydantic config actually uses."""
    cfg = _Block(scoring=_Scoring(output_transform="ifft_magnitude"))

    assert u._scoring_leaf(cfg, "output_transform") == "ifft_magnitude"


def test_scoring_leaf_prefers_scoring_over_a_stale_flat_attribute():
    """If both are present the canonical one wins — a flat leftover must not shadow it."""
    cfg = _Block(
        scoring=_Scoring(output_transform="ifft_magnitude"),
        output_transform="stale",
    )

    assert u._scoring_leaf(cfg, "output_transform") == "ifft_magnitude"


def test_scoring_leaf_still_reads_a_flat_only_stand_in():
    """Hand-built stubs and dicts that spell the leaf flat keep working.

    The fallback is deliberate; it just must never be the ONLY read.
    """
    assert u._scoring_leaf(_Block(output_transform="fft"), "output_transform") == "fft"
    assert u._scoring_leaf({"output_transform": "fft"}, "output_transform") == "fft"


def test_scoring_leaf_reads_a_dict_shaped_scoring_block():
    assert (
        u._scoring_leaf({"scoring": {"domain": "image"}}, "domain") == "image"
    )


def test_scoring_leaf_returns_the_default_when_neither_spelling_resolves():
    assert u._scoring_leaf(_Block(), "output_transform", "fallback") == "fallback"
    assert u._scoring_leaf(None, "output_transform") is None
