"""Regression: ``B0Estimator`` paired-input contract.

The 2026-05-10 ``exp_51_b0_estimation`` smoke crashed at iter 1 with
``AssertionError: Expected at least 2 channels for concatenated input,
got 1`` because the YAML configures a 1-channel image dataset
(``coil_processing_mode: rss_image``, ``target_channels: 1``) but the
B0Estimator's standard-pipeline path expects ``[B, 2, H, W]`` —
[moving | fixed] concatenated along the channel dim.

Two-sided fix (2026-05-11):

1. ``B0Estimator.forward`` now accepts the fixed image via either
   ``fixed_3T`` (explicit), ``kwargs["target"]`` (kwarg propagation
   from a strategy), or ``moving_64mT[:, :1]/[:, 1:2]`` (legacy
   pre-concatenated form). When none of those are supplied for a
   single-channel input, it raises a clear ``ValueError`` explaining
   *why* the model needs paired data and *how* to wire it.

2. ``PhysicsDrivenTrainingStrategy._generate_predictions`` now
   propagates ``batch_context["hr"]`` as ``forward_kwargs["target"]``
   so paired-physics models like ``b0_estimator`` opt in without a
   strategy-side branch on model type. Generators that reject
   ``target`` (no ``**kwargs`` catch-all) get a graceful retry.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.models.geometric.b0_estimator import B0Estimator  # noqa: E402


def _tiny_model() -> B0Estimator:
    return B0Estimator(input_shape=(32, 32), input_channels=2)


def test_forward_with_explicit_fixed_kwarg() -> None:
    """Standard separated-args form: ``forward(moving, fixed_3T=...)``."""
    model = _tiny_model().eval()
    moving = torch.randn(2, 1, 32, 32)
    fixed = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        corrected, flow = model(moving, fixed_3T=fixed)
    assert corrected.shape == (2, 1, 32, 32)
    assert flow.shape == (2, 2, 32, 32)


def test_forward_with_target_kwarg_fallback() -> None:
    """The fix: ``forward(moving, target=fixed)`` — what the strategy uses.

    ``PhysicsDrivenTrainingStrategy`` propagates ``batch_context["hr"]``
    into ``forward_kwargs["target"]``; the B0Estimator must read it.
    """
    model = _tiny_model().eval()
    moving = torch.randn(2, 1, 32, 32)
    fixed = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        corrected, flow = model(moving, target=fixed)
    assert corrected.shape == (2, 1, 32, 32)


def test_forward_with_preconcatenated_input() -> None:
    """Legacy form: ``forward(concatenated)`` with ``[B, 2, H, W]``."""
    model = _tiny_model().eval()
    cat_input = torch.randn(2, 2, 32, 32)
    with torch.no_grad():
        corrected, flow = model(cat_input)
    assert corrected.shape == (2, 1, 32, 32)


def test_forward_raises_clearly_when_no_pair_available() -> None:
    """1-ch input with no fixed/target kwarg raises a model-specific
    error message that says WHY (B0 needs paired scans) and HOW to fix.
    A silent fallback would produce a meaningless B0 estimate — exactly
    the kind of issue CLAUDE.md pitfall #9 forbids.
    """
    model = _tiny_model().eval()
    moving = torch.randn(2, 1, 32, 32)
    with pytest.raises(ValueError, match="paired"):
        model(moving)
