"""Regression tests for ModelValidationMixin.validation_step.

The load-bearing case here is the `batch_context` re-forwarding bug: line 95 calls
``self._validation_forward(input_batch, batch_context, **kwargs)``. If
``validation_step`` only ``.get``s ``batch_context`` (instead of ``.pop``), a caller
that supplies ``batch_context=`` (the cross-field / field-flow / field-bridge
``validation_step`` overrides do) leaves it in ``**kwargs`` too, so it is passed both
positionally and by keyword -> ``TypeError: ... multiple values for argument
'batch_context'``. This crashed EVERY validation pass for those three strategies, and
their per-strategy tests masked it by calling ``_validation_forward`` directly. This
test drives the FULL ``validation_step`` chain so it can never regress silently again.
"""

from __future__ import annotations

import types

from tests.utils.block_config_stub import ValidationConfigStub

import torch

from spectramr.infrastructure.training.strategies.mixins.model_validation import (
    ModelValidationMixin,
)


class _Stub(ModelValidationMixin):
    """Minimal carrier: image metrics are turned off so we exercise only the
    unpack -> batch_context -> _validation_forward path that holds the bug."""

    def __init__(self) -> None:
        self.generator_model = types.SimpleNamespace(eval=lambda: None)
        self.state = types.SimpleNamespace(
            config=types.SimpleNamespace(
                training=None,
                validation=ValidationConfigStub(scoring={"enable_image_metrics": False}),
            )
        )
        self.received: dict | None = None

    def _validation_forward(self, input_batch, batch_context, **kwargs):
        self.received = {"bc": batch_context, "kw": dict(kwargs)}
        return torch.zeros_like(input_batch)


def test_validation_step_pops_batch_context_no_double_pass() -> None:
    s = _Stub()
    out = s.validation_step(
        torch.zeros(1, 1, 4, 4),
        torch.zeros(1, 1, 4, 4),
        batch_context={"k": 1},
        field_strength=torch.tensor([1.0]),
    )
    # compute_image_metrics is False -> no metric machinery -> {} (and no crash).
    assert out == {}
    # batch_context reached _validation_forward exactly once (positionally)...
    assert s.received is not None
    assert s.received["bc"] == {"k": 1}
    # ...and was NOT also left in **kwargs (the double-pass that raised TypeError).
    assert "batch_context" not in s.received["kw"]
    # unrelated kwargs (per-sample fields) still flow through.
    assert "field_strength" in s.received["kw"]


def test_validation_step_without_batch_context_uses_default() -> None:
    # The .get -> .pop change must not regress the common no-batch_context path:
    # the mixin synthesises a default context.
    s = _Stub()
    s.validation_step(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    assert s.received is not None
    assert s.received["bc"] == {"use_dc": False, "measured_kspace": None}


class _MetricStub(ModelValidationMixin):
    """Image metrics ON but no computer wired, so validation_step skips the metric
    machinery and falls through to the val_loss emission we exercise here."""

    def __init__(self) -> None:
        self.generator_model = types.SimpleNamespace(eval=lambda: None)
        self.state = types.SimpleNamespace(
            config=types.SimpleNamespace(
                training=None,
                validation=ValidationConfigStub(scoring={"enable_image_metrics": True}),
            )
        )
        self.validation_metrics_computer = None  # -> image-metric block is skipped

    def _validation_forward(self, input_batch, batch_context, **kwargs):
        return torch.zeros_like(input_batch) + 0.5


def test_validation_step_emits_val_loss() -> None:
    # fig_1_2_learning_curves needs a validation loss series; the mixin emits one
    # (magnitude-matched L1 on pred/target) so the curve is no longer train-only.
    out = _MetricStub().validation_step(torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4))
    assert "val_loss" in out
    assert abs(out["val_loss"] - 0.5) < 1e-6  # L1(0.5, 1.0) over a constant field


def test_val_loss_absent_when_image_metrics_disabled() -> None:
    # When validation metric computation is turned off, no val_loss is emitted
    # (keeps the compute_image_metrics=False contract -> empty metrics).
    assert _Stub().validation_step(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)) == {}


class _DistHeadStub(ModelValidationMixin):
    """A heteroscedastic strategy: the model emits ``[mean, logvar]``, not real/imag.

    ``predicts_distribution_params`` is the existing declaration (``base.py``;
    ``heteroscedastic_ulf_strategy.py`` sets it True) that the two channel-collapse
    fallbacks never consulted.
    """

    def __init__(self, dist_head: bool) -> None:
        self.predicts_distribution_params = dist_head
        self.generator_model = types.SimpleNamespace(eval=lambda: None)
        self.state = types.SimpleNamespace(
            config=types.SimpleNamespace(
                training=None,
                validation=ValidationConfigStub(scoring={"enable_image_metrics": True}),
            )
        )
        self.validation_metrics_computer = None

    def _validation_forward(self, input_batch, batch_context, **kwargs):
        b, _, h, w = input_batch.shape
        mean = torch.full((b, 1, h, w), 0.5)
        logvar = torch.full((b, 1, h, w), -4.0)  # near-constant, as logvar is
        return torch.cat([mean, logvar], dim=1)


def _val_loss(dist_head: bool) -> float:
    s = _DistHeadStub(dist_head)
    target = torch.full((1, 1, 4, 4), 0.5)
    out = s.validation_step(torch.zeros(1, 1, 4, 4), target)
    return float(out["val_loss"])


def test_distribution_head_is_not_rss_combined() -> None:
    """[mean, logvar] must collapse to the MEAN, never sqrt(mean^2 + logvar^2).

    RSS-ing a distribution head mixes an intensity with a log-variance. Because
    logvar is near-constant across the image, the result is dragged toward an
    input-independent field — the b29 wrong-channel metric, which reported a
    large val_loss for a prediction that exactly matched its target.
    """
    # mean == target exactly, so a correct collapse gives ~0 loss.
    assert _val_loss(dist_head=True) < 1e-6, (
        "distribution head was RSS-combined: sqrt(0.5^2 + (-4)^2) ~= 4.03 is graded "
        "against a target of 0.5, so a perfect prediction scores as a large error"
    )


def test_complex_two_channel_output_still_rss_combines() -> None:
    """The guard must not disturb genuine real/imag outputs."""
    got = _val_loss(dist_head=False)
    expected = abs((0.5**2 + 4.0**2) ** 0.5 - 0.5)
    assert got > 1.0, f"real/imag path must still RSS; got {got}"
    assert abs(got - expected) < 0.5, f"expected ~{expected}, got {got}"


# ---------------------------------------------------------------------------
# One `val_psnr`, not two (issue #179 / #180 follow-through).
#
# When the metrics computer raised, this mixin recomputed `val_psnr` from
# `target.max() - target.min()` -- a PER-IMAGE extent -- and wrote it under the
# SAME key the primary path uses. So one key carried two incomparable definitions,
# selected by an exception nobody observes, and early stopping plus
# `final_metrics.best` consumed it as if it were one quantity.
# ---------------------------------------------------------------------------


def test_fallback_no_longer_derives_its_own_data_range():
    """Source guard: the per-image `max - min` derivation must not come back.

    Pinned at the source because the defect is a *definition* divergence — two
    numeric paths that agree on the fixtures you happen to pick and diverge on a
    dark slice. A value assertion alone would not have caught the original bug.
    """
    import inspect

    from spectramr.infrastructure.training.strategies.mixins import model_validation

    src = inspect.getsource(model_validation)
    assert "resolve_image_data_range" in src, (
        "the fallback must share the registered psnr's data_range resolver"
    )
    assert ".max().item() - " not in src, (
        "a per-image `max - min` data_range reappeared in the fallback; it makes "
        "val_psnr mean something different depending on which branch produced it"
    )


def test_fallback_and_canonical_psnr_agree_on_a_dark_slice():
    """The case that separates the two definitions.

    A slice spanning [0, 0.3] scored ~10 dB better under the per-image extent than
    under the normalization contract — the exact "flattered dark image" failure.
    """
    import math

    from spectramr.core.metrics.evaluation_metrics import (
        PSNR,
        resolve_image_data_range,
    )

    gen = torch.Generator().manual_seed(0)
    target = torch.rand(1, 1, 32, 32, generator=gen) * 0.3
    pred = (target + 0.02 * torch.randn(1, 1, 32, 32, generator=gen)).clamp(0, 0.3)

    canonical = float(PSNR()(pred, target))
    mse = torch.nn.functional.mse_loss(pred, target).item()
    fallback = 10.0 * math.log10(
        resolve_image_data_range(target, None, metric_name="val_psnr") ** 2 / mse
    )
    assert abs(canonical - fallback) < 1e-3, (
        f"fallback {fallback:.3f} dB disagrees with canonical {canonical:.3f} dB"
    )

    # And the old derivation is what it must NOT reproduce.
    old = 10.0 * math.log10(
        (target.max().item() - target.min().item()) ** 2 / mse
    )
    assert abs(old - canonical) > 5.0, (
        "fixture no longer separates the two definitions — pick a darker slice"
    )


def test_fallback_declines_rather_than_inventing_on_unnormalized_data():
    from spectramr.core.metrics.evaluation_metrics import resolve_image_data_range
    from spectramr.core.metrics.outcome import MetricNotApplicableError

    import pytest

    unnormalized = torch.rand(1, 1, 16, 16, generator=torch.Generator().manual_seed(0)) * 2479
    with pytest.raises(MetricNotApplicableError):
        resolve_image_data_range(unnormalized, None, metric_name="val_psnr")
