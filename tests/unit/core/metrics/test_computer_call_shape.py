"""The metric call-shape seam (``computer.py``).

Why this file exists: ``MetricsRegistry.requires_reference`` is a *declaration*
and a metric's ``forward`` signature is the *implementation*, and the two
disagree for 69 of the 211 registered metrics -- always in the direction
"declared no-reference, signature still takes (and ignores) a target". Any
dispatch that trusts the declaration alone hands those 69 one argument too few
and converts a working measurement into a swallowed ``TypeError`` -> NaN.

These tests plant both shapes, so the seam is exercised on the case it exists
for and on the case it must not break.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.computer import (
    MetricCallShape,
    ValidationMetricsComputer,
    resolve_metric_call_shape,
)
from spectramr.core.metrics.outcome import (
    MetricContractError,
    MetricNotApplicableError,
    NotApplicableReason,
)
from spectramr.core.metrics.registry import MetricsRegistry
from spectramr.core.metrics.types import MetricSpec, ValidationMetricsConfig


class _NoRefSingleArg(torch.nn.Module):
    """The genuinely target-less shape (``negative_voxels``, ``ndc_diffusion``)."""

    def forward(self, preds: torch.Tensor) -> torch.Tensor:
        return preds.mean()


class _NoRefTakesTarget(torch.nn.Module):
    """Declares no-reference, still takes a target positionally -- 69 metrics."""

    def forward(
        self, preds: torch.Tensor, target: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        return preds.mean()


class _FullReference(torch.nn.Module):
    def forward(
        self, preds: torch.Tensor, target: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        return (preds - target).abs().mean()


class _NoRefNamedKwargOnly(torch.nn.Module):
    """No ``**kwargs``: unsupported context kwargs must be filtered, not passed."""

    def forward(self, preds: torch.Tensor, domain: str = "image") -> torch.Tensor:
        return preds.mean()


def _compute(name: str, **computer_kwargs: object) -> dict[str, float]:
    cfg = ValidationMetricsConfig(metrics=[MetricSpec(name=name)])
    computer = ValidationMetricsComputer(config=cfg, device="cpu", **computer_kwargs)
    return computer.compute(torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8))


# --------------------------------------------------------------------------- #
# resolve_metric_call_shape
# --------------------------------------------------------------------------- #
def test_single_arg_forward_is_reported_as_target_less() -> None:
    shape = resolve_metric_call_shape(_NoRefSingleArg().forward)
    assert shape.accepts_target is False


def test_two_positional_forward_is_reported_as_target_taking() -> None:
    shape = resolve_metric_call_shape(_NoRefTakesTarget().forward)
    assert shape.accepts_target is True
    assert shape.accepted_kwargs is None  # declares **kwargs


def test_var_positional_counts_as_target_taking() -> None:
    def fn(preds: torch.Tensor, *args: object) -> torch.Tensor:
        return preds.mean()

    assert resolve_metric_call_shape(fn).accepts_target is True


def test_named_kwargs_are_enumerated_when_there_is_no_varkw() -> None:
    shape = resolve_metric_call_shape(_NoRefNamedKwargOnly().forward)
    assert shape.accepted_kwargs == frozenset({"preds", "domain"})


def test_uninspectable_callable_falls_back_to_the_permissive_shape(monkeypatch) -> None:
    """5 of the 211 registered metrics (e.g. ``fid``) cannot be introspected.

    The fallback must be the shape they were already being called with, so an
    unknown callable is not silently handed a narrower contract than before.
    """
    import inspect as _inspect

    def _boom(_obj: object) -> None:
        raise ValueError("no signature found")

    monkeypatch.setattr(
        "spectramr.core.metrics.computer.inspect.signature", _boom, raising=True
    )
    assert _inspect is not None  # the module itself is untouched elsewhere
    shape = resolve_metric_call_shape(_NoRefSingleArg().forward)
    assert shape == MetricCallShape(accepts_target=True, accepted_kwargs=None)


# --------------------------------------------------------------------------- #
# dispatch through the computer
# --------------------------------------------------------------------------- #
def test_target_less_metric_is_invoked_and_scores(registered_metric) -> None:
    """The defect this seam was built for: a real number instead of NaN."""
    name = registered_metric(_NoRefSingleArg, requires_reference=False)
    result = _compute(name)
    assert not torch.isnan(torch.tensor(result[name]))


def test_no_reference_declaration_does_not_starve_a_target_taking_metric(
    registered_metric,
) -> None:
    """Anti-regression for the 69: the declaration must not drop their target."""
    name = registered_metric(_NoRefTakesTarget, requires_reference=False)
    result = _compute(name)
    assert not torch.isnan(torch.tensor(result[name]))


def test_full_reference_metric_still_receives_its_target(registered_metric) -> None:
    name = registered_metric(_FullReference, requires_reference=True)
    result = _compute(name)
    assert result[name] > 0.0


def test_declared_reference_with_target_less_signature_raises(
    registered_metric,
) -> None:
    """Declaration and signature are the same claim; disagreement is a defect.

    Silently dropping the target would report a no-reference number under a
    full-reference name -- a wrong number, which this framework ranks above a
    crash.
    """
    name = registered_metric(_NoRefSingleArg, requires_reference=True)
    with pytest.raises(MetricContractError, match="requires_reference=True"):
        _compute(name)


def test_declared_data_range_is_never_silently_dropped(registered_metric) -> None:
    """A configured ``data_range`` a metric cannot accept is a declared N/A.

    Falling back to the metric's built-in range would substitute a default for
    a value the run declared (non-negotiable 3b).
    """
    name = registered_metric(_NoRefNamedKwargOnly, requires_reference=False)
    cfg = ValidationMetricsConfig(metrics=[MetricSpec(name=name)])
    computer = ValidationMetricsComputer(config=cfg, device="cpu", data_range=2.0)
    result = computer.compute(torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8))
    assert torch.isnan(torch.tensor(result[name]))
    assert (
        computer.last_not_applicable[name]
        is NotApplicableReason.DECLARED_KWARG_UNSUPPORTED
    )


def test_metric_not_applicable_is_the_declared_error_type() -> None:
    err = MetricNotApplicableError(
        "x", NotApplicableReason.DECLARED_KWARG_UNSUPPORTED, "detail"
    )
    assert err.reason is NotApplicableReason.DECLARED_KWARG_UNSUPPORTED


@pytest.fixture
def registered_metric():
    """Register a throwaway metric and unregister it afterwards."""
    registered: list[str] = []

    def _register(cls: type, *, requires_reference: bool) -> str:
        name = f"_probe_{cls.__name__.lower()}_{int(requires_reference)}"
        MetricsRegistry.register(name, requires_reference=requires_reference)(cls)
        registered.append(name)
        return name

    yield _register

    for name in registered:
        MetricsRegistry._metrics.pop(name, None)
        MetricsRegistry._requires_reference.pop(name, None)
