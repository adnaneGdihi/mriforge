"""``cubical_ph_w2_distance`` (geomamba_ulf review 2026-09-03).

The registration, the direction contract and the batch/shape handling are
checked everywhere (the loss is stubbed); the optional-dependency contract is
planted by blanking the loss module's ``_gudhi``; the numerical test needs the
``[topology]`` extra (gudhi + POT), which has no aarch64 wheel, so it skips on
the review box and runs on the cluster.
"""

from __future__ import annotations

import subprocess
import sys
from typing import ClassVar

import pytest

torch = pytest.importorskip("torch")

from spectramr.core.metrics.quantitative.cubical_ph_w2_distance import (  # noqa: E402
    CubicalPHW2DistanceMetric,
)


def test_registered_in_a_cold_interpreter() -> None:
    """The name and its aliases resolve after the registry is populated the way the computer does it."""
    code = (
        "from spectramr.core.metrics.registry import MetricsRegistry\n"
        "import spectramr.core.metrics\n"
        "print(MetricsRegistry.is_registered('cubical_ph_w2_distance'), "
        "MetricsRegistry.is_registered('ph_w2_distance'), "
        "MetricsRegistry.is_registered('topology_w2'))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.strip().endswith("True True True"), out.stdout


def test_missing_topology_extra_raises_not_zero(monkeypatch) -> None:
    """Planted violation shape: a silent 0.0 would make the topological claim vacuous."""
    import spectramr.models.losses.cubical_ph_w2_loss as m

    monkeypatch.setattr(m, "_gudhi", None)
    with pytest.raises(ImportError, match="topology"):
        CubicalPHW2DistanceMetric()


def test_direction_is_a_bool_on_the_class_not_a_property() -> None:
    """The resolver constructs a metric whose direction is a property, and construction
    needs the extra: with a property, ``val_cubical_ph_w2_distance`` is unresolvable on
    every box without gudhi (planted: the review draft declared a property)."""
    from spectramr.core.metrics.metric_directions import metric_higher_is_better

    declared = CubicalPHW2DistanceMetric.__dict__.get("higher_is_better")
    assert isinstance(declared, bool) and declared is False
    assert CubicalPHW2DistanceMetric.name == "cubical_ph_w2_distance"
    assert metric_higher_is_better("cubical_ph_w2_distance") is False
    assert metric_higher_is_better("val_cubical_ph_w2_distance") is False


class _RecordingLoss:
    """Stands in for the loss: records what it was handed, returns a fixed tensor."""

    calls: ClassVar[list[dict]] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def __call__(self, pred, target, mask=None, **kwargs):
        type(self).calls.append({"pred": pred, "target": target, "mask": mask})
        return torch.tensor(0.25, requires_grad=True) * pred.new_ones(())


@pytest.fixture
def stubbed_loss(monkeypatch):
    import spectramr.models.losses.cubical_ph_w2_loss as m

    _RecordingLoss.calls = []
    monkeypatch.setattr(m, "CubicalPHWassersteinLoss", _RecordingLoss)
    return _RecordingLoss


def test_call_hands_the_loss_a_detached_batched_magnitude_and_reduces_to_float(
    stubbed_loss,
) -> None:
    """The fires-test that runs without the extra: shapes, complex -> magnitude, no
    gradient, the computer's kwargs absorbed, and a float back."""
    metric = CubicalPHW2DistanceMetric(wasserstein_p=2, device="cpu")
    assert metric._loss.kwargs == {"wasserstein_p": 2}

    x = torch.rand(8, 8, requires_grad=True)
    value = metric(x, x, device="cpu", domain="image", data_range=1.0)
    assert value == pytest.approx(0.25)
    assert isinstance(value, float)
    call = stubbed_loss.calls[-1]
    assert call["pred"].shape == (1, 1, 8, 8) and not call["pred"].requires_grad
    assert call["mask"] is None

    z = torch.complex(torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8))
    metric(z, z)
    call = stubbed_loss.calls[-1]
    assert not call["pred"].is_complex() and call["pred"].dtype == torch.float32
    assert torch.allclose(call["pred"], z.abs())


def test_a_foreground_mask_is_expanded_to_the_prediction_shape(stubbed_loss) -> None:
    metric = CubicalPHW2DistanceMetric()
    pred = torch.rand(2, 3, 8, 8)
    mask = torch.ones(2, 1, 8, 8)
    metric(pred, pred, mask=mask)
    assert stubbed_loss.calls[-1]["mask"].shape == pred.shape
    with pytest.raises(RuntimeError):  # a mask that cannot broadcast raises here, not in the loss
        metric(pred, pred, mask=torch.ones(2, 2, 8, 8))


def test_shape_mismatch_raises_before_the_loss_runs(stubbed_loss) -> None:
    metric = CubicalPHW2DistanceMetric()
    with pytest.raises(ValueError, match="matching shapes"):
        metric(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 9))
    assert stubbed_loss.calls == []


def test_identity_is_zero_and_a_perturbation_is_positive() -> None:
    pytest.importorskip("gudhi", reason="needs the [topology] extra (no aarch64 wheel)")
    pytest.importorskip("ot", reason="needs the [topology] extra")

    metric = CubicalPHW2DistanceMetric()
    torch.manual_seed(0)
    x = torch.rand(1, 1, 16, 16)
    assert metric(x, x) == pytest.approx(0.0, abs=1e-6)
    y = x.clone()
    y[0, 0, 4:8, 4:8] += 0.5  # a new sublevel-set feature
    assert metric(y, x) > 0.0
