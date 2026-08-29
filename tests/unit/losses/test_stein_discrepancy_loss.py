"""Unit tests for SteinDiscrepancyLoss (stein_federated façade closure).

Covers: registration, scalar-tensor output, near-zero discrepancy when the
prediction and target are drawn from the same distribution, larger discrepancy
when they differ, exercise of the real KSD U-statistic machinery, reduction
modes, and shape-mismatch error paths. Direct-import per the source<->test
pairing rule.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.losses.stein_discrepancy_loss import SteinDiscrepancyLoss


def test_registered_in_loss_registry() -> None:
    from mriforge.models.losses.registry import LossRegistry, create_loss

    assert "stein_discrepancy" in LossRegistry.list_available()
    loss = create_loss("stein_discrepancy")
    assert isinstance(loss, SteinDiscrepancyLoss)


def test_aliases_resolve() -> None:
    from mriforge.models.losses.registry import create_loss

    for alias in ("ksd", "kernelized_stein_discrepancy"):
        assert isinstance(create_loss(alias), SteinDiscrepancyLoss)


def test_domain_metadata_is_image() -> None:
    from mriforge.models.losses.registry import LossRegistry

    assert LossRegistry._loss_domains["stein_discrepancy"]["domain"] == "image"


def test_returns_scalar_tensor() -> None:
    loss = SteinDiscrepancyLoss()
    torch.manual_seed(0)
    pred = torch.randn(8, 1, 4, 4)
    target = torch.randn(8, 1, 4, 4)
    out = loss(pred, target)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 0
    assert torch.isfinite(out)


def test_discrepancy_small_when_same_distribution() -> None:
    """Identical samples => prediction matches the target distribution => KSD ~ 0."""
    loss = SteinDiscrepancyLoss()
    torch.manual_seed(1)
    target = torch.randn(16, 1, 6, 6)
    same = loss(target.clone(), target)
    assert same.item() < 1e-2, f"KSD on identical samples too large: {same.item()}"


def test_discrepancy_larger_when_distributions_differ() -> None:
    """A shifted/scaled prediction is a worse fit to the target distribution."""
    loss = SteinDiscrepancyLoss()
    torch.manual_seed(2)
    target = torch.randn(24, 1, 6, 6)
    same = loss(target.clone(), target)
    shifted = loss(target + 5.0, target)
    assert shifted.item() > same.item(), (
        f"shifted prediction KSD {shifted.item()} should exceed "
        f"matched KSD {same.item()}"
    )


def test_ksd_machinery_is_exercised(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real kernelised_stein_discrepancy U-statistic must be called."""
    import mriforge.models.losses.stein_discrepancy_loss as mod

    calls = {"n": 0}
    real = mod.kernelised_stein_discrepancy

    def spy(*args: object, **kwargs: object):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "kernelised_stein_discrepancy", spy)
    loss = SteinDiscrepancyLoss()
    torch.manual_seed(3)
    target = torch.randn(10, 1, 5, 5)
    loss(target.clone(), target)
    assert calls["n"] >= 1, "KSD U-statistic estimator was never invoked"


def test_reduction_sum_vs_mean_batched() -> None:
    loss_mean = SteinDiscrepancyLoss(reduction="mean", batch_chunks=2)
    loss_sum = SteinDiscrepancyLoss(reduction="sum", batch_chunks=2)
    torch.manual_seed(4)
    pred = torch.randn(12, 1, 5, 5)
    target = torch.randn(12, 1, 5, 5)
    m = loss_mean(pred, target)
    s = loss_sum(pred, target)
    # sum over the 2 chunks == 2 * mean over the 2 chunks.
    assert s.item() == pytest.approx(2.0 * m.item(), rel=1e-4, abs=1e-4)


def test_gradient_flows_to_prediction() -> None:
    """A real trainable loss: gradient must reach the prediction and be finite."""
    loss = SteinDiscrepancyLoss()
    torch.manual_seed(5)
    pred = torch.randn(20, 1, 5, 5, requires_grad=True)
    target = torch.randn(20, 1, 5, 5)
    out = loss(pred, target)
    out.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0


def test_shape_mismatch_raises() -> None:
    loss = SteinDiscrepancyLoss()
    pred = torch.randn(8, 1, 4, 4)
    target = torch.randn(8, 1, 5, 5)
    with pytest.raises(ValueError):
        loss(pred, target)


def test_too_few_samples_raises() -> None:
    loss = SteinDiscrepancyLoss()
    pred = torch.randn(1, 1, 4, 4)
    target = torch.randn(1, 1, 4, 4)
    with pytest.raises(ValueError):
        loss(pred, target)


def test_bad_reduction_raises() -> None:
    with pytest.raises(ValueError):
        SteinDiscrepancyLoss(reduction="median")
