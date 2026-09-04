r"""Unit tests for the DL-BAE objective terms.

Targets ``spectramr.models.losses.image.multifield_data_consistency`` and
``spectramr.models.losses.dispersion_monotonicity_loss``.

The data-consistency term is what forces one field-invariant latent to explain
*every* field, so it must genuinely aggregate across the field axis rather than
collapse to whichever field dominates. The monotonicity term is a one-sided
hinge: exactly zero on a physical solution (so it adds no gradient there) and
positive the moment :math:`\partial T_1/\partial B_0 < 0`.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.dispersion_monotonicity_loss import DispersionMonotonicity
from spectramr.models.losses.image.multifield_data_consistency import (
    MultiFieldDataConsistency,
)

pytestmark = pytest.mark.unit


class TestMultiFieldDataConsistency:
    def test_zero_on_exact_reconstruction(self) -> None:
        loss = MultiFieldDataConsistency()
        x = torch.rand(2, 5, 4, 4)
        assert loss(x, x).item() == pytest.approx(0.0, abs=1e-12)

    def test_every_field_contributes(self) -> None:
        """An error confined to one field must still register."""
        loss = MultiFieldDataConsistency()
        target = torch.zeros(1, 4, 4, 4)
        for field in range(4):
            pred = torch.zeros(1, 4, 4, 4)
            pred[:, field] = 1.0
            assert loss(pred, target).item() > 0.0, field

    def test_field_weights_reweight_the_sum(self) -> None:
        """Down-weighting a noisy low-field term must reduce its influence."""
        target = torch.zeros(1, 2, 2, 2)
        pred = torch.zeros(1, 2, 2, 2)
        pred[:, 0] = 1.0  # error lives entirely in field 0
        heavy = MultiFieldDataConsistency(field_weights=(1.0, 0.0))
        light = MultiFieldDataConsistency(field_weights=(0.0, 1.0))
        assert heavy(pred, target).item() > light(pred, target).item()

    def test_l2_mode_penalises_large_errors_more(self) -> None:
        target = torch.zeros(1, 2, 2, 2)
        pred = torch.full((1, 2, 2, 2), 2.0)
        l1 = MultiFieldDataConsistency(reduction_p=1)(pred, target)
        l2 = MultiFieldDataConsistency(reduction_p=2)(pred, target)
        assert l2.item() > l1.item()

    def test_shape_mismatch_raises(self) -> None:
        loss = MultiFieldDataConsistency()
        with pytest.raises(ValueError, match="matching shapes"):
            loss(torch.rand(1, 5, 4, 4), torch.rand(1, 4, 4, 4))

    def test_non_4d_raises(self) -> None:
        loss = MultiFieldDataConsistency()
        with pytest.raises(ValueError, match="4-D multi-field stack"):
            loss(torch.rand(1, 5, 4), torch.rand(1, 5, 4))

    def test_weight_count_mismatch_raises(self) -> None:
        loss = MultiFieldDataConsistency(field_weights=(1.0, 1.0))
        with pytest.raises(ValueError, match="field_weights has"):
            loss(torch.rand(1, 5, 4, 4), torch.rand(1, 5, 4, 4))

    def test_bad_reduction_raises(self) -> None:
        with pytest.raises(ValueError, match="reduction_p"):
            MultiFieldDataConsistency(reduction_p=3)


class TestDispersionMonotonicity:
    def test_zero_on_monotone_t1(self) -> None:
        """A physical solution must attract no penalty at all."""
        loss = DispersionMonotonicity()
        t1 = torch.linspace(0.3, 1.2, 5).reshape(1, 5, 1, 1).expand(1, 5, 3, 3)
        assert loss(t1).item() == pytest.approx(0.0, abs=1e-12)

    def test_positive_on_decreasing_t1(self) -> None:
        loss = DispersionMonotonicity()
        t1 = torch.linspace(1.2, 0.3, 5).reshape(1, 5, 1, 1).expand(1, 5, 3, 3)
        assert loss(t1).item() > 0.0

    def test_sorts_by_field_when_given(self) -> None:
        """An unsorted field axis is reordered, not misread as a violation."""
        loss = DispersionMonotonicity()
        fields = torch.tensor([3.0, 0.3, 1.5])
        t1_by_field = torch.tensor([1.0, 0.4, 0.7]).reshape(1, 3, 1, 1).expand(1, 3, 2, 2)
        assert loss(t1_by_field, fields).item() == pytest.approx(0.0, abs=1e-12)
        assert loss(t1_by_field).item() > 0.0  # unsorted, read literally

    def test_single_field_is_zero_but_differentiable(self) -> None:
        """One field carries no monotonicity info; the graph must survive anyway."""
        loss = DispersionMonotonicity()
        t1 = torch.rand(1, 1, 2, 2, requires_grad=True)
        out = loss(t1)
        assert out.item() == pytest.approx(0.0, abs=1e-12)
        out.backward()
        assert t1.grad is not None

    def test_squared_mode_is_smoother_but_still_zero_when_physical(self) -> None:
        loss = DispersionMonotonicity(squared=True)
        t1 = torch.linspace(0.3, 1.2, 4).reshape(1, 4, 1, 1).expand(1, 4, 2, 2)
        assert loss(t1).item() == pytest.approx(0.0, abs=1e-12)

    def test_field_count_mismatch_raises(self) -> None:
        loss = DispersionMonotonicity()
        with pytest.raises(ValueError, match="fields has"):
            loss(torch.rand(1, 3, 2, 2), torch.tensor([1.0, 2.0]))

    def test_negative_tolerance_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            DispersionMonotonicity(tolerance=-1.0)

    def test_gradient_pushes_toward_monotonicity(self) -> None:
        loss = DispersionMonotonicity()
        t1 = torch.tensor([1.0, 0.5]).reshape(1, 2, 1, 1).clone().requires_grad_(True)
        loss(t1).backward()
        # Raising the later (higher-field) value reduces the violation.
        assert t1.grad[0, 1, 0, 0].item() < 0.0
