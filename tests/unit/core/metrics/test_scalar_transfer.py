"""The fusion has to actually fuse, and it must not answer the caller's question.

Two independent properties, and both have failed in this repo before:

1. **One sync, not N.** The idiom this module extracts existed at three sites
   because the obvious spelling (``.item()`` per entry) is a GPU sync each. A
   "fused" helper that quietly falls back to per-tensor transfers would keep
   every caller's timing exactly as bad while reading as fixed -- so the
   single-``torch.stack`` property is asserted, not assumed.

2. **It refuses to reduce.** ``mean()`` over a non-scalar is correct for a
   per-sample loss and a *defect* for a metric that promised a scalar. A helper
   that picked one would silently answer both questions the same way, which is
   how a vector-returning metric turns into a plausible number instead of a
   flagged failure (pitfall #18).
"""

from __future__ import annotations

import re
from unittest import mock

import pytest
import torch

from mriforge.core.metrics import scalar_transfer
from mriforge.core.metrics.scalar_transfer import fuse_to_host


class TestTransfersCorrectValues:
    def test_empty_in_empty_out(self):
        assert fuse_to_host([]) == []

    def test_values_and_order_are_preserved(self):
        out = fuse_to_host([torch.tensor(3.0), torch.tensor(-1.5), torch.tensor(0.0)])
        assert out == [3.0, -1.5, 0.0]

    def test_returns_python_floats_not_tensors(self):
        (value,) = fuse_to_host([torch.tensor(2.5)])
        assert type(value) is float

    def test_single_element_non_zero_dim_tensor_is_accepted(self):
        """``metric(...)`` often returns shape ``(1,)``, not ``()``."""
        assert fuse_to_host([torch.tensor([7.0])]) == [7.0]

    def test_nan_survives_the_transfer(self):
        """NaN is this codebase's 'not computed' marker -- it must not be lost."""
        (value,) = fuse_to_host([torch.tensor(float("nan"))])
        assert value != value  # NaN self-inequality is the check

    def test_mixed_amp_dtypes_do_not_raise(self):
        """The trap that forces the float64 widening.

        A bare ``torch.stack`` of an fp16 and an fp32 scalar raises; under
        autocast that mix is the NORMAL case, so the naive fusion would fail on
        exactly the runs it was written for.
        """
        out = fuse_to_host([torch.tensor(1.0, dtype=torch.float16), torch.tensor(2.0)])
        assert out == [1.0, 2.0]

    def test_widening_is_lossless_for_float32(self):
        """float64 is a WIDENING, so the value must be bit-identical."""
        value = 0.1 + 0.2  # not representable; catches a stray round-trip
        t = torch.tensor(value, dtype=torch.float32)
        assert fuse_to_host([t]) == [float(t.item())]


class TestTheFusionIsReal:
    def test_n_scalars_cost_exactly_one_stack(self):
        with mock.patch.object(
            scalar_transfer.torch, "stack", wraps=torch.stack
        ) as spy:
            fuse_to_host([torch.tensor(float(i)) for i in range(8)])
        assert spy.call_count == 1

    def test_no_per_tensor_item_on_the_happy_path(self):
        """``.item()`` per entry is the anti-pattern; it must not survive."""
        with mock.patch.object(
            torch.Tensor, "item", autospec=True, side_effect=AssertionError("synced")
        ):
            assert fuse_to_host([torch.tensor(1.0), torch.tensor(2.0)]) == [1.0, 2.0]

    def test_mixed_device_falls_back_rather_than_raising(self):
        """Correctness outranks the fusion when the stack cannot be formed."""
        with mock.patch.object(
            scalar_transfer.torch, "stack", side_effect=RuntimeError("mixed devices")
        ):
            assert fuse_to_host([torch.tensor(4.0), torch.tensor(5.0)]) == [4.0, 5.0]


class TestItRefusesToPickAReduction:
    def test_non_scalar_raises_and_says_why(self):
        with pytest.raises(ValueError, match="single-element"):
            fuse_to_host([torch.tensor([1.0, 2.0])])

    def test_the_message_names_the_caller_policy(self):
        """A bare 'bad shape' would invite the wrong fix (silently mean it)."""
        with pytest.raises(ValueError, match="loss"):
            fuse_to_host([torch.zeros(4)])

    def test_complex_raises_rather_than_taking_real(self):
        with pytest.raises(ValueError, match=r"(?i)complex"):
            fuse_to_host([torch.tensor(1 + 2j)])

    def test_non_tensor_raises(self):
        with pytest.raises(ValueError, match=re.escape("torch.Tensor")):
            fuse_to_host([1.0])  # type: ignore[list-item]

    def test_the_offending_index_is_reported(self):
        """With 20 metrics in flight, 'one of them was wrong' is not actionable."""
        with pytest.raises(ValueError, match=r"\[2\]"):
            fuse_to_host([torch.tensor(1.0), torch.tensor(2.0), torch.zeros(3)])


class TestDetachment:
    def test_a_grad_tracking_tensor_is_accepted(self):
        """Metrics are computed under no_grad in most paths, but not all."""
        t = (torch.tensor([2.0], requires_grad=True) * 3).sum()
        assert fuse_to_host([t]) == [6.0]
