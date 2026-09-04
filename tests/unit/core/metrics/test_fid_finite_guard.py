"""FID must reject non-finite inputs at the metric boundary.

FID accumulates Inception features across batches and reduces once at the end,
so a single NaN-bearing batch silently corrupts the whole score with no clear
error. The same canonical finiteness guard used by LPIPS must fire on FID
inputs, before the Inception backbone, so a diverged model is reported as the
cause (CLAUDE.md pitfall #9). The stubbed backbone means torchmetrics is not
required to run this test.
"""

import pytest

from tests.utils.optional_backends import requires_torch_fidelity
import torch

from spectramr.core.metrics.evaluation_metrics import FID


class _ExplodingFID(torch.nn.Module):
    """Stand-in for torchmetrics FrechetInceptionDistance — fails if reached.

    Must be an `nn.Module`: `FID` is one, so `metric.fid = ...` goes through
    `nn.Module.__setattr__`, which rejects a plain object with "cannot assign
    '_ExplodingFID' as child module 'fid'". The stub predates `FID` becoming a
    Module and failed on the assignment rather than on the behaviour it pins.
    """

    def update(self, *_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("FID backbone must not be reached on NaN input")

    @property
    def device(self):
        return torch.device("cpu")

    def to(self, *_a, **_k):
        return self


@requires_torch_fidelity
class TestFidFiniteGuard:
    @pytest.mark.unit
    @requires_torch_fidelity
    def test_fid_raises_on_nan_pred_before_backbone(self):
        metric = FID(device="cpu")
        metric.use_torchmetrics = True
        metric.fid = _ExplodingFID()

        preds = torch.rand(2, 1, 16, 16)
        preds[0, 0, 0, 0] = float("nan")
        target = torch.rand(2, 1, 16, 16)

        with pytest.raises(ValueError, match="non-finite"):
            metric.compute_metric(preds, target)
