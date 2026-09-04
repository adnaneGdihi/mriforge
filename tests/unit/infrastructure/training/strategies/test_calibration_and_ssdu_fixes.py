"""Regression tests for the 2026-05-31 audit re-run criticals:

* Calibration strategies' ``train_step`` must return a step-config LIST (empty
  = no optimizer steps), not a bare dict — the dict return hit
  ``KeyError('optimizer')`` in ``Trainer.execute_step`` (trainer.py:95).
* ``ssdu.split_acquired_mask`` built its RNG on CPU but drew with
  ``device=acq.device`` → ``RuntimeError`` on CUDA. The fix draws on the
  generator's device then moves the result, so it is device-agnostic.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.training.strategies.conformal_calibration_strategy import (
    ConformalCalibrationStrategy,
)
from spectramr.infrastructure.training.strategies.equivariance_conformal_strategy import (
    EquivarianceConformalCalibrationStrategy,
)
from spectramr.infrastructure.training.strategies.ssdu_strategy import split_acquired_mask


class TestCalibrationNoopTrainStep:
    def test_conformal_calibration_returns_empty_list(self):
        s = object.__new__(ConformalCalibrationStrategy)
        s.device = torch.device("cpu")
        out = s.train_step(batch=None, epoch=0, step=0)
        assert out == []  # list, not a dict -> no KeyError in execute_step

    def test_equivariance_conformal_returns_empty_list(self):
        s = object.__new__(EquivarianceConformalCalibrationStrategy)
        s.device = torch.device("cpu")
        out = s.train_step(batch=None, epoch=0, step=0)
        assert out == []


class TestSSDUSplitDeviceAgnostic:
    def test_split_is_disjoint_and_covers_acquired(self):
        g = torch.Generator(device="cpu")
        g.manual_seed(0)
        acq = torch.ones(4, 4)
        lam, theta = split_acquired_mask(acq, theta_fraction=0.5, generator=g)
        # lambda + theta == acquired ; lambda * theta == 0 (disjoint)
        assert torch.equal(lam + theta, acq)
        assert torch.count_nonzero(lam * theta) == 0
        assert int(theta.sum()) == 8  # 0.5 * 16 acquired cells
        # result rides the input device (the fix: no generator/tensor device clash)
        assert lam.device == acq.device
