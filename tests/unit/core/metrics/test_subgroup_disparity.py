"""Tests for fairness / subgroup-disparity metrics (PR-3).

Covers ``subgroup_psnr_disparity`` and ``subgroup_ssim_gap``: registration,
direction, single-group degeneracy, the systematically-worse-subgroup case,
and shape-mismatch error handling.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.core.metrics import MetricsRegistry, get_metric


class TestSubgroupDisparityRegistration:
    def test_both_metrics_registered(self) -> None:
        assert MetricsRegistry.is_registered("subgroup_psnr_disparity")
        assert MetricsRegistry.is_registered("subgroup_ssim_gap")

    def test_aliases_registered(self) -> None:
        assert MetricsRegistry.is_registered("psnr_disparity")
        assert MetricsRegistry.is_registered("ssim_disparity")

    def test_higher_is_better_false(self) -> None:
        assert get_metric("subgroup_psnr_disparity").higher_is_better is False
        assert get_metric("subgroup_ssim_gap").higher_is_better is False

    def test_name_property(self) -> None:
        assert get_metric("subgroup_psnr_disparity").name == "subgroup_psnr_disparity"
        assert get_metric("subgroup_ssim_gap").name == "subgroup_ssim_gap"


class TestSubgroupPSNRDisparity:
    def test_returns_float(self) -> None:
        pred = torch.rand(4, 1, 16, 16)
        target = torch.rand(4, 1, 16, 16)
        val = get_metric("subgroup_psnr_disparity")(
            pred, target, subgroup=["a", "a", "b", "b"]
        )
        assert isinstance(val, float)

    def test_no_subgroup_kwarg_returns_zero(self) -> None:
        pred = torch.rand(4, 1, 16, 16)
        target = torch.rand(4, 1, 16, 16)
        assert get_metric("subgroup_psnr_disparity")(pred, target) == 0.0

    def test_single_group_returns_zero(self) -> None:
        pred = torch.rand(4, 1, 16, 16)
        target = torch.rand(4, 1, 16, 16)
        val = get_metric("subgroup_psnr_disparity")(
            pred, target, subgroup=["s0", "s0", "s0", "s0"]
        )
        assert val == 0.0

    def test_identical_per_group_zero_disparity(self) -> None:
        # Same prediction quality across both groups => ~zero gap.
        target = torch.rand(4, 1, 16, 16)
        pred = target.clone()
        val = get_metric("subgroup_psnr_disparity")(
            pred, target, subgroup=["a", "a", "b", "b"]
        )
        assert val == pytest.approx(0.0, abs=1e-4)

    def test_systematically_worse_subgroup_positive_disparity(self) -> None:
        # Group "a": perfect reconstruction. Group "b": heavy noise => worse PSNR.
        target = torch.rand(4, 1, 16, 16)
        pred = target.clone()
        pred[2:] = pred[2:] + 0.5 * torch.rand(2, 1, 16, 16)
        val = get_metric("subgroup_psnr_disparity")(
            pred, target, subgroup=["a", "a", "b", "b"]
        )
        assert val > 0.0

    def test_tensor_subgroup_labels(self) -> None:
        target = torch.rand(4, 1, 16, 16)
        pred = target.clone()
        pred[2:] = pred[2:] + 0.5 * torch.rand(2, 1, 16, 16)
        labels = torch.tensor([0, 0, 1, 1])
        val = get_metric("subgroup_psnr_disparity")(pred, target, subgroup=labels)
        assert val > 0.0

    def test_shape_mismatch_raises(self) -> None:
        pred = torch.rand(4, 1, 16, 16)
        target = torch.rand(4, 1, 8, 8)
        with pytest.raises(ValueError):
            get_metric("subgroup_psnr_disparity")(
                pred, target, subgroup=["a", "a", "b", "b"]
            )

    def test_label_length_mismatch_raises(self) -> None:
        pred = torch.rand(4, 1, 16, 16)
        target = torch.rand(4, 1, 16, 16)
        with pytest.raises(ValueError):
            get_metric("subgroup_psnr_disparity")(pred, target, subgroup=["a", "b"])


class TestSubgroupSSIMGap:
    def test_returns_float(self) -> None:
        pred = torch.rand(4, 1, 16, 16)
        target = torch.rand(4, 1, 16, 16)
        val = get_metric("subgroup_ssim_gap")(
            pred, target, subgroup=["a", "a", "b", "b"]
        )
        assert isinstance(val, float)

    def test_no_subgroup_kwarg_returns_zero(self) -> None:
        pred = torch.rand(4, 1, 16, 16)
        target = torch.rand(4, 1, 16, 16)
        assert get_metric("subgroup_ssim_gap")(pred, target) == 0.0

    def test_single_group_returns_zero(self) -> None:
        pred = torch.rand(4, 1, 16, 16)
        target = torch.rand(4, 1, 16, 16)
        val = get_metric("subgroup_ssim_gap")(
            pred, target, subgroup=["g", "g", "g", "g"]
        )
        assert val == 0.0

    def test_systematically_worse_subgroup_positive_gap(self) -> None:
        target = torch.rand(4, 1, 16, 16)
        pred = target.clone()
        pred[2:] = pred[2:] + 0.5 * torch.rand(2, 1, 16, 16)
        val = get_metric("subgroup_ssim_gap")(
            pred, target, subgroup=["a", "a", "b", "b"]
        )
        assert val > 0.0

    def test_shape_mismatch_raises(self) -> None:
        pred = torch.rand(4, 1, 16, 16)
        target = torch.rand(4, 1, 8, 8)
        with pytest.raises(ValueError):
            get_metric("subgroup_ssim_gap")(pred, target, subgroup=["a", "a", "b", "b"])
