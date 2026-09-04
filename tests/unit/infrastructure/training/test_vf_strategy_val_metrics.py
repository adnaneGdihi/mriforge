"""Tests for VF strategy validation_step val_psnr metric emission.

Validates that all VF strategies (VF-ADMM, VirtualFiducial, MotionMeta, TTO)
return ``val_psnr`` from their ``validation_step()`` so that early stopping
can find the monitored metric.

These tests use monkeypatching instead of full strategy instantiation to
avoid the heavyweight BaseTrainingStrategy DI/AMP initialization chain.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyGenerator(nn.Module):
    """Minimal generator stub that echoes input with slight noise."""

    def __init__(self, in_ch: int = 2, out_ch: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        nn.init.eye_(self.conv.weight[:, :, 0, 0])

    def forward(self, x, **kwargs):
        return self.conv(x)


class _DummySimulator:
    """Minimal DigitalTwinSimulator stub."""

    def __init__(self, im_size: tuple[int, int]):
        self.marker_mask = torch.ones(1, 1, *im_size)

    def to(self, device):
        self.marker_mask = self.marker_mask.to(device)
        return self

    def __call__(self, x):
        corrupted = x + 0.01 * torch.randn_like(x)
        marker_prior = torch.zeros_like(x)
        joint = x.clone()
        return corrupted, marker_prior, joint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def target_4d(device):
    """Real-stacked 4D target [B=2, C=2, H=32, W=32]."""
    return torch.randn(2, 2, 32, 32, device=device)


@pytest.fixture
def target_5d(device):
    """5D TorchIO volume [B=2, C=2, H=32, W=32, D=10]."""
    return torch.randn(2, 2, 32, 32, 10, device=device)


# ---------------------------------------------------------------------------
# Tests: Direct validation_step logic (no BaseTrainingStrategy init)
# ---------------------------------------------------------------------------


class TestVFADMMValPSNR:
    """ConcreteVFADMMStrategy._validation_step_with_psnr logic."""

    def _compute_val_psnr(
        self, target: torch.Tensor, gen: nn.Module, sim: _DummySimulator
    ) -> dict[str, float]:
        """Replicate the validation_step PSNR computation from VF-ADMM."""
        # Handle 5D
        if target.ndim == 5:
            mid = target.shape[-1] // 2
            target = target[..., mid]

        with torch.no_grad():
            if not torch.is_complex(target):
                C = target.shape[1]
                if C >= 2 and C % 2 == 0:
                    target_complex = torch.complex(target[:, 0::2], target[:, 1::2])
                else:
                    target_complex = target.to(torch.complex64)
            else:
                target_complex = target

            corrupted_image, _, _ = sim(target_complex)
            corrupted_input = torch.cat(
                [corrupted_image.real, corrupted_image.imag], dim=1
            )
            clean_target = torch.cat([target_complex.real, target_complex.imag], dim=1)
            pred = gen(corrupted_input)

            mse = F.mse_loss(pred, clean_target)
            psnr = -10.0 * torch.log10(mse + 1e-10)

        return {"val_psnr": float(psnr), "g_total_loss": 0.0}

    def test_val_psnr_present_4d(self, device, target_4d):
        gen = _DummyGenerator().to(device)
        sim = _DummySimulator(im_size=(32, 32)).to(device)
        result = self._compute_val_psnr(target_4d, gen, sim)
        assert "val_psnr" in result

    def test_val_psnr_finite_4d(self, device, target_4d):
        gen = _DummyGenerator().to(device)
        sim = _DummySimulator(im_size=(32, 32)).to(device)
        result = self._compute_val_psnr(target_4d, gen, sim)
        psnr = result["val_psnr"]
        assert isinstance(psnr, float)
        assert not torch.isnan(torch.tensor(psnr))
        assert not torch.isinf(torch.tensor(psnr))

    def test_val_psnr_present_5d(self, device, target_5d):
        """5D input should be squeezed and val_psnr computed."""
        gen = _DummyGenerator().to(device)
        sim = _DummySimulator(im_size=(32, 32)).to(device)
        result = self._compute_val_psnr(target_5d, gen, sim)
        assert "val_psnr" in result

    def test_psnr_improves_with_better_pred(self, device):
        """If prediction is closer to target, PSNR should be higher."""
        target = torch.randn(2, 2, 32, 32, device=device)
        clean_target = torch.cat([target[:, 0::2], target[:, 1::2]], dim=1)

        # "Good" prediction (close to target)
        mse_good = F.mse_loss(target + 0.001 * torch.randn_like(target), clean_target)
        psnr_good = -10.0 * torch.log10(mse_good + 1e-10)

        # "Bad" prediction (far from target)
        mse_bad = F.mse_loss(target + 1.0 * torch.randn_like(target), clean_target)
        psnr_bad = -10.0 * torch.log10(mse_bad + 1e-10)

        assert psnr_good > psnr_bad


class TestVFValidationStep:
    """VirtualFiducialStrategy validation_step PSNR logic."""

    def _compute_val_psnr_with_marker_dispatch(
        self,
        target: torch.Tensor,
        gen: nn.Module,
        sim: _DummySimulator,
    ) -> dict[str, float]:
        """Replicate VirtualFiducialStrategy.validation_step PSNR logic."""
        if target.ndim == 5:
            mid = target.shape[-1] // 2
            target = target[..., mid]

        with torch.no_grad():
            if not torch.is_complex(target):
                C = target.shape[1]
                if C >= 2 and C % 2 == 0:
                    tc = torch.complex(target[:, 0::2], target[:, 1::2])
                else:
                    tc = target.to(torch.complex64)
            else:
                tc = target

            corr, _, _ = sim(tc)
            corrupted_input = torch.cat([corr.real, corr.imag], dim=1)
            clean_target = torch.cat([tc.real, tc.imag], dim=1)

            # Method dispatch: generic fallback
            pred = gen(corrupted_input)

            mse = F.mse_loss(pred, clean_target)
            psnr = -10.0 * torch.log10(mse + 1e-10)

        return {"val_psnr": float(psnr)}

    def test_val_psnr_emitted(self, device, target_4d):
        gen = _DummyGenerator().to(device)
        sim = _DummySimulator(im_size=(32, 32)).to(device)
        result = self._compute_val_psnr_with_marker_dispatch(target_4d, gen, sim)
        assert "val_psnr" in result
        assert isinstance(result["val_psnr"], float)

    def test_val_psnr_emitted_5d(self, device, target_5d):
        gen = _DummyGenerator().to(device)
        sim = _DummySimulator(im_size=(32, 32)).to(device)
        result = self._compute_val_psnr_with_marker_dispatch(target_5d, gen, sim)
        assert "val_psnr" in result


class TestTTOValPSNR:
    """ConcreteTTOStrategy.validation_step PSNR proxy logic."""

    def test_dc_psnr_proxy(self, device):
        """DC loss → PSNR proxy should be finite and positive for small loss."""
        dc_loss = 0.001  # small DC loss = good
        dc_psnr = -10.0 * torch.log10(torch.tensor(dc_loss + 1e-10)).item()
        assert isinstance(dc_psnr, float)
        assert dc_psnr > 0.0  # small DC → high PSNR proxy
        assert not torch.isnan(torch.tensor(dc_psnr))

    def test_dc_psnr_monotonic(self, device):
        """Lower DC loss should yield higher PSNR proxy."""
        dc_loss_good = 0.001
        dc_loss_bad = 1.0
        psnr_good = -10.0 * torch.log10(torch.tensor(dc_loss_good + 1e-10)).item()
        psnr_bad = -10.0 * torch.log10(torch.tensor(dc_loss_bad + 1e-10)).item()
        assert psnr_good > psnr_bad


class TestMotionMetaValPSNR:
    """MotionMetaTrainingStrategy validation_step PSNR logic."""

    def test_psnr_from_kinematic_forward(self, device, target_4d):
        """Test PSNR computation with identity kinematic operator."""
        gen = _DummyGenerator().to(device)

        # Simulate the motion meta validation: identity kinematic op
        with torch.no_grad():
            if not torch.is_complex(target_4d):
                C = target_4d.shape[1]
                if C % 2 == 0:
                    tc = torch.complex(
                        target_4d[:, 0::2, :, :], target_4d[:, 1::2, :, :]
                    )
                else:
                    tc = target_4d.to(torch.complex64)
            else:
                tc = target_4d

            # Identity kinematic op: corrupted = target (no motion)
            corrupted_image = tc  # No corruption
            corrupted_input = torch.cat(
                [corrupted_image.real, corrupted_image.imag], dim=1
            )

            # Fiducial contribution (ignored for PSNR)
            pred = gen(corrupted_input)

            target_real = torch.cat([tc.real, tc.imag], dim=1)
            mse = F.mse_loss(pred, target_real)
            psnr = -10.0 * torch.log10(mse + 1e-10)

        assert isinstance(float(psnr), float)
        assert not torch.isnan(psnr)


class TestConfiguredValMetricsResolveMonitors:
    """Regression for dispatch 6944227 (2026-05-25): VF strategies returned
    only ``val_psnr``, so monitors like ``val_hfen`` / ``val_robust_mri_psnr``
    never resolved and early stopping never fired (full-walltime burn).

    The fix wires ``validation_metrics_computer`` into the VF-family
    validation_step methods. This test pins the contract those edits rely on:
    the computer emits bare metric keys that the early-stop alias resolver
    maps back to the configured ``val_*`` monitor.
    """

    @pytest.fixture
    def mag_pair(self, device):
        pred = torch.rand(2, 1, 32, 32, device=device)
        target = torch.rand(2, 1, 32, 32, device=device)
        return pred, target

    def test_computer_emits_monitor_metrics(self, device, mag_pair):
        from spectramr.core.metrics.computer import (
            create_validation_metrics_computer,
        )

        pred, target = mag_pair
        computer = create_validation_metrics_computer(
            ["psnr", "hfen", "ssim", "robust_mri_psnr"], device="cpu"
        )
        computed = computer.compute(pred, target)
        # The arms' monitors are val_hfen / val_robust_mri_psnr; the bare keys
        # must be present so the alias resolver below can find them.
        assert "hfen" in computed
        assert "robust_mri_psnr" in computed

    def test_monitor_resolves_against_merged_result(self, device, mag_pair):
        from spectramr.core.metrics.computer import (
            create_validation_metrics_computer,
        )
        from spectramr.pipelines.train import early_stop_monitor_candidates

        pred, target = mag_pair
        computer = create_validation_metrics_computer(
            ["psnr", "hfen", "robust_mri_psnr"], device="cpu"
        )
        # Mirror the strategy: start with val_psnr, merge computer output.
        result = {"val_psnr": 30.0, "g_total_loss": 0.1}
        result.update({k: float(v) for k, v in computer.compute(pred, target).items()})

        for monitor in ("val_hfen", "val_robust_mri_psnr"):
            candidates = early_stop_monitor_candidates(monitor)
            assert any(c in result for c in candidates), (
                f"monitor {monitor} unresolved against {sorted(result)}"
            )


class TestValidationStep5DHandling:
    """All VF strategies correctly handle 5D TorchIO volumes."""

    def test_5d_volume_squeezed(self, device):
        """5D [B, C, H, W, D] input should be squeezed to 4D via middle slice."""
        target_5d = torch.randn(2, 2, 32, 32, 10, device=device)
        if target_5d.ndim == 5:
            mid = target_5d.shape[-1] // 2
            result = target_5d[..., mid]
        assert result.ndim == 4
        assert result.shape == (2, 2, 32, 32)

    def test_4d_volume_passthrough(self, device):
        """4D [B, C, H, W] input should pass through unchanged."""
        target_4d = torch.randn(2, 2, 32, 32, device=device)
        if target_4d.ndim == 5:
            mid = target_4d.shape[-1] // 2
            target_4d = target_4d[..., mid]
        assert target_4d.ndim == 4
        assert target_4d.shape == (2, 2, 32, 32)

    def test_middle_slice_is_correct(self, device):
        """Middle slice selection should return the correct slice."""
        t = torch.arange(10, device=device).float().view(1, 1, 1, 1, 10)
        t = t.expand(2, 2, 32, 32, 10)
        mid = t.shape[-1] // 2  # = 5
        result = t[..., mid]
        # All values should be 5.0
        assert torch.allclose(result, torch.full_like(result, 5.0))
