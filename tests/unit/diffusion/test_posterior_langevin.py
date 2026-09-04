"""Unit tests for PosteriorLangevinEnsemble."""


import pytest
import torch
import torch.nn as nn


class DummyScoreModel(nn.Module):
    """Minimal score model for testing: returns -x (denoising score)."""

    def forward(self, x_t: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -x_t


class TestPosteriorLangevinEnsemble:
    @pytest.fixture
    def ensemble(self):
        from spectramr.models.diffusion.posterior_langevin import PosteriorLangevinEnsemble
        score = DummyScoreModel()
        return PosteriorLangevinEnsemble(
            score_model=score, num_samples=4, num_steps=5, sde_type="vp",
        )

    def test_output_shapes(self, ensemble):
        y = torch.randn(2, 1, 8, 8)
        result = ensemble(y)
        assert result.mean.shape == (2, 1, 8, 8)
        assert result.variance.shape == (2, 1, 8, 8)
        assert result.samples.shape == (4, 2, 1, 8, 8)

    def test_variance_nonnegative(self, ensemble):
        y = torch.randn(1, 1, 8, 8)
        result = ensemble(y)
        assert (result.variance >= 0).all()

    def test_samples_differ(self, ensemble):
        y = torch.randn(1, 1, 8, 8)
        result = ensemble(y)
        # Each sample should be different (stochastic process)
        diffs = (result.samples[0] - result.samples[1]).abs().sum()
        assert diffs > 1e-4, "Independent samples should differ"

    def test_mean_is_average(self, ensemble):
        y = torch.randn(1, 1, 8, 8)
        result = ensemble(y)
        manual_mean = result.samples.mean(dim=0)
        assert torch.allclose(result.mean, manual_mean, atol=1e-6)

    def test_override_num_samples(self, ensemble):
        y = torch.randn(1, 1, 8, 8)
        result = ensemble(y, num_samples=2)
        assert result.samples.shape[0] == 2

    def test_ve_schedule(self):
        from spectramr.models.diffusion.posterior_langevin import PosteriorLangevinEnsemble
        ensemble = PosteriorLangevinEnsemble(
            score_model=DummyScoreModel(), num_samples=2, num_steps=3, sde_type="ve",
        )
        y = torch.randn(1, 1, 8, 8)
        result = ensemble(y)
        assert result.mean.shape == (1, 1, 8, 8)

    def test_batched_mode(self):
        from spectramr.models.diffusion.posterior_langevin import PosteriorLangevinEnsemble
        ensemble = PosteriorLangevinEnsemble(
            score_model=DummyScoreModel(), num_samples=3, num_steps=3,
            sde_type="vp", batched=True,
        )
        y = torch.randn(2, 1, 8, 8)
        result = ensemble(y)
        assert result.samples.shape == (3, 2, 1, 8, 8)

    def test_with_dc_projector(self):
        from spectramr.models.diffusion.posterior_langevin import PosteriorLangevinEnsemble

        class IdentityDC(nn.Module):
            def forward(self, x, kspace, mask):
                return x

        ensemble = PosteriorLangevinEnsemble(
            score_model=DummyScoreModel(), num_samples=2, num_steps=3,
            dc_projector=IdentityDC(),
        )
        y = torch.randn(1, 1, 8, 8)
        mk = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
        mask = torch.ones(1, 1, 8, 8)
        result = ensemble(y, measured_kspace=mk, mask=mask)
        assert result.mean.shape == (1, 1, 8, 8)
