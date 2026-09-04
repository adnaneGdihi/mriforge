from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from spectramr.models.diffusion.cold_diffusion import ColdDiffusion


class MockModule(nn.Module):
    def __init__(self, side_effect=None):
        super().__init__()
        self.mock = MagicMock()
        if side_effect:
            self.mock.side_effect = side_effect
        self.device = torch.device("cpu")

    def forward(self, *args, **kwargs):
        return self.mock(*args, **kwargs)


class MockAccelerator:
    def get_acceleration_mask(self, shape, timestep, device):
        # Return a mask of shape (C, H, W)
        # For testing, return all ones or zeros depending on timestep
        return torch.ones(shape, device=device)


class TestColdDiffusion:
    @pytest.fixture
    def mock_model(self):
        def side_effect(x, t, cond=None):
            # Return x_0 prediction of same shape as x
            return torch.zeros_like(x)

        return MockModule(side_effect=side_effect)

    @pytest.fixture
    def mock_accelerator(self):
        return MockAccelerator()

    def test_init(self):
        cd = ColdDiffusion(timesteps=100, cold_schedule="linear", device="cpu")
        assert len(cd.cold_schedule) == 100

        cd_cosine = ColdDiffusion(timesteps=100, cold_schedule="cosine", device="cpu")
        assert len(cd_cosine.cold_schedule) == 100

        cd_power = ColdDiffusion(timesteps=100, cold_schedule="power_law", device="cpu")
        assert len(cd_power.cold_schedule) == 100

    def test_degrade_noise(self):
        cd = ColdDiffusion(timesteps=100, degradation_type="noise", device="cpu")
        x_0 = torch.randn(2, 2, 32, 32)  # 2 channels (real/imag)
        t = torch.tensor([50, 50])

        x_t = cd._degrade(x_0, t)
        assert x_t.shape == x_0.shape

    def test_degrade_blur(self):
        cd = ColdDiffusion(timesteps=100, degradation_type="blur", device="cpu")
        x_0 = torch.randn(2, 1, 32, 32)
        t = torch.tensor([50, 50])

        x_t = cd._degrade(x_0, t)
        assert x_t.shape == x_0.shape

    def test_degrade_kspace_mask(self, mock_accelerator):
        cd = ColdDiffusion(
            timesteps=100,
            degradation_type="kspace_mask",
            accelerator=mock_accelerator,
            device="cpu",
        )
        x_0 = torch.randn(2, 1, 32, 32)
        t = torch.tensor([50, 50])

        x_t = cd._degrade(x_0, t)
        assert x_t.shape == x_0.shape

    def test_symmetrize_kspace(self):
        # Input (B, 2, H, W)
        z = torch.randn(2, 2, 32, 32)
        sym_z = ColdDiffusion.symmetrize_kspace(z)
        assert sym_z.shape == z.shape
        # Check symmetry property:
        # If we IFFT sym_z, imaginary part should be 0.
        from spectramr.infrastructure.physics.fft_ops import ifft2c

        z_c = torch.complex(sym_z[:, 0], sym_z[:, 1])
        img = ifft2c(z_c)
        assert torch.allclose(img.imag, torch.zeros_like(img.imag), atol=1e-5)

    def test_p_sample(self, mock_model):
        cd = ColdDiffusion(timesteps=100, model=mock_model, device="cpu")
        x = torch.randn(2, 2, 32, 32)  # 2 channels for symmetry
        t = torch.tensor([99, 99])

        # Test basic p_sample
        out = cd.p_sample(mock_model, x, t, t_index=99)
        assert out.shape == x.shape

    def test_p_sample_loop(self, mock_model):
        cd = ColdDiffusion(timesteps=5, model=mock_model, device="cpu")
        shape = (2, 2, 16, 16)  # 2 channels for symmetry check

        out = cd.p_sample_loop(mock_model, shape)
        assert out.shape == shape

    def test_p_losses(self, mock_model):
        cd = ColdDiffusion(timesteps=100, model=mock_model, device="cpu")
        x_start = torch.randn(2, 2, 32, 32)  # 2 channels
        t = torch.tensor([50, 50])

        loss = cd.p_losses(mock_model, x_start, t, loss_type="l1")
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_generate(self, mock_model):
        cd = ColdDiffusion(timesteps=5, model=mock_model, device="cpu")
        input_data = torch.randn(2, 2, 16, 16)

        out = cd.generate(input_data)
        assert out.shape == input_data.shape

    def test_p_losses_unsupported_loss_type_raises_notimplemented(self, mock_model):
        """Unsupported loss_type must raise NotImplementedError, not AttributeError.

        Regression: the final raise interpolated an undefined
        ``self.supported_loss_types`` attribute, so the f-string evaluation
        raised AttributeError *before* NotImplementedError was constructed,
        masking the real cause.
        """
        cd = ColdDiffusion(timesteps=100, model=mock_model, device="cpu")
        x_start = torch.randn(2, 2, 32, 32)  # 2 channels
        t = torch.tensor([50, 50])

        with pytest.raises(NotImplementedError):
            cd.p_losses(mock_model, x_start, t, loss_type="nope")

        # Explicitly assert the bug (AttributeError) is gone.
        try:
            cd.p_losses(mock_model, x_start, t, loss_type="nope")
        except AttributeError as exc:  # pragma: no cover - fails the test path
            pytest.fail(f"p_losses raised AttributeError instead of NotImplementedError: {exc}")
        except NotImplementedError:
            pass


class _HalfInputModel(nn.Module):
    def forward(self, x, t, cond=None):
        return 0.5 * x


class TestPLossesRegistryRouting:
    """WS-6 plan #6: elementary + log_spectral dispatch is registry-routed,
    numerically identical to the replaced inline implementations, and still
    raises ``NotImplementedError`` on unknown names."""

    def _setup(self):
        cd = ColdDiffusion(timesteps=10, device="cpu")
        model = _HalfInputModel()
        torch.manual_seed(0)
        x_start = torch.randn(1, 2, 8, 8)
        # A degradation mask short-circuits q_sample -> deterministic x_noisy.
        mask = (torch.rand_like(x_start) > 0.5).float()
        t = torch.tensor([4])
        predicted = 0.5 * (x_start * mask)
        return cd, model, x_start, mask, t, predicted

    def test_matches_replaced_functionals(self):
        import torch.nn.functional as F

        cd, model, x_start, mask, t, predicted = self._setup()
        expected = {
            "l1": F.l1_loss(predicted, x_start),
            "l2": F.mse_loss(predicted, x_start),
            "huber": F.smooth_l1_loss(predicted, x_start),
        }
        for name, exp in expected.items():
            got = cd.p_losses(model, x_start, t, degradation_mask=mask, loss_type=name)
            assert torch.allclose(got, exp), name

    def test_l1_smooth_formula_preserved(self):
        cd, model, x_start, mask, t, predicted = self._setup()
        diff = torch.abs(x_start - predicted)
        expected = torch.mean(torch.sqrt(diff**2 + 1e-8) - 1e-4)
        got = cd.p_losses(model, x_start, t, degradation_mask=mask, loss_type="l1_smooth")
        assert torch.allclose(got, expected)

    def test_log_spectral_routes_through_registry(self):
        from spectramr.models.losses.registry import create_loss

        cd, model, x_start, mask, t, predicted = self._setup()
        expected = create_loss("log_spectral_phase", phase_weight=0.1)(predicted, x_start)
        got = cd.p_losses(model, x_start, t, degradation_mask=mask, loss_type="log_spectral")
        assert torch.allclose(got, expected)


# --------------------------------------------------------------------------- #
# #1339 -- the progressive mask ran backwards
# --------------------------------------------------------------------------- #
class TestProgressiveMaskDirection:
    """`t=0` must be (near) the identity and `t=T` the degraded end.

    The method's own docstring states this, and the fixed-mask branch directly
    above it already interpolates that way. The random-mask branch used the
    degradation factor as a KEEP fraction, so the two branches of one method
    disagreed about which end of the schedule was degraded.
    """

    @staticmethod
    def _process(num_timesteps: int = 10):
        import torch

        from spectramr.models.diffusion.cold_diffusion import ColdDiffusion

        torch.manual_seed(0)
        proc = ColdDiffusion(
            timesteps=num_timesteps, degradation_type="mask", device="cpu"
        )
        proc.timesteps = num_timesteps
        return proc

    def test_t_zero_keeps_essentially_all_of_the_signal(self):
        import torch

        proc = self._process()
        x = torch.ones(1, 1, 8, 8)
        out = proc.apply_progressive_mask(x, torch.zeros(1, dtype=torch.long))
        assert out.abs().sum() > 0.9 * x.abs().sum()

    def test_last_timestep_keeps_almost_none_of_it(self):
        import torch

        proc = self._process()
        x = torch.ones(1, 1, 8, 8)
        t = torch.full((1,), proc.timesteps - 1, dtype=torch.long)
        out = proc.apply_progressive_mask(x, t)
        assert out.abs().sum() < 0.1 * x.abs().sum()

    def test_kept_signal_decreases_monotonically_with_t(self):
        import torch

        proc = self._process()
        x = torch.ones(1, 1, 16, 16)
        kept = [
            proc.apply_progressive_mask(x, torch.full((1,), t, dtype=torch.long))
            .abs()
            .sum()
            .item()
            for t in range(proc.timesteps)
        ]
        assert kept == sorted(kept, reverse=True), kept

    def test_the_random_branch_agrees_with_the_fixed_mask_branch(self):
        """Both branches of one method must degrade in the same direction."""
        import torch

        proc = self._process()
        x = torch.ones(1, 1, 8, 8)
        fixed = torch.zeros(1, 1, 8, 8)  # fully masked measurement
        t0 = torch.zeros(1, dtype=torch.long)
        t_last = torch.full((1,), proc.timesteps - 1, dtype=torch.long)

        for t, expect_more in ((t0, True), (t_last, False)):
            random_branch = proc.apply_progressive_mask(x, t).abs().sum().item()
            fixed_branch = (
                proc.apply_progressive_mask(x, t, mask=fixed).abs().sum().item()
            )
            half = 0.5 * x.abs().sum().item()
            assert (random_branch > half) is expect_more
            assert (fixed_branch > half) is expect_more
