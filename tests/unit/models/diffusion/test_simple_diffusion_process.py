import torch

from spectramr.models.diffusion.simple_diffusion_process import SimpleDiffusionProcess


class TestSimpleDiffusionProcess:
    def test_init(self):
        diff = SimpleDiffusionProcess(timesteps=100)
        assert diff.timesteps == 100
        assert len(diff.betas) == 100

    def test_sample_timesteps(self):
        diff = SimpleDiffusionProcess(timesteps=100)
        t = diff.sample_timesteps(batch_size=5)
        assert t.shape == (5,)
        assert torch.all(t >= 0) and torch.all(t < 100)

    def test_add_noise(self):
        diff = SimpleDiffusionProcess(timesteps=100)
        x = torch.randn(2, 1, 32, 32)
        t = torch.randint(0, 100, (2,))
        noisy_x, noise = diff.add_noise(x, t)
        assert noisy_x.shape == x.shape
        assert noise.shape == x.shape

    def test_predict_noise_shape_mismatch_fixed(self):
        # This test used to fail when x.shape[1] != 32, but now it should pass
        diff = SimpleDiffusionProcess(timesteps=100)
        x = torch.randn(2, 1, 32, 32)  # 1 channel
        t = torch.randint(0, 100, (2,))

        pred_noise = diff.predict_noise(x, t)
        assert pred_noise.shape == x.shape

    def test_predict_noise_32_channels(self):
        diff = SimpleDiffusionProcess(timesteps=100)
        x = torch.randn(2, 32, 32, 32)  # 32 channels
        t = torch.randint(0, 100, (2,))

        pred_noise = diff.predict_noise(x, t)
        assert pred_noise.shape == x.shape
