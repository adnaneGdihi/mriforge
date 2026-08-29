import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from mriforge.infrastructure.inference.diffusion_inference_strategy import (
    DiffusionInferenceStrategy,
)


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)

    def forward(self, x, t=None, **kwargs):
        return x


class TestDiffusionInferenceStrategy(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = MockModel()
        self.config = {
            "diffusion": {
                "num_inference_steps": 10,
                "beta_schedule": "linear",
                "beta_start": 0.0001,
                "beta_end": 0.02,
            }
        }

    def test_initialization(self):
        strategy = DiffusionInferenceStrategy(self.model, self.device, self.config)
        self.assertTrue(hasattr(strategy, "betas"))
        self.assertEqual(len(strategy.betas), 10)
        self.assertTrue(hasattr(strategy, "alphas_cumprod"))

    def test_noise_schedule_linear(self):
        strategy = DiffusionInferenceStrategy(self.model, self.device, self.config)
        # Check start and end betas
        self.assertAlmostEqual(strategy.betas[0].item(), 0.0001, places=6)
        self.assertAlmostEqual(strategy.betas[-1].item(), 0.02, places=6)

    def test_run_inference_shape(self):
        strategy = DiffusionInferenceStrategy(self.model, self.device, self.config)
        # Improve mock model to handle the input shape expected by inference
        # Inference creates random noise of shape (batch, channels, h, w)
        # Default in_channels=1, image_size=256
        # We need to mock performance_optimizer.run_optimized_inference to avoid actual model call issues if shapes mismatch

        strategy.performance_optimizer = MagicMock()
        strategy.performance_optimizer.run_optimized_inference.return_value = (
            torch.zeros(1, 1, 32, 32)
        )

        # Override config for smaller image to speed up
        strategy.diffusion_config["image_size"] = 32

        # Prepare input noise for unconditional inference
        input_noise = torch.randn(1, 1, 32, 32)
        output = strategy.run_inference(
            input_tensor=input_noise, batch_size=1, unconditional=True
        )
        self.assertIsInstance(output, torch.Tensor)
        # Output is (1, 1, 32, 32)
        self.assertEqual(output.shape, (1, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# _resolve_diffusion_config + eta wiring (2026-06-12 audit, pitfalls #15/#16)
# ---------------------------------------------------------------------------

import inspect  # noqa: E402

from mriforge.infrastructure.inference import diffusion_inference_strategy as _dis  # noqa: E402
from mriforge.infrastructure.inference.diffusion_inference_strategy import (  # noqa: E402
    _resolve_diffusion_config,
)


class TestResolveDiffusionConfig(unittest.TestCase):
    """The strategy must fall back to the training.diffusion SSOT, mapped."""

    def test_top_level_diffusion_preferred(self):
        cfg = {"diffusion": {"num_inference_steps": 50, "beta_schedule": "cosine"}}
        out = _resolve_diffusion_config(cfg)
        self.assertEqual(out["num_inference_steps"], 50)
        self.assertEqual(out["beta_schedule"], "cosine")

    def test_falls_back_to_training_diffusion_with_key_mapping(self):
        cfg = {
            "training": {
                "diffusion": {"sampling_steps": 28, "noise_schedule": "cosine"}
            }
        }
        out = _resolve_diffusion_config(cfg)
        # sampling_steps → num_inference_steps; noise_schedule → beta_schedule
        self.assertEqual(out["num_inference_steps"], 28)
        self.assertEqual(out["beta_schedule"], "cosine")

    def test_timesteps_used_when_no_sampling_steps(self):
        cfg = {"training": {"diffusion": {"timesteps": 16}}}
        out = _resolve_diffusion_config(cfg)
        self.assertEqual(out["num_inference_steps"], 16)

    def test_empty_when_neither_present(self):
        self.assertEqual(_resolve_diffusion_config({}), {})
        self.assertEqual(_resolve_diffusion_config({"training": {}}), {})

    def test_noise_schedule_enum_is_coerced_to_str(self):
        class _Enum:
            value = "cosine"

        cfg = {"training": {"diffusion": {"noise_schedule": _Enum()}}}
        out = _resolve_diffusion_config(cfg)
        self.assertEqual(out["beta_schedule"], "cosine")


class TestEtaWiredIntoReverseStep(unittest.TestCase):
    """eta must scale the stochastic term (was validated-but-unused)."""

    def test_reverse_step_multiplies_noise_by_eta(self):
        src = inspect.getsource(_dis.DiffusionInferenceStrategy)
        # The stochastic term is gated by self.eta now, not a bare sqrt(beta).
        self.assertIn("self.eta * torch.sqrt(beta_t) * noise", src)
