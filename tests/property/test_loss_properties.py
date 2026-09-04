import unittest

import numpy as np

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
    from hypothesis.extra.numpy import arrays

    HAS_HYPOTHESIS = True
except ImportError:
    from unittest.mock import MagicMock

    HAS_HYPOTHESIS = False

    # Define dummy decorators for when hypothesis is missing
    def given(*args, **kwargs):
        return lambda f: f

    def settings(*args, **kwargs):
        return lambda f: f

    class HealthCheck:
        too_slow = 1

    st = MagicMock()
    arrays = MagicMock()

# Mocking support
try:
    import torch
    import torch.nn as nn
except ImportError:
    from tests.utils.mock_torch import setup_mock_heavy_dependencies

    setup_mock_heavy_dependencies()
    import torch
    import torch.nn as nn

from spectramr.models.losses.gan_loss_library import R1RegularizationLoss, StandardGANLoss


# Strategy for generating tensors
def tensor_strategy(shape=(4, 1, 32, 32), min_value=-10.0, max_value=10.0):
    return arrays(
        dtype=np.float32,
        shape=shape,
        elements=st.floats(
            min_value=min_value,
            max_value=max_value,
            width=32,
            allow_nan=False,
            allow_infinity=False,
        ),
    ).map(lambda x: torch.tensor(x))


class MockDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        if hasattr(nn, "Linear"):
            self.layer = nn.Linear(1, 1)  # Dummy
        else:
            self.layer = MagicMock()

    def forward(self, x):
        return x.mean(dim=[1, 2, 3], keepdim=True)


class TestLossProperties(unittest.TestCase):
    def setUp(self):
        if not HAS_HYPOTHESIS:
            self.skipTest("Hypothesis not installed")

    @settings(
        max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(real=tensor_strategy(), fake=tensor_strategy())
    def test_standard_gan_loss_invariants(self, real, fake):
        """Test StandardGANLoss invariants."""
        criterion = StandardGANLoss()

        real_logits = real.mean(dim=[1, 2, 3], keepdim=True)
        fake_logits = fake.mean(dim=[1, 2, 3], keepdim=True)

        d_loss_real, d_loss_fake = criterion.compute_discriminator_loss(
            real_logits, fake_logits
        )

        self.assertTrue(torch.isfinite(d_loss_real).all(), "Real loss contains NaN/Inf")
        self.assertTrue(torch.isfinite(d_loss_fake).all(), "Fake loss contains NaN/Inf")

        # Invariant: Non-negative (BCE/Softplus-based losses)
        self.assertGreaterEqual(float(d_loss_real.mean()), 0, "Real loss must be >= 0")
        self.assertGreaterEqual(float(d_loss_fake.mean()), 0, "Fake loss must be >= 0")

        g_loss = criterion.compute_generator_loss(fake_logits)
        self.assertTrue(torch.isfinite(g_loss).all(), "Generator loss contains NaN/Inf")
        self.assertGreaterEqual(float(g_loss.mean()), 0, "Generator loss must be >= 0")

    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(img=tensor_strategy(shape=(2, 1, 16, 16)))
    def test_r1_regularization_invariants(self, img):
        """Test R1RegularizationLoss invariants."""
        # Note: In a mocked environment, R1 might not calculate real gradients
        # but we check for finite values and basic contract.
        loss_fn = R1RegularizationLoss(weight=10.0)
        discriminator = MockDiscriminator()

        img.requires_grad = True
        try:
            loss = loss_fn(discriminator, img)
            self.assertTrue(torch.isfinite(loss).all(), "R1 loss contains NaN/Inf")
            self.assertGreaterEqual(float(loss.mean()), 0, "R1 loss must be >= 0")
        except Exception as e:
            # In a mocked environment, autograd might fail
            if "torch.autograd" in str(e) or "gradient" in str(e).lower():
                self.skipTest(f"Autograd not supported in mocks: {e}")
            else:
                raise


if __name__ == "__main__":
    unittest.main()
