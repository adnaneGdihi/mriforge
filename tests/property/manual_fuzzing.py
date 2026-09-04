import logging
import random
import sys

import torch
import torch.nn as nn

from spectramr.models.losses.gan_loss_library import R1RegularizationLoss, StandardGANLoss

# Setup Basic Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Fuzzing")


class MockDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)  # Dummy

    def forward(self, x):
        # Return something dependent on input norm to simulate scoring
        # Output shape (B, 1) if input is (B, C, H, W)
        return x.mean(dim=[1, 2, 3], keepdim=True)


def check_invariants(name, result, min_val=None, max_val=None):
    """Assert standard invariants for a loss check."""
    if not torch.isfinite(result).all():
        raise AssertionError(f"[{name}] FAILED: Result contains NaN/Inf: {result}")

    if min_val is not None:
        if (result < min_val).any():
            raise AssertionError(f"[{name}] FAILED: Result < {min_val}: {result.min()}")

    if max_val is not None:
        if (result > max_val).any():
            raise AssertionError(f"[{name}] FAILED: Result > {max_val}: {result.max()}")


def fuzz_standard_gan_loss(iterations=100):
    logger.info(f"Fuzzing StandardGANLoss for {iterations} iterations...")
    criterion = StandardGANLoss()

    for i in range(iterations):
        # Generate random shapes
        B = random.randint(1, 16)

        # Random Logits
        real_logits = torch.randn(B, 1) * random.uniform(0.1, 10.0)
        fake_logits = torch.randn(B, 1) * random.uniform(0.1, 10.0)

        # Test D Loss
        d_loss_real, d_loss_fake = criterion.compute_discriminator_loss(
            real_logits, fake_logits
        )

        check_invariants(f"StandardGANLoss_D_Real_Iter{i}", d_loss_real, min_val=0.0)
        check_invariants(f"StandardGANLoss_D_Fake_Iter{i}", d_loss_fake, min_val=0.0)

        # Test G Loss
        g_loss = criterion.compute_generator_loss(fake_logits)
        check_invariants(f"StandardGANLoss_G_Iter{i}", g_loss, min_val=0.0)

    logger.info("StandardGANLoss: PASSED")


def fuzz_r1_loss(iterations=100):
    logger.info(f"Fuzzing R1RegularizationLoss for {iterations} iterations...")
    loss_fn = R1RegularizationLoss(weight=10.0)
    discriminator = MockDiscriminator()

    for i in range(iterations):
        B = random.randint(1, 8)
        C = 3
        H = random.randint(16, 64)
        W = random.randint(16, 64)

        img = torch.randn(B, C, H, W, requires_grad=True)

        loss = loss_fn(discriminator, img)
        check_invariants(f"R1_Iter{i}", loss, min_val=0.0)

    logger.info("R1RegularizationLoss: PASSED")


if __name__ == "__main__":
    try:
        fuzz_standard_gan_loss()
        fuzz_r1_loss()
        logger.info("ALL FUZZING TESTS PASSED")
        sys.exit(0)
    except Exception as e:
        logger.error(f"FUZZING FAILED: {e}")
        sys.exit(1)
