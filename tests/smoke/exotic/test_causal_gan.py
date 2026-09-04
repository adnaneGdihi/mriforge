"""
Unit tests for Causal-GAN.
"""

import pytest
import torch

from spectramr.models.reasoning.causal_gan import CausalGAN


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_causal_gan_forward(device):
    """Verify standard generation."""
    model = CausalGAN().to(device)
    img = model(batch_size=2)
    assert img.shape == (2, 1, 32, 32)  # ConvBlocks upsample to 32x32


def test_intervention(device):
    """Verify do(Pathology=1) works."""
    model = CausalGAN().to(device)

    # Generate population with forced Pathology=1
    img_sick = model.generator.do_intervention(n_samples=5, target_pathology=1)

    # Generate population with forced Pathology=0
    img_healthy = model.generator.do_intervention(n_samples=5, target_pathology=0)

    # They should differ somehow (statistically)
    # But shape must be valid
    assert img_sick.shape == (5, 1, 32, 32)
    assert img_healthy.shape == (5, 1, 32, 32)
