import math

import torch
import torch.nn.functional as F

from mriforge.data.transforms.benchmarking_degradations import (
    B1TransmitInhomogeneity,
    KSpaceComplexGaussianNoise,
    KSpaceGhosting,
    KSpaceSpikes,
)


def test_kspace_complex_gaussian_noise():
    """Verify that lower SNR strictly increases error against pristine image."""
    img = torch.ones(1, 1, 32, 32)

    transform_high_snr = KSpaceComplexGaussianNoise(target_snr_db=40.0, seed=42)
    transform_low_snr = KSpaceComplexGaussianNoise(target_snr_db=10.0, seed=42)

    noisy_high = transform_high_snr(img)
    noisy_low = transform_low_snr(img)

    error_high = F.mse_loss(noisy_high, img)
    error_low = F.mse_loss(noisy_low, img)

    # Check shape preserving
    assert noisy_high.shape == img.shape
    # Error should be strictly higher for lower SNR
    assert error_low > error_high
    assert not torch.is_complex(noisy_high)  # Input was real


def test_b1_transmit_inhomogeneity():
    """Verify central brightening and edge darkening."""
    # Flat image at 0.5
    img = torch.ones(1, 1, 32, 32) * 0.5

    transform = B1TransmitInhomogeneity(bias_strength=0.2, gamma=1.0)
    biased_img = transform(img)

    # Center should be brighter than edges
    center_val = biased_img[:, :, 16, 16].item()
    corner_val = biased_img[:, :, 0, 0].item()

    assert center_val > img[0, 0, 16, 16].item()  # Brightened
    assert corner_val < img[0, 0, 0, 0].item()  # Darkened
    assert center_val > corner_val


def test_kspace_ghosting():
    """Verify N/2 ghosting energy shift."""
    # Create single centered dot
    img = torch.zeros(1, 1, 32, 32)
    img[:, :, 16, 16] = 1.0

    # Math.pi phase shift gives maximum ghosting (image splits in half strictly)
    transform = KSpaceGhosting(phase_shift_diff=math.pi)
    ghosted = transform(img)

    # Original center should be significantly attenuated or zeroed out
    center_val_ghosted = ghosted[:, :, 16, 16].item()
    # Ghosting shifts by N/2 in phase encode (dim 2, which is height 32, so +/- 16)
    # 16 + 16 = 32 -> wraps to 0. It should perfectly ghost the dot to 0 (top) and 31 (bottom) or similar depending on centering.
    assert center_val_ghosted < 0.1  # Most energy moved away
    # Check total energy roughly conserved
    assert torch.allclose(torch.sum(img**2), torch.sum(ghosted**2), atol=1e-5)


def test_kspace_spikes():
    """Verify spikes cause striping patterns."""
    img = torch.ones(1, 1, 32, 32) * 0.5
    # A perfectly flat image has all energy at DC (16, 16)
    # Spiking it somewhere else should cause a periodic sine/cosine wave
    transform = KSpaceSpikes(num_spikes=1, spike_intensity_ratio=1.0, seed=42)
    spiked = transform(img)

    assert spiked.shape == img.shape
    assert not torch.allclose(spiked, img)

    # Check variance is high because of the injected spatial sine wave striping
    assert torch.var(spiked) > 0.0
