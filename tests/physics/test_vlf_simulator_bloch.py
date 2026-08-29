import pytest
import torch

from mriforge.data.synthesis.vlf_simulator import VLFSimulator


@pytest.fixture
def simulator():
    return VLFSimulator()


def test_contrast_remapping(simulator):
    """Test standard contrast remapping from 3T to 64mT."""
    # Create a synthetic phantom with GM/WM values (approx normalized T1w)
    # Background=0, CSF=0.2, GM=0.5, WM=0.9

    B, C, H, W = 1, 1, 64, 64
    phantom = torch.zeros(B, C, H, W)

    # Draw simple regions
    # CSF center
    phantom[:, :, 28:36, 28:36] = 0.2
    # GM Ring
    phantom[:, :, 20:28, 20:44] = 0.5
    phantom[:, :, 36:44, 20:44] = 0.5
    phantom[:, :, 28:36, 20:28] = 0.5
    phantom[:, :, 28:36, 36:44] = 0.5
    # WM rest (approx)
    phantom[:, :, 10:20, 10:54] = 0.9

    # Run remapping
    # Standard HF image -> LF image
    # Note: heuristic segmentation uses thresholds: CSF [0.1, 0.3), GM [0.3, 0.65), WM >= 0.65
    # Our values 0.2, 0.5, 0.9 match these perfectly.

    vlf_contrast = simulator.simulate_contrast_remapping(phantom)

    # Check shape
    assert vlf_contrast.shape == phantom.shape

    # Check that values changed
    assert not torch.allclose(vlf_contrast, phantom)

    # Check that Background (0) remains 0
    assert torch.all(vlf_contrast[phantom == 0] == 0)


def test_heuristic_segmentation(simulator):
    """Test heuristic segmentation logic."""
    image = torch.tensor([[[[0.05, 0.2, 0.5, 0.9]]]])  # BG, CSF, GM, WM

    seg = simulator.estimate_segmentation_heuristic(image)

    expected = torch.tensor([[[[0, 1, 2, 3]]]])
    assert torch.equal(seg, expected)


def test_full_pipeline(simulator):
    """Test full __call__ pipeline."""
    image = torch.rand(1, 1, 64, 64)
    out = simulator(image, enable_contrast_remapping=True)

    assert out.shape == image.shape
    # Check it's not identical (noise, warp, contrast)
    assert not torch.allclose(out, image)
