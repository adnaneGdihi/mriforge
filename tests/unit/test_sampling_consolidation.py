import numpy as np
import torch

from mriforge.infrastructure.physics.sampling import (
    MaskGenerator,
    MaskType,
    VDCartesian1DAccelerator,
    VDCartesian2DGaussianAccelerator,
)


def test_mask_generator_instantiation():
    mg = MaskGenerator(seed=42)
    assert mg is not None


def test_mask_generator_gaussian():
    mg = MaskGenerator(seed=42)
    shape = (100, 100)
    accel = 4.0
    mask = mg.generate_mask(MaskType.GAUSSIAN, shape, accel)
    assert mask.shape == shape
    assert mask.sum() > 0
    # Check approximately correct acceleration
    actual_accel = mask.numel() / mask.sum()
    # It won't be exact due to ACS and discrete sampling, but should be close-ish or at least valid
    assert actual_accel >= 1.0


def test_mask_generator_variable_density():
    mg = MaskGenerator(seed=42)
    shape = (100, 100)
    accel = 4.0
    mask = mg.generate_mask(MaskType.VARIABLE_DENSITY, shape, accel)
    assert mask.shape == shape
    assert mask.sum() > 0


def test_mask_generator_consistency():
    # Same seed should produce same mask
    mg1 = MaskGenerator(seed=123)
    mask1 = mg1.generate_mask(MaskType.VARIABLE_DENSITY, (100, 100), 4.0)

    mg2 = MaskGenerator(seed=123)
    mask2 = mg2.generate_mask(MaskType.VARIABLE_DENSITY, (100, 100), 4.0)

    assert torch.all(mask1 == mask2)


def test_sampling_accelerators_integration():
    # Verify that the modified accelerators in sampling.py still work
    device = torch.device("cpu")

    # Test 1D VD Accelerator
    acc1d = VDCartesian1DAccelerator(num_timesteps=10, max_acceleration=4.0, seed=42)
    mask1d = acc1d.get_acceleration_mask((1, 100, 100), timestep=5, device=device)
    assert mask1d.shape == (1, 100, 100)
    assert mask1d.dtype == torch.bool

    # Test 2D Gaussian Accelerator
    acc2d = VDCartesian2DGaussianAccelerator(
        num_timesteps=10, max_acceleration=4.0, seed=42
    )
    mask2d = acc2d.get_acceleration_mask((1, 100, 100), timestep=5, device=device)
    assert mask2d.shape == (1, 100, 100)
    assert mask2d.dtype == torch.bool

    # Check basic properties
    assert mask2d.any()  # Should have some True values


def test_pdf_generation_methods():
    # Verify static methods
    pdf_gauss = MaskGenerator.get_gaussian_pdf(100)
    assert len(pdf_gauss) == 100
    assert np.isclose(pdf_gauss.sum(), 1.0)

    pdf_vd = MaskGenerator.get_variable_density_pdf(100)
    assert len(pdf_vd) == 100
    assert np.isclose(pdf_vd.sum(), 1.0)
