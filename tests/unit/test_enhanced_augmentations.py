import pytest
import torch
import torchio as tio

from spectramr.config.schemas.augmentation import AugmentationConfigSchema
from spectramr.data.transforms.augmentation_factory import TorchIOAugmentationFactory


@pytest.fixture
def sample_subject():
    """Create a dummy TorchIO subject."""
    return tio.Subject(mri=tio.ScalarImage(tensor=torch.rand(1, 64, 64, 32)))


def test_ghosting_integration(sample_subject):
    """Test RandomGhosting integration."""
    config = AugmentationConfigSchema(
        enabled=True,
        probability=1.0,  # Force application
        enable_ghosting=True,
        num_ghosts=(4, 10),
        ghosting_intensity=(0.5, 1.0),
    )
    pipeline = TorchIOAugmentationFactory.build(config)
    assert pipeline is not None

    transformed = pipeline(sample_subject)
    assert transformed.mri.data.shape == sample_subject.mri.data.shape
    assert not torch.allclose(transformed.mri.data, sample_subject.mri.data)


def test_spike_integration(sample_subject):
    """Test RandomSpike integration."""
    config = AugmentationConfigSchema(
        enabled=True,
        probability=1.0,
        enable_spike=True,
        num_spikes=3,
        spike_intensity=(0.5, 1.0),
    )
    pipeline = TorchIOAugmentationFactory.build(config)
    assert pipeline is not None

    transformed = pipeline(sample_subject)
    assert transformed.mri.data.shape == sample_subject.mri.data.shape
    # Spikes add noise/stripes
    assert not torch.allclose(transformed.mri.data, sample_subject.mri.data)


def test_anisotropy_integration(sample_subject):
    """Test RandomAnisotropy integration."""
    config = AugmentationConfigSchema(
        enabled=True,
        probability=1.0,
        enable_anisotropy=True,
        anisotropy_downsampling=(2.0, 2.0),
    )
    pipeline = TorchIOAugmentationFactory.build(config)
    assert pipeline is not None

    transformed = pipeline(sample_subject)
    assert transformed.mri.data.shape == sample_subject.mri.data.shape
    # Anisotropy blurs/pixelates in some dimensions
    assert not torch.allclose(transformed.mri.data, sample_subject.mri.data)


def test_multiple_augmentations(sample_subject):
    """Test chaining multiple augmentations."""
    config = AugmentationConfigSchema(
        enabled=True,
        probability=1.0,
        enable_ghosting=True,
        enable_spike=True,
        enable_anisotropy=True,
    )
    pipeline = TorchIOAugmentationFactory.build(config)
    assert isinstance(pipeline, tio.Compose)
    assert len(pipeline.transforms) == 3
