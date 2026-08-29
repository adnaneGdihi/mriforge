"""Tests for the config-driven coil pipeline (compress → estimate → combine).

Exercises ``apply_coil_processing`` (the SSOT free function) and the
``DataPipelineDirector.coil_process`` seam against the real physics primitives.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.config.schemas.physics import CoilProcessingConfig
from mriforge.infrastructure.builders.directors.data_pipeline_director import (
    apply_coil_processing,
)


def _ks(c: int = 4) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.complex(torch.randn(1, c, 32, 32), torch.randn(1, c, 32, 32))


def test_estimation_on_attaches_smaps_and_target() -> None:
    cfg = CoilProcessingConfig(
        estimation={"method": "power_iter", "enabled": True},
        combine={"method": "rss"},
    )
    out = apply_coil_processing(_ks(), cfg)
    assert out["smaps"] is not None
    assert out["target"].shape == (1, 1, 32, 32)
    # target is a real magnitude image.
    assert not torch.is_complex(out["target"])


def test_estimation_off_no_smaps() -> None:
    cfg = CoilProcessingConfig(
        estimation={"method": "none", "enabled": False},
        combine={"method": "rss"},
    )
    out = apply_coil_processing(_ks(), cfg)
    assert out["smaps"] is None
    assert out["target"].shape == (1, 1, 32, 32)


def test_sense_without_estimation_raises() -> None:
    cfg = CoilProcessingConfig(
        estimation={"method": "none", "enabled": False},
        combine={"method": "sense"},
    )
    with pytest.raises(ValueError, match="sense.*requires"):
        apply_coil_processing(_ks(), cfg)


def test_sense_with_estimation_combines() -> None:
    cfg = CoilProcessingConfig(
        estimation={"method": "power_iter", "enabled": True},
        combine={"method": "sense"},
    )
    out = apply_coil_processing(_ks(), cfg)
    assert out["smaps"] is not None
    assert out["target"].shape == (1, 1, 32, 32)


def test_svd_compression_reduces_coils() -> None:
    cfg = CoilProcessingConfig(
        compression={"method": "svd", "num_virtual_coils": 2},
        estimation={"method": "none", "enabled": False},
        combine={"method": "rss"},  # explicit combine over the virtual coils
    )
    out = apply_coil_processing(_ks(c=4), cfg)
    assert out["kspace"].shape[1] == 2
    # RSS combine runs over the compressed virtual coils → single image.
    assert out["target"].shape == (1, 1, 32, 32)


def test_gcc_compression_raises_not_implemented() -> None:
    cfg = CoilProcessingConfig(compression={"method": "gcc"})
    with pytest.raises(NotImplementedError, match="gcc"):
        apply_coil_processing(_ks(), cfg)


def test_default_config_is_passthrough_none() -> None:
    # A bare block is a no-op: no compression, combine='none' (default) → coils
    # are kept, so the "target" is the multi-coil images, not a single image.
    cfg = CoilProcessingConfig()
    out = apply_coil_processing(_ks(c=4), cfg)
    assert out["kspace"].shape[1] == 4  # no compression
    assert out["target"].shape == (1, 4, 32, 32)  # combine=none → coils kept


def test_explicit_rss_combine_collapses_to_single_image() -> None:
    cfg = CoilProcessingConfig(combine={"method": "rss"})
    out = apply_coil_processing(_ks(c=4), cfg)
    assert out["target"].shape == (1, 1, 32, 32)


def test_director_coil_process_reads_config() -> None:
    """The director seam pulls physics.coil_processing from the frozen config."""
    from mriforge.config.settings import TrainingSettings
    from mriforge.infrastructure.builders.directors.data_pipeline_director import (
        DataPipelineDirector,
    )

    config = TrainingSettings(
        data={"train_path": "/tmp/t", "val_path": "/tmp/v", "batch_size": 1},
        model={"model_type": "unet"},
        optimization={"learning_rate": 1e-4},
        logging={},
        physics={
            "coil_processing": {
                "compression": {"method": "svd", "num_virtual_coils": 2},
                "estimation": {"method": "none", "enabled": False},
            }
        },
    )
    director = DataPipelineDirector(config)
    out = director.coil_process(_ks(c=4))
    assert out["kspace"].shape[1] == 2
