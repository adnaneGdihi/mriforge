import logging
import os
import sys

import pytest
import torch

# Add src to path if needed (though pytest usually handles it if conftest is set up)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.builders.loss_builder import LossBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_loss_builder_experiment_11():
    """Verify LossBuilder works with Experiment 11 config and builds expected losses."""
    config_path = "experiments/training/experiment_11_kspace_cold_diffusion.yaml"

    if not os.path.exists(config_path):
        pytest.skip(f"Config {config_path} not found")

    print(f"Loading config from {config_path}")
    try:
        config = TrainingSettings.from_yaml(config_path)
    except Exception as e:
        pytest.fail(f"Failed to load config: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing LossBuilder on {device}...")

    builder = LossBuilder(config, device)

    losses = (
        builder.build_reconstruction_losses()
        .build_adversarial_losses()
        .build_physics_losses()
        .build_regularization_losses()
        .build_diffusion_losses()
        .build_latent_losses()
        .build_ssl_losses()
        .build()
    )

    # Assertions
    assert len(losses) > 0, "Builder returned 0 losses"

    # Check specific losses required by Exp 11
    # reconstruction.enable_complex_l1: true
    assert "complex_l1" in losses, "complex_l1 loss missing"

    # reconstruction.enable_log_spectral: true
    assert "log_spectral" in losses, "log_spectral loss missing"

    # reconstruction.enable_energy_conservation: true
    assert "energy_conservation" in losses, "energy_conservation loss missing"

    # reconstruction.enable_frequency_domain: true
    assert "frequency_domain" in losses, "frequency_domain loss missing"

    print(f"Successfully built {len(losses)} losses: {list(losses.keys())}")
