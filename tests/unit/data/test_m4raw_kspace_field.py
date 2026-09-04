#!/usr/bin/env python
"""Unit test: M4Raw kspace field must point to averaged target, not noisy input."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest
import torch

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.builders.context import BuilderContext
from spectramr.infrastructure.builders.directors.data_pipeline_director import (
    DataPipelineDirector,
)


# Exercises the deprecated ConsolidatedDatasetFactory on purpose (legacy path);
# H3 added a DeprecationWarning there which strict filterwarnings would promote
# to an error. This test asserts data-correctness, not the deprecation, so we
# silence it locally.
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_m4raw_kspace_field_is_averaged_target():
    """Test that M4Raw batch['kspace'] points to the averaged target, not the noisy input.

    Regression test for bug where kspace was assigned to input_t instead of target_t.
    """
    config_path = (
        Path(__file__).parent
        / "experiments/training/experiment_11_kspace_cold_diffusion.yaml"
    )
    if not config_path.exists():
        pytest.skip(f"Config not found at {config_path}")

    settings = TrainingSettings.from_yaml(str(config_path))

    # Create validation dataloader
    train_loader, val_loader = DataPipelineDirector(
        BuilderContext(config=settings)
    ).build_dataloaders()

    # Get first batch
    batch = next(iter(val_loader))

    assert isinstance(batch, dict), "Batch should be a dict"
    assert "input" in batch, "Batch should have 'input' key"
    assert "target" in batch, "Batch should have 'target' key"
    assert "kspace" in batch, "Batch should have 'kspace' key"

    input_t = batch["input"]
    target_t = batch["target"]
    kspace_t = batch["kspace"]

    # Check that kspace matches target (averaged), not input (noisy)
    assert torch.allclose(target_t, kspace_t, rtol=1e-5, atol=1e-5), (
        "BUG: batch['kspace'] should be identical to batch['target'] (both averaged). "
        "Instead, kspace appears to point to the non-averaged input!"
    )

    # Verify that target is actually different from input (averaging happened)
    assert not torch.allclose(
        input_t, target_t, rtol=1e-5, atol=1e-5
    ), "Input and target should be different (averaging should have occurred)"

    # Additional check: target should have lower magnitude than input (noise reduction)
    if torch.is_complex(input_t):
        input_mag = input_t.abs()
        target_mag = target_t.abs()
    else:
        # Handle real/imaginary channel representation
        if input_t.shape[1] % 2 == 0:
            B, C, *spatial = input_t.shape
            input_real = input_t.view(B, C // 2, 2, *spatial)[..., 0, *spatial]
            input_imag = input_t.view(B, C // 2, 2, *spatial)[..., 1, *spatial]
            input_mag = torch.sqrt(input_real**2 + input_imag**2)

            target_real = target_t.view(B, C // 2, 2, *spatial)[..., 0, *spatial]
            target_imag = target_t.view(B, C // 2, 2, *spatial)[..., 1, *spatial]
            target_mag = torch.sqrt(target_real**2 + target_imag**2)
        else:
            input_mag = input_t.abs()
            target_mag = target_t.abs()

    target_std = target_mag.std().item()
    input_std = input_mag.std().item()

    # Target should have reduced noise (smaller std) due to averaging
    assert target_std < input_std, (
        f"Target magnitude std ({target_std:.6f}) should be less than "
        f"input magnitude std ({input_std:.6f}) due to repetition averaging. "
        "Averaging does not appear to be working!"
    )

    # Should be roughly 50-70% reduction in noise (depending on number of repetitions)
    noise_reduction_ratio = target_std / input_std
    assert 0.3 < noise_reduction_ratio < 0.8, (
        f"Target/Input std ratio is {noise_reduction_ratio:.2f}, expected 0.3-0.8 "
        "(50-70% noise reduction from averaging multiple repetitions)"
    )

    print("✓ PASS: batch['kspace'] correctly points to averaged target")
    print(f"  - Input std: {input_std:.6f}")
    print(f"  - Target std: {target_std:.6f}")
    print(f"  - Noise reduction: {(1 - noise_reduction_ratio) * 100:.1f}%")


if __name__ == "__main__":
    test_m4raw_kspace_field_is_averaged_target()
    print("\n✓ All tests passed!")
