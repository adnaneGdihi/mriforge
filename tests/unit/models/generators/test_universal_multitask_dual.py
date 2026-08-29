"""Tests for the paired-contrast fusion variant of UniversalMultitask.

Pins the dual-contrast contract: ``in_channels == out_channels``, both
even, and forward shape-checks the input. This is what makes
experiment_130 honest about its "pair accelerated T1 + T2 → fill both
k-spaces" intent — any single-contrast tensor leaking in raises before
the convolutions run.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.generators.universal_multitask import (
    UniversalMultitaskDualContrastGenerator,
)


def _small_kwargs() -> dict:
    return {
        "backbone_dim": 16,
        "num_layers": 2,
        "shared_blocks": 2,
        "num_task_heads": 2,
    }


class TestUniversalMultitaskDualContract:
    def test_forward_4ch_paired(self) -> None:
        m = UniversalMultitaskDualContrastGenerator(
            in_channels=4, out_channels=4, **_small_kwargs()
        )
        y = m(torch.randn(2, 4, 32, 32))
        assert y.shape == (2, 4, 32, 32)

    def test_forward_8ch_paired_4coil(self) -> None:
        """Real M4Raw single-contrast layout per side: 4 virtual coils ×
        2 R/I = 8 channels per contrast → 16 paired."""
        m = UniversalMultitaskDualContrastGenerator(
            in_channels=16, out_channels=16, **_small_kwargs()
        )
        y = m(torch.randn(1, 16, 32, 32))
        assert y.shape == (1, 16, 32, 32)

    def test_odd_in_channels_rejected(self) -> None:
        with pytest.raises(ValueError, match="even in_channels"):
            UniversalMultitaskDualContrastGenerator(
                in_channels=3, out_channels=4, **_small_kwargs()
            )

    def test_odd_out_channels_rejected(self) -> None:
        with pytest.raises(ValueError, match="even out_channels"):
            UniversalMultitaskDualContrastGenerator(
                in_channels=4, out_channels=3, **_small_kwargs()
            )

    def test_mismatched_in_out_rejected(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            UniversalMultitaskDualContrastGenerator(
                in_channels=4, out_channels=2, **_small_kwargs()
            )

    def test_wrong_input_shape_rejected_at_forward(self) -> None:
        m = UniversalMultitaskDualContrastGenerator(
            in_channels=4, out_channels=4, **_small_kwargs()
        )
        with pytest.raises(ValueError, match="paired contrasts"):
            m(torch.randn(1, 2, 32, 32))  # single-contrast leakage
        with pytest.raises(ValueError, match="paired contrasts"):
            m(torch.randn(1, 4, 32))  # 3-D, missing W

    def test_name_property(self) -> None:
        m = UniversalMultitaskDualContrastGenerator(
            in_channels=4, out_channels=4, **_small_kwargs()
        )
        assert m.name == "universal_multitask_dual"

    def test_registry_metadata(self) -> None:
        """The registry decoration must declare ``requires_paired_data``
        so the data pipeline / audit can reason about the channel layout
        without inspecting the class."""
        from mriforge.models.registry import MODEL_REGISTRY

        entry = MODEL_REGISTRY.get("universal_multitask_dual")
        assert entry is not None, "model not registered"
        # Capability flags live on the ``ModelCapabilities`` dataclass at
        # ``entry["capabilities"]``.
        caps = entry["capabilities"]
        assert caps.requires_paired_data is True
