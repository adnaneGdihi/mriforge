"""Lock test for the end-to-end coil-processing exercise config.

`experiments/inprogress/coil_processing/exp_coil_svd_espirit_sense.yaml` exercises
the full wired chain — SVD compression (with an ACS calibration band) → ESPIRiT
estimation → SENSE combine. This test asserts it loads, resolves the chain onto
the legacy knobs, and passes the audit-time coil-processing consistency check.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mriforge.config.settings import TrainingSettings
from mriforge.infrastructure.validation.config_health_checker import (
    ConfigHealthChecker,
)

_CFG = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "inprogress"
    / "coil_processing"
    / "exp_coil_svd_espirit_sense.yaml"
)


@pytest.fixture(scope="module")
def settings() -> TrainingSettings:
    if not _CFG.exists():
        pytest.skip(f"e2e coil config not found at {_CFG}")
    return TrainingSettings.from_yaml(str(_CFG))


def test_full_chain_resolves(settings: TrainingSettings) -> None:
    cp = settings.physics.coil_processing
    assert cp.compression.method == "svd"
    assert cp.compression.num_virtual_coils == 4
    assert cp.compression.calibration_lines == 24
    assert cp.estimation.method == "espirit"
    assert cp.estimation.enabled is True
    assert cp.combine.method == "sense"


def test_compression_synced_onto_legacy_knobs(settings: TrainingSettings) -> None:
    # The loader sync drives the legacy SVD machinery from the new block.
    assert settings.data.coils.processing_mode == "svd"
    assert settings.data.coils.num_virtual_coils == 4
    assert settings.data.coils.svd_calibration_lines == 24


def test_passes_coil_processing_audit_check(settings: TrainingSettings) -> None:
    res = ConfigHealthChecker().check_coil_processing_consistency(settings)
    assert res.passed, res.message
