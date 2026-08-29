"""Tests for the DiceRiskCertificateConfig mount (B-1.7)."""

from __future__ import annotations

import pytest

from mriforge.config.schemas.certification import (
    CertificationConfigSchema,
    DiceRiskCertificateConfig,
)


def test_dice_risk_mounted_on_certification() -> None:
    assert "dice_risk" in CertificationConfigSchema.model_fields
    cfg = CertificationConfigSchema()
    assert cfg.dice_risk.enabled is False  # opt-in
    assert cfg.dice_risk.segmenter_backend == "label_dice"


def test_dice_risk_config_validates() -> None:
    c = DiceRiskCertificateConfig(enabled=True, alpha=0.1, delta=0.05, n_classes=14)
    assert c.alpha == 0.1 and c.n_classes == 14
    with pytest.raises(Exception):
        DiceRiskCertificateConfig(alpha=1.5)  # out of (0,1)
    with pytest.raises(Exception):
        DiceRiskCertificateConfig(n_classes=1)  # < 2
