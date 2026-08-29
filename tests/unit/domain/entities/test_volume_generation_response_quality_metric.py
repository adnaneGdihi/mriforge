"""WS-3: ``VolumeGenerationResponse.get_quality_metric`` must raise KeyError for
an unrecorded metric rather than silently returning None (NN#3)."""

from __future__ import annotations

import pytest

from mriforge.domain.entities.entities_3d import VolumeGenerationResponse


def test_get_recorded_metric_returns_value() -> None:
    r = VolumeGenerationResponse(request_id="r")
    r.add_quality_metric("psnr", 31.2)
    assert r.get_quality_metric("psnr") == 31.2


def test_get_missing_metric_raises_keyerror() -> None:
    r = VolumeGenerationResponse(request_id="r")
    with pytest.raises(KeyError, match="not recorded"):
        r.get_quality_metric("ssim")
