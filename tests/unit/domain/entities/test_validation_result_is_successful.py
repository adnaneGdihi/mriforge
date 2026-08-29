"""WS-3 fix: ``ValidationResultEntity.is_successful`` must NOT report success for
an empty ``metrics`` dict (``all(...)`` over an empty iterable is vacuously True).
"""

from __future__ import annotations

from mriforge.domain.entities.training import ValidationResultEntity


def _result(metrics: dict[str, float]) -> ValidationResultEntity:
    return ValidationResultEntity(
        result_id="v", session_id="s", epoch=0, metrics=metrics
    )


def test_empty_metrics_is_not_successful() -> None:
    assert _result({}).is_successful is False


def test_all_positive_metrics_is_successful() -> None:
    assert _result({"psnr": 25.3, "ssim": 0.85}).is_successful is True


def test_any_non_positive_metric_is_not_successful() -> None:
    assert _result({"psnr": 25.3, "loss": 0.0}).is_successful is False
    assert _result({"psnr": -1.0}).is_successful is False
