"""WS1: feature-insertion hallucination test (validation.hallucination_test)."""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.hallucination_test import (
    feature_preservation_index,
    insert_synthetic_features,
    run_hallucination_test,
)


@pytest.mark.parametrize("method", ["feature_insertion", "lesion_swap", "edge_jitter"])
def test_insert_produces_nonempty_mask_and_changes_image(method: str) -> None:
    img = torch.zeros(2, 1, 32, 32)
    perturbed, mask = insert_synthetic_features(img, n_features=4, method=method)
    assert perturbed.shape == img.shape
    assert mask.shape == (2, 1, 32, 32) and mask.dtype == torch.bool
    assert mask.any()
    assert not torch.equal(perturbed, img)  # features were inserted


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown hallucination method"):
        insert_synthetic_features(torch.zeros(1, 1, 8, 8), 1, "nope")  # type: ignore[arg-type]


def test_preservation_index_perfect_when_reconstruction_matches() -> None:
    img = torch.zeros(1, 1, 16, 16)
    perturbed, mask = insert_synthetic_features(img, 3, "feature_insertion")
    assert feature_preservation_index(perturbed, perturbed, mask).item() == pytest.approx(
        1.0
    )


def test_preservation_index_low_when_features_dropped() -> None:
    img = torch.zeros(1, 1, 16, 16)
    perturbed, mask = insert_synthetic_features(img, 3, "feature_insertion")
    # A reconstruction that drops the inserted features (returns the clean image).
    score = feature_preservation_index(img, perturbed, mask).item()
    assert score < 0.2


def test_runner_identity_predict_is_high_preservation() -> None:
    target = torch.rand(2, 1, 24, 24)
    out = run_hallucination_test(lambda x: x, target, n_features=4)
    assert out["hallucination_preservation_index"] == pytest.approx(1.0)
    assert out["method"] == "feature_insertion" and out["n_features"] == 4


def test_runner_dropping_predict_is_low_preservation() -> None:
    target = torch.zeros(2, 1, 24, 24)
    out = run_hallucination_test(lambda x: torch.zeros_like(x), target, n_features=4)
    assert out["hallucination_preservation_index"] < 0.2
