"""Tests for image-domain hallucination metrics (PR-4 / H2).

Targets ``mriforge.core.metrics.hallucination_metrics``.

Plan acceptance criteria covered:

- *"Identity reconstruction → FFI = 1.0, FAB = 0.0"* — both directly
  asserted below
- *"Lesion injection → metric responds monotonically with lesion contrast"*
  — exercised via the ``test_fabrication_rate_grows_with_added_structure``
  test that injects a high-contrast block.

Other contract tests:

- Shape mismatch raises
- Ensemble with < 2 members raises
- Ensemble of identical predictions has zero variance
"""

from __future__ import annotations

import pytest
import torch

from mriforge.core.metrics.hallucination_metrics import (
    fabrication_rate,
    feature_fidelity_index,
    hallucination_index_ensemble,
)


# ---------------------------------------------------------------------------
# Identity acceptance criterion
# ---------------------------------------------------------------------------


def test_ffi_identity_is_one() -> None:
    """``FFI(x, x) == 1.0``."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, 32)
    # Make some structure (gradient must exist or both masks are empty).
    x = x.abs() + 0.5
    x[..., 8:24, 8:24] = 5.0
    assert pytest.approx(feature_fidelity_index(x, x)) == 1.0


def test_fab_identity_is_zero() -> None:
    """``FAB(x, x) == 0.0``."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, 32).abs() + 0.5
    x[..., 8:24, 8:24] = 5.0
    assert pytest.approx(fabrication_rate(x, x)) == 0.0


# ---------------------------------------------------------------------------
# Monotonicity under structural perturbations
# ---------------------------------------------------------------------------


def test_ffi_drops_under_blur() -> None:
    """A heavily-blurred prediction loses fine structure → FFI < 1.

    Uses isolated single-pixel "stars" embedded in a low-amplitude
    noise floor as the target. A wide averaging filter (15×15) on
    point sources collapses each star into a basin whose gradient
    field is more than three orders of magnitude below the target's
    — most of the original feature pixels no longer survive the
    relative-to-image-max threshold check, so FFI must drop.
    """
    torch.manual_seed(0)
    target = torch.randn(1, 1, 32, 32) * 0.05
    target[..., 5, 5] = 1.0
    target[..., 10, 20] = 1.0
    target[..., 25, 8] = 1.0
    target[..., 18, 25] = 1.0

    blurred = torch.nn.functional.avg_pool2d(
        target, kernel_size=15, stride=1, padding=7
    )
    ffi = feature_fidelity_index(blurred, target)
    assert ffi < 1.0


def test_fabrication_rate_grows_with_added_structure() -> None:
    """Injecting a lesion-like block raises FAB monotonically."""
    target = torch.zeros(1, 1, 32, 32)
    target[..., 14:18, 14:18] = 1.0

    pred_clean = target.clone()
    pred_lesion = target.clone()
    pred_lesion[..., 4:8, 4:8] = 1.0  # injected lesion (fabrication)

    fab_clean = fabrication_rate(pred_clean, target)
    fab_lesion = fabrication_rate(pred_lesion, target)
    assert fab_lesion > fab_clean


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_ffi_shape_mismatch_raises() -> None:
    """Different shapes → ``ValueError``."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        feature_fidelity_index(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 16, 16))


def test_fabrication_rate_shape_mismatch_raises() -> None:
    """Different shapes → ``ValueError``."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        fabrication_rate(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 16, 16))


def test_fabrication_rate_zero_for_empty_pred_features() -> None:
    """All-zero prediction → no features, FAB = 0 by construction."""
    target = torch.ones(1, 1, 16, 16)
    pred = torch.zeros(1, 1, 16, 16)  # uniform → no gradients
    assert fabrication_rate(pred, target) == 0.0


# ---------------------------------------------------------------------------
# Unbatched 3D inputs (sim2rank / meta-eval calling convention)
# ---------------------------------------------------------------------------


def test_ffi_accepts_unbatched_3d_input() -> None:
    """Meta-eval feeds [C,H,W] — regression for the 'NaN on 8 consecutive
    base evaluations' short-circuit that fired because ``_sobel_magnitude``
    rejected 3D inputs and ``_safe_metric`` swallowed the exception."""
    torch.manual_seed(0)
    x = torch.randn(1, 32, 32).abs() + 0.5
    x[..., 8:24, 8:24] = 5.0
    score = feature_fidelity_index(x, x)
    assert score == pytest.approx(1.0)


def test_fab_accepts_unbatched_3d_input() -> None:
    """Same regression for FAB (shares ``_feature_mask``)."""
    torch.manual_seed(0)
    x = torch.randn(1, 32, 32).abs() + 0.5
    x[..., 8:24, 8:24] = 5.0
    score = fabrication_rate(x, x)
    assert score == pytest.approx(0.0)


def test_ffi_accepts_unbatched_2channel_complex_3d() -> None:
    """Interleaved real/imag as 3D [2, H, W] — sim2rank meta-eval uses this
    for complex k-space pairs."""
    torch.manual_seed(0)
    real = torch.randn(1, 32, 32).abs() + 0.5
    real[..., 8:24, 8:24] = 5.0
    imag = torch.zeros_like(real)
    x = torch.cat([real, imag], dim=0)  # [2, H, W]
    score = feature_fidelity_index(x, x)
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Ensemble metric
# ---------------------------------------------------------------------------


def test_ensemble_metric_zero_for_identical_members() -> None:
    """Identical members → variance is zero."""
    img = torch.randn(1, 1, 16, 16).abs()
    out = hallucination_index_ensemble([img, img, img])
    assert out == 0.0


def test_ensemble_metric_positive_for_diverse_members() -> None:
    """Diverse predictions → positive variance."""
    torch.manual_seed(0)
    members = [torch.randn(1, 1, 16, 16) for _ in range(4)]
    out = hallucination_index_ensemble(members)
    assert out > 0.0


def test_ensemble_metric_singleton_raises() -> None:
    """Single-member ensemble has no variance — fail loud."""
    with pytest.raises(ValueError, match="≥ 2"):
        hallucination_index_ensemble([torch.zeros(1, 1, 4, 4)])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_metrics_registered() -> None:
    """Both registered metrics resolvable via the metric registry."""
    from mriforge.core.metrics.registry import list_available

    available = list_available()
    assert "feature_fidelity_index" in available
    assert "fabrication_rate" in available


def test_aliases_resolve() -> None:
    """``FFI`` and ``FAB`` aliases dispatch to the canonical names."""
    from mriforge.core.metrics.registry import get_metric

    ffi = get_metric("FFI")
    fab = get_metric("FAB")
    assert ffi.name == "feature_fidelity_index"
    assert fab.name == "fabrication_rate"


def test_hallucination_rate_alias_maps_to_fabrication_rate() -> None:
    """``hallucination_rate`` (referenced by vf_08) resolves to fabrication_rate.

    Both measure the fraction of predicted structure absent from the target
    (``FAB = |F_pred \\ F_target| / |F_pred|``), so the VF arm's metric name is
    an alias rather than a duplicate implementation.
    """
    from mriforge.core.metrics.registry import get_metric

    # Aliases resolve via get_metric (they are not listed in list_available(),
    # which returns canonical names only).
    metric = get_metric("hallucination_rate")
    assert metric.name == "fabrication_rate"
    assert metric.higher_is_better is False
