r"""Image-domain hallucination metrics (PR-4 / H2).

Implements three registered metrics that quantify *fabricated* vs.
*missed* anatomical content in a reconstruction. The radiomic-CCC
profile already shipped in
``spectramr.infrastructure.reporting.metrics.hallucination`` is feature-domain;
these metrics are image-domain and computable on the fly during
validation.

Definitions
-----------
A *feature* here is a thresholded local-gradient maximum: a pixel is
"feature-bearing" iff its Sobel gradient magnitude exceeds the
configured threshold. This is a CT/MRI-friendly surrogate for fine
structure (vessel walls, trabecular bone, lesion boundaries).

Three metrics:

- **FFI — Feature Fidelity Index** (higher is better, ``∈ [0, 1]``):

  .. math::

      \text{FFI} = \frac{|F_{pred} \cap F_{target}|}
                        {|F_{target}|}

  Fraction of target-image features that the prediction also flags.
  ``FFI = 1`` ⇔ no missed anatomy. Identity reconstruction → 1.

- **FAB — Fabrication Rate** (lower is better, ``∈ [0, 1]``):

  .. math::

      \text{FAB} = \frac{|F_{pred} \setminus F_{target}|}
                        {|F_{pred}|}

  Fraction of predicted features that are *not* in the target — i.e.
  fabricated content. ``FAB = 0`` ⇔ no hallucinations. Identity → 0.

- **HIE — Hallucination Index, Ensemble** (lower is better, ``∈ [0, ∞)``):

  Standard deviation of feature counts across an ensemble of
  predictions. High variance under stochastic decoding signals
  unstable fabrications. Computed only when an ensemble is supplied.

Reference: Antun et al. 2020 *PNAS* (instabilities of deep learning
in image reconstruction).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from spectramr.core.metrics.registry import register_metric


def _to_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Return real-valued magnitude for complex / 2-channel real / multi-channel input."""
    if x.is_complex():
        return x.abs()
    if x.shape[-3] == 2:  # interleaved real/imag
        return torch.sqrt(x[..., 0:1, :, :] ** 2 + x[..., 1:2, :, :] ** 2 + 1e-12)
    return x.abs()


def _sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude per pixel (per-channel grouped)."""
    # Build per-input-channel grouped Sobel kernels on the fly.
    c = x.shape[1]
    kx = (
        torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=x.dtype,
            device=x.device,
        )
        .view(1, 1, 3, 3)
        .expand(c, 1, 3, 3)
    )
    ky = (
        torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=x.dtype,
            device=x.device,
        )
        .view(1, 1, 3, 3)
        .expand(c, 1, 3, 3)
    )
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _feature_mask(
    x: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Boolean mask of feature-bearing pixels.

    Accepts both batched ``[B, C, H, W]`` and unbatched ``[C, H, W]``
    inputs — the meta-eval dataset (sim2rank) hands us unbatched
    tensors, and a 3D ``F.conv2d`` call inside :func:`_sobel_magnitude`
    would raise, which ``_safe_metric`` would silently swallow to NaN.

    Returns an all-False mask when the image has no usable gradient
    signal (uniform / near-zero input). The +1e-12 in
    :func:`_sobel_magnitude` keeps the magnitude differentiable, but
    its noise floor must not be allowed to flag *every* pixel as a
    "feature" — that turns identity / blank reconstructions into
    100% fabrication. We guard against that by requiring the
    per-image max to exceed a small absolute floor before any pixel
    is considered feature-bearing.
    """
    added_batch = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        added_batch = True
    mag = _to_magnitude(x)
    grad = _sobel_magnitude(mag.float())
    # Threshold relative to per-image max so the metric is scale-invariant.
    per_image_max = grad.flatten(1).max(dim=1, keepdim=True).values
    per_image_max = per_image_max.view(-1, 1, 1, 1)
    # Hard floor: if the image has no gradient signal worth speaking of,
    # nothing is a feature. 1e-6 is well above the sqrt(1e-12)≈3.16e-6
    # noise from the magnitude stabiliser, so a *real* edge survives.
    sentinel = torch.full_like(per_image_max, 1e-5)
    has_signal = per_image_max > sentinel
    safe_max = per_image_max.clamp_min(1e-12)
    result = has_signal & (grad > threshold * safe_max)
    if added_batch:
        result = result.squeeze(0)
    return result


def feature_fidelity_index(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.10,
) -> float:
    r"""``FFI = |F_pred ∩ F_target| / |F_target|``.

    Identity reconstruction gives ``FFI = 1.0``. A blurred prediction
    that loses fine structure gives ``FFI < 1``.

    Raises:
        ValueError: shape mismatch.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
        )
    f_pred = _feature_mask(pred, threshold)
    f_target = _feature_mask(target, threshold)
    intersection = (f_pred & f_target).float().sum()
    target_count = f_target.float().sum().clamp_min(1.0)
    return float((intersection / target_count).item())


def fabrication_rate(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.10,
) -> float:
    r"""``FAB = |F_pred \\ F_target| / |F_pred|``.

    Identity reconstruction gives ``FAB = 0.0``. A prediction that
    invents structure absent from the target raises FAB.

    Returns 0.0 when ``F_pred`` is empty (no features → no fabrication
    by construction).
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
        )
    f_pred = _feature_mask(pred, threshold)
    f_target = _feature_mask(target, threshold)
    fabricated = (f_pred & ~f_target).float().sum()
    pred_count = f_pred.float().sum()
    if pred_count.item() == 0:
        return 0.0
    return float((fabricated / pred_count).item())


def hallucination_index_ensemble(
    ensemble_preds: Sequence[torch.Tensor],
    threshold: float = 0.10,
) -> float:
    """Standard deviation of feature counts across stochastic predictions.

    Args:
        ensemble_preds: Iterable of equal-shape predictions (e.g.
            multiple decoder seeds, MC-dropout samples, or different
            sampling temperatures).
        threshold: Same gradient threshold as the other metrics.

    Returns:
        ``std(feature_count_per_member)``. ``0`` when the ensemble is
        deterministic.

    Raises:
        ValueError: when fewer than two members are supplied (a
            single-sample ensemble has no variance).
    """
    if len(ensemble_preds) < 2:
        raise ValueError("hallucination_index_ensemble needs ≥ 2 ensemble members.")
    counts = []
    for p in ensemble_preds:
        counts.append(int(_feature_mask(p, threshold).sum().item()))
    counts_t = torch.tensor(counts, dtype=torch.float64)
    return float(counts_t.std(unbiased=False).item())


# ---------------------------------------------------------------------------
# Registry adapters — each metric is exposed under @register_metric.
# ---------------------------------------------------------------------------


@register_metric(name="feature_fidelity_index", aliases=["FFI", "ffi"])
class FeatureFidelityIndexMetric:
    """Registered metric wrapping :func:`feature_fidelity_index`."""

    def __init__(self, threshold: float = 0.10) -> None:
        self.threshold = threshold

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, **_: object) -> float:
        return feature_fidelity_index(prediction, target, threshold=self.threshold)

    @property
    def name(self) -> str:
        return "feature_fidelity_index"

    @property
    def higher_is_better(self) -> bool:
        return True


@register_metric(name="fabrication_rate", aliases=["FAB", "fab", "hallucination_rate"])
class FabricationRateMetric:
    """Registered metric wrapping :func:`fabrication_rate`."""

    def __init__(self, threshold: float = 0.10) -> None:
        self.threshold = threshold

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, **_: object) -> float:
        return fabrication_rate(prediction, target, threshold=self.threshold)

    @property
    def name(self) -> str:
        return "fabrication_rate"

    @property
    def higher_is_better(self) -> bool:
        return False


__all__ = [
    "FabricationRateMetric",
    "FeatureFidelityIndexMetric",
    "fabrication_rate",
    "feature_fidelity_index",
    "hallucination_index_ensemble",
]
