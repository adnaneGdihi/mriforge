r"""Non-conformity score functions for split-conformal prediction.

Each score returns a tensor of *non-conformity values* — larger
indicates "less expected under the calibration distribution". The
calibrator computes the empirical :math:`(1-\alpha)`-quantile of these
values and uses it as the prediction-band radius.

Three functional forms are exposed (all stateless):

- :func:`absolute_residual` — :math:`s = |y - \hat y|`. Symmetric
  band around the point estimate; the simplest score and the right
  choice when the heteroscedasticity of the residuals is mild.
- :func:`quantile_regression` — :math:`s = \max(\hat q_{lo} - y,\, y - \hat q_{hi})`
  given pre-trained quantile estimators. Adaptive band shape; needs
  a model that produces ``(y_hat_lo, y_hat_hi)``.
- :func:`conformalised_quantile_regression` — same form, but the
  output is wrapped so the calibrator only sees the quantile-aware
  residual. Convenience alias for the same numerics.

References
----------
Vovk, Gammerman, Shafer (2005), *Algorithmic Learning in a Random World*.
Romano, Patterson, Candès (2019), *Conformalized quantile regression*.
"""

from __future__ import annotations

import torch


def absolute_residual(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Symmetric absolute-residual non-conformity score.

    Args:
        pred: Predicted point estimate.
        target: Ground-truth label.

    Returns:
        Tensor with the same shape as ``pred``: ``|pred - target|``.

    Raises:
        ValueError: when ``pred`` and ``target`` shapes disagree.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
        )
    return (pred - target).abs()


def quantile_regression(
    pred_lower: torch.Tensor,
    pred_upper: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Quantile-regression conformity score (Romano et al. 2019).

    .. math::

        s = \\max\\bigl(\\hat q_{lo} - y,\\, y - \\hat q_{hi}\\bigr)

    Strictly negative inside the predicted band, strictly positive
    outside. The calibrator's :math:`(1-\\alpha)` quantile of these
    scores yields a slack that, when symmetrically subtracted from
    ``pred_lower`` and added to ``pred_upper``, gives the conformalised
    band with marginal coverage :math:`\\geq 1 - \\alpha`.

    Raises:
        ValueError: shape mismatch, or ``pred_lower > pred_upper`` anywhere
            (would yield an empty / invalid band).
    """
    if not (pred_lower.shape == pred_upper.shape == target.shape):
        raise ValueError(
            f"Shape mismatch: lower {tuple(pred_lower.shape)} / "
            f"upper {tuple(pred_upper.shape)} / target {tuple(target.shape)}"
        )
    if (pred_lower > pred_upper).any():
        raise ValueError(
            "pred_lower must be ≤ pred_upper element-wise; got "
            "swapped quantiles. Check the upstream model output order."
        )
    return torch.maximum(pred_lower - target, target - pred_upper)


def conformalised_quantile_regression(
    pred_lower: torch.Tensor,
    pred_upper: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Alias for :func:`quantile_regression` — exposed under the canonical CQR name."""
    return quantile_regression(pred_lower, pred_upper, target)


# ---------------------------------------------------------------------------
# Physics-residual non-conformity score (PR-CC)
# ---------------------------------------------------------------------------

# Clean contrast labels -> the MultiPhysicsBlochLayer integer sequence enum
# (0=SE, 1=IR/FLAIR, 2=GRE/SPGR). We dispatch on an explicit table and *raise*
# on an unknown label rather than relying on the layer's loose substring
# heuristic, which would silently mis-route (e.g. "spin_echo" matches neither
# "se" nor "t2"). No silent fallback — pitfall #9.
_CONTRAST_TO_SEQ: dict[str, int] = {
    "spin_echo": 0,
    "se": 0,
    "t2": 0,
    "t2w": 0,
    "inversion_recovery": 1,
    "ir": 1,
    "flair": 1,
    "t1_ir": 1,
    "gradient_echo": 2,
    "gre": 2,
    "spgr": 2,
    "t1w_gre": 2,
    "mprage": 2,
}


def physics_residual(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    acquisition: dict[str, float] | None = None,
    contrast_type: str = "spin_echo",
    **_: object,
) -> torch.Tensor:
    r"""Physics-consistency non-conformity score for PR-CC.

    Re-renders the *observed* contrast from an estimated tissue-parameter map
    through the Bloch forward (the physics single-source-of-truth) and returns
    the per-voxel residual against the observed image:

    .. math::

        s(v) = \bigl|\mathcal{A}(\hat{\boldsymbol\xi};\boldsymbol\varphi, B_0)(v)
               - \mathbf{y}(v)\bigr|.

    Because :math:`\mathcal{A}` (the Bloch relation) holds at every field, the
    score is anchored to a field-invariant model: its validity as a conformal
    score needs only residual exchangeability, not image-distribution
    exchangeability (the property PR-CC exploits for cross-field certificates).

    Args:
        prediction: Estimated tissue parameters ``[B, 3, ...]`` channel-stacked
            as ``(rho, T1_ms, T2_ms)``.
        target: Observed (normalised) magnitude image ``[B, 1, ...]``.
        acquisition: Sequence parameters ``{"TR", "TE", "TI"}`` in ms. Required;
            a physics residual without an acquisition is undefined.
        contrast_type: One of the keys of :data:`_CONTRAST_TO_SEQ`.

    Returns:
        ``|render - target|``, same shape as ``target``.

    Raises:
        ValueError: ``acquisition is None``; unknown ``contrast_type``;
            ``prediction`` does not carry exactly 3 parameter channels.
    """
    if acquisition is None:
        raise ValueError(
            "physics_residual requires `acquisition` (TR/TE/TI in ms); a physics "
            "residual without acquisition parameters is undefined."
        )
    if prediction.ndim < 2 or prediction.shape[1] != 3:
        raise ValueError(
            "physics_residual expects `prediction` with exactly 3 parameter "
            f"channels (rho, T1, T2); got shape {tuple(prediction.shape)}."
        )
    seq = _CONTRAST_TO_SEQ.get(str(contrast_type).lower())
    if seq is None:
        raise ValueError(
            f"Unknown contrast_type {contrast_type!r}; expected one of "
            f"{sorted(set(_CONTRAST_TO_SEQ))}."
        )

    # Lazy import keeps this scores module light and avoids pulling the physics
    # stack into every conformal import (the established hot-path pattern).
    from mriforge.infrastructure.physics.multi_physics_bloch import (
        MultiPhysicsBlochLayer,
    )

    layer = MultiPhysicsBlochLayer(learnable_flip_angle=False).to(prediction.device)
    tissue = {
        "rho": prediction[:, 0:1],
        "t1": prediction[:, 1:2],
        "t2": prediction[:, 2:3],
    }
    metadata: dict[str, object] = {
        "TR": float(acquisition["TR"]),
        "TE": float(acquisition["TE"]),
        "TI": float(acquisition.get("TI", 0.0)),
        "contrast_type": int(seq),
    }
    rendered = layer(tissue, metadata)
    return (rendered - target).abs()


# ---------------------------------------------------------------------------
# Cold-diffusion trust scores (C7) — per-image, label-free
# ---------------------------------------------------------------------------
def _per_sample_norm(x: torch.Tensor) -> torch.Tensor:
    """L2 norm over all non-batch dims → ``[B]`` (complex-aware)."""
    reduce_dims = tuple(range(1, x.dim()))
    if torch.is_complex(x):
        return torch.sqrt((x.real**2 + x.imag**2).sum(dim=reduce_dims))
    return torch.sqrt((x**2).sum(dim=reduce_dims))


def kappa_residual(
    prediction: torch.Tensor,
    measurement: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Per-image :math:`\kappa` — relative residual on the observed support.

    ``prediction`` and ``measurement`` are k-space tensors ``[B, ...]``;
    ``measurement`` is the ACQUIRED k-space (not a ground-truth label), so
    this score is label-free and computable at deployment. ``mask=None``
    means fully sampled. Returns one score per batch element (``[B]``) —
    trust conformalises at image level, not pixel level.
    """
    if prediction.shape != measurement.shape:
        raise ValueError(
            f"Shape mismatch: prediction {tuple(prediction.shape)} vs "
            f"measurement {tuple(measurement.shape)}"
        )
    if mask is not None:
        m = mask.real if torch.is_complex(mask) else mask
        prediction = prediction * m
        measurement = measurement * m
    return _per_sample_norm(prediction - measurement) / (_per_sample_norm(measurement) + eps)


def null_energy(
    prediction: torch.Tensor,
    *,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Per-image :math:`\eta^{\mathrm{null}}` — invented null-band energy fraction.

    ``‖(1−M)·x̂‖ / (‖x̂‖ + eps)`` per batch element: how much of the
    reconstruction's k-space energy lives where nothing was measured.
    No-reference; requires the sampling mask (a null band is undefined
    without one, so ``mask`` is mandatory here).
    """
    m = mask.real if torch.is_complex(mask) else mask
    return _per_sample_norm(prediction * (1.0 - m)) / (_per_sample_norm(prediction) + eps)


__all__ = [
    "absolute_residual",
    "conformalised_quantile_regression",
    "kappa_residual",
    "null_energy",
    "physics_residual",
    "quantile_regression",
]
