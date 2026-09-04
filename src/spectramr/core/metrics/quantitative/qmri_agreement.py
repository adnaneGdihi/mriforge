"""Quantitative-agreement battery for qMRI field/parameter estimates.

The VF / multi-acquisition arms estimate a physical field (B0 in Hz, B1+ scale,
EPI displacement, …) against a known reference (the digital twin's applied
field). Image PSNR cannot tell whether the *quantity* agrees — the standard
qMRI-validation statistics do: Bland-Altman bias + limits of agreement, the
two-way mixed single-measure intraclass correlation ICC(3,1), and the
coefficient of variation. This is the metric battery the scientific-validation
backlog (`deferred_source_backlog.json`) repeatedly defers to
("run the qMRI metrics battery on (pred_field, field)").

It is a pure tensor→dict function so strategies compute it inline (mirroring
``ConcreteMultiAcquisitionStrategy.validation_step``), rather than a registered
``@register_metric`` — the inputs are a *paired* (prediction, reference) field,
not the single (pred, target) image the metric registry contract assumes.

Scope: on M4Raw the reference is the simulator's OWN field, so these grade
inverter SELF-CONSISTENCY, not real-scanner accuracy (see
``docs/vf_advanced.rst`` — ``vf_field_self_consistency``).
"""

from __future__ import annotations

import torch

__all__ = ["qmri_agreement_metrics"]


def qmri_agreement_metrics(pred: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """Bland-Altman + ICC(3,1) + CoV agreement of a predicted field vs reference.

    Treats every element of the (broadcast-matched, flattened) field as one
    paired observation — ``n`` subjects, ``k = 2`` measurements (pred, ref) —
    which is the standard construction for a per-voxel agreement readout.

    Args:
        pred: Estimated field ``[B, ...]`` (any shape; flattened).
        ref:  Reference field, broadcastable to ``pred``.

    Returns:
        ``bland_altman_bias`` — mean(pred − ref); 0 is perfect.
        ``limits_of_agreement_upper`` / ``_lower`` — bias ± 1.96·SD(diff).
        ``icc_3_1`` — two-way mixed, single-measure, consistency ICC in [−1, 1];
            1 is perfect agreement.
        ``coefficient_of_variation`` — SD(diff) / mean(|all measurements|).
        ``mae`` — mean(|pred − ref|).

    All values are python floats (one GPU sync); call from a no-grad validation
    path, never the training loop (performance.md).
    """
    p = pred.reshape(-1).float()
    r = ref.expand_as(pred).reshape(-1).float()
    n = int(p.numel())
    if n == 0:
        return {
            "bland_altman_bias": 0.0,
            "limits_of_agreement_upper": 0.0,
            "limits_of_agreement_lower": 0.0,
            "icc_3_1": 0.0,
            "coefficient_of_variation": 0.0,
            "mae": 0.0,
        }

    diff = p - r
    bias = diff.mean()
    sd = diff.std(unbiased=True) if n > 1 else torch.zeros_like(bias)
    mae = diff.abs().mean()
    grand_abs = torch.cat([p, r]).abs().mean()
    cov = sd / (grand_abs + 1e-8)

    # ICC(3,1): two-way mixed, single measure, consistency. k=2 measures
    # (pred, ref) over n subjects. ICC = (MSR − MSE) / (MSR + (k−1)·MSE).
    measures = torch.stack([p, r], dim=1)  # [n, 2]
    k = 2
    grand = measures.mean()
    row_mean = measures.mean(dim=1)  # per-subject
    col_mean = measures.mean(dim=0)  # per-rater
    ss_total = ((measures - grand) ** 2).sum()
    ss_rows = k * ((row_mean - grand) ** 2).sum()
    ss_cols = n * ((col_mean - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / max(n - 1, 1)
    ms_err = ss_err / max((n - 1) * (k - 1), 1)
    icc = (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err + 1e-12)

    return {
        "bland_altman_bias": float(bias),
        "limits_of_agreement_upper": float(bias + 1.96 * sd),
        "limits_of_agreement_lower": float(bias - 1.96 * sd),
        "icc_3_1": float(icc.clamp(-1.0, 1.0)),
        "coefficient_of_variation": float(cov),
        "mae": float(mae),
    }
