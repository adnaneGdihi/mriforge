"""Map-then-mask reductions: the exact way to restrict a metric to a region.

For a metric with a per-pixel error map, the region-restricted value is the map
reduced over the mask. This is **exact** for MSE/MAE/PSNR, and for SSIM it is the
*right* answer rather than an approximation: the SSIM map at an ROI pixel uses an
11x11 Gaussian neighbourhood that may extend outside the ROI, and that is correct
-- it is the structure the pixel actually sits in. Cropping truncates it, and on a
40x40 ROI the 11-px boundary band is ~65% of the pixels.

The normalisation rule (load-bearing)
-------------------------------------

**Error terms are masked; normalisation constants are computed once on the FULL
slice.** PSNR's peak, NMSE's denominator, SSIM's ``data_range`` (which sets C1/C2).
If ``data_range`` were recomputed from the ROI, then "SSIM on GM" and "SSIM on the
lesion bbox" would use *different C1/C2* and would not be comparable across
regions -- the metric would have silently changed per region, which is exactly the
confound the region axis exists to measure.

Every reducer here takes its normalisation from the full slice.

The proof obligation
--------------------

A metric only joins the map tier if
``reduce(maps(pred, target), ones_mask) == registry_metric(pred, target)``.
Without that, the region tier would be silently computing a *different* metric
under the same name -- and nobody would notice, because the numbers would still
look plausible. ``tests/unit/core/metrics/regions/test_reductions.py`` enforces it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import NamedTuple

import torch

__all__ = [
    "MAP_REDUCERS",
    "MapReducer",
    "masked_mean",
]


def _abs_diff(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``|pred - target|`` -- the COMPLEX difference, then magnitude.

    Not ``|pred| - |target|``. The registry's MSE computes ``mean(|p - t|^2)``, so
    taking magnitudes first would silently compute a different quantity on complex
    input (phase error would vanish). Same trap the map tier exists to avoid.
    """
    return (pred - target).abs()


def _magnitude(x: torch.Tensor) -> torch.Tensor:
    return x.abs() if torch.is_complex(x) else x


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over ``mask``. Raises on an empty mask -- 0/0 is not 0.

    The mask must be EXPANDED to ``values``'s shape before summing. Unsqueezing alone
    leaves it ``[1, 1, H, W]``, so ``m.sum()`` counted only the mask's H*W pixels while
    ``(values * m).sum()`` broadcast over B and C -- so every map-tier value came back
    multiplied by exactly ``B * C``. That is the default case, not a corner: C=2 is the
    ordinary interleaved real/imag layout and any batched evaluation has B>1. A
    [4, 2, H, W] batch produced an SSIM of **7.62** (impossible), and PSNR was shifted by
    -10*log10(B*C). It went unnoticed because every fixture in the T2-identity test was
    [1, 1, 64, 64], where B*C == 1.
    """
    m = mask.to(dtype=values.dtype, device=values.device)
    while m.dim() < values.dim():
        m = m.unsqueeze(0)
    m = m.expand_as(values)
    denom = m.sum()
    if float(denom) == 0.0:
        raise ValueError("masked_mean over an empty mask: 0/0 is not a measurement")
    return (values * m).sum() / denom


class MapReducer(NamedTuple):
    """A metric decomposed into full-FOV maps plus a mask-aware reduction.

    Args:
        maps: ``(pred, target) -> {name: full-FOV map}``. A dict because some
            metrics need more than one map: NMSE needs the masked squared error
            *and* the masked squared target.
        reduce: ``(maps, mask) -> float``. Consumes only the masked maps and
            full-slice normalisation constants baked in by ``maps``.
    """

    maps: Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]
    reduce: Callable[[dict[str, torch.Tensor], torch.Tensor], float]


# --- pixel-error family -----------------------------------------------------


def _sq_err(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"sq": _abs_diff(pred, target) ** 2}


def _abs_err(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"abs": _abs_diff(pred, target)}


def _nmse_maps(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """NMSE needs both the squared error AND the squared target, each masked.

    The denominator is the *masked* target energy. Normalising by the full-slice
    energy instead would make a small ROI's NMSE depend on how much background the
    slice happens to contain -- a region effect manufactured out of the FOV.
    """
    return {"sq": _abs_diff(pred, target) ** 2, "sq_target": _magnitude(target) ** 2}


def _psnr_maps(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """PSNR's data_range comes from the FULL slice, never the ROI.

    The registry's default is a **static assumption** (``2.0`` if the target has
    negatives, else ``1.0``) -- *not* ``max - min``, and not the ROI's local max.
    T2 caught that difference: using ``max - min`` on a ``rand`` target gives
    ``0.9997`` instead of ``1.0`` and shifts PSNR by ~0.003 dB. Small, systematic,
    and entirely wrong.

    Deriving the peak from the ROI would be far worse: a dark ROI has a small local
    max, which would inflate that region's PSNR for *every* metric and *every*
    artifact -- a region effect manufactured out of the FOV.
    """
    from mriforge.core.metrics.registry import get_metric

    psnr = get_metric("psnr")
    t = target
    if psnr.data_range is not None:
        dr = float(psnr.data_range)
    elif psnr.use_target_max:
        dr = float(t.max())
        if dr < 1e-6:
            dr = 1.0
    else:
        dr = 2.0 if float(t.min()) < 0 else 1.0
    return {
        "sq": _abs_diff(pred, target) ** 2,
        "dr": torch.tensor(dr, dtype=torch.float32),
    }


def _psnr_reduce(maps: dict[str, torch.Tensor], mask: torch.Tensor) -> float:
    """Mirrors the registry PSNR exactly: the 1e-10 guard, the clamp, the mse==0 case."""
    mse = masked_mean(maps["sq"], mask)
    if float(mse) == 0.0:
        return 100.0
    dr = maps["dr"].to(mse.device)
    psnr = 20.0 * torch.log10(dr / (mse.sqrt() + 1e-10))
    return float(psnr.clamp(min=-30.0, max=100.0))


def _ssim_maps(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Full-FOV SSIM map, using the registry SSIM's OWN window and data_range.

    Reuses ``compute_ssim_map`` rather than reimplementing SSIM -- a second SSIM
    would be a second answer. Note the registry's ``data_range`` is a *static
    assumption* (``2.0`` if the target has negatives else ``1.0``), not the data
    max. Deriving it from the ROI instead would give "SSIM on GM" and "SSIM on the
    lesion bbox" different C1/C2 constants, making them incomparable -- exactly the
    confound the region axis exists to measure.
    """
    from mriforge.core.metrics.evaluation_metrics import compute_ssim_map
    from mriforge.core.metrics.registry import get_metric

    ssim = get_metric("ssim")
    p, t = _magnitude(pred), _magnitude(target)
    ws = ssim.window_size
    dr = ssim.data_range if ssim.data_range is not None else (2.0 if float(t.min()) < 0 else 1.0)
    window = ssim.window.to(p.device)
    channels = p.size(1)
    if window.shape[0] != channels:
        window = window.expand(channels, 1, ws, ws)
    return {"ssim": compute_ssim_map(p, t, window, ws, dr)}


MAP_REDUCERS: Mapping[str, MapReducer] = {
    "mse": MapReducer(_sq_err, lambda m, k: float(masked_mean(m["sq"], k))),
    "mae": MapReducer(_abs_err, lambda m, k: float(masked_mean(m["abs"], k))),
    "rmse": MapReducer(_sq_err, lambda m, k: float(masked_mean(m["sq"], k).sqrt())),
    "nmse": MapReducer(
        _nmse_maps,
        lambda m, k: float(
            masked_mean(m["sq"], k) / masked_mean(m["sq_target"], k).clamp_min(1e-12)
        ),
    ),
    "psnr": MapReducer(_psnr_maps, _psnr_reduce),
    "ssim": MapReducer(_ssim_maps, lambda m, k: float(masked_mean(m["ssim"], k))),
}
