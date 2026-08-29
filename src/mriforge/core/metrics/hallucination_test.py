"""Feature-insertion hallucination test — clinical-defensibility hook.

CLAUDE.md #10 / plan §13 / A5.9. Image-prior paradigms (cold diffusion, latent
diffusion, GANs) can *hallucinate* plausible-but-fabricated structure or, just as
dangerously, *drop* genuine small features that the measurement under-constrains.
This module implements the configured ``validation.hallucination_test`` sweep:

1. insert N synthetic features into a clean target,
2. let the model reconstruct (from the degraded perturbed input),
3. score how faithfully the inserted features survive (preservation index).

The perturbation methods and the scoring are pure tensor functions (no model,
no infrastructure imports), so they are unit-testable in isolation; the runner
orchestrates them around a caller-supplied ``predict`` callable, which the
training-strategy validation hook provides.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import torch

HallucinationMethod = Literal["feature_insertion", "lesion_swap", "edge_jitter"]
_METHODS: tuple[str, ...] = ("feature_insertion", "lesion_swap", "edge_jitter")


def _rand_centers(n: int, h: int, w: int, generator: torch.Generator | None) -> torch.Tensor:
    """``n`` random (row, col) centres with a margin from the border."""
    m = 3
    rows = torch.randint(m, max(m + 1, h - m), (n,), generator=generator)
    cols = torch.randint(m, max(m + 1, w - m), (n,), generator=generator)
    return torch.stack([rows, cols], dim=1)


def insert_synthetic_features(
    image: torch.Tensor,
    n_features: int,
    method: HallucinationMethod = "feature_insertion",
    *,
    amplitude: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert ``n_features`` synthetic features into ``image`` (``B, C, H, W``).

    Returns ``(perturbed, mask)`` where ``mask`` (``B, 1, H, W`` bool) marks the
    inserted-feature support. The same feature pattern is applied to every item
    in the batch so the mask is shared. ``amplitude`` is scaled by the image's
    dynamic range so the features are visible regardless of normalisation.
    """
    if method not in _METHODS:
        raise ValueError(f"unknown hallucination method {method!r}; expected one of {_METHODS}")
    if image.ndim != 4:
        raise ValueError(f"expected a (B, C, H, W) image, got shape {tuple(image.shape)}")

    b, _c, h, w = image.shape
    scale = amplitude * float(image.amax() - image.amin() + 1e-6)
    perturbed = image.clone()
    mask = torch.zeros((b, 1, h, w), dtype=torch.bool, device=image.device)
    centers = _rand_centers(n_features, h, w, generator)

    if method == "edge_jitter":
        # High-contrast step edges (a few-pixel-wide bright column) — stress the
        # model's high-frequency fidelity.
        for r, col in centers.tolist():
            c0, c1 = max(0, col - 1), min(w, col + 2)
            perturbed[:, :, :, c0:c1] += scale
            mask[:, :, :, c0:c1] = True
        return perturbed, mask

    # feature_insertion (small Gaussian blobs) and lesion_swap (larger filled
    # discs) share the radial-bump machinery; lesion_swap is brighter + wider.
    radius = 4 if method == "lesion_swap" else 2
    yy = torch.arange(h, device=image.device).view(h, 1)
    xx = torch.arange(w, device=image.device).view(1, w)
    lvl = scale * (1.5 if method == "lesion_swap" else 1.0)
    for r, col in centers.tolist():
        dist2 = (yy - r) ** 2 + (xx - col) ** 2
        bump = torch.exp(-dist2.float() / (2.0 * radius**2))
        perturbed += (lvl * bump).view(1, 1, h, w)
        mask |= (dist2 <= radius**2).view(1, 1, h, w)
    return perturbed, mask


def feature_preservation_index(
    reconstruction: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Fraction of inserted-feature signal preserved in the reconstruction.

    Computed only inside ``mask`` as ``1 - ||recon - reference|| / ||reference||``
    (relative L2), clamped to ``[0, 1]``. ``1.0`` = features faithfully preserved;
    ``→ 0`` = features dropped or replaced by hallucinated structure. ``reference``
    is the perturbed (features-present) target.
    """
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    m = mask.expand_as(reference)
    ref = reference[m]
    if ref.numel() == 0:
        return torch.zeros((), device=reference.device)
    rec = reconstruction[m]
    rel = torch.linalg.vector_norm(rec - ref) / (torch.linalg.vector_norm(ref) + eps)
    return torch.clamp(1.0 - rel, min=0.0, max=1.0)


def run_hallucination_test(
    predict: Callable[[torch.Tensor], torch.Tensor],
    target: torch.Tensor,
    *,
    n_features: int = 5,
    method: HallucinationMethod = "feature_insertion",
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Run the sweep: perturb ``target``, reconstruct via ``predict``, score.

    ``predict`` maps the perturbed target to a reconstruction (the strategy
    supplies a closure that applies its own degradation + reconstruction). Returns
    a dict with ``hallucination_preservation_index`` (higher = safer) and the
    ``method`` / ``n_features`` used.
    """
    perturbed, mask = insert_synthetic_features(target, n_features, method, generator=generator)
    reconstruction = predict(perturbed)
    if reconstruction.shape != perturbed.shape:
        # Align spatial dims defensively (e.g. a model that returns magnitude).
        reconstruction = reconstruction[..., : perturbed.shape[-2], : perturbed.shape[-1]]
    pidx = feature_preservation_index(reconstruction, perturbed, mask)
    return {
        "hallucination_preservation_index": float(pidx),
        "method": method,
        "n_features": int(n_features),
    }


__all__ = [
    "feature_preservation_index",
    "insert_synthetic_features",
    "run_hallucination_test",
]
