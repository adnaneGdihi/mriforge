r"""VF-residual non-conformity score for split-conformal calibration.

The score is the energy of the reconstruction *projected onto the
marker subspace*: with marker basis :math:`M \in \mathbb{C}^{N \times k}`
(columns are orthonormal markers injected into the image during
training) and orthogonal projector :math:`P_M = M (M^H M)^{-1} M^H`,
the score on a reconstruction :math:`\hat{x}` is

.. math::

    s(\hat{x}) = \| P_M \hat{x} \|_2.

Why this works for split-conformal: under the standard
exchangeability hypothesis (Vovk-Gammerman-Shafer 2005), any
*deterministic* function of the reconstruction is a valid score.
The marker-subspace projection has the practical virtue that it
*does not require a clean reference* --- :math:`P_M` is fixed by the
marker design --- so the calibration set can be drawn from raw acquired
data, not from a paired-ground-truth corpus.

Identifiability / consistency:

* When the marker basis is empty (:math:`k=0`), :math:`P_M = 0` and
  :math:`s \equiv 0`. The Tier-1 audit
  ``vf_residual_marker_matrix_well_conditioned`` rejects this case
  to prevent a silently-trivial calibration.
* When :math:`M` is rank-deficient or numerically ill-conditioned
  (:math:`\kappa(M^H M) > 10^6`), the audit fires; the registered
  score raises :class:`numpy.linalg.LinAlgError` rather than masking
  the failure.

Implementation note: ``P_M`` is *never* materialised as a dense
:math:`N \times N` matrix --- that is :math:`32~\mathrm{GB}` of
complex64 at the :math:`256 \times 256` M4Raw patch size. Instead we
cache the basis :math:`M` and the Gram inverse
:math:`(M^H M)^{-1}` (size :math:`k \times k`) and apply
:math:`\| P_M v \|^2 = w^H (M^H M)^{-1} w` with
:math:`w = M^H v \in \mathbb{C}^k`. Cost: :math:`O(N k + k^2)` per
sample, independent of :math:`N` for fixed :math:`k`.

Storage: the marker basis is stored as a single ``.pt`` artefact
(complex tensor of shape ``(N, k)``) at
``data.markers.marker_basis_path``, consistent with the
:mod:`vf_m7_phase_steganography` experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mriforge.infrastructure.calibration.score_registry import (
    register_calibration_score,
)

# Cache stores (basis, gram_inv) per resolved path. Both are small
# (N x k and k x k) so caching a few bases costs only a few MB.
_MARKER_BASIS_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _load_marker_basis_factors(
    path: str | Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load (basis, gram_inv) for the marker at ``path``, cached by path."""
    key = str(Path(path).resolve())
    cached = _MARKER_BASIS_CACHE.get(key)
    if cached is not None:
        basis, gram_inv = cached
        return basis.to(device), gram_inv.to(device)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Marker basis artefact not found at {path}. Generate it "
            f"with `python scripts/data/generate_marker_artefacts.py` "
            f"or pass an explicit path."
        )
    basis = torch.load(path, map_location="cpu", weights_only=True)
    if basis.ndim != 2:
        raise ValueError(f"Marker basis must be 2-D (N, k); got shape {tuple(basis.shape)}.")
    gram = basis.conj().T @ basis
    gram_inv = torch.linalg.inv(gram)
    _MARKER_BASIS_CACHE[key] = (basis, gram_inv)
    return basis.to(device), gram_inv.to(device)


def _load_marker_basis(path: str | Path, device: torch.device) -> torch.Tensor:
    """Back-compat helper: return the basis tensor."""
    basis, _ = _load_marker_basis_factors(path, device)
    return basis


def _projector_for_basis(path: str | Path, device: torch.device) -> torch.Tensor:
    r"""Materialise the dense projector :math:`P_M = M (M^H M)^{-1} M^H`.

    .. warning::

        Allocates an :math:`N \times N` complex tensor. At
        :math:`N = 65536` (M4Raw 256x256) this is :math:`32~\mathrm{GB}`
        and will OOM. Use only for small test bases; the production
        score path :func:`vf_residual_score` does **not** call this.
    """
    basis, gram_inv = _load_marker_basis_factors(path, device)
    return basis @ gram_inv @ basis.conj().T


def marker_basis_condition_number(path: str | Path) -> float:
    r"""Return :math:`\kappa(M^H M)` for the marker basis at ``path``.

    Used by the Tier-1 audit ``vf_residual_marker_matrix_well_conditioned``.
    """
    basis, _ = _load_marker_basis_factors(path, torch.device("cpu"))
    gram = basis.conj().T @ basis
    return float(torch.linalg.cond(gram).real.item())


@register_calibration_score("vf_residual")
def vf_residual_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    marker_basis_path: str | None = None,
    **_: Any,
) -> torch.Tensor:
    r"""Marker-subspace projection energy: :math:`s = \| P_M \hat{x} \|_2`.

    Computed implicitly as :math:`\| P_M v \|^2 = w^H (M^H M)^{-1} w`
    with :math:`w = M^H v`, avoiding the dense :math:`N \times N`
    projector.

    The ``target`` argument is accepted for signature compatibility with
    the registry's ``(prediction, target) -> tensor`` protocol but is
    IGNORED --- the whole point of the VF-residual score is that it does
    not need a clean reference.

    Args:
        prediction: Complex reconstruction tensor of shape
            ``(..., H, W)``. Flattened to ``(..., N)`` with
            ``N = H * W`` to apply the marker operator.
        target: ignored.
        marker_basis_path: filesystem path to the marker basis ``.pt``.

    Returns:
        Per-sample non-conformity score of shape equal to the leading
        batch dimensions of ``prediction``.
    """
    if marker_basis_path is None:
        raise ValueError(
            "vf_residual_score requires marker_basis_path. Pass it via "
            "the certification.conformal.score_kwargs block in YAML."
        )
    basis, gram_inv = _load_marker_basis_factors(marker_basis_path, prediction.device)
    flat = prediction.reshape(*prediction.shape[:-2], -1).to(basis.dtype)
    # w = M^H v as a batched matmul: (..., N) @ (N, k) -> (..., k)
    w = flat @ basis.conj()
    # quadratic form w^H gram_inv w = sum_i conj(w_i) (gram_inv w)_i
    gw = w @ gram_inv.T
    quad = (w.conj() * gw).sum(dim=-1).real
    return quad.clamp_min(0.0).sqrt()


__all__ = [
    "_load_marker_basis",
    "_load_marker_basis_factors",
    "_projector_for_basis",
    "marker_basis_condition_number",
    "vf_residual_score",
]
