r"""SPD-manifold field-transport harmonization (SPD-FT).

Several MRI estimands are manifold-valued -- diffusion tensors, structure
tensors, conductivity tensors, per-voxel quantitative covariances all live on the
cone :math:`\mathrm{Sym}^+(n)` -- and they are field-dependent. Euclidean
harmonization (ComBat) ignores the manifold and can return indefinite or
volumetrically swollen tensors. SPD-FT models the field effect
:math:`B_0\!\to\!B_0'` on a tissue's tensor as a congruence
:math:`\Sigma\mapsto E\,\Sigma\,E^\top` and harmonises by affine-invariant
transport with

.. math::

   E = \Sigma_{B_0'}^{1/2}\,\Sigma_{B_0}^{-1/2},

the real, PD-preserving transport operator that maps the source tissue mean
exactly onto the target mean (it equals the proposal's :math:`(\Sigma_{B_0'}\Sigma
_{B_0}^{-1})^{1/2}` whenever the two commute). The transport preserves
positive-definiteness, unlike Euclidean shifts that can leave the cone, and
reduces to a multiplicative (log-domain ComBat) scalar correction in the
commuting-scalar limit.

Reuses the affine-invariant geometry of
:class:`spectramr.infrastructure.physics.manifolds.spd_manifold.SPDManifold`.

References
----------
X. Pennec, P. Fillard, N. Ayache, "A Riemannian framework for tensor computing,"
*IJCV* 66(1):41-66, 2006.
O. Yair, M. Ben-Chen, R. Talmon, "Parallel transport on the cone manifold of SPD
matrices for domain adaptation," *IEEE TSP* 67(7):1797-1811, 2019.
"""

from __future__ import annotations

import torch


def _spd_power(matrix: torch.Tensor, power: float, eps: float = 1e-10) -> torch.Tensor:
    """Symmetric matrix power :math:`M^p` for an SPD matrix via eigendecomposition."""
    sym = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigvals, eigvecs = torch.linalg.eigh(sym)
    eigvals = eigvals.clamp_min(eps)
    return (eigvecs * eigvals.pow(power).unsqueeze(-2)) @ eigvecs.transpose(-1, -2)


def affine_invariant_transport_operator(
    sigma_source_mean: torch.Tensor,
    sigma_target_mean: torch.Tensor,
) -> torch.Tensor:
    r"""Transport operator :math:`E=\Sigma_{tgt}^{1/2}\Sigma_{src}^{-1/2}`.

    Satisfies :math:`E\,\Sigma_{src}\,E^\top = \Sigma_{tgt}` exactly, stays real,
    and preserves the cone.

    Args:
        sigma_source_mean: Source-field tissue mean tensor ``[n, n]`` (SPD).
        sigma_target_mean: Target-field tissue mean tensor ``[n, n]`` (SPD).

    Returns:
        Transport operator ``E [n, n]``.
    """
    return _spd_power(sigma_target_mean, 0.5) @ _spd_power(sigma_source_mean, -0.5)


def spd_field_transport(sigma: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    r"""Apply the congruence transport :math:`\Gamma(\Sigma)=E\,\Sigma\,E^\top`."""
    return e @ sigma @ e.transpose(-1, -2)


def estimate_transport_from_pairs(
    source_tensors: torch.Tensor,
    target_tensors: torch.Tensor,
) -> torch.Tensor:
    r"""Estimate :math:`E` from paired traveling/phantom measurements.

    The Procrustes/transport solution on the cone is the mean-matching operator
    :math:`E=\bar\Sigma_{tgt}^{1/2}\bar\Sigma_{src}^{-1/2}` evaluated at the
    arithmetic tissue means; it is identifiable up to the stabilizer of
    orthogonal conjugations commuting with the mean.

    Args:
        source_tensors: ``[N, n, n]`` SPD tensors at the source field.
        target_tensors: ``[N, n, n]`` paired SPD tensors at the target field.

    Returns:
        Transport operator ``E [n, n]``.

    Raises:
        ValueError: shape mismatch between the paired stacks.
    """
    if source_tensors.shape != target_tensors.shape:
        raise ValueError(
            f"paired stacks must match; got {tuple(source_tensors.shape)} vs "
            f"{tuple(target_tensors.shape)}."
        )
    return affine_invariant_transport_operator(
        source_tensors.mean(dim=0), target_tensors.mean(dim=0)
    )


def is_spd(matrix: torch.Tensor, tol: float = 0.0) -> bool:
    """Check that ``matrix`` is symmetric positive-definite (all eigenvalues > tol)."""
    sym = 0.5 * (matrix + matrix.transpose(-1, -2))
    return bool(torch.all(torch.linalg.eigvalsh(sym) > tol))


__all__ = [
    "affine_invariant_transport_operator",
    "estimate_transport_from_pairs",
    "is_spd",
    "spd_field_transport",
]
