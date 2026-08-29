r"""Cellular-sheaf cohomology for k-space completion & multi-contrast.

A *cellular sheaf* :math:`\mathcal{F}` on a graph :math:`G=(V,E)` assigns

* to each vertex :math:`v` a stalk :math:`\mathcal{F}(v)=\mathbb{R}^{N_v}`,
* to each edge :math:`e=(u,v)` a pair of restriction maps
  :math:`\mathcal{F}_{u\trianglelefteq e},\mathcal{F}_{v\trianglelefteq e}`,
  acting on the stalks.

The **sheaf coboundary**
:math:`\delta_{\mathcal{F}}: C^0(G;\mathcal{F})\to C^1(G;\mathcal{F})`

.. math::

    (\delta_{\mathcal{F}}\,x)_e = \mathcal{F}_{v\trianglelefteq e}\,x_v
                                 - \mathcal{F}_{u\trianglelefteq e}\,x_u

and the **sheaf Laplacian** :math:`L_{\mathcal{F}}=\delta_{\mathcal{F}}^{\top}\delta_{\mathcal{F}}`
[Hansen-Ghrist 2019] are the algebraic objects that detect *globally
inconsistent* sections: :math:`\langle x,L_{\mathcal{F}}x\rangle=0` iff
:math:`x\in H^0(G;\mathcal{F})` (the space of global sections).

Two applications:

* **Sheaf k-space completion** (Batch-1 #3) — Hankel-low-rank pairwise
  consistency on overlapping k-space patches; the Mayer-Vietoris sequence
  exposes the obstruction class in :math:`H^1` as a defensibility flag.
* **Multi-contrast sheaf regularisation** (Batch-2 #4) — contrasts are
  vertices; voxel-wise FiLM-style restriction maps cross-condition the
  reconstruction; :math:`H^1` localises *which contrast pair disagrees*.

Exports:

* :func:`sheaf_coboundary` — apply :math:`\delta_{\mathcal{F}}` given edge-restriction maps.
* :func:`sheaf_laplacian_quadratic_form` — :math:`\langle x,L_{\mathcal{F}}x\rangle`.
* :func:`global_sections_rank` — :math:`\dim H^0(G;\mathcal{F})` via the
  kernel of :math:`L_{\mathcal{F}}`.
* :func:`mayer_vietoris_obstruction_norm` — quantitative
  :math:`\|[w]\|_{H^1}` for a two-patch cover.
"""

from __future__ import annotations

import torch

__all__ = [
    "global_sections_rank",
    "mayer_vietoris_obstruction_norm",
    "sheaf_coboundary",
    "sheaf_laplacian_quadratic_form",
]


def sheaf_coboundary(
    vertex_signals: torch.Tensor,
    edges: torch.Tensor,
    restriction_left: torch.Tensor,
    restriction_right: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate :math:`(\delta_{\mathcal{F}}\,x)_e=\mathcal{F}_{v\trianglelefteq e}x_v-\mathcal{F}_{u\trianglelefteq e}x_u`.

    Args:
        vertex_signals: ``(V, n)`` stalk values per vertex.
        edges: ``(E, 2)`` long tensor of (u, v) vertex pairs.
        restriction_left: ``(E, n, n)`` restriction map
            :math:`\mathcal{F}_{u\trianglelefteq e}` (applied to the
            *first* endpoint of each edge).
        restriction_right: ``(E, n, n)`` restriction map
            :math:`\mathcal{F}_{v\trianglelefteq e}` (applied to the
            *second* endpoint).

    Returns:
        ``(E, n)`` edge-cochain :math:`\delta_{\mathcal{F}}\,x`.
    """
    if vertex_signals.ndim != 2:
        raise ValueError(f"vertex_signals must be (V, n); got {tuple(vertex_signals.shape)}.")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must be (E, 2); got {tuple(edges.shape)}.")
    if restriction_left.shape != restriction_right.shape:
        raise ValueError("restriction_left and restriction_right must have same shape.")
    n = vertex_signals.shape[1]
    if restriction_left.shape != (edges.shape[0], n, n):
        raise ValueError(
            f"restriction maps must be (E, n, n) = ({edges.shape[0]}, {n}, {n}); "
            f"got {tuple(restriction_left.shape)}."
        )
    x_u = vertex_signals[edges[:, 0]]  # (E, n)
    x_v = vertex_signals[edges[:, 1]]  # (E, n)
    return torch.einsum("eij,ej->ei", restriction_right, x_v) - torch.einsum(
        "eij,ej->ei", restriction_left, x_u
    )


def sheaf_laplacian_quadratic_form(
    vertex_signals: torch.Tensor,
    edges: torch.Tensor,
    restriction_left: torch.Tensor,
    restriction_right: torch.Tensor,
) -> torch.Tensor:
    r"""Compute :math:`\langle x, L_{\mathcal{F}}\,x\rangle=\sum_e\|(\delta_{\mathcal{F}}\,x)_e\|^2`.

    Zero iff the section :math:`x` is globally compatible under the
    restriction maps — i.e. :math:`x\in H^0(G;\mathcal{F})`.
    """
    coboundary = sheaf_coboundary(vertex_signals, edges, restriction_left, restriction_right)
    return coboundary.pow(2).sum()


def global_sections_rank(
    edges: torch.Tensor,
    restriction_left: torch.Tensor,
    restriction_right: torch.Tensor,
    num_vertices: int,
    eps: float = 1e-6,
) -> int:
    r"""Compute :math:`\dim H^0(G;\mathcal{F})` as the kernel rank of :math:`L_{\mathcal{F}}`.

    Builds the dense block-Laplacian :math:`L_{\mathcal{F}}\in\mathbb{R}^{Vn\times Vn}`
    and returns the count of near-zero eigenvalues. For graphs with
    :math:`V\le 64` this is the practical default; for larger graphs
    consider a Krylov approach (not implemented here).
    """
    n = restriction_left.shape[-1]
    laplacian = torch.zeros(num_vertices * n, num_vertices * n, dtype=restriction_left.dtype)
    # Build block-Laplacian: L_{uu} += R_l^T R_l, L_{vv} += R_r^T R_r,
    # L_{uv} -= R_l^T R_r, L_{vu} -= R_r^T R_l (symmetric).
    for e_idx in range(edges.shape[0]):
        u, v = int(edges[e_idx, 0].item()), int(edges[e_idx, 1].item())
        r_l = restriction_left[e_idx]
        r_r = restriction_right[e_idx]
        ll = r_l.T @ r_l
        rr = r_r.T @ r_r
        lr = r_l.T @ r_r
        laplacian[u * n : (u + 1) * n, u * n : (u + 1) * n] += ll
        laplacian[v * n : (v + 1) * n, v * n : (v + 1) * n] += rr
        laplacian[u * n : (u + 1) * n, v * n : (v + 1) * n] -= lr
        laplacian[v * n : (v + 1) * n, u * n : (u + 1) * n] -= lr.T
    eigs = torch.linalg.eigvalsh(laplacian)
    return int((eigs.abs() < eps).sum().item())


def mayer_vietoris_obstruction_norm(
    patch_a: torch.Tensor,
    patch_b: torch.Tensor,
    overlap_a: torch.Tensor,
    overlap_b: torch.Tensor,
) -> torch.Tensor:
    r"""Quantitative obstruction :math:`\|[w]\|_{H^1}` on a two-patch cover.

    For a cover :math:`\Omega=U_\alpha\cup U_\beta` of an Abelian sheaf,
    the Mayer-Vietoris sequence yields the exact

    .. math::

        H^0(\Omega) \to H^0(U_\alpha)\oplus H^0(U_\beta)
        \to H^0(U_\alpha\cap U_\beta) \to H^1(\Omega) \to 0

    so the obstruction to glueing :math:`(s_\alpha,s_\beta)` into a global
    section is the residual on :math:`U_\alpha\cap U_\beta`:

    .. math::

        \|[w]\|_{H^1} = \|\mathrm{overlap}_a - \mathrm{overlap}_b\|_2.

    Args:
        patch_a: ``(N_a,)`` section on :math:`U_\alpha`.
        patch_b: ``(N_b,)`` section on :math:`U_\beta`.
        overlap_a: ``(N_{ab},)`` restriction of ``patch_a`` to the overlap.
        overlap_b: ``(N_{ab},)`` restriction of ``patch_b`` to the overlap.

    Returns:
        Scalar :math:`\|[w]\|_{H^1}\ge 0`. Zero iff the two sections agree on
        the overlap — i.e. they patch to a global section.

    Notes:
        The ``patch_a`` / ``patch_b`` arguments are reserved for downstream
        diagnostics (e.g. to flag *which* patch is responsible for a large
        residual). They are not used in the scalar return value.
    """
    if overlap_a.shape != overlap_b.shape:
        raise ValueError(
            f"overlap_a {tuple(overlap_a.shape)} and overlap_b "
            f"{tuple(overlap_b.shape)} must have identical shape."
        )
    _ = patch_a  # acknowledge — used only for diagnostic API symmetry
    _ = patch_b
    return (overlap_a - overlap_b).reshape(-1).norm(p=2)
