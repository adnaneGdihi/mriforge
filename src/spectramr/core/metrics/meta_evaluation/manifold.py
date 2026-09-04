"""Manifold geometry helpers for LGDR.

Builds a k-nearest-neighbor graph in the encoder's latent space, computes
the diffusion-map eigenstructure (Coifman & Lafon 2006) at multiple times,
and provides Dijkstra-based geodesic distances on ``-log w`` weights.

We deliberately avoid scikit-learn / scipy dependencies and run everything
on plain torch — keeps the footprint minimal and lets the unit tests run
inside the project's CPU sandbox.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import torch


@dataclass
class ManifoldConfig:
    k_neighbors: int = 8
    epsilon: float | None = None  # Gaussian kernel bandwidth; auto if None
    n_eigen: int = 16
    diffusion_times: tuple[int, ...] = (1, 4, 16)


def _auto_epsilon(distances: torch.Tensor) -> float:
    """Set epsilon to the median of squared k-NN distances (Lafon 2006)."""
    finite = distances[torch.isfinite(distances) & (distances > 0)]
    if finite.numel() == 0:
        return 1.0
    return float(torch.median(finite).item())


def build_knn_graph(
    embeddings: torch.Tensor, config: ManifoldConfig
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Compute symmetric kNN graph in the embedding space.

    Returns
    -------
    weights : [N, N] sparse-like dense weight matrix.
    distances : [N, N] squared Euclidean distances (full).
    epsilon : the kernel bandwidth used.
    """
    n = embeddings.shape[0]
    if n == 0:
        return (
            torch.zeros((0, 0)),
            torch.zeros((0, 0)),
            1.0,
        )
    # Pairwise squared distances.
    sq = torch.cdist(embeddings.float(), embeddings.float(), p=2) ** 2
    # k-NN truncation per row.
    k = min(config.k_neighbors + 1, n)
    knn_vals, knn_idx = torch.topk(-sq, k=k, dim=1)
    knn_vals = -knn_vals  # smallest squared distances
    # Use the kth-NN distances (excluding self) as the bandwidth pool.
    pool = knn_vals[:, 1:].flatten() if k > 1 else knn_vals.flatten()
    epsilon = config.epsilon if config.epsilon is not None else _auto_epsilon(pool)
    epsilon = max(epsilon, 1e-9)

    weights = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        for jpos in range(k):
            j = int(knn_idx[i, jpos].item())
            if i == j:
                continue
            w = float(torch.exp(-knn_vals[i, jpos] / epsilon).item())
            weights[i, j] = max(weights[i, j].item(), w)
    weights = torch.maximum(weights, weights.t())
    return weights, sq, epsilon


def diffusion_eigenstructure(
    weights: torch.Tensor, n_eigen: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (eigenvalues, eigenvectors) of the row-stochastic transition.

    Symmetrized via ``D^{-1/2} W D^{-1/2}`` to allow stable real eigh.
    Returns eigenvalues sorted descending in absolute value (so lambda_0 ≈ 1
    comes first) and the corresponding right-eigenvectors of P (already
    converted from the symmetric basis).
    """
    n = weights.shape[0]
    if n == 0:
        return torch.zeros(0), torch.zeros(0, 0)
    deg = weights.sum(dim=1)
    deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    sym = deg_inv_sqrt.unsqueeze(1) * weights * deg_inv_sqrt.unsqueeze(0)
    eigvals, eigvecs = torch.linalg.eigh(sym)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    n_eigen = min(n_eigen, n)
    eigvals = eigvals[:n_eigen]
    eigvecs = eigvecs[:, :n_eigen]
    # Convert eigenvectors of the symmetric matrix to right-eigenvectors of
    # P = D^{-1} W: psi = D^{-1/2} v.
    psi = deg_inv_sqrt.unsqueeze(1) * eigvecs
    return eigvals, psi


def diffusion_distance(psi: torch.Tensor, eigvals: torch.Tensor, t: int) -> torch.Tensor:
    """[N, N] diffusion distance matrix at diffusion time ``t``.

    ``psi[:, 0]`` corresponds to the trivial eigenvalue 1 and is excluded.
    """
    if psi.shape[1] <= 1:
        return torch.zeros((psi.shape[0], psi.shape[0]), dtype=psi.dtype)
    coeff = (eigvals[1:] ** (2 * t)).unsqueeze(0)  # [1, k-1]
    proj = psi[:, 1:] * torch.sqrt(coeff)  # [N, k-1]
    sq = torch.cdist(proj, proj, p=2) ** 2
    return sq.clamp(min=0).sqrt()


def dijkstra_geodesics(weights: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """All-pairs shortest paths with edge cost ``-log w_ij``.

    Returns an [N, N] matrix; unreachable pairs get ``inf``.
    """
    n = weights.shape[0]
    if n == 0:
        return torch.zeros((0, 0))
    # Build adjacency.
    edges: list[list[tuple[float, int]]] = [[] for _ in range(n)]
    for i in range(n):
        nz = torch.nonzero(weights[i] > 0, as_tuple=False).flatten()
        for j in nz.tolist():
            if j == i:
                continue
            w = float(weights[i, j].item())
            edges[i].append((-float(torch.log(torch.tensor(max(w, eps))).item()), j))
    out = torch.full((n, n), float("inf"), dtype=torch.float64)
    for src in range(n):
        out[src, src] = 0.0
        heap: list[tuple[float, int]] = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > out[src, u].item():
                continue
            for cost, v in edges[u]:
                nd = d + cost
                if nd < out[src, v].item():
                    out[src, v] = nd
                    heapq.heappush(heap, (nd, v))
    return out
