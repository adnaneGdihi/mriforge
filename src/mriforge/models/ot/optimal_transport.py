"""Optimal Transport Module for Unpaired MRI Translation.

This module implements Optimal Transport (OT) algorithms for unpaired
image-to-image translation, enabling training without paired datasets.

Key Components:
- SinkhornDistance: Entropic regularized Wasserstein distance
- DynamicOTFlow: Benamou-Brenier formulation for continuous transport
- KIDOTLoss: Knowledge-Informed Dynamic OT with physics constraints

References:
- Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport.
- Benamou, J.D. & Brenier, Y. (2000). A computational fluid mechanics solution to the OT problem.
"""

import torch
import torch.nn as nn


def sinkhorn_knopp(
    cost_matrix: torch.Tensor,
    reg: float = 0.1,
    max_iter: int = 100,
    stop_thr: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sinkhorn-Knopp algorithm for entropic regularized OT.

    Computes the optimal transport plan between two discrete distributions
    using the Sinkhorn algorithm with entropic regularization.

    Args:
        cost_matrix: [N, M] pairwise cost matrix C(x_i, y_j)
        reg: Entropic regularization parameter (ε)
        max_iter: Maximum iterations
        stop_thr: Stopping threshold for convergence

    Returns:
        transport_plan: [N, M] optimal coupling matrix
        wasserstein_dist: Scalar Wasserstein distance
    """
    N, M = cost_matrix.shape
    device = cost_matrix.device
    dtype = cost_matrix.dtype

    # Uniform marginals (can be extended to support arbitrary marginals)
    mu = torch.ones(N, device=device, dtype=dtype) / N
    nu = torch.ones(M, device=device, dtype=dtype) / M

    # Gibbs kernel
    K = torch.exp(-cost_matrix / reg)

    # Initialize scaling vectors
    u = torch.ones(N, device=device, dtype=dtype)
    v = torch.ones(M, device=device, dtype=dtype)

    for _ in range(max_iter):
        u_prev = u.clone()

        # Sinkhorn iterations
        u = mu / (K @ v + 1e-10)
        v = nu / (K.T @ u + 1e-10)

        # Check convergence
        err = (u - u_prev).abs().max()
        if err < stop_thr:
            break

    # Optimal transport plan
    transport_plan = torch.diag(u) @ K @ torch.diag(v)

    # Wasserstein distance
    wasserstein_dist = (transport_plan * cost_matrix).sum()

    return transport_plan, wasserstein_dist


def entropic_gromov_wasserstein(
    c1: torch.Tensor,
    c2: torch.Tensor,
    reg: float = 0.05,
    max_iter: int = 50,
    gw_iter: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Entropic Gromov-Wasserstein coupling (squared loss), Peyre-Cuturi 2016 Alg. 1.

    Couples two metric spaces by their INTRA-domain distance structures (no cross-domain cost):

    .. math::
        GW(C_1, C_2) = \min_T \sum_{i,k,j,l} (C_1[i,k] - C_2[j,l])^2\, T_{ij} T_{kl}

    The outer loop linearises the quadratic objective at the current coupling and solves an
    entropic OT sub-problem with :func:`sinkhorn_knopp` (uniform marginals ``p=1/n, q=1/m`` —
    consistent with the inner solver). For the squared loss the linearised cost factorises as
    ``constC - 2 C1 T C2`` with ``constC[i,j] = (C1^2 p)[i] + (C2^2 q)[j]`` (``C2`` symmetric).

    Args:
        c1: ``(n, n)`` symmetric intra-domain distance matrix of the source descriptors.
        c2: ``(m, m)`` symmetric intra-domain distance matrix of the target descriptors.
        reg: Sinkhorn entropic regularisation.
        max_iter: inner Sinkhorn iterations.
        gw_iter: outer GW linearisation iterations.

    Returns:
        ``(coupling T [n, m], gw_value scalar)`` — the GW objective at the final coupling.
    """
    if c1.ndim != 2 or c1.shape[0] != c1.shape[1]:
        raise ValueError(f"c1 must be square (n,n); got {tuple(c1.shape)}.")
    if c2.ndim != 2 or c2.shape[0] != c2.shape[1]:
        raise ValueError(f"c2 must be square (m,m); got {tuple(c2.shape)}.")
    n, m = c1.shape[0], c2.shape[0]
    device, dtype = c1.device, c1.dtype
    p = torch.ones(n, device=device, dtype=dtype) / n
    q = torch.ones(m, device=device, dtype=dtype) / m
    const_c = (c1.pow(2) @ p)[:, None] + (c2.pow(2) @ q)[None, :]  # (n, m), T-independent
    t = p[:, None] * q[None, :]  # independent-coupling init
    for _ in range(gw_iter):
        cost = const_c - 2.0 * (c1 @ t @ c2)  # linearised GW cost at current t
        t, _ = sinkhorn_knopp(cost, reg=reg, max_iter=max_iter)
    gw_value = ((const_c - 2.0 * (c1 @ t @ c2)) * t).sum()
    return t, gw_value


def _gw_biased_value(
    feat_a: torch.Tensor, feat_b: torch.Tensor, reg: float, max_iter: int, gw_iter: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Biased entropic-GW value (differentiable in the descriptors) + coupling determinism.

    Builds normalised intra-domain squared-euclidean distance matrices, solves the entropic GW
    coupling ``T`` on the DETACHED matrices, then evaluates the GW value at the fixed ``T`` —
    differentiable in the descriptors (envelope theorem: only the transport COST carries
    gradients, not the Sinkhorn convergence history; Genevay 2018 / Titouan 2019). Per-matrix
    max-normalisation keeps the Gibbs kernel ``exp(-C/reg)`` from under/overflowing across scales.
    """

    def _norm_cdist(f: torch.Tensor) -> torch.Tensor:
        c = torch.cdist(f, f).pow(2)
        return c / (c.max() + 1e-8)

    c1 = _norm_cdist(feat_a)
    c2 = _norm_cdist(feat_b)
    t, _ = entropic_gromov_wasserstein(c1.detach(), c2.detach(), reg, max_iter, gw_iter)
    t = t.detach()
    a = t.sum(1)
    b = t.sum(0)
    value = (
        (c1.pow(2) * torch.outer(a, a)).sum()
        - 2.0 * (c1 * (t @ c2 @ t.t())).sum()
        + (c2.pow(2) * torch.outer(b, b)).sum()
    )
    n, m = t.shape
    uniform = 1.0 / (n * m)
    determinism = (t - uniform).abs().sum() / (2.0 * (1.0 - uniform))
    return value, determinism.detach()


def gromov_wasserstein_feature_loss(
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    reg: float = 0.05,
    max_iter: int = 50,
    gw_iter: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Differentiable DEBIASED Gromov-Wasserstein structure distance between two descriptor sets.

    Returns the Sinkhorn-GW divergence (Genevay 2018)::

        GW_div(A, B) = GW(A, B) - 0.5 GW(A, A) - 0.5 GW(B, B)

    The debiasing removes the entropic bias of the raw value (the biased ``GW(A,A)`` is a large,
    input-dependent floor — empirically it can even exceed ``GW(A, unrelated)``, so the raw value
    is not a usable distance). The divergence is ``0`` for identical/relabeled structures and
    grows with genuine structural (intra-domain geometry) mismatch.

    .. important::
       Gromov-Wasserstein matches INTRA-domain distance structure and is therefore invariant to
       isometries (translations / rotations / reflections / relabelings) of the descriptor cloud.
       This is the right inductive bias for UNPAIRED matching across spaces with no shared
       coordinate frame, but makes GW UNSUITABLE for paired, registered, same-grid image
       translation: a spatially DISPLACED structure (e.g. anatomy moved across the FOV) is an
       isometry, so GW scores it ~0 — it cannot penalise displacement, and on a shared grid the
       position component is identical across domains (degenerate). Reserve this primitive for
       genuinely unpaired distribution-level coupling.

    Args:
        feat_a: ``(n, d)`` first descriptor set.
        feat_b: ``(m, d)`` second descriptor set.

    Returns:
        ``(gw_divergence scalar, coupling_determinism scalar)`` — determinism in ``[0, 1]`` is how
        far the A-B coupling is from the degenerate uniform plan (``0`` = uniform, larger = peaked
        structured matching). It is the mechanism-fires witness (coupling ENTROPY is NOT valid —
        the uniform plan has MAX entropy).
    """
    if feat_a.ndim != 2 or feat_b.ndim != 2:
        raise ValueError(
            f"feat_a/feat_b must be (n,d)/(m,d); got {tuple(feat_a.shape)}, {tuple(feat_b.shape)}."
        )
    gw_ab, determinism = _gw_biased_value(feat_a, feat_b, reg, max_iter, gw_iter)
    gw_aa, _ = _gw_biased_value(feat_a, feat_a, reg, max_iter, gw_iter)
    gw_bb, _ = _gw_biased_value(feat_b, feat_b, reg, max_iter, gw_iter)
    return gw_ab - 0.5 * gw_aa - 0.5 * gw_bb, determinism


class VelocityFieldNetwork(nn.Module):
    """Neural network to parameterize velocity field v(t, x).

    For the dynamic OT formulation, we learn a time-conditioned
    velocity field that transports mass from source to target.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            hidden_dim (int): Description.
            num_layers (int): Description.
        """
        super().__init__()

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        layers = []
        for i in range(num_layers):
            in_dim = in_channels + hidden_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else in_channels
            layers.append(nn.Conv2d(in_dim, out_dim, 3, padding=1))
            if i < num_layers - 1:
                layers.append(nn.GroupNorm(8, out_dim))
                layers.append(nn.SiLU())

        self.net = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute velocity field at position x and time t.

        Args:
            x: [B, C, H, W] spatial position
            t: [B] or scalar time in [0, 1]

        Returns:
            v: [B, C, H, W] velocity field

        forward method for VelocityFieldNetwork.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # Time embedding
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(B)
        t_emb = self.time_embed(t.unsqueeze(-1))  # [B, hidden]
        t_emb = t_emb[:, :, None, None].expand(-1, -1, H, W)

        # Concatenate and process
        h = torch.cat([x, t_emb], dim=1)

        for layer in self.net:
            h = layer(h)

        return h


# Module exports
# ``SinkhornDistance``, ``DynamicOTFlow``, ``KIDOTLoss`` moved to
# ``src/models/losses/optimal_transport_losses.py`` 2026-05-14 per
# ``TODO/backlog_ssot_and_layering_cleanup.md`` Phase 5.
__all__ = [
    "VelocityFieldNetwork",
    "entropic_gromov_wasserstein",
    "gromov_wasserstein_feature_loss",
    "sinkhorn_knopp",
]
