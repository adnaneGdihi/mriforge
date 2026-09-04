r"""Reporting / analysis for the identified degradation operator (Proposal 1).

The headline figures called for by the implementation plan are produced here:

* the **recovered severity field** :math:`s(c,\beta)` — what severity the
  conditioner assigns to each catalog mode across the contrast vocabulary
  and the field coordinate :math:`\beta`; and
* the **commutator-interaction heatmap** :math:`\lVert[L_j,L_k]\rVert_F` —
  which generator pairs interact most strongly, i.e. where the
  Baker-Campbell-Hausdorff correction beyond first order actually matters.

The two computations are deliberately split:

* :func:`commutator_interaction_matrix` is *pure physics* — it depends only
  on the generator basis built from a ``mode_dictionary`` and is therefore
  reproducible without a trained model (handy as a static diagnostic of how
  non-commutative a catalog is before any fitting).
* :func:`severity_field` reads a trained
  :class:`~spectramr.models.operator_id.bch_conditioner.BCHOperatorConditioner`
  on a ``(contrast, field)`` grid.

Plotting is optional: the ``plot_*`` helpers import matplotlib lazily and
raise a clear error if it is unavailable, so importing this module never
pulls in a plotting backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from spectramr.infrastructure.physics.degradation_generators import build_generators
from spectramr.infrastructure.physics.magnus_exponential import commutator, dense_matrix

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectramr.models.operator_id.bch_conditioner import BCHOperatorConditioner

__all__ = [
    "commutator_interaction_matrix",
    "plot_commutator_heatmap",
    "plot_severity_field",
    "severity_field",
]


def commutator_interaction_matrix(
    mode_dictionary: list[str] | tuple[str, ...],
    height: int = 16,
    width: int = 16,
    *,
    normalize: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex128,
) -> tuple[torch.Tensor, list[str]]:
    r"""Frobenius norms of all pairwise commutators of the generator basis.

    For the deterministic generators :math:`\{L_k\}` of ``mode_dictionary``
    this materialises each :math:`[L_j,L_k]` densely (small ``height`` /
    ``width`` only — this is an off-line diagnostic, never a hot path) and
    returns the symmetric matrix :math:`M_{jk}=\lVert[L_j,L_k]\rVert_F`.

    Noise-kind modes carry no generator and are excluded; the returned label
    list names the deterministic modes in basis order.

    Args:
        mode_dictionary: Ordered mode names (the operator basis).
        height, width: Grid size for densifying the generators. Keep small
            (``N = H*W`` columns are materialised per operator).
        normalize: When true, divide by the largest off-diagonal entry so
            the heatmap reads as a relative-interaction map in ``[0, 1]``.
            A basis whose generators all commute returns an all-zero matrix
            (left un-normalised to avoid ``0/0``).
        device, dtype: Device and complex dtype for the dense materialisation.

    Returns:
        ``(matrix, det_modes)`` where ``matrix`` is a real ``[K, K]`` tensor
        (``K`` = number of deterministic modes) with a zero diagonal, and
        ``det_modes`` lists the corresponding mode names.
    """
    generators, det_modes, _noise = build_generators(
        list(mode_dictionary), height, width, device=device, dtype=dtype
    )
    k = len(generators)
    shape = (1, height, width)
    matrix = torch.zeros(k, k, dtype=torch.float64, device=device)
    for j in range(k):
        for kk in range(j + 1, k):
            comm = commutator(generators[j], generators[kk])
            dense = dense_matrix(comm, shape, dtype=dtype, device=device)
            norm = torch.linalg.matrix_norm(dense, ord="fro").real.to(torch.float64)
            matrix[j, kk] = norm
            matrix[kk, j] = norm
    if normalize:
        peak = matrix.max()
        if peak > 0:
            matrix = matrix / peak
    return matrix, det_modes


@torch.no_grad()
def severity_field(
    conditioner: BCHOperatorConditioner,
    *,
    contrast_indices: list[int] | tuple[int, ...] | None = None,
    field_coords: list[float] | tuple[float, ...] | torch.Tensor = (0.0,),
) -> torch.Tensor:
    r"""Recovered severity field :math:`s(c,\beta)` over a conditioning grid.

    Evaluates the conditioner at every ``(contrast, field)`` pair and stacks
    the per-mode severity heads.

    Args:
        conditioner: A trained (or freshly built) ``BCHOperatorConditioner``.
        contrast_indices: Contrast ids to sweep; defaults to the full
            vocabulary ``range(conditioner.num_contrasts)``.
        field_coords: Field coordinates :math:`\beta` to sweep (e.g.
            ``log10(field_strength_tesla)``).

    Returns:
        Real tensor ``[n_contrasts, n_fields, K]`` of non-negative
        severities, with ``K = conditioner.num_modes``.
    """
    if contrast_indices is None:
        contrast_indices = list(range(conditioner.num_contrasts))
    betas = torch.as_tensor(
        list(field_coords) if not isinstance(field_coords, torch.Tensor) else field_coords,
        dtype=torch.float32,
    ).reshape(-1)
    device = conditioner.basis_real.device

    rows = []
    for c in contrast_indices:
        cols = []
        for beta in betas:
            cidx = torch.full((1,), int(c), dtype=torch.long, device=device)
            fcoord = beta.reshape(1).to(device)
            out = conditioner(contrast_idx=cidx, field_coord=fcoord)
            cols.append(out["severities"].reshape(-1))
        rows.append(torch.stack(cols, dim=0))  # [n_fields, K]
    return torch.stack(rows, dim=0)  # [n_contrasts, n_fields, K]


# ── plotting (optional matplotlib) ────────────────────────────────────────
def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without mpl
        raise ImportError(
            "Plotting requires matplotlib. Install it (e.g. `pip install matplotlib`) "
            "or use the numeric helpers (commutator_interaction_matrix / severity_field)."
        ) from exc
    return plt


def plot_commutator_heatmap(
    mode_dictionary: list[str] | tuple[str, ...],
    *,
    height: int = 16,
    width: int = 16,
    ax: object | None = None,
):
    r"""Render the commutator-interaction heatmap :math:`\lVert[L_j,L_k]\rVert_F`.

    Returns the matplotlib ``Axes``. Diagonal is zero by construction.
    """
    plt = _require_matplotlib()
    matrix, labels = commutator_interaction_matrix(mode_dictionary, height, width)
    data = matrix.cpu().numpy()
    if ax is None:
        _fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 1.0, 1.4 * len(labels) + 1.0))
    im = ax.imshow(data, cmap="magma", vmin=0.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(r"Commutator interaction $\|[L_j,L_k]\|_F$")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_severity_field(
    conditioner: BCHOperatorConditioner,
    *,
    mode_names: list[str] | tuple[str, ...] | None = None,
    contrast_names: list[str] | tuple[str, ...] | None = None,
    field_coords: list[float] | tuple[float, ...] = (0.0,),
    ax: object | None = None,
):
    r"""Render the recovered severity field as a contrast-by-mode heatmap.

    Severities are averaged over the swept ``field_coords`` so the figure is
    a compact ``[n_contrasts, K]`` map; sweep a single field for a snapshot.
    Returns the matplotlib ``Axes``.
    """
    plt = _require_matplotlib()
    field = severity_field(conditioner, field_coords=field_coords)  # [C, F, K]
    data = field.mean(dim=1).cpu().numpy()  # [C, K]
    n_contrasts, k = data.shape
    modes = list(mode_names) if mode_names is not None else [f"mode_{i}" for i in range(k)]
    contrasts = (
        list(contrast_names)
        if contrast_names is not None
        else [f"c{i}" for i in range(n_contrasts)]
    )
    if ax is None:
        _fig, ax = plt.subplots(figsize=(1.4 * k + 1.0, 0.8 * n_contrasts + 1.0))
    im = ax.imshow(data, cmap="viridis", aspect="auto", vmin=0.0)
    ax.set_xticks(range(k))
    ax.set_yticks(range(n_contrasts))
    ax.set_xticklabels(modes, rotation=45, ha="right")
    ax.set_yticklabels(contrasts)
    ax.set_title(r"Recovered severity field $s(c,\beta)$")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax
