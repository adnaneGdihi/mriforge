"""SVD-based coil compression for high-channel arrays.

Reduces a multi-coil k-space tensor of shape ``[B, C, H, W]`` (real or
complex) into a smaller virtual-coil set of shape ``[B, V, H, W]`` where
``V <= C``, by projecting along principal singular vectors estimated from
a low-frequency calibration region.

Two policies are exposed:

- ``SVDCoilCompressor`` (autograd-friendly, runtime use): differentiable,
  supports per-batch coil mixing matrices, never materialises the full
  flattened ``[C, H*W]`` matrix when ``H*W`` is large (uses Gram-matrix
  eigendecomposition fallback).

- ``estimate_compression_matrix`` (offline calibration): returns the
  compression matrix from a representative ACS region for caching.

Conventions:

- Complex k-space input: shape ``[B, C, H, W]`` complex64/128.
- Real interleaved k-space: shape ``[B, 2*C, H, W]`` real (the function
  detects the layout via ``torch.is_complex``).
- Output keeps the input dtype (complex stays complex, real stays real).

References:

- Buehrer, M. et al. (2007). "Array compression for MRI with large coil
  arrays." MRM 57(6):1131-1139.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _to_complex(t: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return (complex_view, was_real_interleaved)."""
    if torch.is_complex(t):
        return t, False
    if t.dim() >= 4 and t.shape[1] % 2 == 0:
        # Treat as interleaved real/imag pairs.
        real = t[:, 0::2]
        imag = t[:, 1::2]
        return torch.complex(real, imag), True
    raise ValueError(
        f"Unsupported tensor layout for coil compression: shape={tuple(t.shape)}, dtype={t.dtype}"
    )


def _from_complex(out_complex: torch.Tensor, was_real_interleaved: bool) -> torch.Tensor:
    """Convert back to interleaved real layout if the input was real."""
    if not was_real_interleaved:
        return out_complex
    real = out_complex.real
    imag = out_complex.imag
    return torch.stack([real, imag], dim=2).flatten(1, 2)


def estimate_compression_matrix(
    kspace: torch.Tensor,
    num_virtual_coils: int,
    acs_size: int | None = None,
) -> torch.Tensor:
    """Estimate a complex-valued ``[V, C]`` mixing matrix from k-space.

    Args:
        kspace: Multi-coil k-space ``[B, C, H, W]`` (complex) or
            ``[B, 2*C, H, W]`` (real interleaved).
        num_virtual_coils: Target number of virtual coils ``V``.
        acs_size: If set, restrict the calibration to a centred ``acs_size x acs_size``
            ACS window. Default: full FoV.

    Returns:
        Complex tensor of shape ``[V, C]`` with orthonormal rows.
    """
    k_c, _ = _to_complex(kspace)
    b, c, h, w = k_c.shape
    if num_virtual_coils > c:
        raise ValueError(f"num_virtual_coils ({num_virtual_coils}) must be <= C ({c})")

    if acs_size is not None and (acs_size < h or acs_size < w):
        cy, cx = h // 2, w // 2
        a_h = min(acs_size, h)
        a_w = min(acs_size, w)
        y0, x0 = cy - a_h // 2, cx - a_w // 2
        k_c = k_c[..., y0 : y0 + a_h, x0 : x0 + a_w]

    # Flatten to [B*N, C] where N = H*W of the (possibly cropped) calibration region.
    flat = k_c.permute(0, 2, 3, 1).reshape(-1, c)
    # Gram matrix [C, C] is small even when N is huge.
    gram = flat.conj().t() @ flat / max(flat.shape[0], 1)
    # Hermitian eigendecomp; eigvals ascending, take top V.
    eigvals, eigvecs = torch.linalg.eigh(gram)
    # Top V columns of eigvecs correspond to largest eigvals (which are at the end).
    top = eigvecs[:, -num_virtual_coils:]  # [C, V]
    return top.conj().t().contiguous()  # [V, C]


def apply_compression_matrix(
    kspace: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Project ``kspace`` through a precomputed ``[V, C]`` mixing matrix."""
    k_c, was_real = _to_complex(kspace)
    b, c, h, w = k_c.shape
    if matrix.shape[1] != c:
        raise ValueError(f"matrix shape {tuple(matrix.shape)} incompatible with C={c}")
    # einsum: bcyx, vc -> bvyx
    out = torch.einsum("bcyx,vc->bvyx", k_c, matrix.to(dtype=k_c.dtype, device=k_c.device))
    return _from_complex(out, was_real)


def fit_svd_basis(
    kspace: torch.Tensor, num_virtual_coils: int, calibration_lines: int | None = None
) -> torch.Tensor:
    """Fit a coil-compression basis ``(C, K)`` from k-space ``(B, C, H, W)``.

    Pure math: fit an energy-ordered virtual-coil basis from a calibration
    region. The basis is formed from the leading ``num_virtual_coils``
    **left**-singular vectors ``U`` of the ``(C, N)`` coil-by-sample matrix —
    these span the dominant coil subspace and are the energy-ordered virtual
    coils. (Naming the result ``V`` would be misleading: ``V`` conventionally
    denotes the *right*-singular matrix, which here spans sample space, not the
    coil basis we return.) The transform layer
    (``data/transforms/coil_compression.py``) wraps these.

    Args:
        kspace: Multi-coil k-space ``(B, C, H, W)`` (complex) or
            ``(B, 2*C, H, W)`` (real interleaved). Layout is detected via
            ``torch.is_complex`` and routed through ``_to_complex`` so the
            real/imag channels are never treated as independent coils.
        num_virtual_coils: Target number of virtual coils ``K``.
        calibration_lines: If set, restrict the calibration to a centred
            ``calibration_lines``-row band along H. Default: full FoV.

    Returns:
        Complex tensor ``(C, K)`` — the leading ``K`` left-singular vectors.

    Raises ValueError if ``num_virtual_coils > C`` or the layout is unsupported.
    """
    if kspace.dim() != 4:
        raise ValueError(f"Expected 4D (B,C,H,W) kspace, got {kspace.dim()}D")
    k_c, _was_real = _to_complex(kspace)
    _b, c, h, _w = k_c.shape
    if num_virtual_coils > c:
        raise ValueError(f"num_virtual_coils={num_virtual_coils} exceeds n_coils={c}")
    src = k_c
    if calibration_lines is not None and calibration_lines < h:
        c0 = h // 2 - calibration_lines // 2
        src = k_c[:, :, c0 : c0 + calibration_lines, :]
    # Coil x sample matrix, SVD → left-singular vectors (U) are the virtual coils.
    # Concatenate all batch samples along the sample axis rather than averaging
    # over the batch: different samples carry different global phase, so a mean
    # partially cancels signal and biases the coil basis. The (C, B·N) matrix has
    # the same left-singular vectors as the Gram accumulation over every sample.
    x = src.movedim(1, 0).reshape(c, -1)  # (C, B·N)
    u, _s, _vh = torch.linalg.svd(x, full_matrices=False)  # u: (C, C)
    return u[:, :num_virtual_coils]  # (C, K)


def apply_svd_compression(kspace: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project ``(B, C, H, W)`` onto ``basis`` ``(C, K)`` → ``(B, K, H, W)``.

    Accepts complex ``(B, C, H, W)`` or real-interleaved ``(B, 2*C, H, W)``
    input (layout detected via ``torch.is_complex``, routed through
    ``_to_complex``/``_from_complex``) so real/imag channels are never mixed
    into independent coils. Output keeps the input dtype.

    Raises ValueError if ``basis.shape[0] != C`` (mirrors the coil-count guard
    in ``apply_compression_matrix``) so a mismatched basis yields a clear
    contract-violation message instead of a cryptic einsum broadcast error.
    """
    k_c, was_real = _to_complex(kspace)
    b, c, h, w = k_c.shape
    if basis.shape[0] != c:
        raise ValueError(
            f"basis shape {tuple(basis.shape)} incompatible with C={c}: "
            f"expected basis.shape[0] == {c}"
        )
    x = k_c.reshape(b, c, -1)  # (B, C, N)
    comp = torch.einsum(
        "ck,bcn->bkn", basis.conj().to(dtype=k_c.dtype, device=k_c.device), x
    )  # (B, K, N)
    out = comp.reshape(b, basis.shape[1], h, w)
    return _from_complex(out, was_real)


class SVDCoilCompressor(nn.Module):
    """Differentiable, batched SVD-based coil compressor.

    For each batch element, estimates a per-sample compression matrix
    from the central ACS window, then applies it. Useful as a drop-in
    preprocessing layer in front of any reconstruction network when
    coil count varies across the dataset.

    Args:
        num_virtual_coils: target virtual coil count ``V``.
        acs_size: side of the calibration window (default 24).
        per_batch: if True, estimate one matrix per batch element. If
            False, average across the batch first (faster, requires that
            all batch elements share a common coil basis).
    """

    def __init__(
        self,
        num_virtual_coils: int,
        acs_size: int = 24,
        per_batch: bool = True,
    ) -> None:
        super().__init__()
        if num_virtual_coils < 1:
            raise ValueError(f"num_virtual_coils must be >= 1, got {num_virtual_coils}")
        if acs_size < 4:
            raise ValueError(f"acs_size must be >= 4, got {acs_size}")
        self.num_virtual_coils = int(num_virtual_coils)
        self.acs_size = int(acs_size)
        self.per_batch = per_batch

    def forward(self, kspace: torch.Tensor) -> torch.Tensor:
        if self.per_batch:
            outs = []
            for i in range(kspace.shape[0]):
                slice_i = kspace[i : i + 1]
                mat = estimate_compression_matrix(slice_i, self.num_virtual_coils, self.acs_size)
                outs.append(apply_compression_matrix(slice_i, mat))
            return torch.cat(outs, dim=0)
        # Shared matrix across batch.
        mat = estimate_compression_matrix(kspace, self.num_virtual_coils, self.acs_size)
        return apply_compression_matrix(kspace, mat)


__all__ = [
    "SVDCoilCompressor",
    "apply_compression_matrix",
    "apply_svd_compression",
    "estimate_compression_matrix",
    "fit_svd_basis",
]
