"""Discrete prolate spheroidal sequences (Slepian tapers), torch-only.

The discrete prolate spheroidal sequences (DPSS), also called Slepian
sequences, are the maximally band-limited / maximally time-concentrated
finite sequences of length ``n`` for a given time-half-bandwidth product
``W`` (the *bandwidth* argument here, in cycles/sample, with ``0 < W < 0.5``).

They are the eigenvectors of the real, symmetric *prolate* (sinc) kernel

.. math::

    B_{ij} = \\frac{\\sin\\!\\big(2\\pi W (i-j)\\big)}{\\pi (i-j)},
    \\qquad B_{ii} = 2W,

whose eigenvalues :math:`\\lambda_k \\in (0,1)` are the fraction of the
sequence's energy that lies inside the band :math:`[-W, W]`. The leading
eigenvalues cluster near 1 (well-concentrated tapers); the count above
:math:`1-\\epsilon` is approximately the Shannon number :math:`2NW`.

This helper is used by the SSEL no-reference spectral metric
(see ``core/metrics/nr_spectral.py``) to build a Slepian subspace and
measure off-band energy leakage. It depends on ``torch`` only — the
kernel is assembled and diagonalised with ``torch.linalg.eigh`` (the
kernel is symmetric, so eigendecomposition is real and stable).

References:
    D. Slepian, H. O. Pollak, "Prolate spheroidal wave functions, Fourier
    analysis and uncertainty—I," Bell Syst. Tech. J., 40(1), 1961.
    D. Slepian, "Prolate spheroidal wave functions, Fourier analysis, and
    uncertainty—V: The discrete case," Bell Syst. Tech. J., 57(5), 1978.
"""

from __future__ import annotations

import torch

__all__ = [
    "prolate_basis",
    "prolate_eigis",
    "prolate_kernel",
    "slepian_concentration",
]


def _validate(n: int, bandwidth: float, n_tapers: int) -> None:
    if n <= 1:
        raise ValueError(f"prolate: n must be > 1, got n={n}")
    if not (0.0 < bandwidth < 0.5):
        raise ValueError(
            f"prolate: bandwidth (time-half-bandwidth product W) must be in "
            f"the open interval (0, 0.5), got bandwidth={bandwidth}"
        )
    if n_tapers <= 0:
        raise ValueError(f"prolate: n_tapers must be >= 1, got n_tapers={n_tapers}")
    if n_tapers > n:
        raise ValueError(f"prolate: n_tapers ({n_tapers}) cannot exceed sequence length n ({n})")


def prolate_kernel(
    n: int,
    bandwidth: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Assemble the symmetric prolate (sinc) kernel matrix ``B`` of size ``[n, n]``.

    The off-diagonal entries are ``sin(2*pi*W*(i-j)) / (pi*(i-j))`` and the
    diagonal entries are ``2*W`` (the ``i == j`` limit of the sinc).

    Args:
        n: Sequence length (> 1).
        bandwidth: Time-half-bandwidth product ``W`` in ``(0, 0.5)``.
        device: Target device.
        dtype: Floating dtype (defaults to ``torch.float64`` for a stable
            eigensolve; callers may request ``float32``).

    Returns:
        Real symmetric kernel ``[n, n]``.
    """
    if n <= 1:
        raise ValueError(f"prolate: n must be > 1, got n={n}")
    if not (0.0 < bandwidth < 0.5):
        raise ValueError(
            f"prolate: bandwidth (time-half-bandwidth product W) must be in "
            f"the open interval (0, 0.5), got bandwidth={bandwidth}"
        )
    if dtype is None:
        dtype = torch.float64
    idx = torch.arange(n, device=device, dtype=dtype)
    diff = idx.unsqueeze(0) - idx.unsqueeze(1)  # [n, n], (i - j)
    pi = torch.pi
    two_w = 2.0 * bandwidth
    # sin(2*pi*W*d) / (pi*d) with the d == 0 limit replaced by 2W.
    numer = torch.sin(2.0 * pi * bandwidth * diff)
    denom = pi * diff
    off = torch.where(
        diff != 0,
        numer / torch.where(diff != 0, denom, torch.ones_like(denom)),
        torch.full_like(diff, two_w),
    )
    # Symmetrise defensively against floating round-off.
    return 0.5 * (off + off.transpose(-1, -2))


def prolate_eigis(
    n: int,
    bandwidth: float,
    n_tapers: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the leading DPSS tapers and their concentration eigenvalues.

    Args:
        n: Sequence length (> 1).
        bandwidth: Time-half-bandwidth product ``W`` in ``(0, 0.5)``.
        n_tapers: Number of leading tapers to return (``1 <= n_tapers <= n``).
        device: Target device.
        dtype: Floating dtype for the eigensolve (defaults to ``float64``).

    Returns:
        ``(eigvecs, eigvals)`` where ``eigvecs`` is ``[n_tapers, n]`` with
        orthonormal rows (the Slepian sequences) and ``eigvals`` is
        ``[n_tapers]`` of concentration ratios in ``(0, 1)``, both sorted in
        **descending** eigenvalue order.
    """
    _validate(n, bandwidth, n_tapers)
    solve_dtype = torch.float64 if dtype is None else dtype
    kernel = prolate_kernel(n, bandwidth, device=device, dtype=solve_dtype)
    # eigh returns ascending eigenvalues; flip to descending.
    evals, evecs = torch.linalg.eigh(kernel)
    evals = torch.flip(evals, dims=(0,))  # [n], descending
    evecs = torch.flip(evecs, dims=(1,))  # [n, n], columns reordered
    # Concentration ratios are bounded in [0, 1]; clamp tiny numerical spill.
    evals = evals.clamp(0.0, 1.0)
    top_vals = evals[:n_tapers].contiguous()
    top_vecs = evecs[:, :n_tapers].transpose(0, 1).contiguous()  # [n_tapers, n]
    # Canonical sign convention: make the largest-magnitude entry positive so
    # the basis is deterministic across platforms/BLAS backends.
    abs_max_idx = top_vecs.abs().argmax(dim=1)
    signs = torch.sign(top_vecs[torch.arange(n_tapers, device=top_vecs.device), abs_max_idx])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    top_vecs = top_vecs * signs.unsqueeze(1)
    if dtype is not None:
        top_vecs = top_vecs.to(dtype)
        top_vals = top_vals.to(dtype)
    return top_vecs, top_vals


def prolate_basis(
    n: int,
    bandwidth: float,
    n_tapers: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return ``[n_tapers, n]`` orthonormal 1-D DPSS sequences (descending λ).

    Convenience wrapper around :func:`prolate_eigis` returning only the
    eigenvectors.
    """
    vecs, _ = prolate_eigis(n, bandwidth, n_tapers, device=device, dtype=dtype)
    return vecs


def slepian_concentration(
    signal_1d: torch.Tensor,
    n_tapers: int,
    bandwidth: float,
) -> torch.Tensor:
    """Fraction of ``signal_1d`` energy captured by the leading DPSS subspace.

    Builds the leading ``n_tapers`` Slepian sequences of length
    ``len(signal_1d)`` and returns

    .. math::

        \\frac{\\sum_{i<K} |\\langle s, \\psi_i \\rangle|^2}{\\lVert s \\rVert_2^2}

    a scalar in ``[0, 1]``. A signal that is well band-limited to ``[-W, W]``
    projects almost entirely onto the leading tapers (ratio → 1), whereas a
    broadband / high-pass signal spills energy outside the subspace (ratio
    well below 1).

    Args:
        signal_1d: Real or complex 1-D tensor (shape ``[n]``).
        n_tapers: Size of the leading Slepian subspace.
        bandwidth: Time-half-bandwidth product ``W`` in ``(0, 0.5)``.

    Returns:
        Scalar tensor (0-dim) in ``[0, 1]``. Returns ``nan`` for a
        zero-energy signal.
    """
    if signal_1d.dim() != 1:
        raise ValueError(
            f"slepian_concentration expects a 1-D signal, got shape {tuple(signal_1d.shape)}"
        )
    n = signal_1d.shape[0]
    _validate(n, bandwidth, n_tapers)

    # Use float64 internally for the basis; cast to the signal's real precision.
    is_complex = torch.is_complex(signal_1d)
    real_dtype = (
        torch.float32
        if signal_1d.dtype in (torch.complex64, torch.float32, torch.float16)
        else torch.float64
    )
    basis = prolate_basis(
        n,
        bandwidth,
        n_tapers,
        device=signal_1d.device,
        dtype=real_dtype,
    )  # [n_tapers, n], real orthonormal

    total = (signal_1d.abs() ** 2).sum()
    if float(total) <= 0.0:
        return torch.tensor(float("nan"), device=signal_1d.device, dtype=real_dtype)

    if is_complex:
        basis_c = basis.to(signal_1d.dtype)
        coeffs = basis_c @ signal_1d  # [n_tapers], complex inner products
    else:
        sig = signal_1d.to(real_dtype)
        coeffs = basis @ sig  # [n_tapers], real

    captured = (coeffs.abs() ** 2).sum()
    ratio = captured / total
    return ratio.clamp(0.0, 1.0).to(real_dtype)
