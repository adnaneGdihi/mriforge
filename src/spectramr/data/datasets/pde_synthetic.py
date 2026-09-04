"""Synthetic PDE benchmarks for neural-operator training.

Two operator-learning problems shipped here:

* **Burgers 1-D** (viscous Burgers' equation):

  .. math::

      \\partial_t u + u\\,\\partial_x u = \\nu\\,\\partial_{xx} u,
      \\qquad x \\in (0, 1),\\ t \\in (0, T]

  The dataset maps ``a(x) := u(x, 0)`` (initial condition) to
  ``u(x, T)`` (solution at the final time). The reference implementation
  uses a Fourier pseudo-spectral scheme with explicit Heun integration.

* **Darcy 2-D** (steady-state Darcy flow):

  .. math::

      -\\nabla\\cdot\\bigl(a(x,y)\\,\\nabla u(x,y)\\bigr) = f(x,y),
      \\qquad (x,y) \\in (0,1)^2,\\quad u|_{\\partial\\Omega} = 0

  Maps the permeability field ``a(x, y)`` to the solution ``u(x, y)``.
  The reference implementation uses a 5-point finite-difference
  discretisation and ``scipy.sparse.linalg.spsolve``.

Inputs are sampled from a Gaussian-random-field (GRF) prior with
fixed length scale, which is the standard PDEBench / FNO convention.
A constant random seed makes generated datasets reproducible across
runs without storing them on disk.

This module is self-contained: no external data downloads required.
The generators are deliberately compact (~30 LOC each) so the focus
stays on neural-operator training, not numerical PDE engineering.
For more complex problems (NS 2-D vorticity-streamfunction,
isotropic turbulence 256³), use real PDEBench / JHTDB shards via a
future preprocessed-NPY adapter.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["BurgersDataset", "DarcyDataset", "make_pde_dataset"]


# ── Gaussian random field sampler (shared by Burgers + Darcy) ────────


def _grf_2d(
    n: int,
    length_scale: float = 0.1,
    seed: int = 0,
    smoothness: float = 4.0,
) -> np.ndarray:
    """Sample a 2-D Gaussian random field on an n×n grid.

    Implementation: white noise convolved with a Matérn-style spectral
    filter ``(1 + ‖k‖² · ℓ²)^{-(smoothness+1)/2}``. ``length_scale``
    controls the correlation distance; ``smoothness`` controls
    differentiability (4.0 ≈ infinitely differentiable).
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, n)).astype(np.float32)
    f = np.fft.fft2(noise)

    # Build the spectral filter.
    kx = np.fft.fftfreq(n, d=1.0 / n)
    ky = np.fft.fftfreq(n, d=1.0 / n)
    kxx, kyy = np.meshgrid(kx, ky, indexing="ij")
    kmag2 = kxx**2 + kyy**2
    spec = (1.0 + kmag2 * length_scale**2) ** (-(smoothness + 1) / 2.0)

    f *= spec
    field = np.real(np.fft.ifft2(f)).astype(np.float32)
    # Normalise to unit variance.
    field = (field - field.mean()) / (field.std() + 1e-12)
    return field


def _grf_1d(
    n: int,
    length_scale: float = 0.05,
    seed: int = 0,
    smoothness: float = 4.0,
) -> np.ndarray:
    """1-D analogue of ``_grf_2d`` for Burgers initial conditions."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n).astype(np.float32)
    f = np.fft.fft(noise)
    k = np.fft.fftfreq(n, d=1.0 / n)
    spec = (1.0 + (k * length_scale) ** 2) ** (-(smoothness + 1) / 2.0)
    f *= spec
    field = np.real(np.fft.ifft(f)).astype(np.float32)
    field = (field - field.mean()) / (field.std() + 1e-12)
    return field


# ── Burgers 1-D solver (Fourier pseudo-spectral, explicit Heun) ──────


def _burgers_solve(
    u0: np.ndarray, nu: float = 1e-3, T: float = 1.0, n_steps: int = 500
) -> np.ndarray:
    """Evolve viscous Burgers from ``u0`` over [0, T].

    Pseudo-spectral derivatives, explicit Heun (RK2) time-stepping
    with 2/3 dealiasing. Sufficient for training-grade data; not a
    research-grade PDE solver.
    """
    n = u0.shape[0]
    k = np.fft.fftfreq(n, d=1.0 / n) * 2 * np.pi  # angular wavenumbers
    ik = 1j * k
    k2 = k**2

    # 2/3-rule dealiasing mask.
    cutoff = int(n / 3)
    mask = np.ones(n, dtype=np.float32)
    mask[cutoff : n - cutoff] = 0.0

    def rhs(uh: np.ndarray) -> np.ndarray:
        u = np.real(np.fft.ifft(uh))
        ux = np.real(np.fft.ifft(ik * uh))
        nonlin = u * ux
        nonlin_h = np.fft.fft(nonlin) * mask
        return -nonlin_h - nu * k2 * uh

    dt = T / n_steps
    uh = np.fft.fft(u0)
    for _ in range(n_steps):
        k1 = rhs(uh)
        k2_ = rhs(uh + dt * k1)
        uh = uh + 0.5 * dt * (k1 + k2_)
    return np.real(np.fft.ifft(uh)).astype(np.float32)


# ── Darcy 2-D solver (5-point FD + scipy sparse) ────────────────────


def _darcy_solve(a: np.ndarray, f_const: float = 1.0) -> np.ndarray:
    """Solve -∇·(a∇u) = f on (0,1)² with zero Dirichlet BC.

    5-point conservative finite difference; sparse direct solve. ``a``
    is the permeability, lifted to be strictly positive; ``f`` is a
    spatially-constant forcing of magnitude ``f_const``.
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    n = a.shape[0]
    h = 1.0 / (n - 1)

    # Lift permeability to be strictly positive: a ∈ [0.1, ~]
    a_pos = np.exp(a)  # log-normal permeability — standard FNO Darcy

    # Interior nodes only — boundaries pinned to 0.
    n_int = n - 2
    N = n_int * n_int

    # Build the sparse matrix row-by-row using the conservative 5-point
    # scheme: -∇·(a∇u) ≈ -[(a_{i+½,j}(u_{i+1,j} - u_{i,j})
    #                       - a_{i-½,j}(u_{i,j} - u_{i-1,j})) / h²
    #                    + same for j].
    # Edge-centred a is computed by simple averaging.

    diag = np.zeros(N, dtype=np.float64)
    east = np.zeros(N - 1, dtype=np.float64)
    west = np.zeros(N - 1, dtype=np.float64)
    north = np.zeros(N - n_int, dtype=np.float64)
    south = np.zeros(N - n_int, dtype=np.float64)

    for i in range(n_int):
        for j in range(n_int):
            ii = i + 1  # global index
            jj = j + 1
            n_idx = i * n_int + j
            a_e = 0.5 * (a_pos[ii, jj] + a_pos[ii, jj + 1])
            a_w = 0.5 * (a_pos[ii, jj] + a_pos[ii, jj - 1])
            a_n = 0.5 * (a_pos[ii, jj] + a_pos[ii + 1, jj])
            a_s = 0.5 * (a_pos[ii, jj] + a_pos[ii - 1, jj])
            diag[n_idx] = (a_e + a_w + a_n + a_s) / h**2
            if j + 1 < n_int:
                east[n_idx] = -a_e / h**2
            if j > 0:
                west[n_idx - 1] = -a_w / h**2
            if i + 1 < n_int:
                south[n_idx] = -a_n / h**2
            if i > 0:
                north[n_idx - n_int] = -a_s / h**2

    A = diags(
        [diag, east, west, north, south],
        offsets=[0, 1, -1, n_int, -n_int],
        format="csr",
    )
    rhs = np.full(N, f_const, dtype=np.float64)

    u_int = spsolve(A, rhs)

    u = np.zeros((n, n), dtype=np.float32)
    u[1:-1, 1:-1] = u_int.reshape(n_int, n_int).astype(np.float32)
    return u


# ── Dataset classes ──────────────────────────────────────────────────


class BurgersDataset(Dataset):
    """Burgers 1-D operator dataset.

    Each sample is ``(input, target)`` of shape ``(1, N)`` where N is
    the configured grid resolution. Input is the initial condition
    ``u(x, 0)``; target is ``u(x, T)``.

    Args:
        n_samples:    Number of (a, u) pairs to generate.
        resolution:   1-D grid size (typically 128, 256, or 1024).
        nu:           Kinematic viscosity. Lower = more shocks.
        T:            Final integration time.
        seed_offset:  Seed offset; useful for train/val splits.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        resolution: int = 128,
        nu: float = 1e-3,
        T: float = 1.0,
        seed_offset: int = 0,
    ) -> None:
        self.n_samples = n_samples
        self.resolution = resolution
        self.nu = nu
        self.T = T
        self.seed_offset = seed_offset

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.utils.data.dataset.T_co:
        u0 = _grf_1d(self.resolution, length_scale=0.05, seed=idx + self.seed_offset)
        u_T = _burgers_solve(u0, nu=self.nu, T=self.T)
        # Lift to TorchIO 4-D layout (C, H, W, D) with H=N, W=1, D=1.
        # This lets the rest of the pipeline (queue, transforms,
        # collation) treat the 1-D problem as a degenerate 2-D image.
        import numpy as np
        import torchio as tio

        affine = np.eye(4)
        ic = torch.from_numpy(u0).reshape(1, -1, 1, 1)
        sol = torch.from_numpy(u_T).reshape(1, -1, 1, 1)
        return tio.Subject(
            input=tio.ScalarImage(tensor=ic, affine=affine),
            target=tio.ScalarImage(tensor=sol, affine=affine),
            mri=tio.ScalarImage(tensor=sol, affine=affine),
            file_id=f"burgers_{idx:05d}",
        )


class DarcyDataset(Dataset):
    """Darcy 2-D operator dataset.

    Each sample is ``(input, target)`` of shape ``(1, H, W)``. Input
    is the log-permeability ``log(a(x, y))``; target is the solution
    ``u(x, y)``.

    Args:
        n_samples:    Number of (a, u) pairs to generate.
        resolution:   Grid side length (square, e.g. 32, 64).
        f_const:      Constant forcing.
        length_scale: GRF correlation length for the permeability.
        seed_offset:  Seed offset.
    """

    def __init__(
        self,
        n_samples: int = 200,
        resolution: int = 32,
        f_const: float = 1.0,
        length_scale: float = 0.2,
        seed_offset: int = 0,
    ) -> None:
        self.n_samples = n_samples
        self.resolution = resolution
        self.f_const = f_const
        self.length_scale = length_scale
        self.seed_offset = seed_offset

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.utils.data.dataset.T_co:
        a = _grf_2d(self.resolution, length_scale=self.length_scale, seed=idx + self.seed_offset)
        u = _darcy_solve(a, f_const=self.f_const)
        import numpy as np
        import torchio as tio

        affine = np.eye(4)
        # (C, H, W, D) with D=1 — TorchIO's image layout.
        a_t = torch.from_numpy(a).unsqueeze(0).unsqueeze(-1)
        u_t = torch.from_numpy(u).unsqueeze(0).unsqueeze(-1)
        return tio.Subject(
            input=tio.ScalarImage(tensor=a_t, affine=affine),
            target=tio.ScalarImage(tensor=u_t, affine=affine),
            mri=tio.ScalarImage(tensor=u_t, affine=affine),
            file_id=f"darcy_{idx:05d}",
        )


def make_pde_dataset(
    problem: Literal["burgers_1d", "darcy_2d"],
    n_samples: int,
    resolution: int,
    seed_offset: int = 0,
    **problem_kwargs,
) -> Dataset:
    """Factory used by the dataset_instantiator dispatch.

    ``problem`` selects the PDE; ``problem_kwargs`` is forwarded to the
    underlying dataset (e.g. ``nu`` for Burgers, ``f_const`` for Darcy).
    """
    if problem == "burgers_1d":
        return BurgersDataset(
            n_samples=n_samples,
            resolution=resolution,
            seed_offset=seed_offset,
            **problem_kwargs,
        )
    if problem == "darcy_2d":
        return DarcyDataset(
            n_samples=n_samples,
            resolution=resolution,
            seed_offset=seed_offset,
            **problem_kwargs,
        )
    raise ValueError(f"Unknown PDE problem {problem!r}. Available: burgers_1d, darcy_2d.")
