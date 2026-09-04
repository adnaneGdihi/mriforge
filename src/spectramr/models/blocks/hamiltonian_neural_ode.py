r"""Hamiltonian Neural ODE for k-space trajectory generation (B-4 / Bundle P7).

This block parameterises a *family* of k-space trajectories through a learnable
scalar potential :math:`V_\theta(\mathbf{k})` rather than parameterising the
trajectory waveform directly (the LOUPE [36] / PILOT [37] approach). The
trajectory is then obtained by integrating Hamilton's equations of motion of a
unit-mass particle moving in that potential.

Mathematical formulation
-------------------------
With k-space position :math:`\mathbf{k}\in\mathbb{R}^d` (the generalised
coordinate) and gradient waveform :math:`\mathbf{G}=\dot{\mathbf{k}}` (the
generalised momentum for unit mass), the Hamiltonian is

.. math::

    H(\mathbf{k},\mathbf{G}) = \tfrac12\lVert\mathbf{G}\rVert_2^2 + V_\theta(\mathbf{k}),

and Hamilton's equations give the dynamics

.. math::

    \dot{\mathbf{k}} = \frac{\partial H}{\partial \mathbf{G}} = \mathbf{G},
    \qquad
    \dot{\mathbf{G}} = -\frac{\partial H}{\partial \mathbf{k}} = -\nabla_{\mathbf{k}} V_\theta(\mathbf{k}).

The kinetic term :math:`\tfrac12\lVert\mathbf{G}\rVert^2` makes :math:`H` the
total mechanical energy, so any structure-preserving (symplectic) integrator
conserves :math:`H` to within a bounded oscillation — there is **no secular
energy drift**. We use the **leapfrog / Stormer-Verlet** integrator, the
canonical symplectic second-order scheme:

.. math::

    \mathbf{G}_{n+1/2} &= \mathbf{G}_n - \tfrac{\Delta t}{2}\,\nabla V_\theta(\mathbf{k}_n) \\
    \mathbf{k}_{n+1}   &= \mathbf{k}_n + \Delta t\,\mathbf{G}_{n+1/2} \\
    \mathbf{G}_{n+1}   &= \mathbf{G}_{n+1/2} - \tfrac{\Delta t}{2}\,\nabla V_\theta(\mathbf{k}_{n+1}).

Why this matters for MRI acquisition
-------------------------------------
* The gradient waveform :math:`\mathbf{G}(t)` and the slew rate
  :math:`\dot{\mathbf{G}}(t) = -\nabla V_\theta(\mathbf{k})` are *outputs* of the
  integration, so hardware-feasibility penalties on :math:`\lvert\mathbf{G}\rvert`
  and :math:`\lvert\dot{\mathbf{G}}\rvert` act directly on physical quantities.
* The energy-conservation property bounds how fast the trajectory can change
  velocity, giving an intrinsic regulariser that PILOT enforces only by an
  explicit slew penalty.

Consistency check (zero potential)
----------------------------------
When :math:`V_\theta\equiv\text{const}`, :math:`\nabla V_\theta=\mathbf 0`, the
momentum is conserved (:math:`\dot{\mathbf G}=\mathbf 0`) and the trajectory is a
**straight line** :math:`\mathbf{k}(t)=\mathbf{k}_0+\mathbf{G}_0 t` — i.e. a
Cartesian readout when :math:`\mathbf{G}_0` points along a single axis. This is
asserted in the unit tests.

The gradient :math:`\nabla_{\mathbf{k}} V_\theta` is computed with
``torch.autograd.grad(..., create_graph=True)`` so the whole integration is
differentiable end-to-end with respect to :math:`\theta` (the potential's
parameters) and the initial conditions.

Exports
-------
* :class:`PotentialNet` — learnable scalar potential :math:`V_\theta`.
* :class:`HamiltonianNeuralODE` — the symplectic-integration block.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

__all__ = ["HamiltonianNeuralODE", "PotentialNet"]


class _SineLayer(nn.Module):
    """Sine-activated linear layer (SIREN, Sitzmann et al. 2020).

    Used as the building block of the ``"siren"`` potential architecture. The
    principled weight init keeps the activation distribution stationary across
    depth, which is essential for the second-order spatial derivatives the
    Hamiltonian dynamics rely on.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.omega_0 = float(omega_0)
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                bound = (6.0 / in_features) ** 0.5 / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class _FourierFeatures(nn.Module):
    """Random Fourier feature lift :math:`\\mathbf{k}\\mapsto[\\sin(B\\mathbf{k}),\\cos(B\\mathbf{k})]`.

    The frequency matrix ``B`` is a fixed (non-learnable) Gaussian buffer so the
    feature map is deterministic given the seed (Tancik et al. 2020).
    """

    def __init__(self, in_features: int, num_frequencies: int, scale: float = 10.0) -> None:
        super().__init__()
        b = torch.randn(in_features, num_frequencies) * scale
        self.register_buffer("freq", b)
        self.out_features = 2 * num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * torch.pi * (x @ self.freq)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class PotentialNet(nn.Module):
    r"""Learnable scalar potential :math:`V_\theta:\mathbb{R}^d\to\mathbb{R}`.

    Maps a k-space position to a scalar potential value. Three architectures are
    supported (selected by ``arch``):

    * ``"mlp"`` — tanh-MLP (smooth, well-behaved second derivatives).
    * ``"siren"`` — sine-activated MLP (sharp, high-frequency potentials).
    * ``"fourier_features_mlp"`` — random-Fourier-feature lift + tanh-MLP.

    Args:
        spatial_dims: Dimensionality :math:`d` of k-space (2 or 3).
        hidden_dim: Width of the hidden layers.
        num_layers: Number of hidden layers (``>= 1``).
        arch: Potential architecture.
        omega_0: First-layer frequency for the SIREN variant.
        num_fourier_features: Number of random Fourier features
            (``"fourier_features_mlp"`` only).
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 4,
        arch: Literal["mlp", "siren", "fourier_features_mlp"] = "siren",
        *,
        omega_0: float = 30.0,
        num_fourier_features: int = 64,
    ) -> None:
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        # NO silent fallback (CLAUDE.md #9): reject unknown architectures loudly.
        if arch not in ("mlp", "siren", "fourier_features_mlp"):
            raise ValueError(
                f"Unknown potential arch {arch!r}; "
                "expected one of {'mlp', 'siren', 'fourier_features_mlp'}."
            )

        self.spatial_dims = spatial_dims
        self.arch = arch

        layers: list[nn.Module] = []
        if arch == "siren":
            layers.append(_SineLayer(spatial_dims, hidden_dim, is_first=True, omega_0=omega_0))
            for _ in range(num_layers - 1):
                layers.append(_SineLayer(hidden_dim, hidden_dim, omega_0=1.0))
            layers.append(nn.Linear(hidden_dim, 1))
        elif arch == "fourier_features_mlp":
            ff = _FourierFeatures(spatial_dims, num_fourier_features)
            layers.append(ff)
            in_dim = ff.out_features
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_dim, 1))
        else:  # "mlp"
            layers.append(nn.Linear(spatial_dims, hidden_dim))
            layers.append(nn.Tanh())
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Evaluate the potential.

        Args:
            k: k-space positions ``[..., spatial_dims]``.

        Returns:
            Scalar potential ``[..., 1]``.
        """
        return self.net(k)


class HamiltonianNeuralODE(nn.Module):
    r"""Symplectic integrator for the trajectory Hamiltonian.

    Integrates :math:`\dot{\mathbf k}=\mathbf G,\ \dot{\mathbf G}=-\nabla V_\theta(\mathbf k)`
    with the energy-conserving leapfrog scheme over ``num_steps`` steps of size
    ``dt``. The number of steps is derived from ``T_total / dt`` if not given
    explicitly.

    Args:
        potential_net: A :class:`PotentialNet` (or any ``nn.Module`` mapping
            ``[..., d] -> [..., 1]``) defining :math:`V_\theta`.
        integrator: ``"leapfrog"`` (Stormer-Verlet, symplectic) or
            ``"rk4_symplectic"`` (a partitioned RK scheme; here the
            position-Verlet variant, also symplectic).
        dt: Integration step (seconds; gradient raster).
        T_total: Total integration window (seconds). ``num_steps`` is
            ``round(T_total / dt)`` unless ``num_steps`` is passed to
            :meth:`forward`.

    Notes:
        * ``forward`` runs natively in the dtype/device of the initial
          conditions; no GPU syncs are introduced.
        * The gradient :math:`\nabla V_\theta` is obtained with
          ``torch.autograd.grad(create_graph=self.training)`` so training keeps
          the second-order graph while evaluation stays cheap.
    """

    def __init__(
        self,
        potential_net: nn.Module,
        integrator: Literal["leapfrog", "rk4_symplectic"] = "leapfrog",
        dt: float = 4e-6,
        T_total: float = 50e-3,  # noqa: N803 — physics symbol (total time T)
    ) -> None:
        super().__init__()
        if integrator not in ("leapfrog", "rk4_symplectic"):
            raise ValueError(
                f"Unknown integrator {integrator!r}; expected 'leapfrog' or 'rk4_symplectic'."
            )
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if T_total <= 0:
            raise ValueError(f"T_total must be > 0, got {T_total}")
        self.potential_net = potential_net
        self.integrator = integrator
        self.dt = float(dt)
        self.T_total = float(T_total)

    # ------------------------------------------------------------------
    def default_num_steps(self) -> int:
        """Number of integration steps implied by ``T_total / dt`` (>= 1)."""
        return max(1, round(self.T_total / self.dt))

    def grad_potential(self, k: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\nabla_{\mathbf k} V_\theta(\mathbf k)`.

        Args:
            k: Positions ``[..., d]``.

        Returns:
            Gradient ``[..., d]`` with the same shape as ``k``.
        """
        create_graph = self.training or k.requires_grad
        with torch.enable_grad():
            k_req = k if k.requires_grad else k.detach().requires_grad_(True)
            v = self.potential_net(k_req).sum()
            (grad,) = torch.autograd.grad(v, k_req, create_graph=create_graph, retain_graph=True)
        return grad

    def energy(self, k: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        r"""Total Hamiltonian :math:`H=\tfrac12\lVert\mathbf G\rVert^2+V_\theta(\mathbf k)`.

        Args:
            k: Positions ``[..., d]``.
            g: Momenta / gradients ``[..., d]``.

        Returns:
            Per-trajectory energy ``[...]`` (the trailing coord dim is reduced).
        """
        kinetic = 0.5 * (g**2).sum(dim=-1)
        potential = self.potential_net(k).squeeze(-1)
        return kinetic + potential

    # ------------------------------------------------------------------
    def _leapfrog_step(
        self, k: torch.Tensor, g: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One Stormer-Verlet (velocity-leapfrog) step."""
        g_half = g - 0.5 * dt * self.grad_potential(k)
        k_next = k + dt * g_half
        g_next = g_half - 0.5 * dt * self.grad_potential(k_next)
        return k_next, g_next

    def _position_verlet_step(
        self, k: torch.Tensor, g: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One position-Verlet step (the ``rk4_symplectic`` option).

        Position-Verlet is the drift-kick-drift partner of velocity-leapfrog;
        it is symplectic and second-order, hence still energy-conserving. (A
        genuine 4th-order symplectic composition can be added later by chaining
        Verlet steps with Yoshida coefficients; the second-order variant is the
        functional, audit-clean implementation.)
        """
        k_half = k + 0.5 * dt * g
        g_next = g - dt * self.grad_potential(k_half)
        k_next = k_half + 0.5 * dt * g_next
        return k_next, g_next

    def _step(
        self, k: torch.Tensor, g: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.integrator == "leapfrog":
            return self._leapfrog_step(k, g, dt)
        return self._position_verlet_step(k, g, dt)

    # ------------------------------------------------------------------
    def forward(
        self,
        k0: torch.Tensor,
        g0: torch.Tensor,
        num_steps: int | None = None,
        *,
        return_energy: bool = False,
    ) -> dict[str, torch.Tensor]:
        r"""Integrate the trajectory from initial conditions.

        Args:
            k0: Initial k-space positions ``[B, d]`` (or ``[..., d]``).
            g0: Initial gradients (momenta) ``[B, d]`` (same shape as ``k0``).
            num_steps: Number of leapfrog steps. Defaults to
                ``round(T_total / dt)``.
            return_energy: If True, also return the per-step Hamiltonian
                ``energy`` ``[B, num_steps + 1]`` for the conservation diagnostic.

        Returns:
            A dict with:

            * ``"k"`` — positions ``[B, num_steps + 1, d]``,
            * ``"g"`` — gradients ``[B, num_steps + 1, d]``,
            * ``"slew"`` — :math:`\dot{\mathbf G}=-\nabla V_\theta(\mathbf k)`
              sampled at every stored knot ``[B, num_steps + 1, d]``,
            * ``"energy"`` — (only if ``return_energy``) ``[B, num_steps + 1]``.
        """
        if k0.shape != g0.shape:
            raise ValueError(
                f"k0 and g0 must share shape; got {tuple(k0.shape)} vs {tuple(g0.shape)}"
            )
        if num_steps is None:
            num_steps = self.default_num_steps()
        dt = self.dt

        ks: list[torch.Tensor] = [k0]
        gs: list[torch.Tensor] = [g0]
        energies: list[torch.Tensor] = []
        if return_energy:
            energies.append(self.energy(k0, g0))

        k, g = k0, g0
        for _ in range(num_steps):
            k, g = self._step(k, g, dt)
            ks.append(k)
            gs.append(g)
            if return_energy:
                energies.append(self.energy(k, g))

        k_traj = torch.stack(ks, dim=1)  # [B, T+1, d]
        g_traj = torch.stack(gs, dim=1)  # [B, T+1, d]
        # slew(t) = dG/dt = -grad V(k(t)); sample at stored knots.
        slew = -self.grad_potential(k_traj)

        out: dict[str, torch.Tensor] = {"k": k_traj, "g": g_traj, "slew": slew}
        if return_energy:
            out["energy"] = torch.stack(energies, dim=1)  # [B, T+1]
        return out
