r"""Hamiltonian-NODE k-space trajectory generator (B-4 / Bundle P7).

A registered model that wraps :class:`HamiltonianNeuralODE` to produce a
*family* of k-space trajectories from a learnable scalar potential
:math:`V_\theta`. Distinct from LOUPE [36] / PILOT [37], which parameterise the
trajectory waveform directly: here only the potential is learned, and hardware
compliance is intrinsic to the energy-conserving symplectic dynamics.

The model holds the only trainable parameters of the acquisition stage (the
potential network); the integrator itself is parameter-free. ``forward`` accepts
per-shot initial conditions :math:`(\mathbf{k}_0,\mathbf{G}_0)` and returns the
integrated trajectory together with the gradient and slew waveforms needed by
the hardware-compliance loss.

Registered as ``hamiltonian_trajectory_generator`` (training_mode
``acquisition_learning``). The *internal* trajectory state — gradient, slew, and
energy waveforms — is physics-domain, but the model's **data I/O contract within
the co-training strategy** is ``input_domain="kspace"`` → ``output_domain="image"``
(matching the sibling learnable-acquisition models ``loupe`` / ``pilot``): the
strategy feeds the k-space target through the generated trajectory's coverage to
produce an image-domain reconstruction. ``accepts_complex=False`` because the
trajectory coordinates the potential acts on are real-valued.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from mriforge.models.blocks.hamiltonian_neural_ode import (
    HamiltonianNeuralODE,
    PotentialNet,
)
from mriforge.models.registry import register_model

__all__ = ["HamiltonianTrajectoryGenerator"]


@register_model(
    name="hamiltonian_trajectory_generator",
    training_mode="acquisition_learning",
    spatial_dims=(2, 3),
    # Data I/O contract of the co-training strategy (matches loupe / pilot):
    # k-space target in, image recon out. The trajectory's gradient/slew/energy
    # waveforms are physics-domain *internals*, not the model's data I/O.
    input_domain="kspace",
    output_domain="image",
    accepts_complex=False,
)
class HamiltonianTrajectoryGenerator(nn.Module):
    r"""Generate k-space trajectories by integrating a learnable Hamiltonian.

    Args:
        spatial_dims: k-space dimensionality :math:`d` (2 or 3).
        potential_arch: Architecture of :math:`V_\theta`
            (``"mlp"`` | ``"siren"`` | ``"fourier_features_mlp"``).
        potential_hidden_dim: Hidden width of the potential net.
        potential_num_layers: Number of hidden layers in the potential net.
        integrator: Symplectic integrator (``"leapfrog"`` | ``"rk4_symplectic"``).
        dt_seconds: Gradient-raster step :math:`\Delta t`.
        T_total_seconds: Total readout window :math:`T`.
        num_shots: Number of independent trajectory arms (interleaves).
        samples_per_shot: Stored knots per shot. When ``None`` the integrator
            uses ``round(T_total / dt)`` steps; otherwise the requested number
            of integration steps is ``samples_per_shot - 1``.
        omega_0: First-layer SIREN frequency (``"siren"`` arch only).
        gamma_hz_per_t: Gyromagnetic ratio :math:`\gamma/2\pi` (Hz/T) used to
            convert gradient waveforms (mT/m) into k-space velocity for the
            default initial conditions. Defaults to the proton value.
        fov_m: Field of view (m); sets the k-space extent
            :math:`k_{\max}=1/(2\,\Delta x)` only used for the default Cartesian
            init. Trajectories are otherwise unitless in normalised k-space.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        potential_arch: Literal["mlp", "siren", "fourier_features_mlp"] = "siren",
        potential_hidden_dim: int = 128,
        potential_num_layers: int = 4,
        integrator: Literal["leapfrog", "rk4_symplectic"] = "leapfrog",
        dt_seconds: float = 4e-6,
        T_total_seconds: float = 50e-3,  # noqa: N803 — physics symbol (total time T)
        num_shots: int = 1,
        samples_per_shot: int | None = 256,
        *,
        omega_0: float = 30.0,
        gamma_hz_per_t: float = 42.577478e6,
        fov_m: float = 0.22,
        **_ignored: Any,
    ) -> None:
        super().__init__()
        if num_shots < 1:
            raise ValueError(f"num_shots must be >= 1, got {num_shots}")
        if samples_per_shot is not None and samples_per_shot < 2:
            raise ValueError(f"samples_per_shot must be >= 2 (>= 1 step), got {samples_per_shot}")

        self.spatial_dims = spatial_dims
        self.num_shots = num_shots
        self.samples_per_shot = samples_per_shot
        self.gamma_hz_per_t = float(gamma_hz_per_t)
        self.fov_m = float(fov_m)

        self.potential_net = PotentialNet(
            spatial_dims=spatial_dims,
            hidden_dim=potential_hidden_dim,
            num_layers=potential_num_layers,
            arch=potential_arch,
            omega_0=omega_0,
        )
        self.ode = HamiltonianNeuralODE(
            potential_net=self.potential_net,
            integrator=integrator,
            dt=dt_seconds,
            T_total=T_total_seconds,
        )

    # ------------------------------------------------------------------
    def _num_steps(self) -> int:
        if self.samples_per_shot is None:
            return self.ode.default_num_steps()
        return self.samples_per_shot - 1

    def default_initial_conditions(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Cartesian-raster initial conditions :math:`(\mathbf k_0,\mathbf G_0)`.

        Each shot starts at one edge of normalised k-space (``-0.5`` along the
        readout axis) with a constant readout velocity that traverses the full
        extent over ``T_total``. Phase-encode shots are offset along the second
        axis. With a zero potential this reproduces a Cartesian raster (the
        consistency check).

        Returns:
            ``(k0, g0)`` each of shape ``[num_shots, spatial_dims]``.
        """
        d = self.spatial_dims
        n = self.num_shots
        k0 = torch.zeros(n, d, device=device, dtype=dtype)
        g0 = torch.zeros(n, d, device=device, dtype=dtype)

        steps = self._num_steps()
        total_t = steps * self.ode.dt
        # Readout along axis 0 from -0.5 to +0.5 over the window.
        k0[:, 0] = -0.5
        g0[:, 0] = 1.0 / max(total_t, 1e-12)
        # Distribute phase-encode lines along axis 1 (if present).
        if d >= 2 and n > 1:
            pe = torch.linspace(-0.5, 0.5, n, device=device, dtype=dtype)
            k0[:, 1] = pe
        return k0, g0

    # ------------------------------------------------------------------
    def forward(
        self,
        k0: torch.Tensor | None = None,
        g0: torch.Tensor | None = None,
        *,
        return_energy: bool = True,
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        r"""Integrate trajectories from initial conditions.

        Args:
            k0: Initial positions ``[num_shots, d]``. Defaults to a Cartesian
                raster init (see :meth:`default_initial_conditions`).
            g0: Initial gradients ``[num_shots, d]``. Defaults as above.
            return_energy: Return the per-step Hamiltonian for the
                conservation diagnostic / loss.

        Returns:
            Dict with ``"k"``, ``"g"``, ``"slew"`` of shape
            ``[num_shots, T+1, d]`` and (optionally) ``"energy"``
            ``[num_shots, T+1]``.
        """
        # Resolve device/dtype from the potential net parameters (SSOT).
        param = next(self.potential_net.parameters())
        if k0 is None or g0 is None:
            k0, g0 = self.default_initial_conditions(device=param.device, dtype=param.dtype)
        else:
            k0 = k0.to(param.device, dtype=param.dtype)
            g0 = g0.to(param.device, dtype=param.dtype)

        return self.ode(k0, g0, num_steps=self._num_steps(), return_energy=return_energy)
