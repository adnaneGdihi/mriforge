"""Learnable-acquisition model family (k-space trajectory generators).

Houses models that *generate* the acquisition trajectory rather than
reconstructing an image. The Hamiltonian-NODE trajectory generator (B-4 /
Bundle P7) lives here.
"""

from mriforge.models.acquisition.hamiltonian_trajectory_generator import (
    HamiltonianTrajectoryGenerator,
)

__all__ = ["HamiltonianTrajectoryGenerator"]
