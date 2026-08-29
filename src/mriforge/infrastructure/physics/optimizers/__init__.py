"""Optimisation solvers for MRI reconstruction problems.

Provides iterative solvers (Split Bregman, FISTA, etc.) for
regularised inverse problems arising in model-based image reconstruction.
"""

from .split_bregman import SplitBregmanSolver

__all__ = ["SplitBregmanSolver"]
