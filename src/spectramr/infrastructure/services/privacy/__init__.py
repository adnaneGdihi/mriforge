"""Differential-privacy primitives.

Implements PR-7 (H5) of
``TODO/backlog_paradigm_expansion_roadmap.md``:

- :class:`RDPAccountant` — Rényi-DP accountant (Mironov 2017)
- :func:`clip_per_sample_grad`, :func:`add_gaussian_noise` — the two
  building blocks of the DP-SGD update (Abadi et al. 2016)
- :func:`compose_dp_sgd_update` — wraps both into the typical
  per-sample-clip + add-noise step

Stand-alone — does not import opacus, so it works in environments
without the optional ``[dp]`` extra installed.
"""

from .dp_sgd import (
    add_gaussian_noise,
    clip_per_sample_grad,
    compose_dp_sgd_update,
)
from .rdp_accountant import RDPAccountant

__all__ = [
    "RDPAccountant",
    "add_gaussian_noise",
    "clip_per_sample_grad",
    "compose_dp_sgd_update",
]
