"""Schedule-certification witnesses for k-space cold diffusion (C1/C4 + inert).

The cold-diffusion papers reduce "will this schedule fabricate?" to checks on
the mask cascade itself, decidable *before training*:

* **C1 — leak-free nesting** (``schedule.nesting_leakfree``): the kept-sets
  must be nested ``K_T ⊆ ... ⊆ K_0``; a re-introduced bin breaks the cocycle
  the reverse theory stands on and forces a schedule-induced fabrication
  floor. Detector: :meth:`KSpaceUndersamplingProcess.nesting_leak_report`.
* **inert levels** (``schedule.no_inert_steps``): a level that removes nothing
  makes the timestep axis degenerate there. Detector:
  :meth:`KSpaceUndersamplingProcess.inert_step_report`.
* **C4 — allocation** (``schedule.line_allocation``): no single level may
  carry the bulk of the removed k-space energy (dump the low-frequency band
  into one level and that level's ambiguity exceeds its budget). Detector:
  :meth:`KSpaceUndersamplingProcess.removed_line_energy_stats` under a
  synthetic ``1/(1+|k|^2)`` spectral prior — a data batch is not available on
  the CONFIG surface, and the prior is the standard natural-image spectrum
  shape, so the verdict is a *diagnostic* of the schedule's geometry, not of a
  dataset.

Two further design corollaries are *estimator-dependent* and can only be
checked against declared offline measurements
(``undersampling.certification.per_level``, see
:func:`_declared_certification_levels`):

* **C2 — step-to-reach cap** (``schedule.step_to_reach_cap``, INFO):
  ``kappa_t = delta_hat/tau_hat <= 1/2`` so every per-step Lipschitz factor
  stays at or below 2.
* **C3 — tangential-defect margin** (``schedule.tangential_defect_margin``):
  ``theta_hat < 1 - kappa_t``, the contraction condition (2.9).

The C1/inert/C4 trio runs at ``Severity.WARNING``: the corpus carries 62 known-defective
ladder arms (issue #534) and this module must not take them offline; the hard
gate remains ``scripts/ci/check_acceleration_ladder_realisable.py`` with its
ratcheted baseline. Config parsing below deliberately mirrors that script's
(``_is_cold_diffusion`` / ``_undersampling`` / ``_matrix`` / ``_timesteps``)
so the two surfaces certify the same construction.
"""

# Facade. The helpers and the five witnesses live in sibling modules (300-LOC
# ceiling, NN20) and are re-exported here, so the test module and any future
# caller resolve them through this path against one definition each (NN17).
# The witnesses register by import either way -- the package walk imports every
# module under checks/ -- so this facade is for readers, not for registration.

from __future__ import annotations

from spectramr.infrastructure.validation.witness.checks.schedule_allocation_checks import (
    schedule_line_allocation,
    schedule_step_to_reach_cap,
    schedule_tangential_defect_margin,
)
from spectramr.infrastructure.validation.witness.checks.schedule_certification_common import (
    build_process_from_config,
    synthetic_spectral_prior,
)
from spectramr.infrastructure.validation.witness.checks.schedule_nesting_checks import (
    schedule_nesting_leakfree,
    schedule_no_inert_steps,
)

__all__ = [
    "build_process_from_config",
    "schedule_line_allocation",
    "schedule_nesting_leakfree",
    "schedule_no_inert_steps",
    "schedule_step_to_reach_cap",
    "schedule_tangential_defect_margin",
    "synthetic_spectral_prior",
]
