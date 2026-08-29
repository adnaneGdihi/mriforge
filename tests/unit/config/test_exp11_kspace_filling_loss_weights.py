"""Regression: no ``experiment_11`` kspace_filling arm carries conflicting loss
weights under a MATERIALISED config load.

The cluster loads a materialised config (every schema default stamped as an
explicit value), so ``build_loss_weight_table`` sees BOTH the ``reconstruction``
and ``physics`` defaults for a shared loss. The ``physics`` section defaults
``bloch_residual``/``physics_constraint``/``snr_preserving`` to non-zero while
``reconstruction`` defaults them to 0.0, and ``reconstruction`` defaults the
``content``/``perceptual`` aliases to 0.0/10.0 — cross-declaration clashes that
abort ``build_loss_weight_table`` at iter 1 (loss-weight SSOT, pitfall #13b). A
``yaml.safe_load``-only check cannot catch this (it never materialises the
defaults), so this test loads the real schema and asserts agreement.

The arms pin each clashing loss so all its declarations agree (unwanted physics/
perceptual terms to 0.0; a term the arm genuinely uses stays at its value). The
schema root cause (divergent per-section defaults) is tracked in issue #421.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mriforge.config.settings import TrainingSettings
from mriforge.models.losses.weights import (
    LAMBDA_SECTIONS,
    _loss_name_for_field,
    build_loss_weight_table,
)
from tests.utils.corpus import tracked_yamls

_ARM_DIR = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "inprogress"
    / "kspace_filling"
)
# Every arm in the family, all sub-cohorts (attention_shootout, dc_shootout,
# attention_enhancements, ablations*, and the loose top-level variants).
_ARMS = tracked_yamls(_ARM_DIR)


def _materialised_conflicts(losses: object) -> dict[str, set[float]]:
    """Group EVERY lambda field (incl. defaults) by canonical loss name and keep
    the names whose declarations disagree — the cluster's materialised view."""
    by_name: dict[str, set[float]] = {}
    for section_name in LAMBDA_SECTIONS:
        section = getattr(losses, section_name, None)
        if section is None or not hasattr(section, "model_dump"):
            continue
        for field, value in section.model_dump().items():
            if field.startswith("lambda_") and isinstance(value, (int, float)):
                by_name.setdefault(_loss_name_for_field(field), set()).add(float(value))
    return {name: vals for name, vals in by_name.items() if len(vals) > 1}


def test_cohort_is_non_empty() -> None:
    assert (
        len(_ARMS) >= 50
    ), f"expected the full kspace_filling family, found {len(_ARMS)}"


@pytest.mark.parametrize(
    "arm", _ARMS, ids=[str(p.relative_to(_ARM_DIR)) for p in _ARMS]
)
def test_no_materialised_loss_weight_conflict(arm: Path) -> None:
    losses = TrainingSettings.from_yaml(str(arm)).losses
    conflicts = _materialised_conflicts(losses)
    assert not conflicts, (
        f"{arm.name}: conflicting materialised loss weights (aborts training at "
        f"iter 1): {conflicts}"
    )
    # And the SSOT table must build (the actual failing call on the cluster).
    build_loss_weight_table(losses)
