"""Layering-rule guard tests.

Locks ``TODO/backlog_ssot_and_layering_cleanup.md`` Phase 2 + Phase 6:

- **Phase 2** — pins the three reverse-layer-import fixes (the step executor,
  formerly ``Trainer``, lives in infrastructure/; dotted_override moved to
  infrastructure/hpo/; profiling_helpers removed).
- **Phase 6** — invokes the ``scripts/ci/check_layering.sh`` audit
  inside pytest so the rules ratchet forward (each remaining Phase 4/5
  violation is documented; new ones fail loudly).

These tests are cheap (grep over ``src/``) and run in the standard unit
suite.
"""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_step_executor_canonical_home_is_infrastructure() -> None:
    """``StepExecutor`` lives in ``mriforge.infrastructure.training.step_executor``.

    Reverse-import cleanup (Phase 2): the class (then named ``Trainer``) was at
    ``mriforge.pipelines.trainer`` where ``base.py`` (infrastructure) had to
    import upward into pipelines. Renamed Trainer→StepExecutor 2026-06-18 to
    free the ``Trainer`` name for the public scripting orchestrator.
    """
    from mriforge.infrastructure.training.step_executor import StepExecutor

    assert StepExecutor.__module__ == "mriforge.infrastructure.training.step_executor"


def test_pipelines_trainer_shim_removed() -> None:
    """The legacy ``mriforge.pipelines.trainer`` re-export shim was removed
    (2026-06-18). Importing it must fail loudly — no silent legacy path lingers."""
    with pytest.raises(ModuleNotFoundError):
        import mriforge.pipelines.trainer  # noqa: F401


def test_base_strategy_imports_step_executor_from_infrastructure() -> None:
    """``BaseTrainingStrategy.__init__`` imports ``StepExecutor`` from the canonical home."""
    from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy

    src = inspect.getsource(BaseTrainingStrategy.__init__)
    assert (
        "from mriforge.infrastructure.training.step_executor import StepExecutor" in src
    )
    assert "from mriforge.pipelines.trainer" not in src


def test_dotted_override_canonical_home_is_infrastructure() -> None:
    """``apply_dotted_override`` lives in ``mriforge.infrastructure.hpo.dotted_override``."""
    from mriforge.infrastructure.hpo.dotted_override import apply_dotted_override

    assert apply_dotted_override.__module__ == (
        "mriforge.infrastructure.hpo.dotted_override"
    )


def test_hpo_search_spaces_is_thin_shim_for_dotted_override() -> None:
    """``mriforge.pipelines.hpo_search_spaces.apply_dotted_override`` re-exports the canonical fn."""
    from mriforge.infrastructure.hpo.dotted_override import (
        apply_dotted_override as canonical,
    )
    from mriforge.pipelines.hpo_search_spaces import apply_dotted_override

    assert apply_dotted_override is canonical


def test_hpo_coordinator_no_longer_imports_dotted_override_from_pipelines() -> None:
    """The coordinator's import is inward (infrastructure/hpo)."""
    from mriforge.infrastructure.coordination import hpo_coordinator

    src = inspect.getsource(hpo_coordinator)
    assert "from mriforge.infrastructure.hpo.dotted_override import" in src
    # No longer imports apply_dotted_override from pipelines/
    assert "from mriforge.pipelines.hpo_search_spaces import apply_dotted_override" not in src


def test_profiling_helpers_no_longer_imports_pipelines() -> None:
    """The thin ``run_*_training`` wrappers were removed; no models→pipelines import."""
    helpers = REPO_ROOT / "src" / "models" / "analysis" / "profiling_helpers.py"
    assert not helpers.exists(), (
        "src/mriforge/models/analysis/profiling_helpers.py was reintroduced — "
        "the file imported upward from pipelines/ (CLAUDE.md layer-direction "
        "violation). Callers should invoke run_training_pipeline directly."
    )


def test_models_analysis_init_does_not_export_profiling_helpers() -> None:
    """The package init no longer re-exports the deleted thin wrappers."""
    from mriforge.models import analysis

    assert "run_gan_training" not in dir(analysis)
    assert "run_diffusion_training" not in dir(analysis)
    assert "run_reconstruction_training" not in dir(analysis)


def test_check_layering_script_exists_and_is_executable() -> None:
    """The Phase 6 lint script is on disk and executable."""
    script = REPO_ROOT / "scripts" / "ci" / "check_layering.sh"
    assert script.is_file()
    import stat

    mode = script.stat().st_mode
    assert mode & stat.S_IXUSR, "check_layering.sh must be executable"


def test_dice_risk_segmentation_primitive_lives_in_core() -> None:
    """The Dice-risk segmentation backends are a pure-torch metric primitive in
    ``core/metrics/`` (moved from ``infrastructure/calibration/`` 2026-06-18);
    ``dice_risk`` imports them rightward (core→core), NOT core→infrastructure.
    """
    from mriforge.core.metrics.quantitative import dice_risk, segmentation

    assert segmentation.__name__ == "mriforge.core.metrics.quantitative.segmentation"
    src = inspect.getsource(dice_risk)
    assert "from mriforge.core.metrics.quantitative.segmentation import" in src
    assert "infrastructure.calibration.segmentation" not in src


def test_layering_ratchet_gate_passes() -> None:
    """The ratcheted layering audit passes: zero NEW (un-baselined) violations.

    The script exits 0 iff every detected violation is already recorded in
    ``scripts/ci/layering_baseline.txt`` — that exit code is the gate CI
    enforces. We assert the EXIT CODE plus the ``0 new`` summary rather than
    per-rule prefix strings: the script's per-rule output format and rule names
    are not a stable contract (they drifted — ``PASS``→``clean``/``found``,
    ``→``→``->``, ``models/``→``models|domain|core``) and pinning them made this
    test rot. The exit code does not rot; a genuine new violation flips it.
    """
    script = REPO_ROOT / "scripts" / "ci" / "check_layering.sh"
    result = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, timeout=60,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, (
        "Layering audit found NEW (un-baselined) violation(s) — fix the import "
        "(invert via a Protocol/DI seam) or, if genuinely accepted debt, add it "
        f"to scripts/ci/layering_baseline.txt.\nFull output:\n{output}"
    )
    # Meaningful, low-drift markers: the gate ran its rules and found nothing new.
    assert "0 new" in output, f"Expected '0 new' in the audit summary.\n{output}"
    assert "non-vacuous" in output, f"Layering gate reported vacuous.\n{output}"
