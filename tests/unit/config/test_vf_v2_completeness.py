"""Regression guard: every vf/_v2.yaml must carry the v2 standard.

The v2 standard (established by the cold-diffusion locus ablation and
applied across the vf/ campaign) requires three deployment-matched
validation fields to be present and enabled:

* ``validation.sampler_steps`` — equal to the deployment sampler steps.
  Without this, the validation curve at 10 steps optimistically
  overstates deployment-time performance at 50 steps.
* ``validation.held_out_severity_eval`` — Boolean opt-in for the
  digital-twin-tagged severity-grid evaluation. Catches the
  cold-diffusion paradigm pitfall where models overfit to the in-
  distribution severity curriculum.
* ``validation.hallucination_test.enabled`` — feature-insertion sweep
  for clinical defensibility.

Plus every v2 YAML must validate cleanly through ``TrainingSettings``
(the load-bearing entrypoint that the orchestrator uses).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mriforge.config.settings import TrainingSettings
from tests.utils.corpus import tracked_yamls

VF_DIR = Path(__file__).resolve().parents[3] / "experiments" / "inprogress" / "vf"


def _v2_yamls() -> list[Path]:
    return tracked_yamls(VF_DIR, "*_v2.yaml", recursive=False)


@pytest.fixture(scope="module", params=_v2_yamls(), ids=lambda p: p.name)
def v2_yaml(request: pytest.FixtureRequest) -> Path:
    """One vf/_v2.yaml per parametrised invocation."""
    return request.param


@pytest.fixture(scope="module")
def v2_settings(v2_yaml: Path) -> TrainingSettings:
    """Parsed TrainingSettings for a v2 arm."""
    return TrainingSettings.from_yaml(v2_yaml)


class TestV2Loads:
    """Every v2 YAML must validate cleanly through TrainingSettings."""

    def test_loads(self, v2_settings: TrainingSettings) -> None:
        assert v2_settings is not None


class TestV2ValidationFields:
    """The three plan-mandated validation fields must be present & set.

    All three moved in phase 10a and are read here at their CANONICAL paths,
    resolved through ``RENAMES`` rather than transcribed::

        validation.sampler_steps          -> validation.sampling.steps
        validation.held_out_severity_eval -> validation.gates.held_out_severity_eval
        validation.hallucination_test     -> validation.gates.hallucination_test

    The YAML side is unaffected: every record is still ``fold`` posture, so the
    arms keep declaring the legacy spelling and the validator relocates it.
    """

    def test_the_canonical_paths_are_the_ones_renames_declares(self) -> None:
        """Anti-transcription: if a record moves again, this fails first."""
        from mriforge.config.schemas.renames import RENAMES

        assert RENAMES["validation.sampler_steps"].canonical == (
            "validation.sampling.steps"
        )
        assert RENAMES["validation.held_out_severity_eval"].canonical == (
            "validation.gates.held_out_severity_eval"
        )
        assert RENAMES["validation.hallucination_test"].canonical == (
            "validation.gates.hallucination_test"
        )

    def test_sampler_steps_set(self, v2_settings: TrainingSettings) -> None:
        assert v2_settings.validation.sampling.steps is not None, (
            "validation.sampler_steps is required so the validation "
            "curve matches the deployment sampler step count."
        )
        assert v2_settings.validation.sampling.steps >= 1

    def test_held_out_severity_eval_enabled(
        self, v2_settings: TrainingSettings
    ) -> None:
        assert v2_settings.validation.gates.held_out_severity_eval is True, (
            "validation.held_out_severity_eval=True is required by the "
            "v2 standard so the cold-diffusion / multi-physics severity "
            "grid is exercised at validation time."
        )

    def test_hallucination_test_enabled(
        self, v2_settings: TrainingSettings
    ) -> None:
        ht = v2_settings.validation.gates.hallucination_test
        assert ht.enabled is True, (
            "validation.hallucination_test.enabled=True is required by "
            "the v2 standard (clinical-defensibility hook)."
        )


class TestProofArmsAreDropped:
    """Proof arms (vf_11..vf_20) must NOT be present as YAMLs.

    The DROPPED_PROOF_ARMS.md note relocated them to
    tests/test_vf_physics_identities.py — each is a deterministic
    mathematical identity, not a learnable problem.
    """

    DROPPED_STEMS = tuple(
        f"exp_vf_{n:02d}" for n in range(11, 21)
    )

    @pytest.mark.parametrize("stem", DROPPED_STEMS)
    def test_proof_arm_not_present(self, stem: str) -> None:
        matches = list(VF_DIR.glob(f"{stem}_*.yaml")) + list(VF_DIR.glob(f"{stem}.yaml"))
        assert not matches, (
            f"Proof arm {stem} should NOT exist in vf/ — it was "
            f"relocated to tests/test_vf_physics_identities.py "
            f"(see vf/DROPPED_PROOF_ARMS.md). Found: {matches}"
        )
