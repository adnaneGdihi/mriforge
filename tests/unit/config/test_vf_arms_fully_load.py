"""Every virtual-fiducial arm must survive a REAL ``TrainingSettings.from_yaml``.

``scripts/ci/check_experiment_configs_load.py`` is named as though it does this
and does not: it calls ``DataConfigSchema.validate_dataset_type`` on the
``dataset_type`` STRING and never constructs a ``TrainingSettings``. A config
whose structure is broken anywhere else passes it.

That gap shipped two unloadable arms in #519 — the acquisition fields of
``relaxometric_calibration.source`` / ``.target`` were indented one level short,
so ``tr_ms`` / ``te_ms`` / ``flip_deg`` landed as siblings of ``source`` and the
schema rejected them as extra inputs. The gate was green throughout.

This test closes that hole for the fiducial cohorts. It is deliberately dumb:
construct the real object, assert nothing about it. Anything that raises is a
config that cannot start.
"""

from __future__ import annotations

import pathlib

import pytest

from tests.utils.corpus import tracked_yamls

REPO = pathlib.Path(__file__).resolve().parents[3]
ARMS = sorted(
    p
    for d in ("vf", "vf_ulf")
    for p in tracked_yamls((REPO / "experiments" / "inprogress" / d), recursive=False)
    if "subvoxel" in p.name or "vfulf" in p.name
)


def test_the_cohorts_are_not_empty() -> None:
    """A glob that silently matches nothing would make every test below vacuous."""
    assert len(ARMS) >= 17, f"expected the vf + vf_ulf arms, found {len(ARMS)}"


@pytest.mark.parametrize("path", ARMS, ids=lambda p: p.stem)
def test_arm_constructs_a_real_training_settings(path: pathlib.Path) -> None:
    from spectramr.config.settings import TrainingSettings

    TrainingSettings.from_yaml(str(path))


@pytest.mark.parametrize("path", ARMS, ids=lambda p: p.stem)
def test_subvoxel_arms_carry_no_anti_super_resolution_smoothness(
    path: pathlib.Path,
) -> None:
    """``lambda_smooth`` is an L1 TV penalty on the PREDICTION, justified by
    "fields are smooth". That holds for the B0/B1 methods this strategy was
    built for and not for super-resolution, where the supervised quantity is
    anatomy: TV(truth) measures 1.66x TV(a blurred reconstruction), so the
    term's optimum is strictly smoother than the target."""
    from spectramr.config.settings import TrainingSettings

    macq = TrainingSettings.from_yaml(str(path)).physics.multi_acquisition
    if macq.method != "subvoxel_sr":
        pytest.skip("not a super-resolution arm")
    assert macq.lambda_smooth == 0.0


@pytest.mark.parametrize("path", ARMS, ids=lambda p: p.stem)
def test_subvoxel_arms_carry_a_high_frequency_term(path: pathlib.Path) -> None:
    """Without it the objective is plain MSE plus a smoothness penalty: nothing
    targets the detail these arms exist to recover, and the strategy never reads
    a `losses:` block so HFEN/adversarial/perceptual are absent by
    construction."""
    from spectramr.config.settings import TrainingSettings

    macq = TrainingSettings.from_yaml(str(path)).physics.multi_acquisition
    if macq.method != "subvoxel_sr":
        pytest.skip("not a super-resolution arm")
    assert macq.band_probe.enabled, "the band partition must be built"
    assert macq.band_probe.lambda_anatomy > 0.0
