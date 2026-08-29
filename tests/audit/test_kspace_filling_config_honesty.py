"""Cohort-wide invariants for the kspace_filling arms: declared == effective.

Complements ``test_kspace_filling_cohort_invariants.py``, which scopes itself to the
top-level arms plus the two attention sub-cohorts. This module parametrises over EVERY
arm under ``experiments/inprogress/kspace_filling`` — including ``dc_shootout`` and the
KAN/EMA ablations — because the defects it guards are config-shape defects that do not
respect cohort boundaries.

Two families:

* **Dead keys.** Keys declared across the cohort that no consumer reads: either
  undeclared on an ``extra="ignore"`` schema (``data.slice_aware``,
  ``data.volume_format``), retired from the schema in 2026-05
  (``loss_logging.{enabled,enable_tracking}``), carried as an untyped extra on
  an ``extra="allow"`` schema (``training.sampling_steps``), superseded
  (``training.num_workers`` -> ``data.num_workers``, issue #557), redundant with the
  field it aliases (``checkpoint.save_dir``), or with zero consumers repo-wide
  (``logging.random_validation_sampling``). Each has to be checked against the FILE
  rather than the resolved settings: the whole failure mode is that they vanish at load,
  so a resolved object cannot witness their absence (pitfall #15).

  ``loss_logging.csv_path`` was on that list and is NOT dead. The 2026-05 sweep
  removed it as never-wired, but it has a producer (``hpo.py`` writes it into
  every trial YAML) and a consumer (``paired_arms_audit.py`` reads it) -- so
  ``extra="ignore"`` was silently discarding a key both ends had agreed on
  (#795). It is a real field again; listing it here would flag a live knob as
  dead, the exact inverse of what this test is for.

* **Mask nesting.** Cold diffusion's forward process assumes ``M_{t+1} subset-of M_t``:
  k-space is removed as ``t`` grows, never added. This is measured by building the real
  ``KSpaceUndersamplingProcess`` through the same resolver the generator uses, not by
  reading the YAML. Arms whose undersampling pattern is the independent variable keep
  their declared mask and carry a RECORDED violation count, so that fixing one of them
  fails this test and forces the record to be updated rather than silently widening the
  exemption.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import torch

from mriforge.config.settings import TrainingSettings
from mriforge.models.diffusion.kspace_process import (
    KSpaceUndersamplingProcess,
    resolve_undersampling_kwargs,
)
from tests.utils.corpus import tracked_yamls

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COHORT_ROOT = _REPO_ROOT / "experiments" / "inprogress" / "kspace_filling"

_ARMS = tracked_yamls(_COHORT_ROOT)
_IDS = [str(p.relative_to(_COHORT_ROOT)) for p in _ARMS]

# (regex against the file, why it is dead). Anchored to the line start with its exact
# indentation so a nested homonym is not matched: ``enable_tracking`` also exists under
# ``metrics:``, where it IS read.
_DEAD_KEYS: tuple[tuple[str, str], ...] = (
    (
        r"^  slice_aware:",
        "undeclared on DataConfigSchema (extra='ignore'); use data.modes.val",
    ),
    (r"^  volume_format:", "undeclared on DataConfigSchema (extra='ignore')"),
    (
        r"^  random_validation_sampling:",
        "zero consumers repo-wide; use validation.shuffle_validation",
    ),
    (r"^  save_dir:", "validation_alias of checkpoint.checkpoint_dir; declared twice"),
)

# Sub-block-scoped dead keys: (top-level section, key regex, why).
_DEAD_IN_SECTION: tuple[tuple[str, str, str], ...] = (
    (
        "training",
        r"^  num_workers:",
        "dead (#557); every DataLoader reads data.num_workers",
    ),
    (
        "training",
        r"^  sampling_steps:",
        "untyped extra on extra='allow'; nothing reads it",
    ),
    (
        "loss_logging",
        r"^  enabled:",
        "field deleted from LossLoggingConfigSchema in 2026-05",
    ),
    # NOT `loss_logging.csv_path` -- restored as a real field in #795; it has a
    # producer (hpo.py) and a consumer (paired_arms_audit.py). See the module
    # docstring.
    (
        "loss_logging",
        r"^  enable_tracking:",
        "field deleted from LossLoggingConfigSchema in 2026-05",
    ),
)

# Arms whose undersampling mask is the INDEPENDENT VARIABLE (or the shared context of a
# within-cohort comparison), mapped to their MEASURED nesting violations at 256x256.
# Non-zero entries are known defects tracked separately; they are recorded, not waived,
# so a change in either direction fails this test.
_MASK_IS_IV: dict[str, int] = {
    # Was 15 as the *equispaced* control of the mask family. The arm is the shared
    # BASELINE the ablations name, so carrying a 15/27 non-monotone ladder made every
    # ablation a comparison against a broken control; it moved to ``random`` (the family
    # the ablations already ran) and now measures 0. The equispaced family itself is
    # still represented by its own sibling below.
    "experiment_11_kspace_cold_diffusion.yaml": 0,
    "experiment_11_kspace_cold_diffusion_radial.yaml": 6,
    "experiment_11_kspace_cold_diffusion_spiral.yaml": 6,
    "experiment_11_kspace_cold_diffusion_gaussian.yaml": 0,
    "experiment_11_kspace_cold_diffusion_variable_density.yaml": 0,
    "experiment_11_kspace_cold_diffusion_random.yaml": 0,
    "experiment_11a_swin_diff_rec.yaml": 0,
    "experiment_11b_diff_varnet.yaml": 0,
    "experiment_11e_nafnet.yaml": 0,
    # dc_shootout: the shared context is ``density_nested``; the IV is the DC mode. Was 1
    # under ``variable_density_1d``, whose 24-LINE ACS costs 9.38% of k-space and caps the
    # realised ladder at 10.7x against a declared 32x -- and which #1054 then drove to 2.
    # A 2D family makes the same ACS a 24x24 SQUARE (0.88%), so the budget fits and the
    # cascade is monotone: all seven now measure 0.
    "experiment_11_dc_adaptive.yaml": 0,
    "experiment_11_dc_hard.yaml": 0,
    "experiment_11_dc_kan_adaptive.yaml": 0,
    "experiment_11_dc_noise_adaptive.yaml": 0,
    "experiment_11_dc_noise_adjusted.yaml": 0,
    "experiment_11_dc_soft.yaml": 0,
    "experiment_11_dc_target_aware_fsdc.yaml": 0,
}

_MATRIX = 256


@pytest.fixture(scope="module", params=_ARMS, ids=_IDS)
def arm(request: pytest.FixtureRequest) -> Path:
    return request.param


def test_arms_discovered() -> None:
    """An empty parametrization would pass every invariant below in silence."""
    assert len(_ARMS) >= 50, f"expected the full kspace_filling cohort, found {len(_ARMS)}"


def _section_of(lines: list[str], index: int) -> str:
    for i in range(index, -1, -1):
        if re.match(r"^[a-z_][a-z_0-9]*:", lines[i]):
            return lines[i].split(":", 1)[0]
    return ""


def test_no_dead_keys_declared(arm: Path) -> None:
    """A declared key that no consumer reads is pitfall #15 in its purest form."""
    lines = arm.read_text().splitlines()
    offenders: list[str] = []
    for i, line in enumerate(lines):
        for pattern, why in _DEAD_KEYS:
            if re.match(pattern, line):
                offenders.append(f"{line.strip()} — {why}")
        for section, pattern, why in _DEAD_IN_SECTION:
            if re.match(pattern, line) and _section_of(lines, i) == section:
                offenders.append(f"{section}.{line.strip()} — {why}")
    assert not offenders, f"{arm.name} declares keys nothing reads:\n  " + "\n  ".join(offenders)


def _model_kwargs(settings: TrainingSettings) -> dict[str, Any]:
    return dict(settings.model.model_kwargs or {})


def test_reverse_step_count_declared_where_it_is_read(arm: Path) -> None:
    """``model_kwargs.sampling_steps`` is the ONLY key the reverse loop reads.

    ``KSpaceColdDiffusionGenerator.__init__`` does ``kwargs.get("sampling_steps", 50)``,
    and ``_resolve_validation_sampling_steps`` cannot forward
    ``training.diffusion.sampling_steps`` because ``PhysicsInformedColdDiffusion.sample``
    has no step-count parameter (the #480 signature-probe mechanism). Absent here, every
    arm silently ran the hardcoded 50.
    """
    settings = TrainingSettings.from_yaml(str(arm))
    mk = _model_kwargs(settings)
    timesteps = mk.get("timesteps")
    assert "sampling_steps" in mk, (
        f"{arm.name}: model.model_kwargs.sampling_steps is absent, so the reverse loop "
        "falls back to the hardcoded default 50 (#566/#570)."
    )
    assert mk["sampling_steps"] == timesteps, (
        f"{arm.name}: reverse steps {mk['sampling_steps']} != timesteps {timesteps}. The "
        "schedule is linspace(T-1, 0, steps+1).long() then set()-deduplicated, so a "
        "larger value silently collapses and the declared number is never the effective one."
    )


def test_validation_is_sliced_not_whole_volume(arm: Path) -> None:
    """M4RawRepetitionDataset serves a whole ``(2,H,W,Slices)`` volume per index.

    Only a TorchIO queue with patch depth 1 cuts it into 2-D slices. ``data.modes.val``
    is the live, schema-validated surface for that; the legacy ``data.slice_aware`` and
    ``data.use_queue_for_validation`` are both dropped by ``extra="ignore"`` before their
    ``hasattr`` / ``getattr`` probes run.
    """
    settings = TrainingSettings.from_yaml(str(arm))
    modes = getattr(settings.data, "modes", None)
    assert modes is not None, f"{arm.name}: data.modes absent — validation runs whole-volume."
    val = modes.get("val") if isinstance(modes, dict) else getattr(modes, "val", None)
    sampler = val.get("sampler") if isinstance(val, dict) else getattr(val, "sampler", None)
    stype = sampler.get("type") if isinstance(sampler, dict) else getattr(sampler, "type", None)
    assert str(stype) == "uniform", (
        f"{arm.name}: data.modes.val.sampler.type is {stype!r}; 'uniform' is the only "
        "patch sampler wired ('full'/'grid' RAISE at torchio_queue_builder.py:481)."
    )


def test_dataloader_workers_not_single(arm: Path) -> None:
    """``training.num_workers`` was dead (#557), so the cohort ran single-worker."""
    settings = TrainingSettings.from_yaml(str(arm))
    workers = settings.data.loader.num_workers
    assert workers >= 4, (
        f"{arm.name}: data.num_workers={workers}. This is the key every DataLoader "
        "actually reads; training.num_workers is dead."
    )


def _nesting_violations(arm: Path) -> tuple[int, int, int]:
    """Build the REAL undersampling process and count ``M_{t+1} not-subset-of M_t``."""
    settings = TrainingSettings.from_yaml(str(arm))
    mk = _model_kwargs(settings)
    kwargs = resolve_undersampling_kwargs(settings.undersampling, mk)
    timesteps = int(mk.get("timesteps", 28))
    process = KSpaceUndersamplingProcess(num_timesteps=timesteps, **kwargs)
    masks = [
        process.mask_generator.generate_batch_masks(
            batch_size=1,
            timesteps=torch.tensor([t]),
            image_shape=(_MATRIX, _MATRIX),
            pattern=process.mask_type,
        )[0, 0]
        > 0
        for t in range(timesteps)
    ]
    added = [int((masks[t + 1] & ~masks[t]).sum()) for t in range(timesteps - 1)]
    return sum(1 for a in added if a > 0), sum(added), timesteps - 1


def test_forward_process_is_nested(arm: Path) -> None:
    """``M_{t+1} subset-of M_t`` — measured through the production resolver.

    A violation means the forward process ADDS k-space as ``t`` grows, which the reverse
    loop has no mechanism to undo: the model is trained to invert a degradation that is
    not monotone. ``equispaced`` breaks this on 15 of 27 steps at 256x256 (every-8th and
    every-10th phase-encode lines are not nested), which is why the cohort moved to
    ``random`` on 2026-07-29 wherever the mask was not the independent variable.
    """
    violations, added, steps = _nesting_violations(arm)
    expected = _MASK_IS_IV.get(arm.name)
    if expected is None:
        assert violations == 0, (
            f"{arm.name}: forward process ADDS k-space on {violations}/{steps} steps "
            f"({added} bins). Cold diffusion assumes M_(t+1) subset-of M_t."
        )
        return
    assert violations == expected, (
        f"{arm.name} is a mask-is-the-IV arm with a RECORDED {expected}/{steps} "
        f"violations; measured {violations}. If the mask was fixed, update the record in "
        "_MASK_IS_IV — do not widen the exemption silently."
    )


# Arms whose realised top rung falls short of the declared ladder, with the MEASURED
# realised acceleration at 256x256. Issue #534's remedy (min_center_fraction < the
# budget) fixed the random-Cartesian arms. Recorded rather than waived: a value that
# moves fails this test.
#
# The seven ``dc_shootout`` arms and the 2-D Gaussian left this dict on 2026-08-12.
# ``9facb123f`` set ``min_center_fraction: 0.02`` on them -- the value their 44 sound
# cohort siblings already used -- and they now realise every declared rung (12.8x and
# 12.5x -> 32.0x, measured). ``0f46db1fd`` drained the same eight from
# ``scripts/ci/acceleration_ladder_baseline.txt`` in the same breath (9 paths -> 1).
# This dict is a SECOND record of that one fact and was missed, so it went on
# asserting a shortfall that no longer exists -- the same shape as the entry above it
# warns about, in the opposite direction.
#
# The two survivors are capped by trajectory geometry rather than by the ACS budget,
# so no ``min_center_fraction`` change reaches them.
_UNDERSHOOTING_LADDERS: dict[str, float] = {
    "experiment_11_kspace_cold_diffusion_spiral.yaml": 22.56,
    "experiment_11_kspace_cold_diffusion_radial.yaml": 29.98,
}


def _realised_top_rung(arm: Path) -> float:
    settings = TrainingSettings.from_yaml(str(arm))
    mk = _model_kwargs(settings)
    kwargs = resolve_undersampling_kwargs(settings.undersampling, mk)
    timesteps = int(mk.get("timesteps", 28))
    process = KSpaceUndersamplingProcess(num_timesteps=timesteps, **kwargs)
    kept = [
        max(
            int(
                (
                    process.mask_generator.generate_batch_masks(
                        batch_size=1,
                        timesteps=torch.tensor([t]),
                        image_shape=(_MATRIX, _MATRIX),
                        pattern=process.mask_type,
                    )[0, 0]
                    > 0
                ).sum()
            ),
            1,
        )
        for t in range(timesteps)
    ]
    return max(_MATRIX * _MATRIX / k for k in kept)


def test_declared_ladder_top_rung_is_realised(arm: Path) -> None:
    """The steepest rung the arm advertises has to be the one the mask delivers.

    This is issue #534's property, measured rather than declared. When the ACS band is
    static it IS the entire sampling budget past ``R = 1/center_fraction``, so every
    nominally-steeper rung realises the same mask: the arm reports a ``val_*_32x`` column
    that was really acquired at 12x, and the timesteps spanning those rungs are physically
    identical, so a timestep-conditioned network spends gradient telling apart inputs that
    do not differ.

    Overshoot is failed too. ``acceleration_range`` and ``max_acceleration`` are separate
    fields and the step schedule honours the former, so a stale bound lets an arm run past
    the acceleration it claims.
    """
    settings = TrainingSettings.from_yaml(str(arm))
    declared = float(settings.undersampling.max_acceleration)
    realised = _realised_top_rung(arm)
    recorded = _UNDERSHOOTING_LADDERS.get(arm.name)
    if recorded is not None:
        assert realised == pytest.approx(recorded, rel=0.02), (
            f"{arm.name} has a RECORDED short ladder (declared {declared:g}x, realised "
            f"{recorded:g}x); measured {realised:.2f}x. If the ladder was fixed, drop the "
            "entry from _UNDERSHOOTING_LADDERS — do not widen the record silently."
        )
        return
    assert realised == pytest.approx(declared, rel=0.05), (
        f"{arm.name}: declares max_acceleration={declared:g}x but realises "
        f"{realised:.2f}x at {_MATRIX}x{_MATRIX}. Every val_*_{{R}}x column is then "
        "labelled with an acceleration the mask did not deliver."
    )
