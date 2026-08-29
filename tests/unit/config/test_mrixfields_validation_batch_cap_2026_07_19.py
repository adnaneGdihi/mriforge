"""Regression guard: arms with an unbounded validation set cap ``num_validation_batches``.

``mrixfields_pairing_policy: all_pairs`` enumerates every *ordered* field pair per
subject, so the validation set grows quadratically with the subject count (~1192-2384
batches on the ChallengeData split). With ``validation.num_validation_batches`` unset
(schema default ``None``) the validation loop walks the *entire* set at every checkpoint
(``pipelines/train.py`` only breaks when ``i >= num_validation_batches``). On the
2026-07-19 cluster run this cgroup-OOM-killed **all 70 dispatched arms at the first
validation (step 1000)**: every ``training_metrics.csv`` stopped at exactly iter 1000, no
``validation_metrics.csv`` was written, and the SLURM logs show
``Detected N oom_kill events ... OOM Killed`` right after ``Validating: 0/2384``. Training
reached iter 1000 fine; validation is what blew up.

The fix is the ``num_validation_batches`` cap that 71 other in-progress arms already carry
(``promoted/`` / ``robustness/`` / ``sparse_frame/`` all use ``8``) — it bounds the val
loop to a fixed batch budget regardless of val-set size, which also bounds the
dataloader-worker RAM that the cgroup OOM tracks.

**Two sweeps, and neither selects arms by the thing it asserts on.** An earlier version of
this guard collected arms via ``isinstance(cfg.get("validation"), dict)``, which meant
deleting the whole ``validation:`` block *removed the arm from its own guard* while
leaving it maximally exposed (no block → ``TrainingSettings.validation is None`` →
``train.py`` resolves no cap → ``strided_validation_subset`` returns the set unchanged →
the original OOM). Arms are now selected by **pairing policy** (the root cause, repo-wide)
and by **cohort directory** (uniformity), so a missing block is a failure rather than an
exemption.

Structural ``yaml.safe_load`` guard (no torch): the full Pydantic load path is already
exercised by ``tests/integration/test_inprogress_yamls.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.utils.corpus import tracked_yamls
from tests.utils.corpus_keys import read_key, spelling_used

REPO = Path(__file__).resolve().parents[3]
EXPERIMENTS = REPO / "experiments"
MRIXFIELDS = EXPERIMENTS / "inprogress" / "mrixfields2026"

# The repo convention (71 arms across promoted/robustness/sparse_frame). 8 batches is a
# bounded, representative validation subset that does not OOM the cgroup.
#
# NOTE: this is a cap in *batches*, not samples. The effective sample budget is
# ``cap * val_batch_size``, and ``val_batch_size`` falls back to ``data.batch_size``
# (data_pipeline_director.py:374) which is NOT uniform across the cohort — arms run at
# batch 4/2/1, giving 32/16/8 validation samples. Uniform *within* an ablation pair
# (both members share a batch size), which is what pitfall #17 requires; not uniform
# across the whole cohort.
EXPECTED_CAP = 8

# All arms known to enumerate pairs quadratically. Selecting on this rather than on the
# directory is what catches out-of-cohort arms (novel_2026/breakthrough_fieldscore_*).
UNBOUNDED_POLICIES = {"all_pairs"}


# Canonical paths. `read_key` resolves each under the canonical spelling or any
# legacy one that still folds, driven by RENAMES. The corpus is mid-drain, so
# `mrixfields2026` spells these canonically while the other cohorts do not, and
# this sweep is repo-wide. Reading one spelling makes every arm on the other
# fall out of its own guard SILENTLY: the lookup returns None, which reads as
# "the arm does not declare this" rather than "I looked in the wrong place".
# That is exactly how this file broke when the cohort was drained -- and the
# reader lives in tests/utils/ so the next rename does not need this edit again.
CAP_PATH = "validation.loader.num_batches"
POLICY_PATH = "data.mrixfields.pairing_policy"


def _load(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    return cfg if isinstance(cfg, dict) else {}


def _arms_with_unbounded_pairing() -> list[Path]:
    """Every arm repo-wide whose pairing policy makes the val set grow quadratically.

    Substring pre-filter before the YAML parse: the sweep is repo-wide (1474 files) but
    only ~80 mention the key, and parsing all of them at collection time made this module
    take minutes to import. The filter uses the canonical leaf, which is a substring of
    the retired spelling (`mrixfields_pairing_policy`), so one pattern catches both.
    """
    arms = []
    for y in tracked_yamls(EXPERIMENTS):
        text = y.read_text()
        if "pairing_policy" not in text:
            continue
        if read_key(yaml.safe_load(text), POLICY_PATH) in UNBOUNDED_POLICIES:
            arms.append(y)
    return arms


def _mrixfields_cohort_arms() -> list[Path]:
    """Every arm in the cohort, by directory membership — not by block presence."""
    return tracked_yamls(MRIXFIELDS)


def _assert_capped(arm: Path, *, reason: str) -> None:
    cap = read_key(_load(arm), CAP_PATH)
    assert cap is not None, (
        f"{arm.relative_to(REPO)}: no validation batch cap under `{CAP_PATH}` "
        f"or any legacy spelling of it. {reason} With no cap resolved the val "
        "loop walks the full set — the 2026-07-19 cgroup OOM."
    )
    assert isinstance(cap, int) and cap > 0, (
        f"{arm.relative_to(REPO)}: cap must be a positive int, got {cap!r}"
    )


def test_unbounded_pairing_sweep_is_non_empty() -> None:
    """Guard against a path/key regression silently making the root-cause sweep vacuous."""
    arms = _arms_with_unbounded_pairing()
    assert len(arms) >= 29, (
        f"expected every `all_pairs` arm repo-wide, found {len(arms)} — the nested "
        "policy lookup or the experiments/ path has regressed"
    )


def test_cohort_sweep_is_non_empty() -> None:
    """Guard against the cohort glob silently going vacuous."""
    arms = _mrixfields_cohort_arms()
    assert (
        len(arms) >= 75
    ), f"expected the full mrixfields2026 cohort, found {len(arms)}"


def test_the_canonical_cap_spelling_is_live() -> None:
    """`read_key`'s canonical branch must not go dead while the fallback hides it.

    This carried a second leg asserting the RETIRED spelling was also live,
    because mid-drain a broken canonical lookup would have fallen through to the
    legacy path and left the suite green on a path the schema no longer matches.
    PR #906 drained every `inprogress/` arm, so in this guard's scope — the
    mrixfields2026 cohort plus the unbounded-pairing arms — no arm spells the cap
    the old way and that leg went vacuous. It was deleted per its own
    instructions rather than loosened.

    The leg is not missed: with the legacy declarations gone, a broken
    `CAP_PATH` makes `spelling_used` return `None` for every arm, so the
    remaining assertion fails on exactly the case the deleted one covered.

    Scope note — `read_key`'s legacy fallback is NOT dead code generally. 19
    loadable arms under `templates/` and `validated/` still declare
    `validation.num_validation_batches`, and 100 rename records are still
    declared corpus-wide. Only this cohort is drained.
    """
    used = [
        spelling_used(_load(arm), CAP_PATH)
        for arm in _mrixfields_cohort_arms() + _arms_with_unbounded_pairing()
    ]
    assert CAP_PATH in used, (
        f"no arm spells the cap canonically — `{CAP_PATH}` is unreachable, so "
        "this guard is reading a path the corpus does not use"
    )


@pytest.mark.parametrize(
    "arm",
    _arms_with_unbounded_pairing(),
    ids=lambda p: str(p.relative_to(EXPERIMENTS)),
)
def test_all_pairs_arm_caps_validation_batches(arm: Path) -> None:
    """Root cause: any `all_pairs` arm, anywhere, must bound its validation loop."""
    _assert_capped(
        arm,
        reason=(
            "This arm sets `mrixfields_pairing_policy: all_pairs`, so its validation set "
            "grows quadratically with subject count."
        ),
    )


@pytest.mark.parametrize(
    "arm", _mrixfields_cohort_arms(), ids=lambda p: str(p.relative_to(MRIXFIELDS))
)
def test_mrixfields_cohort_caps_uniformly(arm: Path) -> None:
    """Cohort uniformity: the validation budget is not an ablated knob (pitfall #17)."""
    _assert_capped(
        arm, reason="Every mrixfields2026 arm carries the cohort validation cap."
    )
    cap = read_key(_load(arm), CAP_PATH)
    assert cap == EXPECTED_CAP, (
        f"{arm.name}: expected the cohort cap {EXPECTED_CAP} (repo convention), got {cap}. "
        "Keep it uniform so ablation pairs validate on the same batch budget (pitfall #17)."
    )
