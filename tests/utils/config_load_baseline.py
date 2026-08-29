"""Which corpus arms are known NOT to construct — one owner for the tests.

The sibling of :mod:`tests.utils.corpus`, which elects one owner for *which*
arms a parametric test sweeps. This elects one owner for the follow-on
question: *which of those arms is already known to be unloadable, and so must
not be reported as a fresh regression.*

Why it exists
-------------
``scripts/ci/config_load_baseline.txt`` is the repo's existing record of arms
that ``TrainingSettings.from_yaml`` cannot construct today. Its own header
states the contract: *"A path here is tracked DEBT, not permission: removing
one is the goal, adding one needs a reason in the PR that adds it."*

Until now exactly one consumer honoured it —
``scripts/ci/check_experiment_configs_load.py``. The three smoke sweeps over
``experiments/active/`` did not, and had no way to express "known debt" at all:
they globbed the directory and called ``from_yaml`` on every hit, so any arm on
the baseline surfaced as a hard failure in each of them.

That went from theoretical to load-bearing on 2026-08-12, when PR #964 made the
deferred W&B knobs *raise* instead of being ignored (#675). The owner's accepted
ruling was to record the 20 ``active/`` + 3 ``validated/`` arms that declare
them as tracked debt (commit ``9208e41a0``) rather than edit a corpus that
``tests/smoke/test_deep_config_integrity.py`` itself documents as "a separate
owner decision from the ``inprogress/`` migration". The CI gate was taught about
the debt; the smoke sweeps were not, and cluster job ``8079300`` reported **161
failures** across the three of them — every one an arm the repo had already
decided to carry.

Why membership, and never a blanket ``try``/``except``
------------------------------------------------------
The point of the baseline is that it is a **ratchet**. An arm that is *not*
listed and fails to construct is a genuine regression and must still fail
loudly. So the exemption here is per-arm membership, checked against the file —
never a broad "loading is allowed to fail" guard, which would retire the
sweep's entire reason for existing while leaving it green.

``strict=False`` on the xfail is deliberate: policing *stale* entries (a
baselined arm that constructs again) already has an owner in
``check_experiment_configs_load.py``, which reports it and offers
``--update-baseline``. A strict xfail here would be a second enforcement point
for one fact, which is the shape ``electing-one-owner`` warns about.

Parsing
-------
The parser below is deliberately a re-read of the *file* rather than an import
of ``check_experiment_configs_load.read_baseline``. The file is the SSOT; the
five-line parse is not worth coupling three smoke modules' **collection** to a
CI script's importability, in a tree where a broken schema import would then
take out the sweeps that exist to detect it. ``tests/unit/utils/
test_config_load_baseline.py`` pins the two parsers against each other so the
format cannot drift apart.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from tests.utils.corpus import repo_root

__all__ = ["baseline_path", "baselined_configs", "corpus_params", "is_baselined"]


def baseline_path() -> Path:
    """Absolute path to the baseline file."""
    return repo_root() / "scripts" / "ci" / "config_load_baseline.txt"


@lru_cache(maxsize=1)
def baselined_configs() -> frozenset[str]:
    """Repo-relative paths of arms known not to construct.

    A missing baseline file yields an empty set — the same posture
    ``check_experiment_configs_load.read_baseline`` takes. That is the safe
    direction: with no recorded debt every arm is held to the strict standard,
    so the failure mode is a noisy sweep, never a silently permissive one.
    """
    path = baseline_path()
    if not path.is_file():
        return frozenset()
    return frozenset(
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )


def is_baselined(config_path: Path | str) -> bool:
    """Whether ``config_path`` is recorded as known-unloadable debt.

    Accepts absolute or repo-relative paths, because the sweeps disagree about
    which they hold: ``tests.utils.corpus`` returns absolute paths while the
    older ``glob.glob`` call sites carry repo-relative strings.
    """
    candidate = Path(config_path)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(repo_root())
        except ValueError:
            return False
    return candidate.as_posix() in baselined_configs()


def corpus_params(config_paths) -> list:
    """``pytest.param`` per arm, xfailing the ones on the baseline.

    Drop-in for the bare list in a ``@pytest.mark.parametrize("config_path",
    ...)``. The test body is left completely unchanged: an arm that cannot
    construct still raises exactly where it did before, and the mark is what
    turns that into recorded debt instead of a regression.
    """
    params = []
    for path in config_paths:
        marks = (
            [
                pytest.mark.xfail(
                    reason=(
                        f"{Path(path).name} is on scripts/ci/config_load_baseline.txt "
                        "— tracked debt, not a regression (see #675)"
                    ),
                    strict=False,
                )
            ]
            if is_baselined(path)
            else []
        )
        params.append(pytest.param(path, marks=marks, id=Path(path).name))
    return params
