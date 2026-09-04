"""Experiment-corpus enumeration for parametric tests — one owner.

The sibling of :mod:`tests.utils.registry_iterators`, which exists so test
modules can iterate the component registries "without duplicating import
logic". This does the same for the YAML corpus under ``experiments/``.

Why it exists
-------------
A test that enumerates the corpus with ``Path.rglob("*.yaml")`` **has a
different subject on every machine.** Cluster CPU job ``8004252`` reported 24
failures against two arms --
``kspace_filling/reverse_sampler_ab/experiment_11_attention_none__rev_additive.yaml``
and ``..._rev_replace_freeze.yaml`` -- that exist in **no git history on any
branch**::

    git log --all --oneline --diff-filter=A -- '*rev_additive*'   # no output

They are output of ``scripts/experiments/exp11_reverse_sampler_ab.py``
(``DEFAULT_OUTDIR``), generated 2026-07-11, never gitignored and never added,
still resident on the cluster's working tree. Six test modules globbed them in
and failed on ``training.seed`` -- a ``posture="raise"`` rename since
2026-07-31. Nobody could reproduce or fix those failures from a clean checkout,
because there was nothing to fix.

The same defect makes a corpus-relative *audit* non-reproducible rather than
merely noisy: ``scripts/data/check_inprogress_dataset_wiring.py``'s
``family_consensus`` rule compares an arm's channel count against its **peers**,
so corpus membership *is* the verdict -- 4 findings on the cluster and 2 on a
dev box, from a byte-identical arm and a byte-identical script.

``CLAUDE.md`` already names the authority for this question, in the corpus
census it publishes: *"committed files via ``git ls-files``, not an on-disk
glob"*. Five test modules had independently reached the same conclusion and
grown their own ``subprocess.run(["git", "ls-files", ...])``; the rest had not.
Two populations answering one question is the shape ``electing-one-owner``
describes, so this module is the election.

The deliberate exemption
------------------------
``tests/unit/experiments/test_kspace_filling_cohort_consistency.py::
test_no_cohort_arm_is_hidden_from_git`` **must keep globbing the filesystem.**
It is the tripwire whose entire purpose is to compare on-disk against tracked
and fail when they diverge; routing it through here would make it compare
tracked against tracked and pass vacuously forever. Its companion
``test_no_cohort_directory_is_covered_by_a_gitignore_rule`` is likewise exempt.

With that split, an untracked arm produces **one** honest, actionable failure
naming the real problem, instead of 24 misattributed ones blaming schema keys.

No opt-in for untracked files
-----------------------------
Deliberately absent. An uncommitted arm cannot be reviewed, reproduced, audited
or run on the cluster by anyone but its author, so a corpus-wide invariant has
no business asserting over it -- and an ``include_untracked=`` switch would be a
knob with no validator and no provenance stamp (pitfall #15). Checking a single
work-in-progress arm is already served by ``spectramr audit <file>``.

Two absences of git, answered differently
-----------------------------------------
``repo_root`` discriminates *no checkout* from *broken checkout*, because they
call for opposite responses and only one of them is a defect:

===============================  ==========================================
Tree                             Response
===============================  ==========================================
no ``.git`` above this module    visible ``pytest.skip`` -- an sdist, tarball
                                 or ZIP. Nothing to enumerate, nothing to fix.
``.git`` present, git fails      ``RuntimeError`` -- git off ``PATH``, a
                                 corrupt repo, a ``safe.directory`` refusal.
===============================  ==========================================

The first row is not the fallback this module forbids. A filesystem glob answers
the corpus question with the **wrong subject**; a skip declines to answer and
says so. The distinction matters because **18 test modules reach this function at
import time**, and pytest answers a collection error by ``Interrupted`` -- one of
them zeroes out the whole suite. ``tests/`` is not excluded from the sdist, so a
hard raise here breaks the suite for exactly the people it was shipped to.

The second row must stay a raise. Relaxing it would convert those same 18 modules
into silent no-ops in a working tree whose git is merely misconfigured.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

__all__ = ["repo_root", "tracked_paths", "tracked_yamls"]


def _git_dir_above(start: Path) -> Path | None:
    """The nearest ``.git`` at or above ``start``, or ``None`` if there is none.

    ``.exists()`` rather than ``.is_dir()``: inside a linked worktree ``.git`` is
    a *file* holding a ``gitdir:`` pointer, and that is a real checkout.
    """
    for directory in (start, *start.parents):
        candidate = directory / ".git"
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Absolute path to the repository root.

    Asks git rather than counting ``parents[n]`` -- a hardcoded index is the
    single most repeated defect in this suite's history (the ``src`` ->
    ``src/spectramr`` refactor invalidated it in four separate files, and a stale
    index is *silent* where a directory is globbed, because a glob over a
    non-existent directory returns nothing and the test passes by scanning zero
    files). ``--show-toplevel`` is also correct inside a linked worktree, where
    ``.git`` is a file rather than a directory.
    """
    here = Path(__file__).resolve().parent
    if _git_dir_above(here) is None:
        # Two different absences, and only one of them is a defect.
        #
        # No ``.git`` anywhere above this module means the tree was never a
        # checkout: a PyPI sdist, a release tarball, a downloaded ZIP. There is
        # no corpus to enumerate and nothing to repair, so the honest answer is
        # to decline visibly. Raising here instead is a *collection* error in
        # the 18 modules that reach this function at import time, and pytest
        # answers a collection error by Interrupting the whole session -- one
        # of those modules zeroes out the entire suite for every non-git
        # consumer. ``tests/`` is not excluded from the sdist, so that is
        # precisely the population the suite ships to.
        #
        # ``allow_module_level`` is what lets ONE owner serve both populations:
        # at import time it skips the module, and inside a test function the
        # flag is inert and this behaves as an ordinary skip. No caller changes.
        #
        # This is not the fallback the module docstring forbids. A filesystem
        # glob answers the question with the WRONG SUBJECT; a skip declines to
        # answer and says so.
        pytest.skip(
            f"tests.utils.corpus enumerates git-tracked files, and {here} sits in "
            f"a tree with no .git -- an sdist, tarball or ZIP rather than a "
            f"checkout. Corpus-dependent tests cannot run here. `git clone` the "
            f"repository to run them.",
            allow_module_level=True,
        )
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        raise RuntimeError(
            f"tests.utils.corpus needs a git checkout to enumerate the experiment "
            f"corpus. A .git exists at {_git_dir_above(here)}, so this IS a "
            f"checkout, but `git rev-parse --show-toplevel` failed in {here} -- "
            f"git missing from PATH, a corrupt repository, or a safe.directory "
            f"refusal. Fix the checkout -- do NOT fall back to a filesystem glob, "
            f"which is the bug this module exists to remove, and do NOT relax this "
            f"into the skip above: a broken worktree that quietly skips turns 18 "
            f"modules into silent no-ops."
        ) from exc
    return Path(out.stdout.strip())


@lru_cache(maxsize=1)
def _tracked() -> frozenset[Path]:
    """Every git-tracked path in the repo, absolute. One subprocess per session.

    Submodule contents are correctly absent: ``git ls-files`` reports a
    submodule as its single gitlink entry, not its working-tree files.
    """
    root = repo_root()
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return frozenset(root / line for line in out.stdout.split("\0") if line)


def tracked_paths(directory: Path | str, pattern: str, *, recursive: bool = True) -> list[Path]:
    """Tracked files under ``directory`` matching ``pattern``, sorted.

    Drop-in for ``sorted(directory.rglob(pattern))`` / ``sorted(directory.glob(
    pattern))``. ``directory`` may be absolute or repo-relative; the returned
    paths are absolute, matching what the glob forms yield.
    """
    base = Path(directory)
    if not base.is_absolute():
        base = repo_root() / base
    base = base.resolve()

    matcher = base.rglob if recursive else base.glob
    # Intersect the glob with the tracked set rather than filtering the tracked
    # set by hand: `pattern` keeps pathlib's exact semantics (character classes,
    # `**`, suffix matching) instead of a second, subtly-different implementation.
    return sorted(p for p in matcher(pattern) if p in _tracked())


def tracked_yamls(
    directory: Path | str, pattern: str = "*.yaml", *, recursive: bool = True
) -> list[Path]:
    """Tracked ``*.yaml`` arms under ``directory``, sorted.

    The common case::

        _ARMS = tracked_yamls(_COHORT_ROOT)                    # was rglob("*.yaml")
        _ARMS = tracked_yamls(_DIR, "exp_*.yaml", recursive=False)   # was glob(...)
    """
    return tracked_paths(directory, pattern, recursive=recursive)
