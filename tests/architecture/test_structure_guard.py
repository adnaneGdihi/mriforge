"""Fitness function: no NEW oversized files or over-deep inheritance towers.

Cached-cascade Phase 0, report-only baseline. Flags source files > 300 LOC and
classes whose inheritance chain *through our own classes* exceeds depth 2
(framework bases like ``nn.Module`` terminate the chain). Existing offenders are
baselined so they don't block; only NEW ones fail. Structural refactors of the
baselined offenders (file splits, inheritance flattening) are FLAGGED follow-ups,
not forced by this gate.
"""

from __future__ import annotations

import pytest

from ._fitness_lib import (
    BASELINE_DIR,
    grown_entries,
    ratchet,
    scan_deep_inheritance,
    scan_dispatch_hell,
    scan_exception_dispatch,
    scan_large_files,
    stale_entries,
)

pytestmark = pytest.mark.architecture

#: Every baseline whose entries a scanner can re-derive, paired with that
#: scanner. Wiring only *some* of these is how ``dispatch_hell.txt`` kept a dead
#: ``SchedulerBuilder.build`` entry (deleted in 8d01b95c5, same commit as the
#: three director entries) — the omission is invisible unless something checks
#: the mapping against the directory, which
#: ``test_every_reproducible_baseline_is_checked`` does.
CURRENT_SCANS = {
    "large_files.txt": lambda: scan_large_files(max_loc=300),
    "deep_inheritance.txt": lambda: scan_deep_inheritance(max_depth=2),
    "dispatch_hell.txt": lambda: scan_dispatch_hell(min_branches=3),
    "exception_dispatch.txt": scan_exception_dispatch,
}

#: Baselines a scanner here cannot re-derive on its own. The two signature-drift
#: baselines are produced by ``test_signature_drift.py`` from collectors that
#: need its canonical-call predicate, so staleness there is that file's to own.
#: ``unresolved_imports.txt`` likewise: deciding whether a name resolves means
#: actually IMPORTING the target module, which every scanner here deliberately
#: avoids (they are pure-AST). Its staleness is owned by
#: ``test_import_names_resolve.py::test_baseline_has_no_stale_entries``.
NOT_REPRODUCIBLE_HERE = {
    "step_signature_drift.txt",
    "builder_signature_drift.txt",
    "unresolved_imports.txt",
}

#: Flat per-baseline slack a hard growth gate would use (00_MASTER.md §5):
#: 25 for LOC, 0 for counts. Flat, never proportional.
SLACK = {"large_files.txt": 25}


def test_no_new_oversized_files() -> None:
    current = scan_large_files(max_loc=300)
    new = ratchet(
        "large_files.txt",
        current,
        header="source files > 300 LOC",
    )
    assert not new, (
        "New source file(s) exceed 300 LOC — split them (or, if justified, "
        "regenerate the baseline):\n  " + "\n  ".join(sorted(new))
    )


def test_no_new_deep_inheritance() -> None:
    current = scan_deep_inheritance(max_depth=2)
    new = ratchet(
        "deep_inheritance.txt",
        current,
        header="classes whose inheritance depth through our classes exceeds 2",
    )
    assert not new, (
        "New over-deep inheritance tower(s) — prefer composition / mixins "
        "(perf_design.md: inheritance depth <= 2):\n  " + "\n  ".join(sorted(new))
    )


# ---------------------------------------------------------------------------
# The two holes the identity-keyed ratchet leaves open
# ---------------------------------------------------------------------------


def test_baselines_have_no_stale_entries() -> None:
    """Hard gate: every baselined identity must still be a live offender.

    ``ratchet`` decides membership on identity alone, so an entry naming a
    **deleted** file keeps that path pre-exempted: re-create
    ``training_pipeline_director.py`` at 900 LOC and it lands green. Three such
    entries stood in ``large_files.txt`` after ``8d01b95c5`` deleted the
    director tree.

    The sibling allowlist has had exactly this gate since #629
    (``test_data_io_allowlist_has_no_stale_entries``); the ratchet baselines
    never grew one, and that asymmetry is the whole finding.
    """
    offenders = {
        name: sorted(stale_entries(name, scan()))
        for name, scan in CURRENT_SCANS.items()
        if stale_entries(name, scan())
    }
    assert not offenders, (
        "Stale baseline entries (no longer offenders — delete the lines; they "
        "silently pre-exempt those identities):\n"
        + "\n".join(f"  {f}:\n    " + "\n    ".join(v) for f, v in offenders.items())
    )


@pytest.mark.debt_tracker
@pytest.mark.xfail(
    strict=False,
    reason="Debt report: red while any baselined offender has grown past its "
    "recorded measurement. Report-only by design — see grown_entries().",
)
def test_no_growth_inside_baselined_files() -> None:
    """Debt report: growth *inside* an already-baselined entry (NN20).

    The ratchet keys on identity alone — the correct #629 fix, since keying on
    the whole string made any edit to a long-standing offender re-report as NEW.
    The price is that a baselined file can double in size without one gate going
    red, and 107 of them have.

    Deliberately NOT a hard gate. Against the recorded values it is a 107-file
    flag day; against refreshed values it would first have to refresh the
    measurements — the ceiling raise NN20 tells reviewers to reject — and would
    then re-create #629 exactly. Read it with ``pytest -m debt_tracker -rx``.
    """
    grown: dict[str, tuple[int, int]] = {}
    beyond_slack = 0
    for name, scan in CURRENT_SCANS.items():
        current = scan()
        grown.update(grown_entries(name, current))
        beyond_slack += len(grown_entries(name, current, slack=SLACK.get(name, 0)))
    if grown:
        worst = sorted(grown.items(), key=lambda kv: kv[1][0] - kv[1][1])
        total = sum(now - was for was, now in grown.values())
        pytest.fail(
            f"{len(grown)} baselined offenders have grown past their recorded "
            f"measurement (+{total} total); {beyond_slack} exceed the flat slack "
            f"a hard gate would use. Worst 10:\n"
            + "\n".join(f"  +{now - was:<5} {was:>5} -> {now:<5}  {k}" for k, (was, now) in worst[:10])
        )


def test_every_reproducible_baseline_is_checked() -> None:
    """Completeness: no baseline file may be silently exempt from the checks above.

    This is the gate on the gate. ``test_baselines_have_no_stale_entries`` first
    shipped wired to 2 of the 6 baselines, which is exactly why
    ``dispatch_hell.txt``'s dead ``SchedulerBuilder.build`` entry survived — a
    partial mapping reads identically to a clean result. Anything new under
    ``baselines/`` must be added to ``CURRENT_SCANS`` or explicitly listed in
    ``NOT_REPRODUCIBLE_HERE`` with a reason.
    """
    on_disk = {p.name for p in BASELINE_DIR.glob("*.txt")}
    accounted = set(CURRENT_SCANS) | NOT_REPRODUCIBLE_HERE
    assert not on_disk - accounted, (
        "Baseline file(s) checked by nothing — add to CURRENT_SCANS, or to "
        "NOT_REPRODUCIBLE_HERE with a reason:\n  "
        + "\n  ".join(sorted(on_disk - accounted))
    )
    assert not accounted - on_disk, (
        "CURRENT_SCANS/NOT_REPRODUCIBLE_HERE names a baseline that does not "
        "exist:\n  " + "\n  ".join(sorted(accounted - on_disk))
    )
