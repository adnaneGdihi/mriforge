"""Regression test for F8-r5 (smoke_audit_20260521 round 5) —
``physics_driven_strategy._generate_predictions`` must progressively
strip rejected kwargs, not just ``target``.

Root cause: ``experiment_41_pimn`` (and similar physics-driven
experiments) hit
``TypeError: PIMN.forward() got an unexpected keyword argument 'target'``
at runtime. The pre-fix code had a single-shot retry that only
stripped ``target``; PIMN actually rejects multiple injected kwargs
(``target``, ``query_coords``, ``kspace_measured``, ``mask``, …)
one at a time. Each rejection raised a fresh ``unexpected keyword
argument 'X'`` and the single retry couldn't catch up — the second
TypeError propagated and crashed training.

Fix: progressive-strip retry loop that parses each
``unexpected keyword argument 'X'`` message, pops X from
``forward_kwargs``, and retries — up to ``len(forward_kwargs)+1``
attempts. If the kwarg in the error message isn't in forward_kwargs
(e.g. a positional-arg complaint), re-raise so genuine bugs surface.

This test pins the loop structure so a future contributor who
"simplifies" it back to a single retry breaks CI.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STRATEGY = (
    REPO / "src" / "mriforge" / "infrastructure" / "training" / "strategies"
    / "physics_driven_strategy.py"
)


def _read() -> str:
    return STRATEGY.read_text()


def test_progressive_kwarg_strip_present() -> None:
    """The retry must loop over multiple attempts AND parse the kwarg
    name from the TypeError message. Both are load-bearing — drop
    either and the fix regresses."""
    text = _read()
    assert "unexpected keyword argument" in text, (
        "physics_driven_strategy must parse "
        "``unexpected keyword argument 'X'`` messages — the pre-F8-r5 "
        "single-target retry couldn't keep up with PIMN's progressive "
        "kwarg rejection. See TODO/audit/smoke_audit_20260521.md §F8-r5."
    )
    # Loop structure: a for/range over forward_kwargs length, with a
    # `continue` on successful strip and `break` on successful call.
    assert re.search(
        r"for\s+_attempt\s+in\s+range\(_max_strip_attempts\)",
        text,
    ), (
        "physics_driven_strategy must loop the retry (range-based "
        "bounded attempts) instead of a single-shot retry."
    )
    # No silent fallback — re-raise when we can't make progress.
    assert "raise" in text, "the fix must re-raise when stripping cannot proceed"


def test_audit_anchor_present() -> None:
    text = _read()
    assert "F8-r5 (2026-05-21" in text, (
        "F8-r5 fix must carry the dated audit-doc anchor for traceability."
    )


def test_no_single_shot_retry_left() -> None:
    """The pre-fix form was a single TypeError except with a one-shot
    pop of 'target'. Pinned negative: that exact structure must NOT
    be present (the new loop replaced it). Catches half-reverts."""
    text = _read()
    # Heuristic: an unconditional ``forward_kwargs.pop("target")`` not
    # inside the new progressive loop is what was buggy.
    # Verify it's specifically the OLD pattern (followed by a single
    # retry on next line and no loop). The new code uses a regex match
    # to determine which kwarg to pop.
    old_pattern = re.compile(
        r'if\s+"target"\s+in\s+forward_kwargs\s+and\s+"target"\s+in\s+str\(exc\)\s*:\s*\n'
        r'\s*forward_kwargs\.pop\("target"\)\s*\n'
        r'\s*hr_fakes\s*=\s*self\.env\.generator',
        re.MULTILINE,
    )
    assert not old_pattern.search(text), (
        "The pre-F8-r5 single-shot 'target'-only retry must not be "
        "present. Use the progressive-strip loop instead."
    )
