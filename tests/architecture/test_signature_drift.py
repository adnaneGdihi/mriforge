"""Fitness function: no NEW train_step/validation_step or builder signature drift.

Cached-cascade Phase 0. A step method is conformant if it is migrated behind
``@accepts_step_io`` OR its params match the canonical signature; a builder
``__init__`` is conformant if migrated behind ``@accepts_builder_context`` OR it
takes a single ``BuilderContext`` (or no args). Every CURRENT divergence is
baselined, so this passes today; a NEW undecorated drift (or a changed signature
on a baselined method that is not migrated) fails. As WS-6/WS-7 migrate methods
behind the adapters, the detector goes green for them automatically.
"""

from __future__ import annotations

import pytest

from ._fitness_lib import (
    CANONICAL_TRAIN_STEP,
    CANONICAL_VALIDATION_STEP,
    accepts_canonical_call,
    builder_key,
    collect_builder_inits,
    collect_step_methods,
    ratchet,
    step_method_key,
)

pytestmark = pytest.mark.architecture

# A migrated builder takes a single BuilderContext; bare bases take no args.
_BUILDER_OK_PARAMS = {(), ("ctx",), ("context",)}


def _canonical(method: str) -> tuple[str, ...]:
    return CANONICAL_TRAIN_STEP if method == "train_step" else CANONICAL_VALIDATION_STEP


def test_no_new_step_signature_drift() -> None:
    current_drift = {
        step_method_key(rec)
        for rec in collect_step_methods()
        if not rec["decorated"]
        and not accepts_canonical_call(
            rec["params"],  # type: ignore[arg-type]
            rec["required"],  # type: ignore[arg-type]
            _canonical(str(rec["method"])),
        )
    }
    new = ratchet(
        "step_signature_drift.txt",
        current_drift,
        header="train_step/validation_step signatures that diverge from canonical "
        "and are not migrated behind @accepts_step_io",
    )
    assert not new, (
        "New train_step/validation_step signature drift — conform to the "
        "canonical signature or migrate behind @accepts_step_io:\n  "
        + "\n  ".join(sorted(new))
    )


def test_no_new_builder_signature_drift() -> None:
    current_drift = {
        builder_key(rec)
        for rec in collect_builder_inits()
        if not rec["decorated"] and rec["params"] not in _BUILDER_OK_PARAMS
    }
    new = ratchet(
        "builder_signature_drift.txt",
        current_drift,
        header="builder __init__ signatures not taking BuilderContext and not "
        "migrated behind @accepts_builder_context",
    )
    assert not new, (
        "New builder __init__ signature drift — take a BuilderContext or "
        "migrate behind @accepts_builder_context:\n  " + "\n  ".join(sorted(new))
    )
