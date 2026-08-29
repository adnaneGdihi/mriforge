"""Regression test for F3/E22 (smoke_audit_20260521) —
``ConcretePMAVarNetStrategy.validation_step`` must unwrap
``TrainingBatch`` before passing to ``_compute_losses_impl``.

Root cause: the strategy's ``validation_step`` passed ``val_batch``
(a ``TrainingBatch`` dataclass) directly as ``target_batch``, which
crashed at line 79 with ``AttributeError: 'TrainingBatch' object has
no attribute 'shape'`` across 4 experiments:

  experiment_pma_01_baseline
  experiment_pma_02_b0_correction
  experiment_pma_03_full_m4raw
  experiment_jfe_varnet_sota

Fix: call ``self._unpack_batch(val_batch)`` (provided by the
``BatchPreparationMixin``) so the TrainingBatch is split into
``(input_batch, target_batch)`` tensors before they reach
``_compute_losses_impl``.

This test pins the unwrap call so a future contributor who reverts
the line gets caught by CI rather than the cluster.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STRATEGY = REPO / "src" / "mriforge" / "infrastructure" / "training" / "strategies" / "pma_varnet_strategy.py"


def _read() -> str:
    return STRATEGY.read_text()


def test_validation_step_unpacks_training_batch() -> None:
    """The ``validation_step`` body must call ``_unpack_batch`` and pass
    the unwrapped tensors to ``_compute_losses_impl``.

    Match the full body so a half-revert (e.g. keeping the
    ``_unpack_batch`` call but reverting the ``_compute_losses_impl``
    args) still fails the test.
    """
    text = _read()
    # The fix must include both the unpack call AND the threaded args.
    pattern = re.compile(
        r"def\s+validation_step\(\s*self,[^)]*\)[^:]*:\s*"
        r"(?:\"\"\"[\s\S]*?\"\"\"\s*)?"  # tolerate docstring
        r"torch\.manual_seed\([^)]+\)\s*\n"
        r"\s*input_batch,\s*target_batch\s*=\s*self\._unpack_batch\(val_batch\)\s*\n"
        r"\s*return\s+self\._compute_losses_impl\(\s*input_batch\s*,\s*target_batch",
        re.MULTILINE,
    )
    assert pattern.search(text), (
        "ConcretePMAVarNetStrategy.validation_step must unwrap the "
        "TrainingBatch via self._unpack_batch(val_batch) and pass the "
        "tensors as positional args to _compute_losses_impl. "
        "See TODO/audit/smoke_audit_20260521.md §F3."
    )


def test_no_raw_val_batch_passed_to_losses() -> None:
    """The buggy form ``self._compute_losses_impl(None, val_batch, ...)``
    must not appear — that's the line that crashed 4 experiments.
    Pinned so a copy-paste revert from git history doesn't re-introduce it.
    """
    text = _read()
    assert "self._compute_losses_impl(None, val_batch" not in text, (
        "The buggy ``self._compute_losses_impl(None, val_batch, …)`` "
        "form passes a TrainingBatch as target_batch and crashes at "
        ".shape access. Always unwrap via _unpack_batch first."
    )


def test_audit_anchor_present() -> None:
    """The F3 fix must carry the dated comment anchor + audit doc
    reference so a future cleanup doesn't strip the unpack call as
    redundant."""
    text = _read()
    assert "F3/E22 (2026-05-21" in text, (
        "F3 fix must reference the audit doc by code+date so the "
        "fix's purpose stays traceable."
    )
