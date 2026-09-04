"""Regression test for the 2026-06 audit fix to ``mrf_acquisition_strategies``.

The dead ``enforce_1d_beltrami_constraint`` import and the unconsumed
strategy-level ``self.tau_eps`` assignment (a silent no-op — the Beltrami eps
lives in the model) were removed.
"""
from __future__ import annotations

import inspect

from spectramr.infrastructure.training.strategies import mrf_acquisition_strategies as mod
from spectramr.infrastructure.training.strategies.mrf_acquisition_strategies import (
    CrossScannerMRFHarmonisationStrategy,
)


def test_dead_beltrami_import_removed() -> None:
    assert not hasattr(mod, "enforce_1d_beltrami_constraint")


def test_no_unconsumed_tau_eps_assignment() -> None:
    src = inspect.getsource(CrossScannerMRFHarmonisationStrategy.__init__)
    # The unconsumed strategy-level knob assignment is gone (the comment may
    # still mention the word, so grep the exact assignment).
    assert "self.tau_eps =" not in src
