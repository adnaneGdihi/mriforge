"""Tests for the (deprecated) ``TrainingState`` EMA construction path.

``TrainingState`` is deprecated in favour of ``TrainingEnvironment`` and is
never instantiated in ``src/``, but its ``initialize_ema`` was the last
surviving caller of ``spectramr.models.utils.adaptive_ema`` — a module deleted in
ff0efff9f. The branch could therefore only ever raise ``ImportError`` and fall
through to standard EMA, silently discarding an arm's whole adaptive-EMA
declaration (#1294). These tests pin the single wired construction path that
replaced it, so the orphan cannot come back.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
import warnings

import pytest
import torch
import torch.nn as nn

from spectramr.config.schemas.ema import EMAConfigSchema
from spectramr.infrastructure.optimization.ema import ModelEma
from spectramr.infrastructure.training.state import TrainingState


class _Cfg:
    """Minimal stand-in for the config SSOT: only ``.ema`` is read here."""

    def __init__(self, ema):
        self.ema = ema


def _make_state(ema_cfg) -> TrainingState:
    """Build a TrainingState with every required field stubbed but the two
    that ``initialize_ema`` actually reads (``config`` and ``generator``)."""
    kwargs = {
        f.name: None
        for f in dataclasses.fields(TrainingState)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    }
    kwargs["config"] = _Cfg(ema_cfg)
    kwargs["generator"] = nn.Linear(4, 4, bias=False)
    kwargs["device"] = torch.device("cpu")
    with warnings.catch_warnings():
        # __post_init__ warns that the whole class is deprecated; that is the
        # subject under test, not a failure.
        warnings.simplefilter("ignore", DeprecationWarning)
        return TrainingState(**kwargs)


def test_disabled_ema_is_not_constructed():
    state = _make_state(EMAConfigSchema(enabled=False))
    assert state.ema is None


def test_standard_ema_is_constructed_from_config():
    state = _make_state(EMAConfigSchema(enabled=True, decay=0.99, warmup=False))
    assert isinstance(state.ema, ModelEma)
    assert state.ema.decay == pytest.approx(0.99)
    assert state.ema.warmup is False
    assert state.ema.adaptive is False


def test_adaptive_declaration_reaches_the_ema_instead_of_being_swallowed():
    """The regression this file exists for.

    Before #1294 this config produced a plain fixed-decay ModelEma: the
    adaptive import raised, the ``except ImportError: pass`` swallowed it, and
    warmup_steps / initial_decay / final_decay were never read by anything.
    """
    state = _make_state(
        EMAConfigSchema(
            enabled=True,
            enable_adaptive_ema=True,
            warmup_steps=250,
            initial_decay=0.5,
            final_decay=0.995,
        )
    )
    assert isinstance(state.ema, ModelEma)
    assert state.ema.adaptive is True
    assert state.ema.warmup_steps == 250
    assert state.ema.initial_decay == pytest.approx(0.5)
    assert state.ema.final_decay == pytest.approx(0.995)
    # ...and the ramp is live, not decorative.
    state.ema.num_updates = 125
    assert state.ema._current_decay() == pytest.approx(0.7475)


def _initialize_ema_ast() -> ast.FunctionDef:
    """Parse ``initialize_ema``'s body.

    Deliberately AST, not ``"needle" in inspect.getsource(...)``: a substring
    pin also matches the explanatory COMMENTS that describe the removed code,
    so it fires on the fix rather than on the regression.
    """
    src = textwrap.dedent(inspect.getsource(TrainingState.initialize_ema))
    return ast.parse(src).body[0]


def test_the_deleted_adaptive_module_is_not_imported_anymore():
    """Guard against the orphaned caller being reintroduced."""
    fn = _initialize_ema_ast()
    imported = {
        alias.name
        for node in ast.walk(fn)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    modules = {node.module for node in ast.walk(fn) if isinstance(node, ast.ImportFrom)}
    assert "create_adaptive_ema_for_model" not in imported
    assert not any("adaptive_ema" in (m or "") for m in modules)


def test_ema_construction_swallows_nothing():
    """No exception handler may stand between a declared EMA and the run.

    The old code had two, and both were silent: ``except ImportError: pass``
    around the adaptive path (which ALWAYS fired, the module being gone) and
    ``except ImportError: self.ema = None`` around the standard one.
    """
    fn = _initialize_ema_ast()
    handlers = [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]
    assert handlers == []


def test_update_ema_no_longer_advertises_the_unwired_stability_signal():
    """``loss_value`` / ``gradient_norm`` fed ``update_stability_score`` on the
    deleted class. Nothing consumes them now, so they must not be accepted
    (non-negotiable 8: never advertise an unread knob)."""
    assert list(inspect.signature(TrainingState.update_ema).parameters) == ["self"]


def test_update_ema_advances_the_shadow():
    state = _make_state(EMAConfigSchema(enabled=True, decay=0.9, warmup=False))
    assert state.ema.num_updates == 0
    state.update_ema()
    assert state.ema.num_updates == 1
