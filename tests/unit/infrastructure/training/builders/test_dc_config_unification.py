"""Tests for the DC-config SSOT enforcement on the generator-kwarg path.

Locks in the contract from ``TODO/backlog_unify_dc_config.md``:
``physics.data_consistency`` is the single source of truth for
data-consistency behaviour. ``model.model_kwargs`` may not also specify
``dc_method`` / ``dc_weight`` / ``use_dc`` with conflicting values — the
resolution must fail loudly on disagreement (CLAUDE.md #9).

Categories:

- ``ValueError`` raised when ``model_kwargs.dc_method`` ≠
  ``physics.data_consistency.method``
- No raise when the YAML keys agree
- No raise when only one source is specified
- Generators that consume ``dc_method`` via ``**kwargs`` (e.g.
  ``KSpaceColdDiffusionGenerator``) are also reconciled — previously they
  silently bypassed SSOT because their ``__init__`` does not list
  ``dc_method`` as an explicit parameter.

These assert **behaviour**, by calling the resolution. They previously
asserted on ``inspect.getsource(ModelBuilder.build_generator)``, which could
not distinguish "the reconciliation runs" from "the word ``_reconcile``
appears in the file", and broke when the logic was extracted to its SSOT
without any behavioural change.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mriforge.infrastructure.builders.generator_kwargs import (
    resolve_generator_kwargs,
)


class _ExplicitDCGenerator:
    """Lists the DC keys as explicit constructor parameters."""

    def __init__(self, use_dc=None, dc_method=None, dc_weight=None):
        pass


class _VarKwargsGenerator:
    """The ``KSpaceColdDiffusionGenerator`` shape: reads DC via ``kwargs.get``."""

    def __init__(self, **kwargs):
        pass


class _NoDCGenerator:
    """Cannot accept DC keys at all."""

    def __init__(self, in_channels=1, out_channels=1):
        pass


def _config(model_kwargs=None, *, method="soft", enabled=True, weight=0.5):
    return SimpleNamespace(
        model=SimpleNamespace(model_type="stub", model_kwargs=dict(model_kwargs or {})),
        physics=SimpleNamespace(
            data_consistency=SimpleNamespace(enabled=enabled, method=method, weight=weight)
        ),
    )


def test_conflicting_dc_method_raises() -> None:
    """The documented failure: two sources, two answers, no error.

    Silent divergence here previously disabled experiment-32a's adversarial
    training (CLAUDE.md pitfall #9).
    """
    with pytest.raises(ValueError, match="Data-consistency configuration conflict"):
        resolve_generator_kwargs(
            _config({"dc_method": "hard"}, method="soft"),
            model_cls=_ExplicitDCGenerator,
        )


def test_conflict_message_names_the_ssot_and_the_remedy() -> None:
    with pytest.raises(ValueError) as excinfo:
        resolve_generator_kwargs(
            _config({"dc_weight": 0.9}, weight=0.5), model_cls=_ExplicitDCGenerator
        )
    message = str(excinfo.value)
    assert "physics.data_consistency" in message
    assert "backlog_unify_dc_config.md" in message


def test_agreeing_values_do_not_raise() -> None:
    kwargs = resolve_generator_kwargs(
        _config({"dc_method": "soft"}, method="soft"), model_cls=_ExplicitDCGenerator
    )
    assert kwargs["dc_method"] == "soft"


def test_single_source_is_injected_without_complaint() -> None:
    kwargs = resolve_generator_kwargs(_config(), model_cls=_ExplicitDCGenerator)
    assert kwargs["use_dc"] is True
    assert kwargs["dc_method"] == "soft"
    assert kwargs["dc_weight"] == 0.5


def test_ssot_wins_over_a_silent_overwrite() -> None:
    """Prior behaviour was ``gen_kwargs['dc_method'] = dc.method`` — a silent
    overwrite. The value must still come from the SSOT, but only after the
    conflict check above has had its say."""
    kwargs = resolve_generator_kwargs(_config(method="soft"), model_cls=_VarKwargsGenerator)
    assert kwargs["dc_method"] == "soft"


def test_var_keyword_generators_are_reconciled() -> None:
    """Generators that take ``**kwargs`` participate in DC reconciliation.

    Regression for the silent-fallback bug where
    ``KSpaceColdDiffusionGenerator.__init__`` consumes ``dc_method`` via
    ``kwargs.get(...)`` rather than as an explicit parameter. Before the fix,
    the contract inspector returned an empty parameter set for that generator,
    so reconciliation skipped it — yielding a silent divergence between
    ``physics.data_consistency.method`` and the generator's internal default
    (``"hard"``). See ``TODO/backlog_unify_dc_config.md``.
    """
    with pytest.raises(ValueError, match="Data-consistency configuration conflict"):
        resolve_generator_kwargs(
            _config({"dc_method": "hard"}, method="soft"),
            model_cls=_VarKwargsGenerator,
        )


def test_generator_that_cannot_accept_dc_is_left_alone() -> None:
    """Negative control: reconciliation is contract-gated, not unconditional.

    Without this, a test suite where every generator accepted everything could
    not tell contract-gating from blanket injection.
    """
    kwargs = resolve_generator_kwargs(
        _config({"dc_method": "hard"}, method="soft"), model_cls=_NoDCGenerator
    )
    # The arm's own model_kwargs entry passes through untouched (the factory's
    # signature filter drops it downstream); what must NOT happen is the SSOT
    # value being injected over it, and no conflict is raised for a generator
    # that could never have consumed either value.
    assert kwargs["dc_method"] == "hard"
    assert "use_dc" not in kwargs
    assert "dc_weight" not in kwargs


def test_absent_physics_block_injects_nothing() -> None:
    config = SimpleNamespace(
        model=SimpleNamespace(model_type="stub", model_kwargs={}), physics=None
    )
    assert resolve_generator_kwargs(config, model_cls=_ExplicitDCGenerator) == {}
