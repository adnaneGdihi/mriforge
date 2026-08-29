"""Tests for the canonical BuilderContext + compat adapter (WS-D)."""

from __future__ import annotations

import dataclasses
import warnings

import pytest

from mriforge.infrastructure.builders.context import (
    BuilderContext,
    accepts_builder_context,
)


def test_from_args_collapses_legacy_shapes() -> None:
    ctx = BuilderContext.from_args("CFG", device="cuda", models={"g": 1}, dataset="DS")
    assert ctx.config == "CFG"
    assert ctx.device == "cuda"
    assert ctx.models == {"g": 1}
    assert ctx.dataset == "DS"
    assert ctx.extras == {}


def test_from_args_routes_unknown_deps_to_extras() -> None:
    ctx = BuilderContext.from_args("CFG", custom_thing=42)
    assert ctx.extras == {"custom_thing": 42}


def test_context_is_frozen() -> None:
    ctx = BuilderContext(config="CFG")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.device = "cpu"  # type: ignore[misc]


class _Builder:
    @accepts_builder_context
    def __init__(self, ctx: BuilderContext) -> None:
        self.ctx = ctx


def test_adapter_accepts_builder_context() -> None:
    ctx = BuilderContext(config="CFG", device="cuda")
    b = _Builder(ctx)
    assert b.ctx is ctx


def test_adapter_accepts_legacy_args() -> None:
    b = _Builder("CFG", device="cuda", models={"g": 1})  # type: ignore[arg-type]
    assert isinstance(b.ctx, BuilderContext)
    assert b.ctx.config == "CFG" and b.ctx.models == {"g": 1}


def test_adapter_silent_on_legacy_by_default() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _Builder("CFG")  # type: ignore[arg-type]  # must not raise


def test_adapter_warn_opt_in() -> None:
    class _WBuilder:
        @accepts_builder_context(warn=True)
        def __init__(self, ctx: BuilderContext) -> None:
            self.ctx = ctx

    with pytest.warns(DeprecationWarning):
        _WBuilder("CFG")  # type: ignore[arg-type]


def test_adapter_marker_present() -> None:
    assert getattr(_Builder.__init__, "__accepts_builder_context__", False) is True
