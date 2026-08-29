"""Unit tests for the extracted gradient-checkpointing wrapper.

Split out of ``generator_kwargs`` in the Wave 0 exit-criterion work (#1400).
The behaviour is preserved verbatim, including the broad ``except`` that
downgrades a wrapping failure to a warning -- a known defect tracked in #1333.
These tests pin what the extraction must not have changed.
"""

from __future__ import annotations

from types import SimpleNamespace

from mriforge.infrastructure.builders.gradient_checkpointing import (
    apply_gradient_checkpointing,
)


def _config(enabled: bool | None) -> SimpleNamespace:
    if enabled is None:
        return SimpleNamespace(optimization=SimpleNamespace(gradient=SimpleNamespace()))
    return SimpleNamespace(
        optimization=SimpleNamespace(gradient=SimpleNamespace(enable_checkpointing=enabled))
    )


class _Native:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    def set_grad_checkpointing(self, flag: bool) -> None:
        self.calls.append(flag)


class TestReExportIdentity:
    def test_generator_kwargs_and_model_builder_share_one_function(self) -> None:
        from mriforge.infrastructure.builders import generator_kwargs as gk
        from mriforge.infrastructure.training.builders import model_builder

        assert gk.apply_gradient_checkpointing is apply_gradient_checkpointing
        assert model_builder.apply_gradient_checkpointing is apply_gradient_checkpointing


class TestGating:
    def test_native_hook_is_used_when_enabled(self) -> None:
        gen = _Native()
        apply_gradient_checkpointing(gen, _config(True))
        assert gen.calls == [True]

    def test_disabled_arm_is_a_no_op(self) -> None:
        gen = _Native()
        apply_gradient_checkpointing(gen, _config(False))
        assert gen.calls == []

    def test_missing_knob_is_a_no_op_rather_than_an_error(self) -> None:
        """A config without the gradient block must not crash the builder."""
        gen = _Native()
        apply_gradient_checkpointing(gen, _config(None))
        assert gen.calls == []

    def test_config_without_optimization_at_all_is_a_no_op(self) -> None:
        gen = _Native()
        apply_gradient_checkpointing(gen, SimpleNamespace())
        assert gen.calls == []
