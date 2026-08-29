"""``supports_schedule_free_modes`` is the SSOT for "does this need train/eval?".

The predicate had two spellings. This one was exported from the package and
called by nothing in ``src/``; ``training_loop._set_optimizer_eval_mode`` carried
its own inline variant. They had already drifted -- the inline one tested only
the method for the direction being toggled, and lacked nothing else; this one
tested both methods but not ``step``, so an ``nn.Module`` satisfied it.

Consolidated onto the strictly-stronger form (train AND eval AND step). These
tests pin the three ways the weaker versions were wrong.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.infrastructure.training.optimizers import (  # noqa: E402
    supports_schedule_free_modes,
)


def _param():
    return torch.nn.Parameter(torch.zeros(1))


class TestScheduleFreePredicate:
    def test_an_ordinary_optimizer_is_not_schedule_free(self) -> None:
        assert not supports_schedule_free_modes(torch.optim.Adam([_param()]))

    def test_an_optimizer_with_both_modes_is(self) -> None:
        class _SF(torch.optim.SGD):
            def train(self): ...
            def eval(self): ...

        assert supports_schedule_free_modes(_SF([_param()], lr=0.1))

    def test_an_nn_module_is_not(self) -> None:
        """``train``/``eval`` alone do not identify an optimizer.

        Every ``nn.Module`` has both, so without the ``step`` clause a model
        that ever reached the ``pipeline.optimizers`` mapping would be toggled
        as though it were a schedule-free optimizer.
        """
        assert not supports_schedule_free_modes(torch.nn.Linear(2, 2))

    def test_half_an_implementation_is_not_enough(self) -> None:
        """Both directions, or neither -- otherwise the run is left in the
        wrong mode after the first validation pass."""

        class _OnlyTrain(torch.optim.SGD):
            def train(self): ...

        class _OnlyEval(torch.optim.SGD):
            def eval(self): ...

        assert not supports_schedule_free_modes(_OnlyTrain([_param()], lr=0.1))
        assert not supports_schedule_free_modes(_OnlyEval([_param()], lr=0.1))

    def test_it_is_the_predicate_the_training_loop_uses(self) -> None:
        """Anti-drift: the consumer must CALL this, not re-derive it.

        A second copy is what let the two disagree in the first place, and the
        disagreement was invisible -- both spellings returned True for the only
        object anyone tested them with.
        """
        import inspect

        from mriforge.pipelines.training_loop import _set_optimizer_eval_mode

        source = inspect.getsource(_set_optimizer_eval_mode)
        assert "supports_schedule_free_modes" in source


class TestExtraBackedOptimizersFailLoudly:
    """A missing extra must not present as an unknown-name typo."""

    @pytest.mark.parametrize("name", ["schedulefree_adamw", "schedulefree_sgd"])
    def test_the_name_is_registered_even_without_the_extra(self, name: str) -> None:
        from mriforge.infrastructure.training.optimizer_registry import (
            OptimizerRegistry,
        )

        assert name in OptimizerRegistry.list_available()

    @pytest.mark.parametrize("name", ["schedulefree_adamw", "schedulefree_sgd"])
    def test_accepted_kwargs_are_declared_not_introspected(self, name: str) -> None:
        """The accepted set must not depend on whether the extra is installed.

        These classes fall back to a torch base when ``schedulefree`` is absent,
        so signature introspection would answer a different question on a
        machine with the extra than on one without.
        """
        from mriforge.infrastructure.training.optimizer_registry import (
            accepted_optimizer_kwargs,
        )

        accepted = accepted_optimizer_kwargs(name)
        assert "lr" in accepted
        assert "warmup_steps" in accepted
