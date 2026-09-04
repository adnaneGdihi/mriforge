"""Optimizer Type Coverage Tests.

PURPOSE:
Every value in the OptimizerType enum must be actually handled by
OptimizationBuilder.create_single_optimizer().  Unhandled values silently fall
through to the else branch which either raises ValueError immediately or uses
an unintended default — both are bugs.

This test suite:
  1. Calls create_single_optimizer() with each enum value
  2. Verifies a valid torch.optim.Optimizer is returned
  3. Verifies param_groups are populated (parameters are registered)
  4. Documents known third-party dependencies (lars, lamb) with xfail markers
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectramr.config.schemas.enums import OPTIMIZER_NAMES

# ---------------------------------------------------------------------------
# Fixture: a tiny model to supply parameters
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_params():
    """Minimal parameter list for optimizer construction."""
    model = nn.Linear(4, 2)
    return list(model.parameters())


# ---------------------------------------------------------------------------
# 1. create_single_optimizer factory coverage
# ---------------------------------------------------------------------------


class TestOptimizerTypeCoverage:
    """Every OptimizerType enum value must produce a valid Optimizer."""

    # (optimizer_type_value, requires_extra_pkg)
    CASES = [
        ("adam", False),
        ("adamw", False),
        ("sgd", False),
        ("rmsprop", False),
        ("lars", False),  # in-repo since 2026-07-30
        ("lamb", False),  # in-repo since 2026-07-30
    ]

    @pytest.mark.parametrize("opt_type,needs_extra", CASES, ids=[c[0] for c in CASES])
    def test_optimizer_created(self, opt_type: str, needs_extra: bool, tiny_params):
        """create_single_optimizer must return a valid Optimizer for each type."""
        from spectramr.infrastructure.training.builders.optimization_builder import (
            OptimizationBuilder,
        )

        if needs_extra:
            pytest.xfail(
                reason=f"'{opt_type}' optimizer requires a third-party package "
                "(apex/timm). This is a DOCUMENTED GAP: the OptimizerType enum "
                "declares it but the builder has no implementation. Either add "
                "the implementation or remove the enum value.",
            )

        optimizer = OptimizationBuilder.create_single_optimizer(
            parameters=tiny_params,
            learning_rate=1e-4,
            optimizer_type=opt_type,
        )
        assert isinstance(
            optimizer, torch.optim.Optimizer
        ), f"Expected torch.optim.Optimizer, got {type(optimizer).__name__}"
        assert (
            len(optimizer.param_groups) > 0
        ), "Optimizer has no param_groups — parameters were not registered."
        assert (
            len(optimizer.param_groups[0]["params"]) > 0
        ), "First param_group has no parameters."

    @pytest.mark.parametrize("opt_type,needs_extra", CASES, ids=[c[0] for c in CASES])
    def test_optimizer_lr_is_respected(
        self, opt_type: str, needs_extra: bool, tiny_params
    ):
        """The configured learning rate must appear in the optimizer's param_groups."""
        from spectramr.infrastructure.training.builders.optimization_builder import (
            OptimizationBuilder,
        )

        if needs_extra:
            pytest.xfail(reason=f"'{opt_type}' requires third-party package")

        target_lr = 3.7e-4
        optimizer = OptimizationBuilder.create_single_optimizer(
            parameters=tiny_params,
            learning_rate=target_lr,
            optimizer_type=opt_type,
        )
        actual_lr = optimizer.param_groups[0]["lr"]
        assert (
            abs(actual_lr - target_lr) < 1e-10
        ), f"Optimizer lr mismatch: expected {target_lr}, got {actual_lr}"


# ---------------------------------------------------------------------------
# 2. OptimizerType enum completeness vs. builder implementation
# ---------------------------------------------------------------------------


class TestOptimizerEnumCompleteness:
    """Document the mapping between OptimizerType enum and builder branches."""

    # Values that are implemented in create_single_optimizer
    IMPLEMENTED = {"adam", "adamw", "sgd", "rmsprop"}

    def test_all_implemented_types_are_in_enum(self):
        """Every implemented type must appear in the OptimizerType enum."""
        from spectramr.config.schemas.enums import OptimizerType

        enum_values = {m.value for m in OptimizerType}
        for impl_type in self.IMPLEMENTED:
            assert impl_type in enum_values, (
                f"'{impl_type}' is handled by OptimizationBuilder but NOT in the "
                "OptimizerType enum — the schema and implementation are inconsistent."
            )

    #: Enum values with no registry entry. **Empty, and it must stay empty.**
    #:
    #: This was a list. `lars` and `lamb` sat in the enum with no implementation
    #: anywhere for the enum's whole life, recorded only as an xfail reading
    #: "needs apex / timm"; the torch-tier names were reachable by reflective
    #: getattr but unknown to the registry that was supposed to validate them.
    #:
    #: `optimizer_registry` now asserts the same thing at IMPORT time, so a
    #: regression is an ImportError rather than a red test. This stays as the
    #: readable statement of the invariant.
    KNOWN_UNIMPLEMENTED: set[str] = set()

    def test_every_enum_value_is_registered(self):
        """The advertised vocabulary and the dispatch table are the same set."""
        from spectramr.config.schemas.enums import OPTIMIZER_NAMES
        from spectramr.infrastructure.training.optimizer_registry import (
            OptimizerRegistry,
        )

        missing = OPTIMIZER_NAMES - set(OptimizerRegistry.list_available())
        assert missing == self.KNOWN_UNIMPLEMENTED, (
            f"OptimizerType advertises {sorted(missing)} with no registry entry. "
            "Register it, or remove the enum member -- an advertised name that "
            "resolves to nothing reports as a user typo."
        )

    def test_the_unimplemented_set_is_a_ratchet_not_a_dumping_ground(self):
        """Any entry must still be a real enum value, so the guard cannot go
        stale the way an allowlist silently does once its entries are fixed."""
        from spectramr.config.schemas.enums import OPTIMIZER_NAMES

        stale = self.KNOWN_UNIMPLEMENTED - OPTIMIZER_NAMES
        assert not stale, f"stale KNOWN_UNIMPLEMENTED entries: {sorted(stale)}"

    def test_unknown_optimizer_raises_value_error(self, tiny_params):
        """An unknown optimizer type must raise ValueError, not silently default."""
        from spectramr.infrastructure.training.builders.optimization_builder import (
            OptimizationBuilder,
        )

        with pytest.raises(ValueError, match="[Uu]nknown"):
            OptimizationBuilder.create_single_optimizer(
                parameters=tiny_params,
                optimizer_type="nonexistent_optimizer_xyz",
            )


# ---------------------------------------------------------------------------
# 3. OptimizationConfigSchema optimizer_type field is consumed correctly
# ---------------------------------------------------------------------------


class TestConfigOptimizerTypePropagation:
    """optimization.optimizer.type in config must flow to the actual optimizer class."""

    @pytest.mark.parametrize(
        "opt_type,expected_cls",
        [
            ("adam", torch.optim.Adam),
            ("adamw", torch.optim.AdamW),
            ("sgd", torch.optim.SGD),
            ("rmsprop", torch.optim.RMSprop),
        ],
    )
    def test_config_optimizer_type_selects_correct_class(
        self, opt_type: str, expected_cls: type, tiny_params
    ):
        """optimizer_type in OptimizationConfigSchema must select the correct torch class."""
        from spectramr.config.schemas.optimization import OptimizationConfigSchema
        from spectramr.infrastructure.training.builders.optimization_builder import (
            OptimizationBuilder,
        )

        opt_config = OptimizationConfigSchema(
            optimizer_type=opt_type, learning_rate=1e-4
        )
        # Use the internal _create_optimizer method which reads from config
        builder = OptimizationBuilder.__new__(OptimizationBuilder)
        builder._config = None
        builder._models = {}
        builder._optimizers = {}
        builder._schedulers = {}
        builder._scaler = None

        optimizer = builder._create_optimizer(tiny_params, opt_config)
        assert isinstance(optimizer, expected_cls), (
            f"optimizer_type='{opt_type}' should produce {expected_cls.__name__}, "
            f"got {type(optimizer).__name__}"
        )


class TestEveryAdvertisedNameActuallyConstructs:
    """The claim ``TestOptimizerTypeCoverage`` makes, delivered for all of them.

    That class asserts "every OptimizerType enum value must produce a valid
    Optimizer" against a hand-maintained six-entry list, so fifteen advertised
    names were never construction-tested -- and ``adafactor`` sat broken behind
    the gap, dying with ``TypeError: 'float' object is not subscriptable``
    because the schema's scalar ``eps`` was forwarded into a constructor
    parameter typed ``tuple[float | None, float]``.

    A literal list is the wrong shape for a coverage claim: it was right when
    written and became wrong the moment another optimizer was registered.
    Parametrising over ``OPTIMIZER_NAMES`` means adding one cannot silently
    reopen the hole.

    Optimizers behind an optional extra are expected to raise ``ImportError``
    carrying an install command -- that is the contract (never a silent
    substitution), so it counts as a pass.
    """

    @pytest.mark.parametrize("opt_type", sorted(OPTIMIZER_NAMES))
    def test_name_builds_through_the_production_path(
        self, opt_type: str, tiny_params
    ) -> None:
        from spectramr.config.schemas.optimization import OptimizationConfigSchema
        from spectramr.infrastructure.training.optimizer_resolution import (
            build_optimizer_from_spec,
            resolve_optimizer_spec,
        )

        model = nn.Linear(4, 2)
        spec = resolve_optimizer_spec(
            OptimizationConfigSchema(optimizer_type=opt_type, learning_rate=1e-4)
        )
        try:
            optimizer = build_optimizer_from_spec(spec, model.named_parameters())
        except ImportError as exc:
            assert "install" in str(exc).lower(), (
                f"{opt_type!r} is behind an optional extra but its ImportError "
                f"does not tell the user how to install it: {exc}"
            )
            return
        assert isinstance(optimizer, torch.optim.Optimizer)

    @pytest.mark.parametrize("opt_type", sorted(OPTIMIZER_NAMES))
    def test_name_takes_a_step_without_raising(
        self, opt_type: str, tiny_params
    ) -> None:
        """Construction is not enough -- ``lbfgs`` and friends only fail on use."""
        from spectramr.config.schemas.optimization import OptimizationConfigSchema
        from spectramr.infrastructure.training.optimizer_resolution import (
            build_optimizer_from_spec,
            resolve_optimizer_spec,
        )

        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        spec = resolve_optimizer_spec(
            OptimizationConfigSchema(optimizer_type=opt_type, learning_rate=1e-4)
        )
        try:
            optimizer = build_optimizer_from_spec(spec, model.named_parameters())
        except ImportError:
            pytest.skip(f"{opt_type} is behind an optional extra")

        if opt_type == "sparseadam":
            pytest.skip("sparseadam requires sparse gradients by construction")

        def _closure():
            optimizer.zero_grad()
            loss = (model(torch.randn(8, 4)) ** 2).mean()
            loss.backward()
            return loss

        if opt_type == "lbfgs":
            optimizer.step(_closure)  # LBFGS re-evaluates, so it needs a closure
        else:
            _closure()
            optimizer.step()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
