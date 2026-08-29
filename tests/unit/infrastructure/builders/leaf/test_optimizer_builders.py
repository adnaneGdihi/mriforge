"""Tests for the leaf optimizer builders (WS-D BuilderContext migration).

Covers the Phase-0 back-compat migration of :class:`GradScalerBuilder` to the
canonical ``def __init__(self, ctx: BuilderContext)`` convention behind the
:func:`accepts_builder_context` shim. The shim must keep the legacy
``GradScalerBuilder(config)`` call site working while also accepting an explicit
``BuilderContext``; both forms must produce equivalent state.
"""

from __future__ import annotations




class _StubConfig:
    """Minimal stand-in for ``TrainingSettings``.

    ``GradScalerBuilder.__init__`` only stashes ``config`` on ``self._config``
    (it never reads the config until ``build``/``validate``), so a plain
    sentinel object is sufficient to exercise both construction shapes without
    pulling in the heavy real schema.
    """


class TestOptimizerBuilderAgreesWithTheConfigResolver:
    """One dispatch table, two kwarg resolvers -- pitfall #13b's shape.

    ``OptimizerBuilder`` routes construction through
    ``build_optimizer_from_spec``, so the *dispatch* is single. But it assembled
    its spec from builder-local defaults (``_weight_decay = 0.0``,
    ``_betas = (0.9, 0.999)``) instead of reading the config the way
    ``resolve_optimizer_spec`` does, so the same ``TrainingSettings`` produced a
    different optimizer depending on which entry point built it.

    The leaf path is not a backwater: ``pipelines/fit.py`` builds BOTH the
    generator and the discriminator optimizer through it, so a scripting
    ``fit(...)`` run silently trained a different objective from the one
    ``mriforge train`` ran off the same YAML.
    """

    @staticmethod
    def _config(**overrides):
        from mriforge.config.schemas.optimization import OptimizationConfigSchema

        class _Cfg:
            def __init__(self, optimization):
                self.optimization = optimization

        return _Cfg(OptimizationConfigSchema(**overrides))

    @staticmethod
    def _group(optimizer):
        keys = ("lr", "betas", "weight_decay", "eps")
        return {k: v for k, v in optimizer.param_groups[0].items() if k in keys}

    def _both_paths(self, **overrides):
        import torch.nn as nn

        from mriforge.infrastructure.builders.leaf.optimizer_builders import (
            OptimizerBuilder,
        )
        from mriforge.infrastructure.training.optimizer_resolution import (
            build_optimizer_from_spec,
            resolve_optimizer_spec,
        )

        config = self._config(**overrides)
        resolver = build_optimizer_from_spec(
            resolve_optimizer_spec(config.optimization),
            nn.Linear(4, 2).named_parameters(),
        )
        leaf = OptimizerBuilder(config).with_model(nn.Linear(4, 2)).build()
        return self._group(resolver), self._group(leaf)

    def test_schema_defaults_reach_both_paths_identically(self) -> None:
        """``beta1`` defaults to 0.5 here, not 0.9 -- the leaf builder's own
        default silently disagreed with the schema for every caller."""
        resolver, leaf = self._both_paths()
        assert leaf == resolver

    def test_declared_weight_decay_and_betas_reach_the_leaf_path(self) -> None:
        resolver, leaf = self._both_paths(
            weight_decay=0.07, beta1=0.85, beta2=0.97, eps=1e-7
        )
        assert leaf == resolver
        assert leaf["weight_decay"] == 0.07
        assert leaf["betas"] == (0.85, 0.97)

    def test_an_explicit_fluent_override_still_wins(self) -> None:
        """Reading the config must not disable the builder's own API."""
        import torch.nn as nn

        from mriforge.infrastructure.builders.leaf.optimizer_builders import (
            OptimizerBuilder,
        )

        config = self._config(weight_decay=0.07)
        optimizer = (
            OptimizerBuilder(config)
            .with_model(nn.Linear(4, 2))
            .with_weight_decay(0.5)
            .build()
        )
        assert optimizer.param_groups[0]["weight_decay"] == 0.5


class TestOptimizerBuilderHonoursTTUR:
    """``pipelines/fit.py`` builds BOTH GAN optimizers through this builder.

    ``optimization.optimizer.discriminator_learning_rate`` is the two-timescale update
    rule: D trains at its own LR. The config-driven builder passes it as
    ``lr_override``; the leaf builder had no way to say "this one is the
    discriminator", so a scripting ``fit(paradigm='gan')`` run trained D at G's
    LR -- TTUR silently off, with nothing in the log to say so.
    """

    @staticmethod
    def _config():
        from mriforge.config.schemas.optimization import OptimizationConfigSchema

        class _Cfg:
            def __init__(self, optimization):
                self.optimization = optimization

        return _Cfg(
            OptimizationConfigSchema(
                learning_rate=1e-4, discriminator_learning_rate=2.5e-5
            )
        )

    def test_discriminator_role_picks_up_the_ttur_learning_rate(self) -> None:
        import torch.nn as nn

        from mriforge.infrastructure.builders.leaf.optimizer_builders import (
            OptimizerBuilder,
        )

        config = self._config()
        opt_g = OptimizerBuilder(config).with_model(nn.Linear(4, 2)).build()
        opt_d = (
            OptimizerBuilder(config)
            .with_model(nn.Linear(4, 2))
            .with_role("discriminator")
            .build()
        )

        assert opt_g.param_groups[0]["lr"] == 1e-4
        assert opt_d.param_groups[0]["lr"] == 2.5e-5

    def test_generator_role_is_unaffected_by_the_discriminator_knob(self) -> None:
        import torch.nn as nn

        from mriforge.infrastructure.builders.leaf.optimizer_builders import (
            OptimizerBuilder,
        )

        opt_g = OptimizerBuilder(self._config()).with_model(nn.Linear(4, 2)).build()
        assert opt_g.param_groups[0]["lr"] == 1e-4

    def test_an_explicit_learning_rate_still_beats_the_role(self) -> None:
        import torch.nn as nn

        from mriforge.infrastructure.builders.leaf.optimizer_builders import (
            OptimizerBuilder,
        )

        opt_d = (
            OptimizerBuilder(self._config())
            .with_model(nn.Linear(4, 2))
            .with_role("discriminator")
            .with_learning_rate(3e-3)
            .build()
        )
        assert opt_d.param_groups[0]["lr"] == 3e-3


class TestTheLearningRateComesFromTheConfig:
    """Regression: the leaf builder read the LR off the WRONG level.

    ``validate()`` resolved an unset LR with
    ``getattr(self._config.optimization, "learning_rate", 1e-3)``. Phase 8 moved
    that field onto ``optimization.optimizer``, and because the read carried a
    literal default it did not raise -- it silently returned 1e-3 for every arm,
    whatever the config said. A config-independent learning rate is pitfall #9
    with the whole run behind it, and nothing in the run would have reported it.

    Both knobs are asserted at a NON-default value, so a read that falls back to
    a literal cannot pass by coincidence.
    """

    @staticmethod
    def _config(**optimizer):
        from mriforge.config.schemas.optimization import OptimizationConfigSchema

        class _Cfg:
            def __init__(self, optimization):
                self.optimization = optimization

        return _Cfg(OptimizationConfigSchema(optimizer=optimizer))

    def _build(self, role="generator", **optimizer):
        import torch.nn as nn

        from mriforge.infrastructure.builders.leaf.optimizer_builders import (
            OptimizerBuilder,
        )

        return (
            OptimizerBuilder(self._config(**optimizer))
            .with_model(nn.Linear(4, 2))
            .with_role(role)
            .build()
        )

    def test_the_configured_learning_rate_reaches_the_optimizer(self) -> None:
        opt = self._build(type="adamw", learning_rate=7.5e-4)
        assert opt.param_groups[0]["lr"] == 7.5e-4

    def test_it_is_not_the_old_literal_fallback(self) -> None:
        """Pins the specific wrong answer, so a regression names itself."""
        opt = self._build(type="adamw", learning_rate=7.5e-4)
        assert opt.param_groups[0]["lr"] != 1e-3

    def test_ttur_gives_the_discriminator_its_own_rate(self) -> None:
        opt = self._build(
            role="discriminator",
            type="adamw",
            learning_rate=7.5e-4,
            discriminator_learning_rate=2.5e-4,
        )
        assert opt.param_groups[0]["lr"] == 2.5e-4

    def test_without_ttur_the_discriminator_shares_the_base_rate(self) -> None:
        opt = self._build(role="discriminator", type="adamw", learning_rate=7.5e-4)
        assert opt.param_groups[0]["lr"] == 7.5e-4
