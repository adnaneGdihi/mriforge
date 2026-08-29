"""Registry dispatch-mechanism tests.

These tests exercise the dispatch *logic* of each registry (model / loss /
metric) and the TrainingStrategyFactory — *not* the enumeration of every
registered entry (that lives in the per-surface contract suites).

Coverage targets
----------------
1. Canary: each registry resolves a known key to a class/callable.
2. Parametrised surface sweep: one representative key per surface.
3. Double-register contract: registering the same key twice with the
   *same* class is idempotent; registering with a *different* class raises
   ValueError (all three registries).  Fixtures use throwaway unique names
   and register unique dummy classes so the global state is only additive
   (no cleanup required, and uniqueness prevents cross-test collisions).
4. Case-sensitivity: model registry is case-sensitive (no normalisation);
   loss and metric registries normalise to lowercase before lookup.
5. Unknown-key guard (pitfall #9 — no silent fallback):
   - get_model_class  → ValueError
   - LossRegistry.create → ValueError
   - MetricsRegistry.get → KeyError
   - TrainingStrategyFactory.get_strategy_class (bad paradigm) → raises
     ConfigurationError (a subclass of Exception)
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unique(prefix: str = "zz_test") -> str:
    """Return a unique name that will never collide with real registrations."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# 1. Canaries — well-known keys resolve
# ---------------------------------------------------------------------------


class TestModelRegistryCanary:
    """MODEL_REGISTRY resolves a known key via get_model_class."""

    @pytest.mark.unit
    def test_known_model_resolves(self) -> None:
        """'domain_discriminator' was registered in domain_adaptation.py."""
        # Ensure the generators sub-package is imported so decorators fire.
        import mriforge.models.domain_adaptation  # noqa: F401
        from mriforge.models.registry import get_model_class

        cls = get_model_class("domain_discriminator")
        assert cls is not None
        assert isinstance(cls, type), "registry must return a class, not an instance"


class TestLossRegistryCanary:
    """LossRegistry resolves a known key via create / list_available."""

    @pytest.mark.unit
    def test_known_loss_resolves(self) -> None:
        """'hfen' is registered in models/losses/hfen_loss.py."""
        # Importing the losses package fires all @register_loss decorators.
        import mriforge.models.losses  # noqa: F401
        from mriforge.models.losses.registry import LossRegistry

        assert "hfen" in LossRegistry.list_available()

    @pytest.mark.unit
    def test_known_loss_instantiates(self) -> None:
        """LossRegistry.create('hfen') must return an nn.Module instance."""
        import mriforge.models.losses  # noqa: F401
        from mriforge.models.losses.registry import LossRegistry

        loss = LossRegistry.create("hfen")
        assert isinstance(loss, nn.Module)


class TestMetricsRegistryCanary:
    """MetricsRegistry resolves 'psnr' — registered in evaluation_metrics.py."""

    @pytest.mark.unit
    def test_known_metric_resolves(self) -> None:
        # Importing mriforge.core.metrics triggers pkgutil walk-discovery.
        from mriforge.core.metrics import get_metric

        metric = get_metric("psnr")
        assert metric is not None

    @pytest.mark.unit
    def test_known_metric_is_callable(self) -> None:
        from mriforge.core.metrics import get_metric

        metric = get_metric("ssim")
        assert callable(metric)


# ---------------------------------------------------------------------------
# 2. Parametrised surface sweep
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "surface, key",
    [
        ("model", "domain_discriminator"),
        ("loss", "hfen"),
        ("metric", "psnr"),
    ],
    ids=["model/domain_discriminator", "loss/hfen", "metric/psnr"],
)
def test_registry_surface_canary(surface: str, key: str) -> None:
    """Each registry surface resolves its representative known key."""
    if surface == "model":
        import mriforge.models.domain_adaptation  # noqa: F401
        from mriforge.models.registry import get_model_class

        result = get_model_class(key)
        assert result is not None and isinstance(result, type)

    elif surface == "loss":
        import mriforge.models.losses  # noqa: F401
        from mriforge.models.losses.registry import LossRegistry

        result = LossRegistry.create(key)
        assert isinstance(result, nn.Module)

    elif surface == "metric":
        from mriforge.core.metrics import get_metric

        result = get_metric(key)
        assert result is not None


# ---------------------------------------------------------------------------
# 3. Double-register contract
# ---------------------------------------------------------------------------


class TestModelRegistryDoubleRegister:
    """Same class → idempotent; different class → ValueError."""

    @pytest.mark.unit
    def test_same_class_idempotent(self) -> None:
        """Re-registering the same class under the same name is allowed."""
        from mriforge.models.registry import MODEL_REGISTRY, register_model

        name = _unique("m_idempotent")

        @register_model(name=name, training_mode="test")
        class _DummyA(nn.Module):
            def forward(self, x: Any) -> Any:
                return x

        # Applying the same decorator again (same class, same name) must not raise.
        # Re-applying is the decorator pattern used on test reloads.
        register_model(name=name, training_mode="test")(_DummyA)

        # Registry entry must still point to the original class.
        assert MODEL_REGISTRY[name]["class"] is _DummyA

    @pytest.mark.unit
    def test_different_class_raises(self) -> None:
        """Registering a different class under an existing name must raise ValueError."""
        from mriforge.models.registry import register_model

        name = _unique("m_collision")

        @register_model(name=name, training_mode="test")
        class _DummyOrig(nn.Module):
            def forward(self, x: Any) -> Any:
                return x

        class _DummyNew(nn.Module):
            def forward(self, x: Any) -> Any:
                return x

        with pytest.raises(ValueError, match=name):
            register_model(name=name, training_mode="test")(_DummyNew)


class TestLossRegistryDoubleRegister:
    """Same class → idempotent; different class → ValueError."""

    @pytest.mark.unit
    def test_same_class_idempotent(self) -> None:
        from mriforge.models.losses.registry import LossRegistry, register_loss

        name = _unique("l_idempotent")

        @register_loss(name=name)
        class _DummyLossA(nn.Module):
            def forward(self, pred: Any, target: Any) -> Any:
                return pred

        # Re-registering same class must not raise and must keep the entry.
        LossRegistry.register(name)(_DummyLossA)
        assert LossRegistry._custom_losses[name] is _DummyLossA

    @pytest.mark.unit
    def test_different_class_raises(self) -> None:
        from mriforge.models.losses.registry import register_loss

        name = _unique("l_collision")

        @register_loss(name=name)
        class _DummyLossOrig(nn.Module):
            def forward(self, pred: Any, target: Any) -> Any:
                return pred

        class _DummyLossNew(nn.Module):
            def forward(self, pred: Any, target: Any) -> Any:
                return pred

        with pytest.raises(ValueError, match=name):
            register_loss(name=name)(_DummyLossNew)


class TestMetricsRegistryDoubleRegister:
    """Same class → idempotent; different class → ValueError."""

    @pytest.mark.unit
    def test_same_class_idempotent(self) -> None:
        from mriforge.core.metrics.registry import MetricsRegistry, register_metric

        name = _unique("met_idempotent")

        @register_metric(name=name)
        class _DummyMetricA:
            @property
            def name(self) -> str:
                return name

            @property
            def higher_is_better(self) -> bool:
                return True

            def __call__(self, pred: Any, target: Any, **kw: Any) -> float:
                return 0.0

        # Re-registering same class must be idempotent (returns class unchanged).
        result = MetricsRegistry.register(name)(_DummyMetricA)
        assert result is _DummyMetricA
        assert MetricsRegistry._metrics[name] is _DummyMetricA

    @pytest.mark.unit
    def test_different_class_raises(self) -> None:
        from mriforge.core.metrics.registry import register_metric

        name = _unique("met_collision")

        @register_metric(name=name)
        class _DummyMetricOrig:
            @property
            def name(self) -> str:
                return name

            @property
            def higher_is_better(self) -> bool:
                return True

            def __call__(self, pred: Any, target: Any, **kw: Any) -> float:
                return 0.0

        class _DummyMetricNew:
            @property
            def name(self) -> str:
                return name

            @property
            def higher_is_better(self) -> bool:
                return True

            def __call__(self, pred: Any, target: Any, **kw: Any) -> float:
                return 0.0

        with pytest.raises(ValueError, match=name):
            register_metric(name=name)(_DummyMetricNew)


# ---------------------------------------------------------------------------
# 4. Case-sensitivity behaviour
# ---------------------------------------------------------------------------


class TestModelRegistryCaseSensitive:
    """MODEL_REGISTRY does not normalise keys — lookup is case-sensitive."""

    @pytest.mark.unit
    def test_uppercase_key_raises(self) -> None:
        """'Domain_Discriminator' (wrong case) must raise ValueError, not return a class."""
        import mriforge.models.domain_adaptation  # noqa: F401
        from mriforge.models.registry import get_model_class

        with pytest.raises(ValueError):
            get_model_class("Domain_Discriminator")

    @pytest.mark.unit
    def test_lowercase_key_resolves(self) -> None:
        """Exact lowercase key resolves correctly."""
        import mriforge.models.domain_adaptation  # noqa: F401
        from mriforge.models.registry import get_model_class

        cls = get_model_class("domain_discriminator")
        assert cls is not None


class TestLossRegistryCaseInsensitive:
    """LossRegistry normalises to lowercase before lookup."""

    @pytest.mark.unit
    def test_uppercase_alias_resolves_via_normalisation(self) -> None:
        """'HFEN' should resolve to the same loss as 'hfen'."""
        import mriforge.models.losses  # noqa: F401
        from mriforge.models.losses.registry import LossRegistry

        # 'hfen' canonical key — verify that uppercase also resolves
        # (normalisation path: name.lower() → 'hfen')
        loss = LossRegistry.create("HFEN")
        assert isinstance(loss, nn.Module)


class TestMetricsRegistryCaseInsensitive:
    """MetricsRegistry normalises to lowercase before lookup."""

    @pytest.mark.unit
    def test_uppercase_name_resolves(self) -> None:
        """'PSNR' (uppercase) must resolve through alias normalisation."""
        from mriforge.core.metrics import get_metric

        metric = get_metric("PSNR")
        assert metric is not None

    @pytest.mark.unit
    def test_mixed_case_name_resolves(self) -> None:
        """'Ssim' must resolve through lowercase normalisation."""
        from mriforge.core.metrics import get_metric

        metric = get_metric("Ssim")
        assert metric is not None


# ---------------------------------------------------------------------------
# 5. Unknown-key guard (pitfall #9 — no silent fallback)
# ---------------------------------------------------------------------------


class TestModelRegistryUnknownKey:
    """get_model_class raises ValueError for unknown keys — no silent fallback."""

    @pytest.mark.unit
    def test_unknown_model_raises_value_error(self) -> None:
        from mriforge.models.registry import get_model_class

        with pytest.raises(ValueError, match="not found in registry"):
            get_model_class("__no_such_model_ever_registered__")

    @pytest.mark.unit
    def test_empty_string_raises(self) -> None:
        from mriforge.models.registry import get_model_class

        with pytest.raises(ValueError):
            get_model_class("")


class TestLossRegistryUnknownKey:
    """LossRegistry.create raises ValueError for unknown keys."""

    @pytest.mark.unit
    def test_unknown_loss_raises_value_error(self) -> None:
        import mriforge.models.losses  # noqa: F401
        from mriforge.models.losses.registry import LossRegistry

        with pytest.raises(ValueError, match="Unknown loss"):
            LossRegistry.create("__no_such_loss_ever_registered__")

    @pytest.mark.unit
    def test_empty_string_raises(self) -> None:
        import mriforge.models.losses  # noqa: F401
        from mriforge.models.losses.registry import LossRegistry

        with pytest.raises(ValueError):
            LossRegistry.create("")


class TestMetricsRegistryUnknownKey:
    """MetricsRegistry.get raises KeyError for unknown keys."""

    @pytest.mark.unit
    def test_unknown_metric_raises_key_error(self) -> None:
        from mriforge.core.metrics.registry import MetricsRegistry

        with pytest.raises(KeyError, match="Unknown metric"):
            MetricsRegistry.get("__no_such_metric_ever_registered__")

    @pytest.mark.unit
    def test_empty_string_raises(self) -> None:
        from mriforge.core.metrics.registry import MetricsRegistry

        with pytest.raises(KeyError):
            MetricsRegistry.get("")


class TestStrategyFactoryUnknownParadigm:
    """TrainingStrategyFactory.get_strategy_class raises on an unresolvable config.

    The factory raises ConfigurationError (subclass of Exception) when
    config provides no recognised strategy_class, no typed schema, and no
    training_mode that appears in STRATEGY_CLASS_PATHS.
    """

    @pytest.fixture()
    def factory(self):
        from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory

        return TrainingStrategyFactory()

    def _make_config(self, **training_attrs: Any) -> MagicMock:
        """Build a minimal mock config with a .training sub-object."""
        training = MagicMock()
        # Attributes the factory inspects during fallback inference:
        training.strategy_class = None
        training.diffusion_type = None
        training.timesteps = None
        training.training_mode = "__no_such_paradigm_9f3a__"
        # Overlay caller-supplied overrides.
        for attr, val in training_attrs.items():
            setattr(training, attr, val)

        config = MagicMock()
        config.training = training
        return config

    @pytest.mark.unit
    def test_unknown_paradigm_raises(self, factory: Any) -> None:
        """A training_mode not in STRATEGY_CLASS_PATHS → ConfigurationError."""
        from mriforge.domain.exceptions import ConfigurationError

        config = self._make_config()
        with pytest.raises(ConfigurationError):
            factory.get_strategy_class(config)

    @pytest.mark.unit
    def test_none_training_config_raises(self, factory: Any) -> None:
        """config.training = None (no training block at all) → ConfigurationError."""
        from mriforge.domain.exceptions import ConfigurationError

        config = MagicMock()
        config.training = None
        with pytest.raises(ConfigurationError):
            factory.get_strategy_class(config)

    @pytest.mark.unit
    def test_known_paradigm_returns_class(self, factory: Any) -> None:
        """A valid strategy_class path in STRATEGY_CLASS_PATHS → resolves without error."""
        from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory

        config = self._make_config(
            strategy_class=None,
            training_mode="reconstruction",
        )
        # Override: training_mode IS in STRATEGY_CLASS_PATHS so Priority 3 fires.
        # But Priority 1 (strategy_class=None) and Priority 2 (no schema fields)
        # must both pass first. We set training_mode to a real entry.
        config.training.training_mode = "reconstruction"
        cls = factory.get_strategy_class(config)
        assert cls is not None
        assert isinstance(cls, type)

    @pytest.mark.unit
    def test_explicit_strategy_class_path_takes_priority(self, factory: Any) -> None:
        """Priority 1: explicit strategy_class path is used before any fallback."""
        full_path = (
            "mriforge.infrastructure.training.strategies.reconstruction"
            ".ReconstructionTrainingStrategy"
        )
        config = self._make_config(strategy_class=full_path)
        cls = factory.get_strategy_class(config)
        assert cls is not None
        assert cls.__name__ == "ReconstructionTrainingStrategy"
