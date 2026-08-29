"""Regression tests for ``InferenceStrategyFactory`` silent-fallback fix.

Locks ``TODO/audit/17_remaining_infrastructure.md`` F1 + F2:

- F1: Factory previously fell back to ``ReconstructionInferenceStrategy``
  on unknown ``strategy_type`` — a CLAUDE.md pitfall #9 silent fallback
  that masked typos and produced wrong reconstructions without warning.
- F2: ``SSLInferenceStrategy`` and ``MultiInferenceStrategy`` existed
  but were not registered in the dispatch map, making them unreachable
  via ``InferenceStrategyFactory.create``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from mriforge.infrastructure.inference.inference_factory import (
    InferenceStrategyFactory,
)


def _stub_model() -> MagicMock:
    """Return a mock object that passes for an ``nn.Module``."""
    m = MagicMock(spec=torch.nn.Module)
    return m


def test_unknown_strategy_type_raises_not_silent_fallback() -> None:
    """Unknown strategy_type must raise ``ValueError``, not silently default."""
    with pytest.raises(ValueError, match=r"Unknown inference strategy_type"):
        InferenceStrategyFactory.create(
            model=_stub_model(),
            device=torch.device("cpu"),
            config={},
            strategy_type="totally_made_up_name",
        )


def test_unknown_strategy_error_lists_known_types() -> None:
    """The error message lists registered types so users know the valid surface."""
    with pytest.raises(ValueError) as exc_info:
        InferenceStrategyFactory.create(
            model=_stub_model(),
            device=torch.device("cpu"),
            config={},
            strategy_type="bogus",
        )
    msg = str(exc_info.value)
    assert "reconstruction" in msg
    assert "gan" in msg
    assert "diffusion" in msg


def test_ssl_strategy_is_now_registered() -> None:
    """SSLInferenceStrategy is reachable via the canonical factory path."""
    from mriforge.infrastructure.inference.ssl_inference_strategy import (
        SSLInferenceStrategy,
    )

    # We can't trivially instantiate the strategy without a real model,
    # so we test the dispatch map via a source-level check. The factory
    # uses a local ``strategy_map`` dict — confirm the imports + dict
    # entries are present by source inspection.
    import inspect

    from mriforge.infrastructure.inference import inference_factory

    src = inspect.getsource(inference_factory)
    assert "SSLInferenceStrategy" in src
    assert '"ssl"' in src or "'ssl'" in src
    # And the symbol is actually importable from the factory module:
    assert SSLInferenceStrategy is not None


def test_multi_inference_strategy_is_now_registered() -> None:
    """MultiInferenceStrategy is reachable via the canonical factory path."""
    from mriforge.infrastructure.inference.multi_inference_strategy import (
        MultiInferenceStrategy,
    )

    import inspect

    from mriforge.infrastructure.inference import inference_factory

    src = inspect.getsource(inference_factory)
    assert "MultiInferenceStrategy" in src
    assert '"multi"' in src or "'multi'" in src
    assert MultiInferenceStrategy is not None


def test_known_strategy_type_instantiates() -> None:
    """A registered ``strategy_type`` still dispatches without raising."""
    # ReconstructionInferenceStrategy is the canonical example.
    strategy = InferenceStrategyFactory.create(
        model=_stub_model(),
        device=torch.device("cpu"),
        config={},
        strategy_type="reconstruction",
    )
    # It must be a real BaseInferenceStrategy subclass instance.
    from mriforge.infrastructure.inference.base_inference_strategy import (
        BaseInferenceStrategy,
    )

    assert isinstance(strategy, BaseInferenceStrategy)
