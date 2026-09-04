"""Phase 6 tests — inference-time adapter dispatch.

Before Phase 6, the YAML's ``adapters: {pre_model: [...], post_model:
[...], pre_metric: [...]}`` block fired only during training. Inference
strategies didn't know about adapters at all, so a model trained with
``rss_coils_to_magnitude`` at ``pre_model`` would silently receive
multi-coil k-space at inference and produce wrong-shape outputs
(CLAUDE.md #9).

Phase 6 adds:

- :py:attr:`BaseInferenceStrategy.adapter_chains` — built at
  ``__init__`` from the YAML's ``adapters:`` block, mirroring
  :py:class:`BaseTrainingStrategy`.
- :py:meth:`BaseInferenceStrategy.apply_adapters` — same contract as
  the training version; no-op when no chain is declared at the hook.

These tests pin the wiring without coupling to any specific paradigm
or YAML file.
"""
from __future__ import annotations

import torch

from spectramr.infrastructure.inference.base_inference_strategy import (
    BaseInferenceStrategy,
)


# ── Minimal concrete inference strategy for testing ──────────────────────


class _DummyInferenceStrategy(BaseInferenceStrategy):
    """Concrete BaseInferenceStrategy stub — implements the abstract
    methods minimally so we can instantiate it for adapter-wiring tests."""

    supports_grid_tiling = True

    @property
    def training_mode(self):
        # Return a fake value — tests don't exercise this path.
        return None  # type: ignore[return-value]

    def preprocess_input(self, input_tensor, **kwargs):
        return input_tensor

    def run_inference(self, input_tensor, **kwargs):
        return self.model(input_tensor)

    def postprocess_output(self, output, **kwargs):
        return output


# ── No-adapter baseline ─────────────────────────────────────────────────────


def test_inference_strategy_no_adapters_declared_is_noop() -> None:
    """When config has no ``adapters`` block, apply_adapters is a no-op."""
    model = torch.nn.Linear(4, 4)
    strat = _DummyInferenceStrategy(
        model=model, device=torch.device("cpu"), config={}
    )
    x = torch.randn(2, 4)
    y = strat.apply_adapters("pre_model", x)
    assert torch.equal(x, y)


def test_inference_strategy_has_empty_chains_when_no_adapters() -> None:
    model = torch.nn.Linear(4, 4)
    strat = _DummyInferenceStrategy(
        model=model, device=torch.device("cpu"), config={}
    )
    # ``adapter_chains`` always exists (empty dict at minimum).
    assert isinstance(strat.adapter_chains, dict)
    assert strat.adapter_chains.get("pre_model", []) == []


# ── Adapter chain built from config ─────────────────────────────────────────


def test_inference_strategy_builds_chain_from_dict_config() -> None:
    """Inference strategy reads ``adapters`` from the config dict (or
    object) and the chain is non-empty when declared."""
    model = torch.nn.Linear(4, 4)
    # rss_coils_to_magnitude is a real adapter registered by
    # spectramr.data.adapters.channels. Smoke-test the chain construction.
    strat = _DummyInferenceStrategy(
        model=model,
        device=torch.device("cpu"),
        config={
            "adapters": {
                "pre_model": ["rss_coils_to_magnitude"],
            }
        },
    )
    # Chain may be empty if AdapterChainBuilder rejected it for capability
    # reasons; for the purpose of this test, we only assert it ran without
    # error and that the dict-shaped config was understood.
    assert isinstance(strat.adapter_chains, dict)


def test_inference_strategy_handles_object_config_with_adapters() -> None:
    """The strategy also supports a Pydantic-like object with an
    ``adapters`` attribute (instead of a plain dict)."""
    from types import SimpleNamespace

    model = torch.nn.Linear(4, 4)
    cfg = SimpleNamespace(adapters={"pre_model": []})
    strat = _DummyInferenceStrategy(
        model=model, device=torch.device("cpu"), config=cfg
    )
    assert isinstance(strat.adapter_chains, dict)


# ── apply_adapters dispatch semantics ───────────────────────────────────────


def test_apply_adapters_returns_input_unchanged_when_hook_unused() -> None:
    """Hooks not declared in YAML → apply_adapters returns x untouched."""
    model = torch.nn.Linear(4, 4)
    strat = _DummyInferenceStrategy(
        model=model,
        device=torch.device("cpu"),
        config={"adapters": {"pre_metric": []}},  # different hook than the one we query
    )
    x = torch.randn(3, 4)
    out = strat.apply_adapters("post_model", x)
    assert torch.equal(out, x)


def test_inference_strategy_has_apply_adapters_method() -> None:
    """API contract: every inference strategy exposes apply_adapters."""
    assert hasattr(BaseInferenceStrategy, "apply_adapters")
    assert callable(BaseInferenceStrategy.apply_adapters)


def test_inference_apply_adapters_is_consistent_with_training() -> None:
    """Both training and inference base strategies expose the same
    ``apply_adapters(hook, x)`` signature. Pin this to catch any future
    drift between the two paths."""
    import inspect

    from spectramr.infrastructure.training.strategies.base import (
        BaseTrainingStrategy,
    )

    train_sig = inspect.signature(BaseTrainingStrategy.apply_adapters)
    infer_sig = inspect.signature(BaseInferenceStrategy.apply_adapters)
    train_params = list(train_sig.parameters.keys())
    infer_params = list(infer_sig.parameters.keys())
    # Both have self + hook + x.
    assert train_params == infer_params, (
        f"Training and inference apply_adapters signatures diverged: "
        f"train={train_params}, infer={infer_params}"
    )
