"""Smoke test: ``ConformalMRFDictlessReconStrategy`` delegates inference
to the KD-tree :class:`MRFDictlessMatcher` (MRF §3 of
``TODO/audit_plan_novel_fmri.md``).

The strategy must:

1. expose ``build_matcher`` and ``match`` methods,
2. cache an :class:`MRFDictlessMatcher` instance after ``build_matcher``,
3. round-trip a small dictionary so each entry matches itself when queried,
4. raise if ``match`` is called before ``build_matcher``.

The test bypasses ``BaseTrainingStrategy.__init__`` (which requires a
full DI environment) by mirroring the ``_alloc`` helper used in
``test_fmri_mrf_components.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from spectramr.domain.services.mrf_dictless_matching import (
    MRFDictlessMatcher,
)
from spectramr.infrastructure.training.strategies.mrf_acquisition_strategies import (
    ConformalMRFDictlessReconStrategy,
)
from spectramr.models.generators.mrf_models import ConformalFPEmbedding


def _alloc_strategy(generator: nn.Module) -> ConformalMRFDictlessReconStrategy:
    """Allocate the strategy without going through the DI container."""
    inst = ConformalMRFDictlessReconStrategy.__new__(ConformalMRFDictlessReconStrategy)
    inst.config = SimpleNamespace(training=SimpleNamespace(seed=0))
    inst.device = torch.device("cpu")
    inst.env = SimpleNamespace(
        models={"generator": generator},
        device=torch.device("cpu"),
        config=inst.config,
    )
    type(inst).generator_model = property(  # type: ignore[assignment]
        lambda s: s.env.models["generator"]
    )
    inst.lambda_c = 0.0
    inst._matcher = None
    return inst


def _make_dictionary(d: int = 16, n: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Tiny synthetic dictionary: random fingerprints + plausible Bloch params."""
    torch.manual_seed(0)
    fingerprints = torch.randn(d, n)
    parameters = torch.stack(
        [
            torch.linspace(0.5, 2.5, d),    # T1
            torch.linspace(0.05, 0.25, d),  # T2
            torch.ones(d),                  # M0
            torch.zeros(d),                 # B0
            torch.ones(d),                  # B1
        ],
        dim=-1,
    )
    return fingerprints, parameters


def test_strategy_exposes_kdtree_interface() -> None:
    """The wired strategy must expose the matcher API."""
    model = ConformalFPEmbedding(
        fingerprint_length=32, embedding_dim=5, hidden=16, n_blocks=2
    )
    strategy = _alloc_strategy(model)
    assert hasattr(strategy, "build_matcher")
    assert hasattr(strategy, "match")
    assert strategy._matcher is None


def test_match_before_build_raises() -> None:
    """Querying before the index is built must fail loudly, not silently."""
    model = ConformalFPEmbedding(
        fingerprint_length=32, embedding_dim=5, hidden=16, n_blocks=2
    )
    strategy = _alloc_strategy(model)
    with pytest.raises(RuntimeError, match="build_matcher"):
        strategy.match(torch.randn(2, 32))


def test_strategy_build_matcher_uses_kdtree() -> None:
    """``build_matcher`` returns an ``MRFDictlessMatcher`` and caches it."""
    model = ConformalFPEmbedding(
        fingerprint_length=32, embedding_dim=5, hidden=16, n_blocks=2
    )
    strategy = _alloc_strategy(model)
    dict_fp, dict_params = _make_dictionary(d=16, n=32)

    matcher = strategy.build_matcher(dict_fp, dict_params, batch_size=8)

    assert isinstance(matcher, MRFDictlessMatcher)
    assert strategy._matcher is matcher
    # Index must contain all D entries.
    assert matcher._dictionary is not None
    assert matcher._dictionary.embeddings.shape == (16, 5)
    assert matcher._dictionary.parameters.shape == (16, 5)


def test_strategy_match_round_trips_dictionary() -> None:
    """Querying with dictionary entries returns each entry as its own NN.

    This is the integration smoke: training loss runs first (proves the
    strategy's training surface works), then we build the KD-tree from
    the trained-once embedding, then verify the matcher round-trips.
    """
    model = ConformalFPEmbedding(
        fingerprint_length=32, embedding_dim=5, hidden=16, n_blocks=2
    )
    strategy = _alloc_strategy(model)
    dict_fp, dict_params = _make_dictionary(d=16, n=32)

    # 1. Training-time loss path is still callable (sanity).
    loss_out = strategy._compute_losses_impl(dict_fp, dict_fp, epoch=0)
    assert torch.isfinite(loss_out["g_total_loss"])

    # 2. Build the KD-tree index from the (untrained-but-deterministic) embedding.
    strategy.build_matcher(dict_fp, dict_params, batch_size=8)

    # 3. Query the dictionary against itself — every entry must self-match.
    out = strategy.match(dict_fp, k=1)
    assert out["indices"].shape == (16, 1)
    assert out["parameters"].shape == (16, 1, 5)
    assert out["distances"].shape == (16, 1)
    for i in range(16):
        assert out["indices"][i, 0] == i, (
            f"self-match failed at row {i}: got NN index {out['indices'][i, 0]}"
        )


def test_strategy_match_k_neighbours() -> None:
    """``k>1`` returns the requested number of neighbours per query."""
    model = ConformalFPEmbedding(
        fingerprint_length=32, embedding_dim=5, hidden=16, n_blocks=2
    )
    strategy = _alloc_strategy(model)
    dict_fp, dict_params = _make_dictionary(d=16, n=32)
    strategy.build_matcher(dict_fp, dict_params, batch_size=8)

    out = strategy.match(dict_fp[:4], k=3)
    assert out["indices"].shape == (4, 3)
    assert out["parameters"].shape == (4, 3, 5)
    # Self-match should be the top-1 for each query.
    for i in range(4):
        assert out["indices"][i, 0] == i
