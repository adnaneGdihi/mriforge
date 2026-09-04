"""Phase 7 — DomainAdaptationTrainingStrategy ↔ Balancer adapter.

Pins the contract of ``DomainAdaptationTrainingStrategy.unpack_balancer_batch``:
it must accept either round-robin or stratified Balancer items and
return ``(input, target)`` in the form the rest of ``train_step``
already expects.
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def da_strategy_proto():
    """Return a minimal DA-strategy-shaped object that exposes the adapter.

    We don't instantiate the full strategy (it requires an env / models
    / optimizers). Instead, we mix in the adapter method on a stub class
    via inheritance — the adapter only depends on ``_unpack_batch`` from
    the BatchPreparationMixin which itself just dispatches on dict shape.
    """
    from spectramr.infrastructure.training.strategies.domain_adaptation import (
        DomainAdaptationTrainingStrategy,
    )
    from spectramr.infrastructure.training.strategies.mixins.batch_preparation import (
        BatchPreparationMixin,
    )

    class _Stub(BatchPreparationMixin):
        # Inherit only the adapter from the real DA strategy — pull it
        # in via direct reference so we don't have to build a full env.
        unpack_balancer_batch = (
            DomainAdaptationTrainingStrategy.unpack_balancer_batch
        )

        def __init__(self):
            self.device = torch.device("cpu")

    return _Stub()


# ── Round-robin format ──────────────────────────────────────────────────────


def test_round_robin_with_paired_batch(da_strategy_proto) -> None:
    """{"domain": "src", "batch": (input, target)} ⇒ (input, target)."""
    inp = torch.zeros(2, 1, 4, 4)
    tgt = torch.ones(2, 1, 4, 4)
    item = {"domain": "src", "batch": (inp, tgt)}
    out_in, out_tgt = da_strategy_proto.unpack_balancer_batch(item)
    assert torch.equal(out_in, inp)
    assert torch.equal(out_tgt, tgt)


def test_round_robin_with_dict_batch(da_strategy_proto) -> None:
    """Round-robin item carrying a dict batch (canonical keys)."""
    inp = torch.zeros(2, 1, 4, 4)
    tgt = torch.ones(2, 1, 4, 4)
    item = {"domain": "tgt", "batch": {"input": inp, "target": tgt}}
    out_in, out_tgt = da_strategy_proto.unpack_balancer_batch(item)
    assert torch.equal(out_in, inp)
    assert torch.equal(out_tgt, tgt)


# ── Stratified / concat format ──────────────────────────────────────────────


def test_stratified_prefers_source_domain(da_strategy_proto) -> None:
    """When 'source' carries paired data, it wins."""
    src_in = torch.zeros(2, 1, 4, 4)
    src_tgt = torch.ones(2, 1, 4, 4)
    tgt_in = torch.full((2, 1, 4, 4), 7.0)
    item = {
        "source": (src_in, src_tgt),
        "target": (tgt_in,),  # input only
    }
    out_in, out_tgt = da_strategy_proto.unpack_balancer_batch(item)
    assert torch.equal(out_in, src_in)
    assert torch.equal(out_tgt, src_tgt)


def test_stratified_falls_back_to_input_only_when_no_target(da_strategy_proto) -> None:
    """No paired batch present ⇒ return (input, None) for the first item."""
    a_in = torch.zeros(2, 1, 4, 4)
    b_in = torch.ones(2, 1, 4, 4)
    item = {"site_a": (a_in,), "site_b": (b_in,)}
    out_in, out_tgt = da_strategy_proto.unpack_balancer_batch(item)
    assert out_in is not None
    # First-found input is acceptable; order depends on dict insertion.
    assert out_tgt is None


# ── Pass-through for single-domain batches ──────────────────────────────────


def test_single_domain_batch_passes_through(da_strategy_proto) -> None:
    """Non-dict batch (legacy single-loader) ⇒ delegate to _unpack_batch."""
    inp = torch.zeros(2, 1, 4, 4)
    tgt = torch.ones(2, 1, 4, 4)
    out_in, out_tgt = da_strategy_proto.unpack_balancer_batch((inp, tgt))
    assert torch.equal(out_in, inp)
    assert torch.equal(out_tgt, tgt)
