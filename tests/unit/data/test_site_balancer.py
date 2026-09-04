"""Phase 4e — Site / domain balancers.

Uses tiny synthetic DataLoaders to exercise the three balancing
strategies (round_robin, stratified, concat). No model / no manifests —
the balancer is pure DataLoader plumbing.
"""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from spectramr.data.builders.site_balancer import (
    Balancer,
    ConcatBalancer,
    RoundRobinBalancer,
    StratifiedBalancer,
    build_balancer,
)


def _make_loader(n: int, value: int, batch_size: int = 2) -> DataLoader:
    """Tiny DataLoader where each sample is the constant `value`."""
    x = torch.full((n,), float(value))
    ds = TensorDataset(x)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ── Base class guards ───────────────────────────────────────────────────────


def test_balancer_requires_at_least_two_loaders() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        RoundRobinBalancer({"only": _make_loader(4, 0)})


# ── RoundRobinBalancer ──────────────────────────────────────────────────────


def test_round_robin_alternates_domains() -> None:
    """First batch from src, second from tgt, third from src, ..."""
    src = _make_loader(4, 1)  # 2 batches
    tgt = _make_loader(4, 2)  # 2 batches
    balancer = RoundRobinBalancer({"src": src, "tgt": tgt})
    domains_seen = [item["domain"] for item in balancer]
    # 4 batches total: src, tgt, src, tgt
    assert domains_seen == ["src", "tgt", "src", "tgt"]


def test_round_robin_emits_domain_tagged_dicts() -> None:
    src = _make_loader(2, 1)
    tgt = _make_loader(2, 2)
    balancer = RoundRobinBalancer({"src": src, "tgt": tgt})
    first = next(iter(balancer))
    assert "domain" in first
    assert "batch" in first
    assert first["domain"] == "src"


def test_round_robin_cycles_shorter_loader() -> None:
    """Shorter loader restarts mid-epoch so cycling stays balanced."""
    src = _make_loader(8, 1)  # 4 batches
    tgt = _make_loader(2, 2)  # 1 batch — will need to repeat
    balancer = RoundRobinBalancer({"src": src, "tgt": tgt})
    items = list(balancer)
    # max_len * num_loaders = 4 * 2 = 8
    assert len(items) == 8


def test_round_robin_len_is_max_times_num_domains() -> None:
    src = _make_loader(6, 1)  # 3 batches
    tgt = _make_loader(4, 2)  # 2 batches
    balancer = RoundRobinBalancer({"src": src, "tgt": tgt})
    assert len(balancer) == 6


# ── StratifiedBalancer ─────────────────────────────────────────────────────


def test_stratified_yields_composite_dicts() -> None:
    src = _make_loader(4, 1)  # 2 batches
    tgt = _make_loader(4, 2)
    balancer = StratifiedBalancer({"src": src, "tgt": tgt})
    items = list(balancer)
    assert len(items) == 2  # min loader length
    for item in items:
        assert set(item.keys()) == {"src", "tgt"}


def test_stratified_each_step_has_one_batch_per_domain() -> None:
    src = _make_loader(4, 1)
    tgt = _make_loader(4, 2)
    balancer = StratifiedBalancer({"src": src, "tgt": tgt})
    first = next(iter(balancer))
    # Each domain's batch is a tuple from TensorDataset.
    src_x = first["src"][0]
    tgt_x = first["tgt"][0]
    assert src_x.shape[0] == 2  # batch size
    assert tgt_x.shape[0] == 2


def test_stratified_len_is_min_loader_length() -> None:
    src = _make_loader(8, 1)  # 4 batches
    tgt = _make_loader(4, 2)  # 2 batches
    balancer = StratifiedBalancer({"src": src, "tgt": tgt})
    assert len(balancer) == 2


# ── ConcatBalancer ─────────────────────────────────────────────────────────


def test_concat_yields_dict_per_step() -> None:
    src = _make_loader(4, 1)
    tgt = _make_loader(4, 2)
    balancer = ConcatBalancer({"src": src, "tgt": tgt})
    items = list(balancer)
    assert len(items) == 2
    for item in items:
        assert "src" in item and "tgt" in item


# ── Factory dispatch ────────────────────────────────────────────────────────


def test_build_balancer_round_robin() -> None:
    src = _make_loader(4, 1)
    tgt = _make_loader(4, 2)
    b = build_balancer({"src": src, "tgt": tgt}, strategy="round_robin")
    assert isinstance(b, RoundRobinBalancer)


def test_build_balancer_stratified() -> None:
    src = _make_loader(4, 1)
    tgt = _make_loader(4, 2)
    b = build_balancer({"src": src, "tgt": tgt}, strategy="stratified")
    assert isinstance(b, StratifiedBalancer)


def test_build_balancer_concat() -> None:
    src = _make_loader(4, 1)
    tgt = _make_loader(4, 2)
    b = build_balancer({"src": src, "tgt": tgt}, strategy="concat")
    assert isinstance(b, ConcatBalancer)


def test_build_balancer_rejects_unknown_strategy() -> None:
    src = _make_loader(4, 1)
    tgt = _make_loader(4, 2)
    with pytest.raises(ValueError, match="Unknown balancing strategy"):
        build_balancer({"src": src, "tgt": tgt}, strategy="random")


# ── Three-domain support ────────────────────────────────────────────────────


def test_round_robin_handles_three_domains() -> None:
    """Domain-adaptation with one source + two targets."""
    a = _make_loader(2, 1)
    b = _make_loader(2, 2)
    c = _make_loader(2, 3)
    balancer = RoundRobinBalancer({"a": a, "b": b, "c": c})
    domains_seen = [item["domain"] for item in balancer]
    # 1 batch each * 3 domains = 3 steps; cycle: a, b, c
    assert domains_seen[:3] == ["a", "b", "c"]


def test_stratified_handles_three_domains() -> None:
    a = _make_loader(4, 1)
    b = _make_loader(4, 2)
    c = _make_loader(4, 3)
    balancer = StratifiedBalancer({"a": a, "b": b, "c": c})
    first = next(iter(balancer))
    assert set(first.keys()) == {"a", "b", "c"}
