"""Tests for the domain/site batch balancers.

Review finding (2026-07-01): ``DomainConfigSchema.weight`` was validated,
stamped, and threaded into ``build_balancer`` — but no balancer consumed it.
``StratifiedBalancer`` stored ``self.weights`` and never read it; the default
``RoundRobinBalancer`` did not even accept it. A domain-adaptation config asking
for 3x oversampling of the target domain silently ran 1:1 (pitfall #15).

Fix: ``RoundRobinBalancer`` (one domain per step) honours weights via an
integer visit schedule; ``StratifiedBalancer`` (one batch per domain per step)
cannot express differential weights and now raises rather than no-op.
"""

from __future__ import annotations

from collections import Counter

import pytest

from mriforge.data.builders.site_balancer import (
    RoundRobinBalancer,
    StratifiedBalancer,
    build_balancer,
)


class _FakeLoader:
    """Minimal DataLoader stand-in: yields ``n`` unique-labelled batches."""

    def __init__(self, name: str, n: int) -> None:
        self._name = name
        self._n = n

    def __iter__(self):
        return iter([f"{self._name}:{i}" for i in range(self._n)])

    def __len__(self) -> int:
        return self._n


def _domain_counts(balancer) -> Counter:
    return Counter(step["domain"] for step in balancer)


def test_round_robin_uniform_weights_is_balanced() -> None:
    """Unit weights reproduce strict 1:1 round-robin (backward compatible)."""
    loaders = {"source": _FakeLoader("source", 4), "target": _FakeLoader("target", 4)}
    counts = _domain_counts(RoundRobinBalancer(loaders))
    assert counts["source"] == counts["target"]


def test_round_robin_honours_weighted_oversampling() -> None:
    """weight 3.0 on 'target' ⇒ ~3x as many target steps as source."""
    loaders = {"source": _FakeLoader("source", 4), "target": _FakeLoader("target", 4)}
    balancer = RoundRobinBalancer(loaders, weights={"source": 1.0, "target": 3.0})
    counts = _domain_counts(balancer)
    assert counts["target"] == 3 * counts["source"], counts
    assert len(balancer) == sum(counts.values())


def test_round_robin_schedule_is_interleaved_not_blocked() -> None:
    """The oversampled domain is spread out, not clustered as [s, t, t, t]."""
    loaders = {"source": _FakeLoader("source", 2), "target": _FakeLoader("target", 2)}
    balancer = RoundRobinBalancer(loaders, weights={"source": 1.0, "target": 3.0})
    # One schedule cycle interleaves as [source, target, target, target].
    assert balancer._schedule == ["source", "target", "target", "target"]


def test_build_balancer_round_robin_threads_weights() -> None:
    """The factory (default strategy) must forward weights to round-robin."""
    loaders = {"source": _FakeLoader("source", 3), "target": _FakeLoader("target", 3)}
    balancer = build_balancer(
        loaders, strategy="round_robin", weights={"source": 1.0, "target": 2.0}
    )
    counts = _domain_counts(balancer)
    assert counts["target"] == 2 * counts["source"]


def test_stratified_rejects_non_uniform_weights() -> None:
    """Stratified cannot express differential weights — it must raise, not no-op."""
    loaders = {"source": _FakeLoader("source", 3), "target": _FakeLoader("target", 3)}
    with pytest.raises(NotImplementedError, match="differentially oversample"):
        StratifiedBalancer(loaders, weights={"source": 1.0, "target": 3.0})


def test_stratified_uniform_weights_ok() -> None:
    """Uniform weights are a well-defined no-op and must still construct."""
    loaders = {"source": _FakeLoader("source", 3), "target": _FakeLoader("target", 3)}
    balancer = StratifiedBalancer(loaders, weights={"source": 2.0, "target": 2.0})
    steps = list(balancer)
    assert len(steps) == 3
    assert all(set(step) == {"source", "target"} for step in steps)


# ── WS8: import-surface guard after the unused-typing.Iterable cleanup ─────────


def test_module_imports_and_exposes_balancers():
    """Guards the WS8 import cleanup (dropped unused ``typing.Iterable``): the
    module must still import and expose its balancer surface."""
    import mriforge.data.builders.site_balancer as sb

    assert hasattr(sb, "build_balancer")
    assert hasattr(sb, "RoundRobinBalancer")
    assert hasattr(sb, "Balancer")
