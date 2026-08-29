"""Site / domain batch balancer (Phase 4e).

Wraps two or more DataLoaders into a single iterator that emits batches
according to a declared :class:`DomainBalancingStrategy`. Three strategies:

- :class:`RoundRobinBalancer` — alternates batches across loaders.
  Step 0: loader[0]; step 1: loader[1]; step 2: loader[0]; ...
  Emits ``{"domain": name, "batch": <loader-batch>}`` dicts so the
  strategy code can dispatch on domain identity.

- :class:`StratifiedBalancer` — yields one composite batch per step that
  contains samples from every loader, weighted by per-loader weights.
  Useful when the GAN discriminator needs a mixed batch per step.

- :class:`ConcatBalancer` — zips loaders without balancing. Each yielded
  dict carries one batch from each loader simultaneously.

The strategy receives a dict keyed by domain name and dispatches the
appropriate model passes. Compare with the legacy code in
``DomainAdaptationTrainingStrategy`` which manually advanced two
DataLoader iterators — easy to desync, no per-batch domain tag.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from torch.utils.data import DataLoader

__all__ = [
    "Balancer",
    "ConcatBalancer",
    "RoundRobinBalancer",
    "StratifiedBalancer",
    "build_balancer",
]


class Balancer:
    """Base class — wraps two or more named loaders and emits domain-tagged batches."""

    def __init__(self, loaders: dict[str, DataLoader]):
        if len(loaders) < 2:
            raise ValueError(f"Balancer needs at least 2 loaders; got {len(loaders)}.")
        self.loaders = loaders
        self.names = list(loaders.keys())

    def __iter__(self) -> Iterator[Any]:
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of steps per epoch — depends on strategy."""
        raise NotImplementedError


def _weights_are_uniform(weights: dict[str, float] | None) -> bool:
    """True when every domain shares the same weight (⇒ weighting is a no-op)."""
    if not weights:
        return True
    values = list(weights.values())
    return all(abs(v - values[0]) < 1e-9 for v in values)


class RoundRobinBalancer(Balancer):
    """Alternates batches across loaders, honouring per-domain weights.

    Each step yields a single dict ``{"domain": name, "batch": ...}``. Because
    only ONE domain appears per step, differential oversampling IS expressible
    here (unlike :class:`StratifiedBalancer`): a domain's weight sets how many
    times it appears per rotation cycle. ``{source: 1.0, target: 3.0}`` visits
    ``target`` 3x as often as ``source`` -- the domain-adaptation oversampling
    the ``DomainConfigSchema.weight`` knob advertises. Weights are quantised to
    integer visit counts per cycle (``round``); uniform weights reproduce the
    original strict round-robin exactly. One epoch = the *longest* loader times
    the visit-schedule length (shorter loaders cycle).
    """

    def __init__(
        self,
        loaders: dict[str, DataLoader],
        weights: dict[str, float] | None = None,
    ):
        super().__init__(loaders)
        self.weights = weights or dict.fromkeys(self.names, 1.0)
        # Visit schedule: each domain appears ``round(weight)`` times, evenly
        # interleaved (not blocked as [a,a,a,b]) so oversampling is spread
        # across the epoch rather than clustered.
        self._schedule = self._build_schedule()

    def _build_schedule(self) -> list[str]:
        counts = {name: max(1, round(self.weights.get(name, 1.0))) for name in self.names}
        schedule: list[str] = []
        remaining = dict(counts)
        while any(remaining.values()):
            for name in self.names:
                if remaining[name] > 0:
                    schedule.append(name)
                    remaining[name] -= 1
        return schedule

    def __iter__(self) -> Iterator[dict[str, Any]]:
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        # One schedule pass per loader-length cycle; a weight-w domain is thus
        # visited ``max_len * w`` times vs. ``max_len`` for a weight-1 domain.
        max_len = max(len(loader) for loader in self.loaders.values())
        for _ in range(max_len):
            for domain in self._schedule:
                try:
                    batch = next(iterators[domain])
                except StopIteration:
                    iterators[domain] = iter(self.loaders[domain])
                    batch = next(iterators[domain])
                yield {"domain": domain, "batch": batch}

    def __len__(self) -> int:
        max_len = max(len(loader) for loader in self.loaders.values())
        return max_len * len(self._schedule)


class StratifiedBalancer(Balancer):
    """Yields one composite batch per step containing samples from every loader.

    Each composite batch is a dict ``{name: batch_from_that_loader}``.
    Strategy code is expected to consume all domains per step (e.g.,
    forward pass on each domain, then a combined loss).

    One epoch = the length of the *shortest* loader (we don't repeat the
    short side mid-batch; the strategy sees each target sample at most once).
    """

    def __init__(
        self,
        loaders: dict[str, DataLoader],
        weights: dict[str, float] | None = None,
    ):
        super().__init__(loaders)
        self.weights = weights or dict.fromkeys(self.names, 1.0)
        # A composite step yields exactly one batch per domain ({name: batch}),
        # so this balancer CANNOT express differential oversampling — a weight
        # of 3.0 on one domain has nowhere to go without changing that contract.
        # The old code silently stored non-uniform weights and ignored them: a
        # config asking for 3x target oversampling ran 1:1 (pitfall #15 -- a
        # validated-but-unwired knob). Fail loud instead; route weighted
        # oversampling through ``round_robin`` (which yields one domain/step and
        # CAN honour weights), or add a list-valued composite balancer.
        if not _weights_are_uniform(self.weights):
            raise NotImplementedError(
                "StratifiedBalancer received non-uniform domain weights "
                f"({self.weights}), but its per-step composite ({{name: batch}}) "
                "yields one batch per domain per step and cannot differentially "
                "oversample. Use uniform weights here, or select 'round_robin' "
                "balancing (which honours per-domain weights)."
            )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        min_len = min(len(loader) for loader in self.loaders.values())
        for _ in range(min_len):
            composite: dict[str, Any] = {}
            for name in self.names:
                composite[name] = next(iterators[name])
            yield composite

    def __len__(self) -> int:
        return min(len(loader) for loader in self.loaders.values())


class ConcatBalancer(Balancer):
    """No balancing — zip the loaders.

    Useful when strategy code wants explicit access to both batches every
    step without any reordering. Identical to :class:`StratifiedBalancer`
    with unit weights — kept as a distinct name for readability in YAMLs.
    """

    def __iter__(self) -> Iterator[dict[str, Any]]:
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        min_len = min(len(loader) for loader in self.loaders.values())
        for _ in range(min_len):
            composite: dict[str, Any] = {}
            for name in self.names:
                composite[name] = next(iterators[name])
            yield composite

    def __len__(self) -> int:
        return min(len(loader) for loader in self.loaders.values())


def build_balancer(
    loaders: dict[str, DataLoader],
    strategy: str = "round_robin",
    weights: dict[str, float] | None = None,
) -> Balancer:
    """Dispatch on strategy string to the right balancer class."""
    if strategy == "round_robin":
        return RoundRobinBalancer(loaders, weights)
    if strategy == "stratified":
        return StratifiedBalancer(loaders, weights)
    if strategy == "concat":
        return ConcatBalancer(loaders)
    raise ValueError(
        f"Unknown balancing strategy: {strategy}. Must be one of: round_robin, stratified, concat."
    )
