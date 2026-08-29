"""What a witness is given to look at, built lazily per surface.

The four surfaces can afford very different things. A CI corpus walk over 400
YAMLs cannot build a model per config; the in-run surface already has a model,
a strategy and a ledger in hand. A single eager container would force the
cheapest surface to pay the most expensive surface's cost.

So the subject holds *factories*, not values, and resolves each kind on first
access. Laziness is two-layer:

1. ``gate.run_witnesses`` pre-filters on ``Witness.subjects``, so a witness that
   needs a module tree is never even invoked under a CI subject;
2. ``get()`` is the backstop. If a witness is mis-scheduled it raises
   :class:`WitnessSubjectUnavailableError` naming exactly which kind it lacked, rather
   than crashing with an ``AttributeError`` deep inside a factory that was never
   supposed to run, or worse, silently returning a pass.

That second layer matters more than it looks: a mis-scheduled witness that
quietly returned "nothing to check" would be a detector that never fires, which
is the failure this package exists to prevent.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mriforge.infrastructure.validation.witness.registry import Subject

logger = logging.getLogger(__name__)

__all__ = ["WitnessSubject", "WitnessSubjectUnavailableError"]


class WitnessSubjectUnavailableError(RuntimeError):
    """A witness asked for a subject kind this surface cannot provide."""


@dataclass
class WitnessSubject:
    """Lazily-resolved bundle of everything a witness might inspect."""

    config_path: str | None
    raw_config: dict[str, Any]
    _factories: dict[Subject, Callable[[], Any]] = field(default_factory=dict)
    _cache: dict[Subject, Any] = field(default_factory=dict, init=False, repr=False)

    # ---- construction per surface -------------------------------------

    @classmethod
    def for_ci(cls, config_path: str | None, raw_config: dict[str, Any]) -> WitnessSubject:
        """Corpus walk: only the raw dict exists.

        No ``TrainingSettings``, no torch, no model. A witness declaring
        ``MODULE_TREE`` run against this subject raises rather than silently
        passing.
        """
        return cls(config_path=config_path, raw_config=raw_config)

    @classmethod
    def for_audit(
        cls,
        config_path: str | None,
        settings: Any,
        *,
        module_tree_factory: Callable[[], Any] | None = None,
    ) -> WitnessSubject:
        """``mriforge audit`` / bootstrap: the resolved settings are available."""
        subject = cls(
            config_path=config_path,
            raw_config=(
                settings.model_dump(mode="json") if hasattr(settings, "model_dump") else {}
            ),
        )
        subject._factories[Subject.SETTINGS] = lambda: settings
        subject._factories[Subject.LEDGER] = _ledger_factory
        if module_tree_factory is not None:
            subject._factories[Subject.MODULE_TREE] = module_tree_factory
        return subject

    # ---- access --------------------------------------------------------

    def get(self, kind: Subject) -> Any:
        if kind is Subject.NONE:
            return None
        if kind is Subject.CONFIG:
            return self.raw_config
        if kind in self._cache:
            return self._cache[kind]
        factory = self._factories.get(kind)
        if factory is None:
            raise WitnessSubjectUnavailableError(
                f"subject kind {str(kind)!r} is unavailable on this surface "
                f"(config_path={self.config_path!r}). A witness declaring it was "
                f"scheduled where it cannot run — fix the witness's `subjects=` "
                f"or the surface that built this subject."
            )
        value = factory()
        self._cache[kind] = value
        return value

    def provides(self, kinds: frozenset[Subject]) -> bool:
        """Whether every requested kind is obtainable here, without building it."""
        return all(
            kind in (Subject.CONFIG, Subject.NONE) or kind in self._factories for kind in kinds
        )

    @property
    def config(self) -> dict[str, Any]:
        return self.raw_config

    @property
    def settings(self) -> Any:
        return self.get(Subject.SETTINGS)

    @property
    def ledger(self) -> Any:
        return self.get(Subject.LEDGER)


def _ledger_factory() -> Any:
    from mriforge.core.execution_ledger import ExecutionLedger

    return ExecutionLedger.current()
