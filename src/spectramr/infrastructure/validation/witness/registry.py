"""The witness registry: one place that knows every silent-failure detector.

Why a registry rather than more checks. There are currently **two disjoint
validation stacks**, and neither knows the other exists:

* ``bootstrap.build_container`` runs ``ValidatorRegistry`` (via
  ``validate_config_at_startup``) and never touches ``ConfigHealthChecker``,
* ``cli/app.py::_audit_one`` runs ``ConfigHealthChecker`` + ``compatibility_matrix``
  and has zero references to ``get_validator_registry``.

So an audit-clean config can hit checks at train time that the audit never ran,
and vice versa. That is not a missing feature, it is a live instance of the
failure class this whole effort is about: the guard itself is silent. Adding
checks to either stack would deepen the split. One registry with several
consumers cannot drift, because there is nothing to drift from.

The second reason is enumerability. A registry can be *asked* what it covers, so
a meta-witness can assert that every taxonomy class has at least one detector.
The 126 ``check_*`` methods on ``ConfigHealthChecker`` cannot answer that
question: they are a hand-written sequence of calls inside one 275-line method,
which is how two of them ended up defined but never invoked.

Registration semantics follow ``core/metrics/registry.py``: a genuine name
collision raises, re-importing the same function is idempotent.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spectramr.infrastructure.validation.witness.subject import WitnessSubject

logger = logging.getLogger(__name__)

__all__ = [
    "Applicability",
    "Severity",
    "Stage",
    "Subject",
    "Tier",
    "Witness",
    "WitnessRegistry",
    "WitnessVerdict",
    "get_witness_registry",
    "register_witness",
]


class Stage(StrEnum):
    """Where in a declared value's life the failure happens.

    The stage is not decoration: it determines the only possible detector. A
    ``DECLARE`` defect is decidable from the corpus, an ``EXECUTE`` defect needs
    the code to actually run, and a guard placed at the wrong stage is not weak
    but incapable. See ``docs/explanation/execution_ledger.md``.
    """

    DECLARE = "declare"
    PARSE = "parse"
    DISPATCH = "dispatch"
    CONSTRUCT = "construct"
    WIRE = "wire"
    EXECUTE = "execute"
    CONTRIBUTE = "contribute"
    MEASURE = "measure"
    REPORT = "report"
    ENVIRONMENT = "environment"
    META = "meta"


class Tier(StrEnum):
    """Which surface can afford to run this witness."""

    T0 = "0"  # schema load
    T1 = "1"  # static cross-validation
    T2 = "2"  # synthetic forward pass
    T2_5 = "2.5"  # mechanism fires (needs a built model + 2 steps)
    T3_LITE = "3-lite"  # first K iterations of the real run
    T3 = "3"  # full instrumented shadow run


class Subject(StrEnum):
    """What a witness needs to look at.

    Declared so the gate can pre-filter: a CI corpus walk must never build a
    model just to discover the witness could not have run anyway.
    """

    CONFIG = "config"  # raw dict
    SETTINGS = "settings"  # resolved TrainingSettings
    LEDGER = "ledger"  # ExecutionLedger records
    MODULE_TREE = "module_tree"  # instantiated model
    TRACE = "trace"  # executed lines / sampled support
    TENSOR = "tensor"  # gradients, energies, measured physics
    ARTIFACTS = "artifacts"  # the run's output directory
    NONE = "none"  # pure introspection; needs no config at all


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Applicability:
    """Dispatch predicate, read from the RAW dict so it never forces a build.

    Generalises ``ValidationRule.applies``, which could only dispatch on
    ``training_mode`` and ``model_type``.
    """

    model_types: frozenset[str] | None = None
    training_modes: frozenset[str] | None = None
    regimes: frozenset[str] | None = None
    paradigm_keys: frozenset[str] | None = None

    _PATHS = (
        ("model_types", ("model", "model_type")),
        ("training_modes", ("training", "training_mode")),
        ("regimes", ("workflow", "name")),
    )

    def matches(self, config: dict[str, Any]) -> bool:
        for attr, path in self._PATHS:
            allowed = getattr(self, attr)
            if allowed is None:
                continue
            node: Any = config
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
            # Compare against the VALUE, never str(member): a (str, Enum) member
            # stringifies as 'Class.MEMBER' and would match nothing.
            value = getattr(node, "value", node)
            if value not in allowed:
                return False
        return True


@dataclass(frozen=True)
class WitnessVerdict:
    """One finding. Deliberately shaped like ``HealthCheckResult``.

    Keeping the field names aligned is what lets the existing audit rendering,
    ``--json`` payload and exit codes consume witness output unchanged, so the
    unification is additive rather than a rewrite of the CLI.
    """

    witness_name: str
    passed: bool
    message: str
    severity: Severity = Severity.ERROR
    category: str = "unclassified"
    class_ids: tuple[str, ...] = ()
    stage: Stage | None = None
    tier: Tier | None = None
    yaml_keys: tuple[str, ...] = ()
    fix_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "witness_name": self.witness_name,
            "passed": self.passed,
            "message": self.message,
            "severity": str(self.severity),
            "category": self.category,
            "class_ids": list(self.class_ids),
            "stage": str(self.stage) if self.stage else None,
            "tier": str(self.tier) if self.tier else None,
            "yaml_keys": list(self.yaml_keys),
            "fix_hint": self.fix_hint,
        }


WitnessFn = Callable[["WitnessSubject"], "WitnessVerdict | list[WitnessVerdict]"]


@dataclass(frozen=True)
class Witness:
    """A registered detector plus the metadata the gate dispatches on."""

    name: str
    fn: WitnessFn
    category: str
    stage: Stage
    tiers: frozenset[Tier]
    subjects: frozenset[Subject]
    class_ids: tuple[str, ...] = ()
    severity: Severity = Severity.ERROR
    description: str = ""
    applicability: Applicability = field(default_factory=Applicability)
    fix_hint: str | None = None


class WitnessRegistry:
    """Dict-backed, enumerable, and able to answer what it covers."""

    def __init__(self) -> None:
        self._witnesses: dict[str, Witness] = {}

    def register(self, witness: Witness, *, override: bool = False) -> None:
        existing = self._witnesses.get(witness.name)
        if existing is not None and not override:
            if existing.fn is witness.fn:
                return  # same function re-imported; idempotent
            raise ValueError(
                f"Witness {witness.name!r} is already registered to a different "
                f"function ({existing.fn!r}). Names must be unique so a verdict "
                f"can be traced back to exactly one detector."
            )
        self._witnesses[witness.name] = witness

    def all(self) -> tuple[Witness, ...]:
        return tuple(self._witnesses.values())

    def get(self, name: str) -> Witness | None:
        return self._witnesses.get(name)

    def for_tier(self, tier: Tier) -> tuple[Witness, ...]:
        return tuple(w for w in self._witnesses.values() if tier in w.tiers)

    def covered_class_ids(self) -> frozenset[str]:
        """Every taxonomy id some registered witness claims to detect."""
        return frozenset(cid for w in self._witnesses.values() for cid in w.class_ids)

    def __len__(self) -> int:
        return len(self._witnesses)


_registry: WitnessRegistry | None = None


def get_witness_registry() -> WitnessRegistry:
    global _registry
    if _registry is None:
        _registry = WitnessRegistry()
    return _registry


def register_witness(
    name: str,
    *,
    category: str,
    stage: Stage,
    tiers: Sequence[Tier],
    subjects: Sequence[Subject],
    class_ids: Sequence[str] = (),
    severity: Severity = Severity.ERROR,
    description: str = "",
    applicability: Applicability | None = None,
    fix_hint: str | None = None,
) -> Callable[[WitnessFn], WitnessFn]:
    """Register a witness. Decorator, fired at import time.

    ``class_ids`` are the taxonomy ids this detector covers. They are what lets
    the coverage meta-witness assert that no class is left with no detector, so
    leaving them empty is a real (if quiet) gap rather than a style choice.
    """

    def decorator(fn: WitnessFn) -> WitnessFn:
        get_witness_registry().register(
            Witness(
                name=name,
                fn=fn,
                category=category,
                stage=stage,
                tiers=frozenset(tiers),
                subjects=frozenset(subjects),
                class_ids=tuple(class_ids),
                severity=severity,
                description=description or (fn.__doc__ or "").strip().split("\n")[0],
                applicability=applicability or Applicability(),
                fix_hint=fix_hint,
            )
        )
        return fn

    return decorator
