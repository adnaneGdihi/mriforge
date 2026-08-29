"""Runtime ledger of every substitution a run made, stamped into its artifacts.

Why this exists. ``tests/unit/config/test_schema_key_consumption.py`` asks
whether a key is *referenced*, and ``tests/unit/config/test_knob_behaviour.py``
asks whether the reference *does anything*. Both are tests: they run at PR time,
and they only cover the knobs somebody thought to put in a table. Neither tells
a **run** anything. So a config could resolve to something nobody declared and
the run would complete, report success, and leave no trace of the substitution:

* ``optimization.scheduler`` was read under the wrong key shape, so 226/226 arms
  trained under ``CosineAnnealingLR(T_max=100)`` (issue #533),
* ``acceleration.min_center_fraction`` was dropped by ``extra="ignore"`` and the
  32x rung realised 12.2x (issues #534/#550),
* ``eta_min: 1e-6`` happened to equal the library default, so even the logged LR
  looked correct.

This module is the runtime half. Every seam that substitutes a value records one
:class:`Substitution` here, and the collected ledger is written into the run's
``resolved_config.json`` under ``_ledger``. ``**kwargs`` flexibility is
deliberately preserved: nothing here forbids a drop, it only makes the drop
*visible*. An unrecorded drop is the failure mode; a recorded one is a decision.

The per-run output is also the candidate list those two hand-maintained test
inventories currently need a human to discover.

Layering. This module is a pure leaf: stdlib only, no ``torch``, and no imports
from anywhere else in ``mriforge``. That is what lets ``config/settings.py``,
``models/factories/``, and ``infrastructure/training/`` all record into it
without violating the inward-only dependency rule (non-negotiable #5). Pydantic
models are handled by duck-typing (``model_fields`` / ``model_extra``) rather
than importing pydantic, which also keeps the diff testable against fakes.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "NO_VALUE",
    "ExecutionLedger",
    "Substitution",
    "SubstitutionClass",
    "diff_declared_vs_resolved",
    "record_default_coincidence",
    "unconsumed_keys",
]


class _NoValue:
    """Sentinel for "not applicable", distinct from a legitimate ``None``.

    ``resolved=None`` means the run really used ``None``. ``resolved=NO_VALUE``
    means the value never arrived at all. Collapsing those two is how a dropped
    knob reads as an intentional null.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "NO_VALUE"


NO_VALUE = _NoValue()

_UNSET_JSON = "__unset__"

#: Env knob: make a failed ledger *flush* abort the run instead of soft-failing.
#: Off by default so a stamping hiccup never kills GPU work; on for CI and
#: cluster-audit contexts that want "a run without a ledger did not happen".
STRICT_ENV = "MRIFORGE_LEDGER_STRICT"


class SubstitutionClass(StrEnum):
    """Stable taxonomy id for one kind of silent substitution.

    ``StrEnum``, deliberately, not ``(str, Enum)``: the latter makes ``str(member)``
    return ``'SubstitutionClass.MEMBER'`` rather than the value, which is the trap
    that made a scoped corpus check select zero arms (see
    ``docs/explanation/execution_ledger.md``). Here ``str()`` gives the value.

    Ids map onto the stages in ``docs/explanation/execution_ledger.md``.
    This is an internal diagnostic vocabulary, not a YAML-facing registered
    option set, so it does not need the audit's raise-on-unknown treatment.
    """

    # Every member here HAS an emitter. Four earlier members
    # (`getattr_default_fired`, `read_not_forwarded`, `enum_no_factory`,
    # `key_only_in_resolved`) were declared with no emitter and no wiring plan,
    # which is pitfall #15 inside the module built to detect pitfall #15: an
    # advertised class that can never appear tells a reader the ledger covers
    # ground it does not. They are deleted rather than kept as aspiration —
    # backlog the class, do not advertise it. (`key_only_in_resolved` is
    # additionally redundant: injected defaults are COUNTED, not itemised.)
    #
    # CONSTRUCT stage
    DROPPED_UNCONSUMED_KWARG = "dropped_unconsumed_kwarg"
    DEFAULT_COINCIDENCE = "default_coincidence"
    # PARSE stage
    RAW_DICT_UNVALIDATED = "raw_dict_unvalidated"
    EXTRA_ALLOW_UNTYPED = "extra_allow_untyped"
    EXTRA_IGNORE_DROPPED = "extra_ignore_dropped"
    VALUE_CHANGED_ON_FINALIZE = "value_changed_on_finalize"


def _jsonable(value: Any) -> Any:
    """Coerce to something ``json.dumps`` will accept, eagerly.

    Coercion happens at record time, not flush time, on purpose: if a caller
    hands us something unserialisable we want the traceback to point at the
    instrumented seam, not at a mysterious failure thousands of iterations later
    when the artifact is written.
    """
    if isinstance(value, _NoValue):
        return _UNSET_JSON
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


@dataclass(frozen=True)
class Substitution:
    """One declared/resolved/dropped value event, diagnosable on its own.

    ``library_default`` is what makes the invisible case visible: when a
    declared value coincides with the receiver's own default, the resolved
    config shows no delta at all, so the drop is only detectable here at the
    seam. That is the ``eta_min: 1e-6`` case from issue #533.
    """

    seq: int
    class_id: SubstitutionClass
    site: str
    stage: str
    path: str
    requested: Any = NO_VALUE
    resolved: Any = NO_VALUE
    reason: str = ""
    severity: str = "warning"
    consumer: str | None = None
    library_default: Any = NO_VALUE

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "class_id": self.class_id.value,
            "site": self.site,
            "stage": self.stage,
            "path": self.path,
            "requested": _jsonable(self.requested),
            "resolved": _jsonable(self.resolved),
            "reason": self.reason,
            "severity": self.severity,
            "consumer": self.consumer,
            "library_default": _jsonable(self.library_default),
        }


_ACTIVE: contextvars.ContextVar[ExecutionLedger | None] = contextvars.ContextVar(
    "mriforge_execution_ledger", default=None
)


@dataclass
class ExecutionLedger:
    """Collects :class:`Substitution` records for one logical run.

    Scoping is a ``ContextVar``, not the DI container: ``build_container``
    calls ``container.clear()`` *after* ``TrainingSettings.from_yaml`` has
    already run, so a DI-registered ledger would lose every config-time record.
    ``config/settings.py`` also cannot reach the container without importing
    leftward.

    One ledger per *logical* run, not per process: ``pipelines/ablation.py`` and
    ``pipelines/hpo.py`` drive many variants in a single interpreter, so each
    must call :meth:`begin_run` or variant A's records leak into variant B's
    artifact. HPO subprocesses and DDP ranks are separate interpreters and
    isolate for free.
    """

    source: str | None = None
    substitutions: list[Substitution] = field(default_factory=list)
    health_report: dict[str, Any] | None = None
    #: Dotted paths of the schema defaults injected, in walk order.
    #:
    #: Was a bare ``int``. The count alone could not answer the question it was
    #: there for -- *which* knobs is this run taking on trust -- and it hid a
    #: defect: the number tracks the config's SPELLING, not the run. Measured on
    #: one arm across the 2026-08-02 canonical-key drain, whose resolved
    #: document is byte-identical either way (``verify_config_migration`` leg
    #: (ii), 58/58)::
    #:
    #:     legacy spellings   defaults_injected = 563
    #:     canonical spellings                  = 625      <- +62, same run
    #:
    #: The walker descends only into ``raw_keys & resolved_keys``, so a
    #: sub-block the YAML never mentions costs **1** and its own fields are
    #: never reached. Folding a leaf into a new sub-block puts that block in
    #: ``raw``, the walker descends, and its remaining fields become countable.
    #: So the total is only comparable between configs written at the same
    #: schema depth. ``defaults_paths`` makes that visible instead of averaging
    #: it into one integer -- the delta is exactly the sub-blocks that appeared.
    #:
    #: Kept as paths, never ``Substitution`` records: those land in
    #: ``substitutions`` and would bury the handful that matter (610 defaults vs
    #: 14 substitutions on a typical arm).
    defaults_paths: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def defaults_injected(self) -> int:
        """How many defaults were injected. Derived, never accumulated.

        A separate counter alongside the list would be a second derivation of
        one number, which is the failure this module exists to catch.
        """
        return len(self.defaults_paths)

    # ---- lifecycle ----------------------------------------------------

    @classmethod
    def begin_run(cls, *, source: str | None = None) -> ExecutionLedger:
        """Start a fresh ledger and make it active for this context."""
        ledger = cls(source=source)
        _ACTIVE.set(ledger)
        return ledger

    @classmethod
    def current(cls) -> ExecutionLedger | None:
        return _ACTIVE.get()

    @classmethod
    def current_or_begin(cls, *, source: str | None = None) -> ExecutionLedger:
        """Adopt the active ledger, or self-arm loudly if nobody armed one.

        A programmatic caller that skipped the normal entry points gets degraded
        coverage (config-time records are already lost) rather than a crash --
        but it is told so, because silently starting an empty ledger would make
        "no substitutions" indistinguishable from "nothing was watching".
        """
        ledger = _ACTIVE.get()
        if ledger is not None:
            return ledger
        logger.warning(
            "ExecutionLedger.begin_run() was never called for this run; "
            "config-load substitutions were NOT recorded. The ledger in this "
            "run's artifacts is incomplete, not empty."
        )
        ledger = cls.begin_run(source=source)
        ledger.notes.append("ledger armed late: config-load substitutions are missing")
        return ledger

    @classmethod
    def recording(cls) -> bool:
        """Whether a ledger is armed. One ``ContextVar`` read.

        This is the entire cost paid by an unarmed caller -- audit, unit tests,
        a notebook doing ``TrainingSettings.from_yaml`` -- so instrumentation
        can be gated on it without measurable overhead.
        """
        return _ACTIVE.get() is not None

    @classmethod
    def reset(cls) -> None:
        """Disarm. Tests use this; production never needs it."""
        _ACTIVE.set(None)

    # ---- recording ----------------------------------------------------

    def record(
        self,
        *,
        class_id: SubstitutionClass,
        site: str,
        stage: str,
        path: str,
        requested: Any = NO_VALUE,
        resolved: Any = NO_VALUE,
        reason: str = "",
        severity: str = "warning",
        consumer: str | None = None,
        library_default: Any = NO_VALUE,
    ) -> Substitution:
        """Append one record. Never soft-fails; see :func:`_jsonable`."""
        sub = Substitution(
            seq=len(self.substitutions),
            class_id=class_id,
            site=site,
            stage=stage,
            path=path,
            requested=requested,
            resolved=resolved,
            reason=reason,
            severity=severity,
            consumer=consumer,
            library_default=library_default,
        )
        # Force serialisation now so a bad payload blames the right call site.
        sub.to_dict()
        self.substitutions.append(sub)
        return sub

    def attach_health_report(self, report: Any) -> None:
        """Persist the config health report, which is otherwise discarded.

        ``validate_config_health`` already runs on every training run and
        already classifies findings under ``category="silent_fallback"``, but
        its report is only ``log_summary()``-ed and then dropped, so a run's own
        fallback findings vanish the moment it starts.
        """
        if report is None:
            return
        to_dict = getattr(report, "to_dict", None)
        payload = to_dict() if callable(to_dict) else {"report": repr(report)}
        results = payload.get("results") if isinstance(payload, dict) else None
        self.health_report = {
            "passed": bool(getattr(report, "passed", True)),
            "n_errors": len(getattr(report, "errors", []) or []),
            "n_warnings": len(getattr(report, "warnings", []) or []),
            "results": results if results is not None else payload,
        }

    # ---- reporting ----------------------------------------------------

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {"error": 0, "warning": 0, "info": 0}
        for sub in self.substitutions:
            out[sub.severity] = out.get(sub.severity, 0) + 1
        return out

    def to_dict(
        self, *, run_id: str | None = None, config_version: str | None = None
    ) -> dict[str, Any]:
        """The ``_ledger`` block written into ``resolved_config.json``.

        Args:
            run_id: The run this ledger belongs to, when known.
            config_version: The audited config's own declared schema tier
                (:data:`~mriforge.config.schemas.base.CANONICAL_CONFIG_VERSION`
                for every loadable arm). Stamped so the block is not the only
                thing in the artifact wearing the word "version".

        **Three unrelated quantities in this artifact are spelled "version",
        and the block used to name only one of them.** ``schema_version`` is the
        version of THIS LEDGER'S FILE FORMAT -- it counts changes to the keys
        below, and has nothing to do with the config. ``config_version`` is the
        audited config's schema tier. ``metadata.version`` (reachable here only
        inside a ``raw_dict_unvalidated`` substitution payload) is a free-form
        AUTHOR field that 478 of 647 ``inprogress/`` arms had filled with a
        retired config tier, because the framework default hardcoded ``"6.0"``.

        Read side by side with no labels, ``schema_version: 2`` beside a
        ``metadata.version`` of ``6.0`` reads as a contradiction. It is not one:
        they measure different things. ``schema_version_of`` says so in the
        artifact rather than only in this docstring, because the audit's
        ``--json`` report keeps ``_ledger`` and discards the resolved config
        around it -- so in that payload these were the only version facts, with
        nothing to distinguish them.
        """
        # schema_version 2: `defaults` is new. `defaults_injected` keeps its name
        # and meaning, and is now len(defaults) rather than a parallel counter --
        # a consumer reading only the count is unaffected.
        #
        # `schema_version_of` / `config_version` are additive: `schema_version`
        # keeps its value AND its meaning, so every existing reader is
        # unaffected and the number is not bent toward the config's tier.
        return {
            "schema_version": 2,
            "schema_version_of": "execution_ledger",
            "config_version": config_version,
            "run_id": run_id,
            "source": self.source,
            "write_status": "ok",
            "counts": self.counts(),
            "defaults_injected": self.defaults_injected,
            "defaults": list(self.defaults_paths),
            "notes": list(self.notes),
            "substitutions": [s.to_dict() for s in self.substitutions],
            "health_report": self.health_report,
        }

    @staticmethod
    def strict() -> bool:
        return os.environ.get(STRICT_ENV, "").strip().lower() in {"1", "true", "yes"}


# ----------------------------------------------------------------------
# Construction-seam helper
# ----------------------------------------------------------------------


def record_default_coincidence(
    *,
    path: str,
    declared: Any,
    library_default: Any,
    site: str,
    stage: str,
    consumer: str,
) -> bool:
    """Record a declared value that happens to equal the receiver's own default.

    This is the invisible case, and the reason ``Substitution.library_default``
    exists. When the two coincide, the resolved config shows NO delta, the logged
    value looks correct, and a reader cannot tell whether the knob was honoured or
    dropped. `eta_min: 1e-6` was exactly this: it equalled the torch default, so
    the oscillating LR in issue #533 looked right in every artifact.

    Returns whether a record was made, so a caller can log it too.
    """
    if declared is None or isinstance(declared, _NoValue):
        return False
    if declared != library_default:
        return False
    ledger = ExecutionLedger.current()
    if ledger is None:
        return False
    ledger.record(
        class_id=SubstitutionClass.DEFAULT_COINCIDENCE,
        site=site,
        stage=stage,
        path=path,
        requested=declared,
        resolved=declared,
        library_default=library_default,
        reason=(
            f"{path} was declared as {declared!r}, which is also {consumer}'s own "
            "default. The value in force is correct, but this artifact cannot "
            "distinguish 'honoured' from 'dropped and defaulted' — so a drop here "
            "would be invisible."
        ),
        severity="info",
        consumer=consumer,
    )
    return True


def unconsumed_keys(
    declared: dict[str, Any],
    accepted: set[str] | frozenset[str],
    *,
    site: str,
    stage: str,
    consumer: str,
    path_prefix: str = "",
    severity: str = "warning",
    class_id: SubstitutionClass = SubstitutionClass.DROPPED_UNCONSUMED_KWARG,
) -> list[str]:
    """Record every declared key the consumer cannot accept, and return them.

    The shared shape behind ``scheduler_resolution.resolve_scheduler_spec`` and
    ``optimization_builder._unconsumed_optimizer_kwargs``, which each hand-rolled
    it. Recording is separated from the decision to raise: some seams must raise
    (an unroutable scheduler knob is a config error) and some must not (the model
    factory's signature filter is a supported flexibility path), but *both* must
    be visible.

    ``class_id`` defaults to the drop class because that is what every original
    caller meant. It is overridable for the one case that is **not** a drop: a
    consumer whose ``__init__`` declares ``**kwargs`` accepts the key and then
    typically never reads it, so the honest record is
    :attr:`SubstitutionClass.EXTRA_ALLOW_UNTYPED` — "accepted, consumer
    unverified" — rather than silence (issue #878).
    """
    unknown = sorted(k for k in declared if k not in accepted)
    if not unknown:
        return []
    ledger = ExecutionLedger.current()
    if ledger is None:
        return unknown
    for key in unknown:
        ledger.record(
            class_id=class_id,
            site=site,
            stage=stage,
            path=f"{path_prefix}{key}" if path_prefix else key,
            requested=declared[key],
            # A dropped key resolves to nothing; a key swallowed by ``**kwargs``
            # keeps its value but has no verified reader, so it is NOT NO_VALUE.
            resolved=(
                declared[key] if class_id is SubstitutionClass.EXTRA_ALLOW_UNTYPED else NO_VALUE
            ),
            reason=(
                (
                    f"{consumer} accepts {key!r} only via **kwargs, so nothing "
                    f"proves it is read; its named parameters are "
                    f"{sorted(accepted)}."
                )
                if class_id is SubstitutionClass.EXTRA_ALLOW_UNTYPED
                else (f"{consumer} does not accept {key!r}; it accepts {sorted(accepted)}.")
            ),
            severity=severity,
            consumer=consumer,
        )
    return unknown


# ----------------------------------------------------------------------
# Declared-vs-resolved tree diff
# ----------------------------------------------------------------------


def _model_fields(obj: Any) -> dict[str, Any] | None:
    """Pydantic-v2 field map for a model instance, or None if not a model."""
    fields_map = getattr(type(obj), "model_fields", None)
    if isinstance(fields_map, dict):
        return fields_map
    return None


def _normalise(value: Any) -> Any:
    """Reduce comparison noise without hiding real changes.

    Tuples and lists are the same YAML shape, and a ``(str, Enum)`` member
    compares equal to its value already. Anything unserialisable falls back to
    ``repr`` so the comparison is at least stable.
    """
    if isinstance(value, tuple):
        return [_normalise(v) for v in value]
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in value.items()}
    if isinstance(value, Enum):
        return value.value
    return value


def _declared_value_was_rewritten(declared: Any, current: Any) -> bool:
    """Did finalisation change anything the config actually DECLARED?

    ``_normalise`` above canonicalises containers and enums but has no case for
    a pydantic model, and the caller's model-recursion branch only fires when the
    field itself is a model — a ``list[LossComponentConfig]`` is a *list*, so it
    fell through to raw equality and compared a list of dicts against a list of
    model objects. Those are never equal, so **every arm declaring a loss list
    was reported as "the declared value was rewritten during finalisation"** at
    severity ``warning``, describing a pure dict→model coercion in which no value
    moved. That is not cosmetic: ``audit`` is ``--strict`` by default and
    warnings exit 2 (CLAUDE.md #4), so two of ``experiment_11_attention_none``'s
    eight gating warnings were this false positive.

    Compares **only the keys the declaration mentions**, which is the question
    the record claims to answer. A full ``model_dump`` would reintroduce the
    inverse false positive: an arm declaring ``{name, weight}`` would be diffed
    against four dumped fields and mismatch on the two defaults it never wrote.

    Real rewrites still fire — a validator that clamps a declared ``weight``, a
    legacy bridge that renames a declared key, an enum coerced to a different
    member all change a declared key's value and are reported exactly as before.
    """
    declared_fields = _model_fields(current)
    if declared_fields is not None and isinstance(declared, dict):
        for key, declared_sub in declared.items():
            resolved_sub = getattr(current, key, NO_VALUE)
            if isinstance(resolved_sub, _NoValue):
                # Declared but not a field of the resolved model: that is an
                # extra/dropped-key disposition, classified by the caller's own
                # walk, not a value rewrite.
                continue
            if _declared_value_was_rewritten(declared_sub, resolved_sub):
                return True
        return False
    if isinstance(declared, list | tuple) and isinstance(current, list | tuple):
        if len(declared) != len(current):
            return True
        return any(
            _declared_value_was_rewritten(d, c) for d, c in zip(declared, current, strict=True)
        )
    return _normalise(declared) != _normalise(current)


def diff_declared_vs_resolved(
    raw: Any,
    instance: Any,
    *,
    ledger: ExecutionLedger,
    stage: str = "config_finalize",
    path_prefix: str = "",
    site: str = "mriforge.config.settings._finalize_from_dict",
    _depth: int = 0,
) -> None:
    """Record every divergence between the declared dict and the resolved model.

    Walks the **pydantic field tree of** ``type(instance)``, not the raw dict's
    own shape, so each sub-block is classified by its own ``model_config``. That
    matters because strictness is per-block here: ``StrictSchema`` forbids
    extras, ``CompatSchema`` ignores them (silently dropping the key -- the
    issue #550 mechanism), and ``StrategySchema`` allows them (so 14 paradigm
    blocks arrive as raw dicts whose own bounds never run).

    Four dispositions per level:

    ``EXTRA_IGNORE_DROPPED``
        declared, not a field, not captured -> the value is simply gone.
    ``EXTRA_ALLOW_UNTYPED``
        declared, not a field, captured in ``model_extra`` -> present but never
        validated.
    ``RAW_DICT_UNVALIDATED``
        declared as a mapping onto a field that is not a model (e.g. a
        ``dict[str, Any]`` like ``model_kwargs``) -> sub-keys bypass pydantic
        entirely and are at the mercy of a downstream signature filter.
    ``VALUE_CHANGED_ON_FINALIZE``
        declared and resolved differ -> something rewrote it, e.g. the
        coil-processing legacy bridges.

    Injected defaults are collected as **paths only** (``ledger.defaults_paths``),
    never as ``Substitution`` records: a typical arm defaults ~610 knobs against
    14 substitutions, so recording them alongside would bury everything that
    matters. Note the paths are only comparable between configs written at the
    same schema depth -- see ``ExecutionLedger.defaults_paths``.
    """
    if _depth > 12:  # pathological nesting guard; real configs are ~5 deep
        return
    fields_map = _model_fields(instance)
    if fields_map is None or not isinstance(raw, dict):
        return

    resolved_keys = set(fields_map)
    # Pydantic aliases: a field may be declared in YAML under its alias.
    #
    # BOTH alias attributes, not just `alias`. `validation_alias` is the
    # input-only spelling, and reading only `alias` misses it entirely -- which
    # is not a corner case here: `checkpoint.checkpoint_dir` carries
    # `validation_alias="save_dir"` with `alias=None`, and it is the ONLY
    # aliased field in the whole schema tree, so missing this attribute missed
    # 100% of the alias surface. 293 corpus arms declare `save_dir`; every one
    # was recorded as EXTRA_IGNORE_DROPPED at severity "error", claiming "the
    # run never sees it" about a value that lands in `checkpoint_dir` intact.
    # That is precisely the cry-wolf failure the `__folded_input_keys__` block
    # below exists to prevent, one attribute name away.
    #
    # `validation_alias` may also be an AliasChoices/AliasPath rather than a
    # str; take every string it offers rather than str()-ing the container into
    # a key that matches nothing.
    for info in fields_map.values():
        for attr in ("alias", "validation_alias"):
            value = getattr(info, attr, None)
            if isinstance(value, str):
                resolved_keys.add(value)
            elif value is not None:
                resolved_keys.update(c for c in getattr(value, "choices", ()) if isinstance(c, str))
    # Staged renames are input-only aliases too. A `fold` record's legacy
    # spelling is present in the raw YAML and absent from model_fields, which
    # without this reads as EXTRA_IGNORE_DROPPED at severity "error" -- 35 of
    # them per arm on every config that has not migrated yet, all claiming "the
    # run never sees it" about a value the run does see. A ledger that cries
    # wolf 35 times stops being read, which costs more than the records are
    # worth. The class publishes the set so `core` need not import `config`.
    #
    # Kept OUT of `resolved_keys` so the defaults count below stays honest: a
    # folded legacy name is not a field, so counting it as one silently inflates
    # `defaults_injected` by up to 35 on every arm.
    accepted_keys = resolved_keys | set(getattr(type(instance), "__folded_input_keys__", ()))
    raw_keys = set(raw)
    extra = getattr(instance, "model_extra", None) or {}

    for key in sorted(raw_keys - accepted_keys):
        path = f"{path_prefix}{key}"
        if key in extra:
            ledger.record(
                class_id=SubstitutionClass.EXTRA_ALLOW_UNTYPED,
                site=site,
                stage=stage,
                path=path,
                requested=raw[key],
                resolved=extra[key],
                reason=(
                    f"{type(instance).__name__} is extra=allow, so {key!r} is "
                    "carried as an untyped extra: its bounds, forbid and frozen "
                    "settings never run."
                ),
                severity="warning",
            )
        else:
            ledger.record(
                class_id=SubstitutionClass.EXTRA_IGNORE_DROPPED,
                site=site,
                stage=stage,
                path=path,
                requested=raw[key],
                resolved=NO_VALUE,
                reason=(
                    f"{type(instance).__name__} does not declare {key!r} and "
                    "ignores extras, so the declared value was discarded. This "
                    "is the issue #550 mechanism: the YAML still shows it, the "
                    "run never sees it."
                ),
                severity="error",
            )

    ledger.defaults_paths.extend(f"{path_prefix}{key}" for key in sorted(resolved_keys - raw_keys))

    # A FOLDED key is accepted above (it was moved, not dropped) but is not in
    # `resolved_keys`, so the loop below never reaches it and nothing inside it
    # is ever audited. `acceleration:` is a whole top-level block folded to
    # `undersampling:`, which is how a British-spelling `min_centre_fraction`
    # was discarded by AccelerationConfigSchema's `extra="ignore"` while the
    # ledger stayed empty -- the exact silent drop the ledger exists to catch,
    # and four independent tests were written to notice it.
    #
    # Recurse into the CANONICAL resolved sub-model, but report the LEGACY dotted
    # prefix: the reader is looking for the key they wrote in the YAML.
    folded_paths = getattr(type(instance), "__folded_input_paths__", {})
    for key in sorted(raw_keys & set(folded_paths)):
        declared = raw[key]
        if not isinstance(declared, dict):
            continue  # a folded scalar has no sub-keys to audit
        target: Any = instance
        for part in folded_paths[key]:
            target = getattr(target, part, NO_VALUE)
            if isinstance(target, _NoValue):
                break
        if isinstance(target, _NoValue) or _model_fields(target) is None:
            continue
        diff_declared_vs_resolved(
            declared,
            target,
            ledger=ledger,
            stage=stage,
            path_prefix=f"{path_prefix}{key}.",
            site=site,
            _depth=_depth + 1,
        )

    for key in sorted(raw_keys & resolved_keys):
        declared = raw[key]
        current = getattr(instance, key, NO_VALUE)
        if isinstance(current, _NoValue):
            continue
        path = f"{path_prefix}{key}"
        if _model_fields(current) is not None:
            if isinstance(declared, dict):
                diff_declared_vs_resolved(
                    declared,
                    current,
                    ledger=ledger,
                    stage=stage,
                    path_prefix=f"{path}.",
                    site=site,
                    _depth=_depth + 1,
                )
            continue
        if isinstance(declared, dict) and current is not None:
            ledger.record(
                class_id=SubstitutionClass.RAW_DICT_UNVALIDATED,
                site=site,
                stage=stage,
                path=path,
                requested=declared,
                resolved=current,
                reason=(
                    f"{type(instance).__name__}.{key} is not a model, so its "
                    "sub-keys are never validated by pydantic and survive only "
                    "as far as the next signature filter."
                ),
                severity="info",
            )
            continue
        if _declared_value_was_rewritten(declared, current):
            ledger.record(
                class_id=SubstitutionClass.VALUE_CHANGED_ON_FINALIZE,
                site=site,
                stage=stage,
                path=path,
                requested=declared,
                resolved=current,
                reason=(
                    "the declared value was rewritten during finalisation "
                    "(coercion, a legacy bridge, or a model validator)."
                ),
                severity="warning",
            )
