"""Audit-ladder regression corpus driver.

PART B — Task III.3: Audit-ladder regression corpus.

Each YAML under ``corpus/failing/`` is named ``<check>__<mode>.yaml`` and must be
**rejected by the audit ladder**, with the rejection attributable to ``<check>``.
Each YAML under ``corpus/passing/`` must load and produce no error-severity result.

Where a failing fixture is caught is declared *in the fixture*, so this driver
holds no hand-maintained table of which entry belongs to which tier:

===========================  ==================================================
header marker                contract
===========================  ==================================================
(none)                       ``TrainingSettings.from_yaml`` must succeed, and
                             ``ConfigHealthChecker`` must emit a non-passing
                             result named ``<check>``.
``# expect: tier0 <leaf>``   ``from_yaml`` must RAISE, and the failure must be
                             located at ``<leaf>``. Tier 0 catching a violation
                             is a *stronger* result than the Tier-1 check
                             firing, not a weaker one — the config never
                             reaches the health checker at all.
``# expect: registry``       The dict-level ``ValidatorRegistry`` rule
                             ``<check>`` must fire on the raw document. Used
                             where no ``ConfigHealthChecker`` check carries the
                             name, so this fixture is excluded from the
                             health-checker parametrisation by *collection*
                             rather than skipped at runtime.
``# known-dead: <reason>``   ``xfail(strict=True)``: the named check is known
                             to be structurally incapable of firing. Repairing
                             the check turns this RED, which is the point — the
                             marker cannot rot into a silent pass.
===========================  ==================================================

Why the driver looks like this (issue #922). It previously built a
``SimpleNamespace`` tree from the raw YAML and wrapped ``run_all_checks`` in::

    except Exception:
        # A crash in the checker itself counts as the check firing
        return

That inference does not hold: the crash came from the *stub* lacking
``data.source``, inside ``check_train_val_split_leakage`` — a check unrelated to
whatever the fixture was named after, and one a perfectly valid config would
crash identically. Every one of the 30 fixtures also still declared
``config_version: '6.0'``, so all 30 aborted before any check ran and all 30
were absorbed: 25 reported ``skipped`` and 5 reported ``passed``. The corpus
verified nothing while looking green.

So: no ``SimpleNamespace``, and no ``except`` that converts an unexpected
exception into evidence. Anything unanticipated propagates.

A ``# expect: tier0`` marker must name the leaf the *fixture's own check* is
about. Three fixtures used to be rejected at Tier 0 for reasons that had nothing
to do with their check (a retired ``objectives:`` block, ``optimization.epochs``,
``validation.val_frequency``); annotating those as tier-0 would have turned them
green while testing nothing. They were repaired instead.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import yaml
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(__file__).parent / "corpus"
PASSING_DIR = CORPUS_DIR / "passing"
FAILING_DIR = CORPUS_DIR / "failing"


# ---------------------------------------------------------------------------
# Fixture-declared expectations
# ---------------------------------------------------------------------------

_TIER0_RE = re.compile(r"^#\s*expect:\s*tier0\s+(\S+)", re.MULTILINE)
_REGISTRY_RE = re.compile(r"^#\s*expect:\s*registry\b", re.MULTILINE)
_KNOWN_DEAD_RE = re.compile(r"^#\s*known-dead:\s*(.+)$", re.MULTILINE)


class Expectation(NamedTuple):
    """Where in the audit ladder a failing fixture declares it is caught."""

    kind: str  # "health" | "tier0" | "registry"
    leaf: str  # tier0 only: the location the failure must be reported at
    dead_reason: str  # non-empty when the named check is known-dead


def _expectation(path: Path) -> Expectation:
    text = path.read_text()
    dead = _KNOWN_DEAD_RE.search(text)
    tier0 = _TIER0_RE.search(text)
    kind = "tier0" if tier0 else ("registry" if _REGISTRY_RE.search(text) else "health")
    return Expectation(
        kind=kind,
        leaf=tier0.group(1) if tier0 else "",
        dead_reason=dead.group(1).strip() if dead else "",
    )


def _load_yaml_raw(path: Path) -> dict[str, Any]:
    """Load YAML as a raw dict (no schema validation)."""
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _check_name_from_path(path: Path) -> str:
    """``<check_name>__<mode>.yaml`` -> ``check_name``."""
    return path.stem.split("__")[0]


def _failure_locations(exc: Exception) -> set[str]:
    """Dotted locations a load failure is reported at.

    Pydantic carries structured ``loc`` tuples, so a tier-0 expectation can be
    matched exactly rather than by substring — ``data`` must be *the* failing
    field, not merely a word occurring somewhere in the message.
    """
    if isinstance(exc, ValidationError):
        return {".".join(str(p) for p in e["loc"]) for e in exc.errors()}
    return set()


def _param(path: Path) -> Any:
    """Attach ``xfail(strict=True)`` to fixtures whose check is known-dead."""
    reason = _expectation(path).dead_reason
    if not reason:
        return path
    return pytest.param(path, marks=pytest.mark.xfail(strict=True, reason=reason))


# ---------------------------------------------------------------------------
# Corpus partition — derived from the fixtures, never hand-listed
# ---------------------------------------------------------------------------

_passing_yamls = sorted(PASSING_DIR.glob("*.yaml"))
_failing_yamls = sorted(FAILING_DIR.glob("*.yaml"))

_health_yamls = [p for p in _failing_yamls if _expectation(p).kind == "health"]
_tier0_yamls = [p for p in _failing_yamls if _expectation(p).kind == "tier0"]
_registry_yamls = [p for p in _failing_yamls if _expectation(p).kind == "registry"]


# ---------------------------------------------------------------------------
# PART B-1: Passing corpus
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("yaml_path", _passing_yamls, ids=[p.stem for p in _passing_yamls])
def test_passing_corpus_loads_without_schema_error(yaml_path: Path) -> None:
    """Each passing corpus YAML must load via ``TrainingSettings.from_yaml``."""
    from spectramr.config.settings import TrainingSettings

    settings = TrainingSettings.from_yaml(yaml_path)
    assert settings is not None


@pytest.mark.unit
@pytest.mark.parametrize("yaml_path", _passing_yamls, ids=[p.stem for p in _passing_yamls])
def test_passing_corpus_health_checker_no_errors(yaml_path: Path) -> None:
    """Each passing corpus YAML must produce zero error-severity failures."""
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    settings = TrainingSettings.from_yaml(yaml_path)
    report = ConfigHealthChecker().run_all_checks(settings)
    assert report.errors == [], f"{yaml_path.name} produced unexpected errors:\n" + "\n".join(
        f"  [{r.check_name}] {r.message}" for r in report.errors
    )


# ---------------------------------------------------------------------------
# PART B-2: Failing corpus — Tier 1 (ConfigHealthChecker)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "yaml_path", [_param(p) for p in _health_yamls], ids=[p.stem for p in _health_yamls]
)
def test_failing_corpus_trips_named_check(yaml_path: Path) -> None:
    """The fixture must load, and its named health check must fire.

    Loading is part of the contract: a fixture that cannot be parsed proves
    nothing about the Tier-1 check it is named after. If a fixture is genuinely
    meant to die at Tier 0, it declares ``# expect: tier0 <leaf>`` and is tested
    by :func:`test_failing_corpus_rejected_at_tier0` instead.
    """
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    check_name = _check_name_from_path(yaml_path)
    settings = TrainingSettings.from_yaml(yaml_path)
    results = ConfigHealthChecker().run_all_checks(settings).results

    for_check = [r for r in results if r.check_name == check_name]
    assert any(not r.passed for r in for_check), (
        f"{yaml_path.name} was expected to trip '{check_name}' but it did not.\n"
        f"Results for '{check_name}' ({len(for_check)}):\n"
        + "\n".join(f"  passed={r.passed} sev={r.severity}: {r.message}" for r in for_check)
        + f"\n(check names present: {sorted({r.check_name for r in results})})"
    )


# ---------------------------------------------------------------------------
# PART B-2b: Failing corpus — Tier 0 (schema rejection)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "yaml_path", [_param(p) for p in _tier0_yamls], ids=[p.stem for p in _tier0_yamls]
)
def test_failing_corpus_rejected_at_tier0(yaml_path: Path) -> None:
    """The fixture must be refused by the schema, at the leaf it declares.

    Asserting merely "it raises" would also pass if the field had been *deleted*
    — the same trap a bare ``pytest.raises`` sets for a rename test. The declared
    leaf is what makes this an assertion about the named violation.
    """
    from spectramr.config.settings import TrainingSettings

    expected = _expectation(yaml_path).leaf
    with pytest.raises((ValidationError, ValueError)) as excinfo:
        TrainingSettings.from_yaml(yaml_path)

    exc = excinfo.value
    locations = _failure_locations(exc)
    located = expected in locations if locations else expected in str(exc)
    assert located, (
        f"{yaml_path.name} declares '# expect: tier0 {expected}' but the load "
        f"failed elsewhere.\n  locations: {sorted(locations) or '(unstructured)'}\n"
        f"  message: {exc}"
    )


# ---------------------------------------------------------------------------
# PART B-3: Failing corpus — ValidatorRegistry cross-check
# ---------------------------------------------------------------------------
#
# Rule names come from the registry itself. A hand-written frozenset of names
# lived here and had drifted to 6 of the 32 registered rules, so most fixtures
# with a registry counterpart were never cross-checked.


def _registry_rule_names() -> frozenset[str]:
    from spectramr.config.schemas.validator_registry import get_validator_registry

    return frozenset(rule.name for rule in get_validator_registry().list_validators())


def _registry_backed(paths: list[Path]) -> list[Path]:
    names = _registry_rule_names()
    return [p for p in paths if _check_name_from_path(p) in names]


_registry_checked = _registry_backed(_failing_yamls)


@pytest.mark.unit
@pytest.mark.parametrize(
    "yaml_path",
    [_param(p) for p in _registry_checked],
    ids=[p.stem for p in _registry_checked],
)
def test_failing_corpus_validator_registry_also_fires(yaml_path: Path) -> None:
    """Every fixture naming a registered rule must trip it at the dict level."""
    from spectramr.config.schemas.validator_registry import get_validator_registry

    check_name = _check_name_from_path(yaml_path)
    raw = _load_yaml_raw(yaml_path)
    raw.pop("config_version", None)

    issues = get_validator_registry().validate(raw)
    matching = [(name, msg) for name, msg in issues if name == check_name]
    assert matching, (
        f"{yaml_path.name}: expected registry rule '{check_name}' to fire, but "
        f"got no matching issues. All issues: {issues}"
    )


@pytest.mark.unit
def test_registry_expectations_name_a_registered_rule() -> None:
    """``# expect: registry`` must name a rule the registry actually has.

    Without this, the marker is an escape hatch: it removes a fixture from the
    health-checker parametrisation, and if no registry rule carries the name
    either, the fixture is tested by nothing at all.
    """
    names = _registry_rule_names()
    orphaned = [p.name for p in _registry_yamls if _check_name_from_path(p) not in names]
    assert orphaned == [], (
        "these fixtures declare '# expect: registry' but no ValidatorRegistry "
        f"rule carries their name, so nothing tests them: {orphaned}"
    )


# ---------------------------------------------------------------------------
# PART B-4: Inventory self-check
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_passing_corpus_is_non_empty() -> None:
    """Sanity: the passing corpus must have at least 2 YAML files."""
    assert len(_passing_yamls) >= 2, (
        f"Expected at least 2 passing YAMLs, found {len(_passing_yamls)}"
    )


@pytest.mark.unit
def test_failing_corpus_is_non_empty() -> None:
    """Sanity: the failing corpus must have at least 5 YAML files."""
    assert len(_failing_yamls) >= 5, (
        f"Expected at least 5 failing YAMLs, found {len(_failing_yamls)}"
    )


@pytest.mark.unit
def test_failing_corpus_filenames_follow_convention() -> None:
    """All failing YAMLs must use ``<check_name>__<mode>.yaml``."""
    bad = [p.name for p in _failing_yamls if "__" not in p.stem]
    assert bad == [], "Failing YAML filenames must contain '__' separator: " + str(bad)


@pytest.mark.unit
def test_every_failing_fixture_is_covered_by_exactly_one_tier() -> None:
    """The partition must be total: no fixture may fall through it.

    ``kind`` is derived from the fixture header, so a typo'd marker would
    silently demote an entry to the default tier rather than erroring. This
    asserts the three parametrised lists reconstruct the directory exactly.
    """
    partitioned = sorted(p.name for p in _health_yamls + _tier0_yamls + _registry_yamls)
    assert partitioned == sorted(p.name for p in _failing_yamls)
    overlap = (
        set(_health_yamls) & set(_tier0_yamls)
        | set(_health_yamls) & set(_registry_yamls)
        | set(_tier0_yamls) & set(_registry_yamls)
    )
    assert overlap == set(), f"fixtures claimed by two tiers: {overlap}"


@pytest.mark.unit
def test_no_failing_fixture_declares_a_refused_config_version() -> None:
    """Every fixture must declare the canonical version.

    All 30 declared ``'6.0'`` while the driver absorbed the resulting load
    failure as a pass, so the corpus was uniformly unloadable and nobody could
    see it. The version is a ratchet here, not a migration.
    """
    from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION

    stale = {}
    for p in _failing_yamls + _passing_yamls:
        declared = _load_yaml_raw(p).get("config_version")
        if declared != CANONICAL_CONFIG_VERSION:
            stale[p.name] = declared
    assert stale == {}, (
        f"corpus fixtures must declare config_version '{CANONICAL_CONFIG_VERSION}': {stale}"
    )
