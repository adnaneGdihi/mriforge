"""Every schema version the CLI *advertises* must be one the loader accepts.

``mriforge audit --help`` told the user to write ``config_version: '6.0'``
against a loader whose ``ACCEPTED_CONFIG_VERSIONS`` is ``{'1.0'}``. Following
that help produced::

    Config version 6.0 not supported in <path>.
    Accepted values: ['1.0']. Please update your configuration.

-- ``audit`` exiting 2 on the first command a new user runs. Nothing saw it: the
string is help text, so no import resolves it, no schema validates it, and
``ruff`` has no opinion about English.

Why the version is a literal in ``app.py`` and the pin lives here
----------------------------------------------------------------
Interpolating :data:`~mriforge.config.schemas.base.CANONICAL_CONFIG_VERSION`
into the help string would be the obvious single-owner fix and is the wrong one:
importing that module pulls **torch** (1366 modules, measured), and parser
construction is deliberately torch-free -- that is the budget
``tests/unit/cli/test_startup_budget.py`` guards and PR #1130 bought. So the
literal stays in ``app.py``, and *this executed test* is the sync point. A test
may import torch; ``--help`` may not.

What this scans, and the blindness it was built around
------------------------------------------------------
The subcommand one-liners -- including the defective one -- do **not** live on
the child parsers. ``argparse`` keeps them on the ``_SubParsersAction``'s
``_choices_actions``. A walk that recurses through ``action.choices`` and reads
each child's ``.description``/``.epilog`` and per-argument ``.help`` therefore
covers 197 strings and **misses all 32 subcommand one-liners**, which is the
population the real defect belonged to. That walk was written first, reported a
clean sweep, and would have shipped green (non-negotiable 15). Counting the
choice help is what takes the surface to 229.

Why the pattern matches the *claim* and not the *number*
--------------------------------------------------------
A bare ``\\bv?\\d+\\.\\d+\\b`` over all 229 strings raises three false alarms --
``hpo --cost-weight`` (``default: 0.0``) and ``meta-evaluate``'s ``--nr-battery``
/ ``--tiers``, which cite document sections ``8.1``-``8.7``. Neither is a schema
claim. Requiring the version token to sit beside ``schema`` or ``config_version``
scores exactly one hit -- the real one -- and zero false positives.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterator

import pytest

from mriforge.config.schemas.base import ACCEPTED_CONFIG_VERSIONS

#: A version token adjacent to the word that makes it a claim about the config
#: schema. Both orders occur in the wild: "the v6.0 schema" and "schema v6.0".
_SCHEMA_VERSION_CLAIM = re.compile(
    r"(?:schema|config[_ ]version)\D{0,20}?v?(\d+\.\d+)"
    r"|v?(\d+\.\d+)\D{0,20}?(?:schema|config[_ ]version)",
    re.IGNORECASE,
)

#: Below this, the walk has lost a surface. 229 measured 2026-08-28; the guard is
#: a floor rather than an equality so adding a subcommand does not fail the suite,
#: while *deleting* the choice-help branch (the blindness this module documents)
#: drops the count by 32 and trips it.
_MIN_ADVERTISED_STRINGS = 200


def _advertised_strings(
    parser: argparse.ArgumentParser, path: str = "mriforge"
) -> Iterator[tuple[str, str, str]]:
    """Every string ``--help`` can put in front of a user, as (where, what, text)."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            # The subcommand one-liners live here, NOT on the child parser.
            # Dropping this loop is what made the first version of this scan
            # blind to the only real violation in the tree.
            for choice_action in action._choices_actions or ():
                if choice_action.help:
                    yield f"{path} {choice_action.dest}", "<subcommand help>", choice_action.help
            for name, subparser in action.choices.items():
                yield from _advertised_strings(subparser, f"{path} {name}")
        elif action.help:
            yield path, "|".join(action.option_strings) or action.dest, action.help
    for label, text in (("<description>", parser.description), ("<epilog>", parser.epilog)):
        if text:
            yield path, label, text


def _build_parser() -> argparse.ArgumentParser:
    from mriforge.cli.app import build_parser

    return build_parser()


def test_every_schema_version_the_cli_advertises_is_one_the_loader_accepts() -> None:
    """Following ``--help`` must not produce a config the loader rejects."""
    offenders = []
    for where, what, text in _advertised_strings(_build_parser()):
        claimed = {group for match in _SCHEMA_VERSION_CLAIM.finditer(text) for group in match.groups() if group}
        for version in sorted(claimed - set(ACCEPTED_CONFIG_VERSIONS)):
            offenders.append(f"{where} {what}: advertises schema {version!r} -- {text.strip()[:110]!r}")

    assert not offenders, (
        "CLI help advertises schema version(s) the loader rejects.\n"
        f"ACCEPTED_CONFIG_VERSIONS = {sorted(ACCEPTED_CONFIG_VERSIONS)}\n"
        + "\n".join(f"  {line}" for line in offenders)
    )


def test_the_scan_reaches_the_subcommand_one_liners() -> None:
    """The surface that held the defect must be in the scan, not merely nearby.

    Without this the module above is satisfiable by a walk that visits zero
    subcommand help strings: no offender, green, blind.
    """
    rows = list(_advertised_strings(_build_parser()))
    assert len(rows) >= _MIN_ADVERTISED_STRINGS, (
        f"only {len(rows)} advertised strings reached -- a surface was dropped from the walk"
    )

    subcommand_help = [r for r in rows if r[1] == "<subcommand help>"]
    assert len(subcommand_help) >= 20, (
        f"only {len(subcommand_help)} subcommand one-liner(s) scanned; argparse keeps "
        "them on _SubParsersAction._choices_actions and a choices-only recursion misses all of them"
    )
    assert any(where.endswith(" audit") for where, _, _ in subcommand_help), (
        "the `audit` one-liner -- the string this module was written for -- is not in the scan"
    )


def test_the_claim_pattern_reads_a_claim_and_not_any_decimal() -> None:
    """Precision pin: without it, tightening the pattern to nothing still passes.

    The three innocents are real strings in this CLI today (``hpo
    --cost-weight``, ``meta-evaluate --nr-battery``/``--tiers``). Matching them
    would make this module cry wolf on every release and get muted.
    """
    def claimed(text: str) -> set[str]:
        return {g for m in _SCHEMA_VERSION_CLAIM.finditer(text) for g in m.groups() if g}

    assert claimed("Audit an experiment YAML against the v6.0 schema") == {"6.0"}
    assert claimed("write config_version: '5.0'") == {"5.0"}
    assert claimed("schema version 6.1 is required") == {"6.1"}

    assert claimed("Cost-weight for multi-objective HPO (default: 0.0).") == set()
    assert claimed("drives the sweep with the audited digital twin (§8.1)") == set()


def test_the_accepted_set_is_not_empty() -> None:
    """A vacuity guard: an empty accepted set would pass the main test trivially."""
    assert ACCEPTED_CONFIG_VERSIONS, "ACCEPTED_CONFIG_VERSIONS is empty"


@pytest.mark.parametrize("version", sorted(ACCEPTED_CONFIG_VERSIONS))
def test_an_accepted_version_is_not_reported_as_an_offender(version: str) -> None:
    """The check must pass what the loader accepts, or the fix is unshippable."""
    text = f"Audit an experiment YAML against the v{version} schema (Tier 0)."
    claimed = {g for m in _SCHEMA_VERSION_CLAIM.finditer(text) for g in m.groups() if g}
    assert claimed == {version}
    assert not (claimed - set(ACCEPTED_CONFIG_VERSIONS))
