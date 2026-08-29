"""Tests for ``config/schemas/defaults_provider.py``.

Created 2026-08-03 — this module had no test partner at all, which is how its
``config_version`` default sat at the *previous* schema version. A defaults
provider is the one place a wrong version is invisible: it only fires for configs
that omit the key, so nothing errors and nothing looks stale.
"""

from __future__ import annotations

from mriforge.config.schemas.base import (
    ACCEPTED_CONFIG_VERSIONS,
    CANONICAL_CONFIG_VERSION,
    LEGACY_CONFIG_VERSIONS,
)
from mriforge.config.schemas.defaults_provider import ConfigDefaultsProvider


def _base_defaults() -> dict:
    return ConfigDefaultsProvider()._get_base_defaults()


def test_the_default_version_is_canonical() -> None:
    """Not merely *accepted*. A defaults provider that emits a legacy version
    manufactures new legacy configs while the corpus is being drained of them —
    the countdown would never reach zero."""
    assert _base_defaults()["config_version"] == CANONICAL_CONFIG_VERSION


def test_the_default_version_is_not_legacy() -> None:
    assert _base_defaults()["config_version"] not in LEGACY_CONFIG_VERSIONS


def test_the_default_version_loads() -> None:
    assert _base_defaults()["config_version"] in ACCEPTED_CONFIG_VERSIONS


def test_the_default_tracks_the_constant_rather_than_a_literal() -> None:
    """The seam, not the value: patching the SSOT must move the default.

    Without this the test above passes on a hard-coded ``"1.0"`` that would go
    stale the next time the canonical version changes — which is exactly how
    this module drifted in the first place.
    """
    import mriforge.config.schemas.defaults_provider as provider

    source = provider.__dict__["CANONICAL_CONFIG_VERSION"]
    assert source == CANONICAL_CONFIG_VERSION
    assert (
        "CANONICAL_CONFIG_VERSION" in provider.__dict__
    ), "the provider must import the constant, not restate the version"


# --------------------------------------------------------------------------
# `metadata.version` is the AUTHOR's revision, not a schema tier
# --------------------------------------------------------------------------
#
# The same dict literal that correctly sources `config_version` from the constant
# above hardcoded `metadata: {"version": "6.0"}` two lines below it -- a *retired,
# unloadable* config tier (6.0 left ACCEPTED_CONFIG_VERSIONS in PR #891, so
# declaring it now raises) standing as the default for a free-form author field.
# In the audit artifact a `metadata.version: "6.0"` sits beside
# `_ledger.schema_version: 2` and reads as a contradiction between two numbers
# that version unrelated things.
#
# SCOPE, measured rather than assumed (2026-08-18): this provider has NO
# production call site -- its only importers are this file and
# `tests/unit/config/test_defaults_provider.py`, plus a prose mention in
# `config_health_checker.py`. So the literal reached no arm, and it is NOT the
# mechanism behind the corpus's 6.x. The 467 of 647 `inprogress/` arms carrying
# `metadata.version: '6.0'` declare it in their own YAML, from a template. These
# tests therefore guard a latent surface: they stop an unwired provider from
# re-teaching the retired spelling the day it is wired.
#
# Two reasons it could sit wrong indefinitely, both still true: a defaults
# provider only fires for configs that OMIT the key, so nothing errors and
# nothing looks stale; and `metadata` has no schema at all (`settings.py`:
# `dict[str, Any] | None`), so no validator could have rejected the value.


def _metadata_defaults() -> dict:
    return _base_defaults()["metadata"]


def test_the_declared_metadata_version_is_not_stated() -> None:
    """`None` == "the author did not say", matching every free-form neighbour
    (`description`, `author`, `created_at`, `timestamp`) in the same block."""
    assert _metadata_defaults()["version"] is None


def test_the_declared_metadata_version_is_never_a_config_tier() -> None:
    """The regression that matters, stated independently of the value above.

    Whatever this block ever comes to hold, it must not be a *schema* version --
    that conflation is the whole defect. Asserting `not in ACCEPTED | LEGACY`
    would still pass on a tier that is merely unrecognised, so this checks the
    shape: a config tier is what `CANONICAL_CONFIG_VERSION` looks like.
    """
    version = _metadata_defaults()["version"]
    tiers = set(ACCEPTED_CONFIG_VERSIONS) | set(LEGACY_CONFIG_VERSIONS)
    assert version not in tiers, (
        f"the defaults provider declares the config tier {version!r} as author "
        "metadata; `metadata.version` is the author's experiment revision"
    )
    assert version != CANONICAL_CONFIG_VERSION, (
        "even the *current* tier is wrong here -- it would make every arm claim "
        "an author revision it never wrote, and the next bump would silently "
        "rewrite it"
    )


def test_the_retired_tier_is_gone_from_the_defaults_entirely() -> None:
    """Belt-and-braces over the whole block, not just the one key.

    `6.0`/`6.1` are unloadable: a legacy declaration now *raises*. Any default
    that reintroduces one manufactures a config the loader refuses, so the
    literal must not appear anywhere in the base defaults' metadata.
    """
    assert "6.0" not in set(map(str, _metadata_defaults().values()))
    assert "6.1" not in set(map(str, _metadata_defaults().values()))


def test_the_free_form_metadata_fields_agree_with_each_other() -> None:
    """Pitfall #17, in miniature: the divergence was one field out of five.

    `version` was the only free-form key in this block carrying a value while its
    four neighbours were `None`. That asymmetry is what would make boilerplate
    read as an author's declaration once this provider is wired.
    """
    metadata = _metadata_defaults()
    free_form = {"description", "author", "created_at", "timestamp", "version"}
    stated = {key for key in free_form if metadata[key] is not None}
    assert stated == set(), (
        f"{sorted(stated)} claim a value the author never wrote, while the rest "
        "of the block correctly says nothing"
    )
