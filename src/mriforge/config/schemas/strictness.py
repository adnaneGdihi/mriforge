"""Shared strictness bases — the single source of truth for the schema policy.

Historically every one of the ~230 config-schema classes hand-declared its own
``model_config``. The policy drifted (15 non-frozen blocks, one with no
``model_config`` at all, one ``model_*`` block missing ``protected_namespaces``)
and ``schemas/__init__.py`` documented an invariant that was not actually true.

These three bases make the policy declarable in one line and impossible to get
subtly wrong. New schemas SHOULD inherit one of them; the package-wide invariant
(frozen everywhere, ``extra='allow'`` only where reviewed,
``protected_namespaces=()`` wherever a ``model_*`` field exists) is enforced for
*all* classes — inherited or inline — by
``tests/unit/config/test_schema_strictness_policy.py``.

Tier guide
----------
``StrictSchema``
    ``extra='forbid'`` — the **default**. An unknown key is a load-time
    ``ValidationError`` rather than a silently-dropped facade knob (CLAUDE.md
    pitfall #9 / #16). Use this unless you have a concrete reason not to.

``CompatSchema``
    ``extra='ignore'`` — reserved for the v6.0<->v6.1 forward-compat blocks
    that must accept *additive* sub-fields from a newer YAML without rejecting an
    older config (e.g. ``model:`` / ``data:`` while ~30 ``inprogress`` configs
    still omit the v6.1 keys). The cost is that a typo in such a block is
    swallowed, so keep these few and prefer tightening to ``StrictSchema`` once
    the corpus has migrated.

``StrategySchema``
    ``extra='allow'`` — reserved for the strategy / adapter / audit blocks that
    *forward* strategy-specific knobs read downstream via ``getattr`` / ``hasattr``.
    This is the facade hole of pitfall #16: an unread/typo'd strategy knob is
    accepted silently. Closing it (a per-``strategy_class`` typed-knob audit) is
    tracked in ``docs/superpowers/specs/2026-06-12-schema-strictness-and-strategy-knob-audit.rst``.
    Every ``extra='allow'`` class is enumerated in the test's ``ALLOW_ALLOWLIST``
    so a new one cannot appear without a conscious review.

All three are ``frozen=True`` (non-negotiable #1 / #4 — config is immutable after
load) and disable the protected ``model_`` namespace (MRI configs legitimately
use ``model_type`` / ``model_kwargs`` / ``model_domain`` field names).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictSchema(BaseModel):
    """Frozen, ``extra='forbid'`` — the default config-schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class CompatSchema(BaseModel):
    """Frozen, ``extra='ignore'`` — forward-compat blocks only (see module docstring)."""

    model_config = ConfigDict(extra="ignore", frozen=True, protected_namespaces=())


class StrategySchema(BaseModel):
    """Frozen, ``extra='allow'`` — strategy/adapter knob-forwarding blocks only."""

    model_config = ConfigDict(extra="allow", frozen=True, protected_namespaces=())


__all__ = ["CompatSchema", "StrategySchema", "StrictSchema"]
