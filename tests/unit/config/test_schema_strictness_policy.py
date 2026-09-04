"""Package-wide enforcement of the config-schema strictness policy.

This is the executable form of the invariant that ``schemas/__init__.py``
*documents* but historically could not guarantee (the policy was hand-declared
on 230 classes and had drifted: 15 non-frozen, one with no ``model_config`` at
all, one ``model_``-field block missing ``protected_namespaces``).

The policy has three tiers (see ``schemas/strictness.py``):

* **StrictSchema**   — ``extra='forbid'`` — the default; a typo is a load-time
  ``ValidationError`` (CLAUDE.md pitfall #9 / #16: no silent-swallow).
* **CompatSchema**   — ``extra='ignore'`` — *only* for the v6.0<->v6.1
  forward-compat blocks that must accept additive keys from a newer YAML.
* **StrategySchema** — ``extra='allow'`` — *only* the strategy / adapter /
  audit blocks that forward strategy-specific knobs read downstream via
  ``getattr``. Every such class is enumerated in ``ALLOW_ALLOWLIST`` so a new
  ``extra='allow'`` block cannot be added without a conscious review (it is the
  facade hole of pitfall #16; closing it is tracked in the Tier-B spec).

Two invariants are enforced for *every* schema class, regardless of whether it
inherits a base or declares ``model_config`` inline:

1. ``frozen=True`` — settings are immutable after load (non-negotiable #1 / #4).
2. ``protected_namespaces == ()`` whenever the class has a ``model_*`` field
   (otherwise Pydantic emits a ``UserWarning`` which pytest promotes to an
   error under the strict ``filterwarnings`` config).
"""

from __future__ import annotations

import importlib
import pkgutil

from pydantic import BaseModel

import spectramr.config.schemas as schemas_pkg


def _discover_schema_classes() -> list[type[BaseModel]]:
    """Force-import every schema module, then collect all BaseModel subclasses
    defined under ``spectramr.config`` (schemas package + the root settings)."""
    for mod in pkgutil.walk_packages(schemas_pkg.__path__, schemas_pkg.__name__ + "."):
        importlib.import_module(mod.name)
    # the root TrainingSettings lives one level up and must also comply
    importlib.import_module("spectramr.config.settings")

    seen: set[type] = set()
    stack: list[type] = [BaseModel]
    while stack:
        parent = stack.pop()
        for sub in parent.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return sorted(
        (c for c in seen if c.__module__.startswith("spectramr.config")),
        key=lambda c: f"{c.__module__}.{c.__qualname__}",
    )


# The ONLY classes permitted to use ``extra='allow'``. Adding to this list is a
# deliberate act that should accompany a knob-audit plan (Tier-B spec). These
# forward strategy/adapter-specific keys read downstream via getattr/hasattr.
ALLOW_ALLOWLIST: frozenset[str] = frozenset(
    {
        # The canonical 'allow' base itself (defines the policy, never loaded directly).
        "spectramr.config.schemas.strictness.StrategySchema",
        "spectramr.config.schemas.adapters.AdapterStepSchema",
        "spectramr.config.schemas.audit.ksd.KSDAuditConfigSchema",
        "spectramr.config.schemas.training.base.DiffusionTrainingConfigSchema",
        # The two paradigm variants that could NOT be tightened when
        # `training.diffusion` became a discriminated union (2026-08-12).
        #
        # The other five variants -- DDPMParams, DDIMParams, ScoreBasedParams,
        # RectifiedFlowParams, ChiSquareParams -- ARE `extra="forbid"`, so their
        # absence from this list is the audit, not an oversight.
        #
        # These two are pinned open by exactly three keys that live arms declare
        # and NOTHING in `src/` reads: `clip_sample`, `clip_sample_range`,
        # `dynamic_thresholding_func`. Every available home is dishonest --
        # a typed field advertises an unread knob (#8), a `raise` rename record
        # makes live `inprogress/` arms unloadable, and `ignore` silently drops
        # them, which is the defect the cleanup exists to remove. Tightening
        # these needs an owner decision on those three keys, not a policy edit.
        "spectramr.config.schemas.training.base.ColdParams",
        "spectramr.config.schemas.training.base.LatentDiffusionParams",
        # Transitional by design: it exists so the 75 arms that declare no
        # paradigm tag keep loading unchanged. Tightening it would make landing
        # the union a breaking change instead of a behaviour-neutral one. It
        # drains as arms declare a tag, and is deleted when the count reaches
        # zero -- the same ratchet shape as `fold` -> `raise`.
        "spectramr.config.schemas.training.base.UnspecifiedParams",
        # Mounted on TrainingSettings.metadata on 2026-09-02. The corpus carries about
        # thirty free-form bookkeeping keys (`note`, `negative_result_plan`,
        # `dataset_baseline`, ...) that no code reads; `forbid` would fail the
        # config-load gate on most arms and `ignore` would drop them from the
        # provenance stamp. The closed field is `status`; `tags.status` is rejected.
        "spectramr.config.schemas.base.ExperimentMetadataSchema",
        "spectramr.config.schemas.training.base.LatentTrainingConfigSchema",
        "spectramr.config.schemas.training.base.TrainingStrategyConfigSchema",
        "spectramr.config.schemas.training.bloch_manifold_dps.TrainingConfigBlochManifoldDPS",
        "spectramr.config.schemas.training.equivariance_conformal.TrainingConfigEquivarianceConformal",
        "spectramr.config.schemas.training.pmps.TrainingConfigCorruptionCalibration",
        "spectramr.config.schemas.training.pmps.TrainingConfigDataEfficiencyHarness",
        "spectramr.config.schemas.training.pmps.TrainingConfigPairedSynthesis",
        "spectramr.config.schemas.training.pmps.TrainingConfigTissueDiffusionPretrain",
        "spectramr.config.schemas.training.phys_residual_conformal.TrainingConfigPhysResidualConformal",
        "spectramr.config.schemas.training.se3_navigator.TrainingConfigSE3Navigator",
        "spectramr.config.schemas.training.vf_advanced.TrainingConfigIBVF",
        "spectramr.config.schemas.training.vf_advanced.TrainingConfigTwinDPS",
    }
)

_SCHEMA_CLASSES = _discover_schema_classes()


def _qual(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def test_discovery_is_non_trivial():
    """Guard against a silent import regression that empties the sweep."""
    assert len(_SCHEMA_CLASSES) > 150


# These are loop-style (not parametrized-per-class) on purpose: the discovery is
# the same set for all three, the per-pytest-item fixture overhead dominates at
# ~700 params (10 min vs ~10 s), and a single failure listing *every* violator is
# more actionable than one red dot per class.


def test_every_schema_is_frozen():
    """Non-negotiable #1/#4: every loaded config block is immutable."""
    offenders = [_qual(c) for c in _SCHEMA_CLASSES if c.model_config.get("frozen") is not True]
    assert not offenders, (
        "These schema classes are not frozen — downstream code could mutate "
        "them. Inherit StrictSchema/CompatSchema/StrategySchema or set "
        "frozen=True:\n  " + "\n  ".join(offenders)
    )


def test_extra_policy_is_known_and_allow_is_allowlisted():
    """extra is one of the three tiers, and 'allow' only where reviewed."""
    unknown, unlisted = [], []
    for c in _SCHEMA_CLASSES:
        extra = c.model_config.get("extra", "ignore")  # pydantic default is 'ignore'
        if extra not in {"forbid", "ignore", "allow"}:
            unknown.append(f"{_qual(c)} (extra={extra!r})")
        elif extra == "allow" and _qual(c) not in ALLOW_ALLOWLIST:
            unlisted.append(_qual(c))
    assert not unknown, "Unknown extra policy:\n  " + "\n  ".join(unknown)
    assert not unlisted, (
        "These classes use extra='allow' (silent-swallow facade, pitfall #16) "
        "but are not in ALLOW_ALLOWLIST. Either tighten to 'forbid'/'ignore' or "
        "add them to the allowlist with a knob-audit rationale:\n  " + "\n  ".join(unlisted)
    )


def test_model_fields_have_protected_namespace_disabled():
    """A ``model_*`` field without protected_namespaces=() warns (→ test error)."""
    offenders = [
        _qual(c)
        for c in _SCHEMA_CLASSES
        if any(name.startswith("model_") for name in c.model_fields)
        and c.model_config.get("protected_namespaces") != ()
    ]
    assert not offenders, (
        "These classes declare a model_* field but do not set "
        "protected_namespaces=() (Pydantic will emit a UserWarning → test "
        "error):\n  " + "\n  ".join(offenders)
    )
