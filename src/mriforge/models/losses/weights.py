"""Loss-weight resolution — the SSOT.

Before this module there were **eight** weight resolvers (two in ``strategies/``,
five across ``losses/computers/``, one in ``loss_folding``), each with its own
precedence and its own magic default table. The same YAML resolved to a different
effective weight depending on which code path a strategy happened to take, and an
undeclared loss silently materialised at 1.0 (pitfall #9: a silent fallback).

This module replaces all of them. Two objects:

* :class:`LossWeightTable` — every declared weight, resolved **once** from the frozen
  config at build time. Not per step: the old ``BaseLossComputer._get_loss_weight``
  called ``config.losses.model_dump()`` per loss per step (``performance.md``).
* :func:`resolve_loss_weight` — a pure lookup on that table.

Declaration surfaces and the conflict rule
------------------------------------------
A weight may be declared on exactly ONE of two surfaces:

1. ``losses.<section>.lambda_<name>`` — the explicit lambda.
2. ``losses.{image,kspace,complex}_losses[].weight`` — the declarative list entry.

Both declaring the same (canonical) loss is legal only when they AGREE. Disagreement
raises: a "warning + pick one" would be pitfall #10 wearing a raise costume. Names are
canonicalised through :class:`~mriforge.models.losses.registry.LossRegistry`, so
``image_losses: [{name: mse}]`` and ``lambda_l2`` are recognised as the same knob.

"Declared" means the author WROTE it (``model_fields_set``), never a schema default —
``lambda_hfen`` defaults to 0.0, and treating that as a declaration is exactly how a
declared ``hfen weight: 0.1`` ended up silently computing at zero.

A loss declared nowhere RAISES (pitfall #9/#15). ``enabled: false`` / ``weight: 0`` is a
declaration — it resolves to 0.0 and never raises.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, get_args

from mriforge.domain.exceptions import ConfigurationError
from mriforge.models.losses.registry import LossRegistry

if TYPE_CHECKING:  # pragma: no cover
    from mriforge.config.schemas.loss import LossConfigSchema

#: Bumped when the resolution semantics change. Stamped into run provenance (#15).
WEIGHT_SEMANTICS_VERSION = "2"

#: Every sub-block of ``losses:`` that may carry ``lambda_<name>`` fields.
#: ``pinn`` is included deliberately: no legacy resolver scanned it, so 48 arms
#: declaring ``losses.pinn.lambda_pde`` were silently ignored.
LAMBDA_SECTIONS: tuple[str, ...] = (
    "reconstruction",
    "latent",
    "adversarial",
    "diffusion",
    "gan",
    "physics",
    "registration",
    "ssl",
    "spatial",
    "evidential",
    "pinn",
)

#: The declarative loss lists (v6.0+).
LOSS_LISTS: tuple[str, ...] = ("image_losses", "kspace_losses", "complex_losses")

#: Spellings of a loss that the registry does not know, because the schema field and the
#: component the computers stack are named differently. Mirrors the remap
#: ``LossConfigSchema.get_enabled_losses()`` already applies when deciding what to BUILD;
#: without it the same knob is invisible when deciding what to WEIGH.
#:
#: This is not cosmetic. Every legacy resolver looked up ``lambda_adversarial``, but the
#: field is ``losses.gan.lambda_adv`` — so the adversarial weight was **never read from
#: config**, and every GAN arm silently ran it at the fallback 1.0. 23 arms happened to
#: declare 1.0 anyway; one declares 0.1 and has been training 10x hot.
#:
#: Applied to BOTH directions (the field ``lambda_adv`` and a computer asking for
#: ``adversarial``), so the two always land on the same entry.
NAME_ALIASES: dict[str, str] = {
    "adv": "adversarial",
    "gp": "gradient_penalty",
    "commit": "commitment",
    # UnifiedGANLossComputer stacks the R1 term as `r1_penalty`; the field is `lambda_r1`.
    "r1_penalty": "r1",
}

#: The historical hardcoded warm-up set, duplicated in ``strategies/base.py`` and
#: ``unified_diffusion_reconstruction.py``. Kept as the DEFAULT for
#: ``losses.reconstruction.warmup_losses`` so behaviour is unchanged, but it is now
#: named in one place and stamped into provenance instead of being implied.
LEGACY_WARMUP_LOSSES: frozenset[str] = frozenset(
    {
        "complex_spatial_gradient",
        "rician_consistency",
        "background_suppression",
        "perceptual",
        "adversarial",
        "l1",
    }
)

DEFAULT_WARMUP_ITERATIONS = 1000


def canonical_loss_name(name: str) -> str:
    """Resolve a loss alias to its canonical registry name (``mse`` -> ``l2``).

    Unregistered names pass through unchanged: a strategy-inline term such as
    ``pre_dc_kspace`` has a lambda but no registry entry, and that is legal.

    A ``lambda_``-prefixed query is the FIELD spelling of the same knob (some callers
    thread the schema field name straight through, e.g. the disentangled computer's
    ``{"hist": "lambda_hist"}`` map), so the prefix is stripped rather than treated as a
    distinct — and therefore undeclared — loss.
    """
    if name.startswith("lambda_"):
        name = name[len("lambda_") :]
    name = NAME_ALIASES.get(name.lower(), name)
    return LossRegistry.canonical_name(name)


def _loss_name_for_field(field: str) -> str:
    """``lambda_adv`` -> ``adversarial``; ``lambda_mse`` -> ``l2``. One place."""
    return canonical_loss_name(field)


@lru_cache(maxsize=1)
def _schema_defaults() -> dict[str, tuple[str, float]]:
    """``{canonical loss -> (section, schema default)}`` for every ``lambda_<n>`` field.

    This is the ONLY sanctioned fallback, and it is not a magic table: the value is the
    Pydantic field default, declared once in ``config/schemas/loss.py`` and visible to
    the reader of the schema. Callers routinely *probe* a term's weight and gate on
    ``> 0`` ("is this configured?"), so an undeclared term with a schema field must
    answer with that default (almost always 0.0 = not requested), not explode.

    A term with NO ``lambda_<n>`` field anywhere is the dangerous case: those are what
    fell through to the three disagreeing hardcoded tables (an undeclared ``adversarial``
    resolved to 1.0 or 0.01; ``kl_divergence`` to 1.0 or 1e-4). Those RAISE.
    """
    from mriforge.config.schemas.loss import LossConfigSchema

    defaults: dict[str, tuple[str, float]] = {}
    for section_name in LAMBDA_SECTIONS:
        field = LossConfigSchema.model_fields.get(section_name)
        if field is None:
            continue
        section_cls = _section_type(field)
        if section_cls is None:
            continue
        for field_name, spec in getattr(section_cls, "model_fields", {}).items():
            if not field_name.startswith("lambda_"):
                continue
            default = spec.default
            if not isinstance(default, (int, float)):
                continue
            name = _loss_name_for_field(field_name)
            # First section wins; a clash is caught as an ambiguity at declaration time.
            defaults.setdefault(name, (section_name, float(default)))
    return defaults


def _section_type(field: Any) -> Any:
    """The concrete ``*LossesConfig`` behind an ``X | None`` annotation."""
    annotation = getattr(field, "annotation", None)
    for candidate in get_args(annotation) or (annotation,):
        if isinstance(candidate, type) and hasattr(candidate, "model_fields"):
            return candidate
    return None


@dataclass(frozen=True, slots=True)
class LossWeightSpec:
    """One resolved loss weight, with the provenance of where it came from."""

    name: str
    """Canonical registry name (aliases resolved)."""

    weight: float
    enabled: bool
    source: str
    """e.g. ``losses.reconstruction.lambda_hfen`` or ``losses.image_losses[hfen].weight``."""

    warmup_gated: bool
    """Precomputed ``name in warmup_losses`` — keeps the hot path allocation-free."""


class LossWeightTable(Mapping[str, LossWeightSpec]):
    """Immutable, fully-resolved weight table. Built once from a frozen config."""

    __slots__ = ("_specs", "warmup_iterations", "warmup_losses")

    def __init__(
        self,
        specs: dict[str, LossWeightSpec],
        *,
        warmup_iterations: int,
        warmup_losses: frozenset[str],
    ) -> None:
        self._specs = specs
        self.warmup_iterations = warmup_iterations
        self.warmup_losses = warmup_losses

    def __getitem__(self, key: str) -> LossWeightSpec:
        return self._specs[canonical_loss_name(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def weight(self, name: str, *, iteration: int = 0) -> float:
        """Static weight for ``name``. Raises when the loss is declared nowhere."""
        return resolve_loss_weight(self, name, iteration=iteration)

    def provenance(self) -> dict[str, Any]:
        """The stamp written into the run record (#15: every knob read AND stamped)."""
        return {
            "semantics_version": WEIGHT_SEMANTICS_VERSION,
            "warmup_iterations": self.warmup_iterations,
            "warmup_losses": sorted(self.warmup_losses),
            "resolved": {
                spec.name: {
                    "weight": spec.weight,
                    "enabled": spec.enabled,
                    "source": spec.source,
                    "warmup_gated": spec.warmup_gated,
                }
                for spec in self._specs.values()
            },
        }


def _declared_lambdas(loss_config: Any) -> dict[str, list[tuple[str, float]]]:
    """Author-written ``lambda_<name>`` fields, keyed by canonical loss name.

    Only fields in ``model_fields_set`` count — a schema default is not a declaration.
    """
    found: dict[str, list[tuple[str, float]]] = {}
    for section_name in LAMBDA_SECTIONS:
        section = getattr(loss_config, section_name, None)
        if section is None:
            continue
        for field in _written_lambda_fields(section):
            value = getattr(section, field, None)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            name = _loss_name_for_field(field)
            source = f"losses.{section_name}.{field}"
            found.setdefault(name, []).append((source, float(value)))
    return found


def _written_lambda_fields(section: Any) -> list[str]:
    """The ``lambda_*`` fields the AUTHOR set on ``section``.

    For a Pydantic model that is ``model_fields_set`` — the whole point, since a schema
    default is not a declaration. Config *doubles* (``SimpleNamespace``, a test's
    ``MagicMock``) have no such attribute; for those, every ``lambda_*`` present in the
    instance dict counts as written, since a double only carries what a test put there.
    Silently treating a double as "declares nothing" would zero its weights.
    """
    written = getattr(section, "model_fields_set", None)
    if isinstance(written, (set, frozenset, list, tuple)):
        return [f for f in written if isinstance(f, str) and f.startswith("lambda_")]
    instance_dict = getattr(section, "__dict__", None)
    if isinstance(instance_dict, dict):
        return [k for k in instance_dict if isinstance(k, str) and k.startswith("lambda_")]
    return []


def _declared_list_entries(
    loss_config: Any,
) -> dict[str, list[tuple[str, float, bool]]]:
    """Declarative list entries, keyed by canonical loss name."""
    found: dict[str, list[tuple[str, float, bool]]] = {}
    for list_name in LOSS_LISTS:
        for entry in getattr(loss_config, list_name, None) or []:
            raw = getattr(entry, "name", None)
            if raw is None and isinstance(entry, dict):
                raw = entry.get("name")
            if raw is None:
                continue
            weight = getattr(entry, "weight", None)
            if weight is None and isinstance(entry, dict):
                weight = entry.get("weight")
            enabled = getattr(entry, "enabled", None)
            if enabled is None and isinstance(entry, dict):
                enabled = entry.get("enabled", True)
            name = canonical_loss_name(str(raw))
            source = f"losses.{list_name}[{raw}].weight"
            found.setdefault(name, []).append(
                (source, 1.0 if weight is None else float(weight), bool(enabled))
            )
    return found


def _agree(values: list[float]) -> bool:
    first = values[0]
    return all(math.isclose(v, first, rel_tol=1e-9, abs_tol=1e-12) for v in values)


def build_loss_weight_table(
    loss_config: LossConfigSchema | None,
) -> LossWeightTable:
    """Resolve every declared loss weight exactly once, at config-freeze time.

    Raises :class:`ConfigurationError` — listing every offender at once, so one run
    surfaces the whole problem — when a canonical loss is declared more than once at
    DIFFERENT weights, whether across sections (``lambda_l1`` in both ``reconstruction``
    and ``latent``), across surfaces (``lambda_hfen`` vs ``image_losses[hfen].weight``),
    or across aliases (``lambda_l2`` vs ``image_losses[mse]``).

    Never raises for an *undeclared* name — that is a lookup-time error (the caller may
    legitimately probe for a term it does not use).
    """
    if loss_config is None:
        return LossWeightTable(
            {},
            warmup_iterations=DEFAULT_WARMUP_ITERATIONS,
            warmup_losses=frozenset(),
        )

    lambdas = _declared_lambdas(loss_config)
    entries = _declared_list_entries(loss_config)

    recon = getattr(loss_config, "reconstruction", None)
    raw_warmup = getattr(recon, "warmup_iterations", None) if recon is not None else None
    # A config double may hand back a non-int here; fall back rather than crash.
    warmup_iterations = (
        int(raw_warmup)
        if isinstance(raw_warmup, int) and not isinstance(raw_warmup, bool)
        else DEFAULT_WARMUP_ITERATIONS
    )
    configured_warmup = getattr(recon, "warmup_losses", None) if recon else None
    if not isinstance(configured_warmup, (list, tuple, set, frozenset)):
        configured_warmup = None
    warmup_losses = frozenset(
        canonical_loss_name(n)
        for n in (LEGACY_WARMUP_LOSSES if configured_warmup is None else configured_warmup)
    )

    conflicts: list[str] = []
    specs: dict[str, LossWeightSpec] = {}

    for name in sorted(set(lambdas) | set(entries)):
        lam = lambdas.get(name, [])
        lst = entries.get(name, [])
        declarations = [(src, val) for src, val in lam] + [(src, val) for src, val, _ in lst]
        values = [val for _, val in declarations]

        if len(values) > 1 and not _agree(values):
            detail = ", ".join(f"{src} = {val}" for src, val in declarations)
            conflicts.append(
                f"  '{name}' is declared {len(values)}x at DIFFERENT weights: {detail}"
            )
            continue

        # `enabled: false` on ANY list entry disables the term; a lambda-only term is
        # enabled by declaration. `weight: 0` is a declaration too (resolves to 0.0).
        enabled = all(en for _, _, en in lst) if lst else True
        weight = values[0]
        source = (
            "+".join(src for src, _ in declarations)
            if len(declarations) > 1
            else declarations[0][0]
        )
        specs[name] = LossWeightSpec(
            name=name,
            weight=0.0 if not enabled else weight,
            enabled=enabled,
            source=source,
            warmup_gated=name in warmup_losses,
        )

    if conflicts:
        raise ConfigurationError(
            "Conflicting loss-weight declarations (a loss may be declared on the "
            "lambda surface OR the declarative-list surface, not both at different "
            "values):\n"
            + "\n".join(conflicts)
            + "\n\nFix: keep ONE declaration per loss, or make the two agree. "
            "Note that aliases collapse (`mse` and `l2` are the same loss)."
        )

    return LossWeightTable(
        specs,
        warmup_iterations=warmup_iterations,
        warmup_losses=warmup_losses,
    )


def resolve_loss_weight(
    table: LossWeightTable,
    name: str,
    *,
    scheduled: Mapping[str, float] | None = None,
    iteration: int = 0,
) -> float:
    """The hot path: a pure lookup. No config walk, no ``model_dump``, no allocation.

    Precedence:
      1. ``scheduled`` — the ``loss_schedule`` curriculum override, which supersedes
         BOTH the warm-up gate and the static weight (a schedule rule must be able to
         enable a spatial term before ``warmup_iterations``).
      2. ``enabled: false`` -> 0.0
      3. the warm-up gate -> 0.0 while ``iteration < warmup_iterations``
      4. the declared static weight

    Raises:
        ConfigurationError: ``name`` is declared nowhere. Never silently 1.0.
    """
    canonical = canonical_loss_name(name)

    if scheduled:
        override = scheduled.get(canonical, scheduled.get(name))
        if override is not None:
            return float(override)

    spec = table.get(canonical)
    if spec is None:
        # Undeclared. Fall back to the `lambda_<n>` SCHEMA default if one exists — that
        # is a single, visible, auditable value (almost always 0.0 = "not requested"),
        # and callers legitimately probe a term's weight before deciding to compute it.
        fallback = _schema_defaults().get(canonical)
        if fallback is not None:
            _section, default = fallback
            if canonical in table.warmup_losses and iteration < table.warmup_iterations:
                return 0.0
            return default

        # No declaration AND no schema field: this is precisely the class that used to
        # fall through to the three disagreeing hardcoded tables — an undeclared
        # `adversarial` resolved to 1.0 or 0.01 (100x apart), `kl_divergence` to 1.0 or
        # 1e-4 (10,000x apart), purely by which computer the strategy happened to build.
        # Refuse to guess (pitfall #9/#15).
        raise ConfigurationError(
            f"Loss '{name}' is active but its weight is declared nowhere, and no "
            f"`lambda_{canonical}` field exists in any losses section to default from. "
            f"Refusing to invent a weight (CLAUDE.md #9/#15).\n"
            f"Fix: declare it explicitly — add `- {{name: {canonical}, weight: <w>}}` to "
            f"`losses.image_losses`, or add a `lambda_{canonical}` field to the schema."
        )

    if not spec.enabled:
        return 0.0
    if spec.warmup_gated and iteration < table.warmup_iterations:
        return 0.0
    return spec.weight
