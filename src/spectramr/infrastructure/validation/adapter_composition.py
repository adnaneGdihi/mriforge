"""Adapter chain composition + audit-time verification.

The audit ``check_data_model_compatibility`` derives a coarse
``DataForm`` from the data block and a ``ModelForm`` from registered
capabilities, then walks each declared chain at the relevant hook to
see whether the gap between them is bridged. Three results possible:

- **bridged** (chain output ⊇ consumer requirement): ✅ info, side-effects
  surfaced.
- **chain doesn't bridge** (output still mismatched): ❌ error, fix-hint
  lists candidate adapters whose ``bridges_from``/``bridges_to`` could.
- **no chain declared**: ❌ error (when capabilities mismatch) with same
  candidate-list hint, OR ✅ pass (when capabilities match).

Adapters never fire silently. The audit can *suggest* a chain in a
fix-hint, never insert one on the user's behalf. See
``docs/superpowers/specs/2026-05-05-experiment-spec-card-and-adapters-design.md``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from spectramr.data.adapters.registry import ADAPTER_REGISTRY


@dataclass
class ChainResult:
    """Outcome of trying to compose an adapter chain at one hook."""

    bridged: bool
    output_form: dict[str, Any]
    side_effects: list[str] = field(default_factory=list)
    error: str | None = None


def _matches(produced: Any, required: Any) -> bool:
    """Does ``produced`` satisfy ``required``?

    - ``None`` on either side means "unconstrained" → matches anything.
    - Tuple/list on the required side means "any one of these works".
    - Everything else: equality.
    """
    if required is None or produced is None:
        return True
    if isinstance(required, (tuple, list, set, frozenset)):
        if isinstance(produced, (tuple, list, set, frozenset)):
            # Both are sets-of-allowed → at least one overlap
            return any(p in required for p in produced)
        return produced in required
    if isinstance(produced, (tuple, list, set, frozenset)):
        return required in produced
    return produced == required


def forms_match(produced: dict[str, Any], required: dict[str, Any]) -> tuple[bool, list[str]]:
    """Compare two form dicts key-by-key.

    Returns ``(all_match, mismatched_keys)``. A missing key on either
    side is treated as "unconstrained" and matches.
    """
    mismatches: list[str] = []
    for key in set(produced) | set(required):
        if not _matches(produced.get(key), required.get(key)):
            mismatches.append(key)
    return (not mismatches, mismatches)


def compose_chain(
    upstream_form: dict[str, Any],
    chain: Iterable[str],
) -> ChainResult:
    """Walk ``chain`` of adapter names and return the resulting form.

    Each adapter's ``bridges_from`` must match the running form;
    otherwise the chain is broken and ChainResult.error is set.
    """
    current = dict(upstream_form)
    side_effects: list[str] = []
    for name in chain:
        entry = ADAPTER_REGISTRY.get(name)
        if entry is None:
            return ChainResult(
                bridged=False,
                output_form=current,
                side_effects=side_effects,
                error=f"adapter {name!r} is not registered",
            )
        caps = entry["capabilities"]
        ok, mismatched = forms_match(produced=current, required=caps.bridges_from)
        if not ok:
            return ChainResult(
                bridged=False,
                output_form=current,
                side_effects=side_effects,
                error=(
                    f"adapter {name!r} expects {caps.bridges_from} but "
                    f"upstream produces {current} (mismatched: {mismatched})"
                ),
            )
        # Apply the adapter's bridges_to over the running form.
        current = {**current, **caps.bridges_to}
        side_effects.extend(caps.side_effects)
    return ChainResult(bridged=True, output_form=current, side_effects=side_effects)


def candidate_adapters_for(
    upstream_form: dict[str, Any],
    required_form: dict[str, Any],
    hook: str | None = None,
) -> list[str]:
    """Return adapter names whose single-step bridge would close the gap.

    Used in fix-hints. Only single-step suggestions; multi-step chains
    are too speculative for an automated hint and the user should
    declare them explicitly.
    """
    candidates: list[str] = []
    for name, entry in ADAPTER_REGISTRY.items():
        caps = entry["capabilities"]
        if hook is not None and caps.insertion_points and hook not in caps.insertion_points:
            continue
        # The adapter must accept the upstream form...
        ok_in, _ = forms_match(produced=upstream_form, required=caps.bridges_from)
        if not ok_in:
            continue
        # ...and after applying it, the result must satisfy the consumer.
        produced_after = {**upstream_form, **caps.bridges_to}
        ok_out, _ = forms_match(produced=produced_after, required=required_form)
        if ok_out:
            candidates.append(name)
    return candidates


__all__ = [
    "ChainResult",
    "candidate_adapters_for",
    "compose_chain",
    "forms_match",
]
