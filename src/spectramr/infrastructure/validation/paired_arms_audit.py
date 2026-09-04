"""Paired-arms audit for single-factor ablation campaigns.

A controlled A/B ablation is only meaningful if the two arms differ in
exactly one declared factor. The cold_diffusion_locus_ablation campaign
(2026-05-19) introduces this discipline: arms tagged with the same
``metadata.group`` and a ``paired_with`` link are checked to ensure the
shared specifications truly are shared.

The audit is intentionally generic — it operates on two raw YAML dicts
(loaded but not yet validated through Pydantic). The
``allowed_diff_paths`` parameter declares the dotted-path keys that
are permitted to differ; everything else must match. The campaign
manifest can declare its own allow-list via
``stage_groups[*].experiments[*].paired_arms_diff_paths`` (future), or
the caller can pass it explicitly.

Use cases:

* Pre-submission audit: ``python -m spectramr.cli audit-pair <a.yaml> <b.yaml>``.
* Campaign-load-time audit: triggered when a CampaignConfigSchema
  validates a stage_group whose experiments declare matching
  ``metadata.group`` strings.

Returns a list of :class:`PairedArmsDiff` entries — each is a single
disagreement between the two arms at a single dotted-path key.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# The allow-list itself lives in a sibling data module (300-LOC ceiling, NN20).
# Re-exported under its historical private name so the one frozenset keeps one
# owner (NN17) while existing importers -- including the retired-spelling pin in
# tests/unit/validation/ -- keep working against it.
from spectramr.infrastructure.validation.paired_arms_diff_paths import (
    DEFAULT_DIFF_PATHS as _DEFAULT_DIFF_PATHS,
)


@dataclass(frozen=True)
class PairedArmsDiff:
    """A single mismatch between two arms at a dotted-path key."""

    path: str
    arm_a_value: Any
    arm_b_value: Any
    severity: str = "error"  # "error" | "warning" | "info"


@dataclass
class PairedArmsAuditResult:
    """Outcome of the paired-arms audit."""

    arm_a_path: Path
    arm_b_path: Path
    group: str
    passed: bool
    diffs: list[PairedArmsDiff] = field(default_factory=list)
    skipped_paths: set[str] = field(default_factory=set)

    def render(self) -> str:
        if self.passed:
            return (
                f"[✓] paired-arms audit PASSED for group "
                f"{self.group!r}: {self.arm_a_path.name} ↔ "
                f"{self.arm_b_path.name}"
            )
        lines = [
            f"[✗] paired-arms audit FAILED for group {self.group!r}:",
            f"    {self.arm_a_path.name} ↔ {self.arm_b_path.name}",
            f"    {len(self.diffs)} unexpected difference(s):",
        ]
        for d in self.diffs:
            lines.append(f"      • {d.path}: A={d.arm_a_value!r} ≠ B={d.arm_b_value!r}")
        return "\n".join(lines)


def _fold_canonical() -> dict[str, str]:
    """Legacy dotted path -> canonical, for every STAGED rename.

    This audit walks the raw YAML, so during a staged migration one arm may
    spell a knob ``optimization.learning_rate`` while its sibling spells it
    ``optimization.optimizer.learning_rate``. Untranslated, the pair reports a
    spurious diff at EVERY moved knob -- and worse, an allow-list entry silences
    only whichever spelling its author happened to write. Both arms are
    normalised to the canonical path so the comparison is about values.
    """
    from spectramr.config.schemas.renames import RENAMES

    return {r.legacy: r.canonical for r in RENAMES.values() if r.posture == "fold"}


def _walk_raw(prefix: str, value: Any) -> Iterable[tuple[str, Any]]:
    """Yield (dotted-path, leaf-value) pairs verbatim, with NO canonicalisation.

    Kept separate from :func:`_canonical_paths` because the collision check
    there needs the spelling the arm actually wrote in order to name it.
    """
    if isinstance(value, Mapping):
        for k, v in value.items():
            sub = f"{prefix}.{k}" if prefix else str(k)
            yield from _walk_raw(sub, v)
    else:
        yield (prefix, value)


_MISSING = object()


def _canonical_paths(doc: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Leaf paths of ``doc``, canonicalised through the rename SSOT.

    A migrated arm and an unmigrated one describe the same knob with the same
    key, so the comparison is about values rather than spellings.

    This replaced a ``dict(_walk(...))`` generator. That form resolved a
    legacy/canonical collision by **insertion order**, so an arm declaring both
    spellings of one knob got a verdict that depended on how the file happened
    to be serialised. ``yaml.safe_dump`` sorts keys, which is enough to flip it:
    the same two values compared equal or unequal by alphabetical order alone.

    The loader refuses such a document outright -- *"`optimization.learning_rate`
    and `optimization.optimizer.learning_rate` disagree ... Declare one of
    them."* An audit that silently picks a winner is therefore reporting on a
    config that cannot run, and is a second resolver disagreeing with the SSOT.
    Mirror the loader and raise (non-negotiable #3).

    Equal values are not a contradiction: the arm is redundant, not ambiguous,
    and the fold resolves it the same way whichever spelling wins.
    """
    fold = _fold_canonical()
    collected: dict[str, Any] = {}
    declared_at: dict[str, str] = {}
    for raw, leaf in _walk_raw("", doc):
        canonical = fold.get(raw, raw)
        prior = collected.get(canonical, _MISSING)
        if prior is not _MISSING and prior != leaf:
            first, second = declared_at[canonical], raw
            raise ValueError(
                f"{arm} declares the same knob twice with different values: "
                f"`{first}`={prior!r} and `{second}`={leaf!r} both resolve to "
                f"`{canonical}`. The loader refuses this document, so the "
                f"paired-arms audit cannot report on it either. Declare one "
                f"of them."
            )
        collected[canonical] = leaf
        declared_at[canonical] = raw
    return collected


def _is_diff_allowed(path: str, allowed_paths: frozenset[str]) -> bool:
    """A path is allowed if it (or an ancestor) is in the allow-list.

    The allow-list contains *prefixes* — declaring ``adapters`` exempts
    every descendant key like ``adapters.0.name``. This is essential
    for nested-block keys whose exact shape differs between arms.
    """
    if path in allowed_paths:
        return True
    for allowed in allowed_paths:
        if path.startswith(allowed + ".") or path.startswith(allowed + "."):
            return True
    return False


def audit_paired_arms(
    arm_a_path: str | Path,
    arm_b_path: str | Path,
    allowed_diff_paths: Iterable[str] | None = None,
) -> PairedArmsAuditResult:
    """Audit two arm YAMLs as a single-factor ablation.

    Args:
        arm_a_path: filesystem path to the Arm A YAML.
        arm_b_path: filesystem path to the Arm B YAML.
        allowed_diff_paths: optional override / extension to the
            default allow-list. When ``None``, the curated default
            (above) is used. When provided as a set, it REPLACES the
            default — pass ``set(_DEFAULT_DIFF_PATHS) | {"foo.bar"}``
            to extend rather than replace.

    Returns:
        :class:`PairedArmsAuditResult` with per-key diffs.
    """
    arm_a_path = Path(arm_a_path)
    arm_b_path = Path(arm_b_path)
    if not arm_a_path.exists():
        raise FileNotFoundError(f"Arm A YAML not found: {arm_a_path}")
    if not arm_b_path.exists():
        raise FileNotFoundError(f"Arm B YAML not found: {arm_b_path}")

    a = yaml.safe_load(arm_a_path.read_text(encoding="utf-8")) or {}
    b = yaml.safe_load(arm_b_path.read_text(encoding="utf-8")) or {}

    if not isinstance(a, dict) or not isinstance(b, dict):
        raise ValueError("Both arm YAMLs must parse to mappings at the top level.")

    group_a = (a.get("metadata") or {}).get("group", "")
    group_b = (b.get("metadata") or {}).get("group", "")
    if group_a != group_b:
        raise ValueError(
            f"Arm A and Arm B have different metadata.group: "
            f"{group_a!r} ≠ {group_b!r}. The paired-arms audit only "
            f"applies to arms that declare the same campaign group."
        )

    allowed = (
        frozenset(allowed_diff_paths) if allowed_diff_paths is not None else _DEFAULT_DIFF_PATHS
    )

    paths_a = _canonical_paths(a, f"Arm A ({arm_a_path.name})")
    paths_b = _canonical_paths(b, f"Arm B ({arm_b_path.name})")
    all_paths = set(paths_a) | set(paths_b)

    diffs: list[PairedArmsDiff] = []
    skipped: set[str] = set()
    sentinel = object()
    for path in sorted(all_paths):
        va = paths_a.get(path, sentinel)
        vb = paths_b.get(path, sentinel)
        if va == vb:
            continue
        if _is_diff_allowed(path, allowed):
            skipped.add(path)
            continue
        diffs.append(
            PairedArmsDiff(
                path=path,
                arm_a_value=va if va is not sentinel else "<missing>",
                arm_b_value=vb if vb is not sentinel else "<missing>",
            )
        )

    return PairedArmsAuditResult(
        arm_a_path=arm_a_path,
        arm_b_path=arm_b_path,
        group=group_a,
        passed=not diffs,
        diffs=diffs,
        skipped_paths=skipped,
    )


__all__ = [
    "PairedArmsAuditResult",
    "PairedArmsDiff",
    "audit_paired_arms",
]
