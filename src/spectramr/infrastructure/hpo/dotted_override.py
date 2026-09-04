"""Dotted-path override helper for HPO and ablation config mutation.

Moved here from ``src/pipelines/hpo_search_spaces.py`` 2026-05-14 per
``TODO/backlog_ssot_and_layering_cleanup.md`` Phase 2 (the
infrastructure-layer HPO coordinator was importing upward into
``pipelines/`` for this helper — that's a layer-direction violation).
``hpo_search_spaces.py`` now re-exports from here for backward-compat.

The helper applies a dotted-path override to a nested dict in place.
Two notations are supported:

* **Plain dotted path** — ``a.b.c`` navigates ``cfg[a][b][c]``. Missing
  intermediate keys are created on the fly.
* **List selector** — ``a.b[name=foo].c`` finds the item in the list
  ``cfg[a][b]`` whose ``name`` field equals ``foo`` (or matches by any
  ``key=value`` pair) and navigates into ``.c`` of that item. Essential
  for the loss YAML format which uses ``[{name: complex_l1, weight: 1.0}, ...]``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _ListSelector:
    """Internal: parsed ``[key=value]`` selector token."""

    key: str
    value: Any


def _tokenize_path(path: str) -> list:
    """Split a dotted path with optional ``[k=v]`` selectors into tokens.

    Examples:
        ``"a.b.c"`` -> ``["a", "b", "c"]``
        ``"a.b[name=foo].c"`` -> ``["a", "b", _ListSelector("name", "foo"), "c"]``
    """
    tokens: list = []
    for seg in path.split("."):
        if not seg:
            continue
        # Multiple selectors per segment are unusual but supported (a.b[k=v][m=n])
        m = re.match(r"^([^\[]+)((?:\[[^\]]+\])*)$", seg)
        if not m:
            tokens.append(seg)
            continue
        head, selectors = m.groups()
        if head:
            tokens.append(head)
        for sel in re.findall(r"\[([^\]]+)\]", selectors or ""):
            if "=" not in sel:
                raise ValueError(f"path {path!r}: selector [{sel}] must be of form [key=value]")
            k, v = sel.split("=", 1)
            # Try to coerce simple types
            if v.lower() in ("true", "false"):
                v_typed: Any = v.lower() == "true"
            else:
                try:
                    v_typed = int(v)
                except ValueError:
                    try:
                        v_typed = float(v)
                    except ValueError:
                        v_typed = v  # plain string
            tokens.append(_ListSelector(k.strip(), v_typed))
    return tokens


def apply_dotted_override(cfg: dict, path: str, value: Any) -> None:
    """Apply a dotted-path override to a nested dict in place.

    Args:
        cfg: Target dict (mutated in place).
        path: Dotted-path with optional ``[key=value]`` list selectors.
        value: Value to write at the resolved path.

    Raises:
        ValueError: If a list selector matches no item or matches multiple items,
            or if the path navigates into a non-container type.

    Examples:
        >>> cfg = {"a": {"b": [{"name": "x", "weight": 1.0}]}}
        >>> apply_dotted_override(cfg, "a.b[name=x].weight", 0.5)
        >>> cfg["a"]["b"][0]["weight"]
        0.5
    """
    parts = _tokenize_path(path)
    target: Any = cfg
    for tok in parts[:-1]:
        if isinstance(tok, _ListSelector):
            if not isinstance(target, list):
                raise ValueError(
                    f"path {path!r}: selector {tok} requires a list at this level, "
                    f"got {type(target).__name__}"
                )
            matches = [
                idx
                for idx, item in enumerate(target)
                if isinstance(item, dict) and item.get(tok.key) == tok.value
            ]
            if len(matches) == 0:
                raise ValueError(f"path {path!r}: no list item with {tok.key}={tok.value!r}")
            if len(matches) > 1:
                raise ValueError(
                    f"path {path!r}: multiple list items match {tok.key}={tok.value!r} "
                    f"(found {len(matches)})"
                )
            target = target[matches[0]]
        else:
            if not isinstance(target, dict):
                raise ValueError(
                    f"path {path!r}: plain key {tok!r} requires a dict at this "
                    f"level, got {type(target).__name__}"
                )
            if tok not in target:
                # Auto-create missing intermediate dicts for the Optuna
                # search-space pattern that injects new nested sub-keys
                # like ``model_kwargs.kan_dual_domain_kwargs``.
                target[tok] = {}
            elif not isinstance(target[tok], (dict, list)):
                raise ValueError(
                    f"path {path!r}: intermediate key {tok!r} holds a "
                    f"{type(target[tok]).__name__} (expected dict or list); "
                    "check the path for an extra nesting level"
                )
            target = target[tok]

    last = parts[-1]
    if isinstance(last, _ListSelector):
        if not isinstance(target, list):
            raise ValueError(
                f"path {path!r}: terminal selector {last} requires a list, "
                f"got {type(target).__name__}"
            )
        matches = [
            idx
            for idx, item in enumerate(target)
            if isinstance(item, dict) and item.get(last.key) == last.value
        ]
        if len(matches) == 0:
            raise ValueError(f"path {path!r}: no terminal list item with {last.key}={last.value!r}")
        if len(matches) > 1:
            raise ValueError(
                f"path {path!r}: multiple terminal list items match {last.key}={last.value!r}"
            )
        target[matches[0]] = value
    else:
        if not isinstance(target, dict):
            raise ValueError(
                f"path {path!r}: terminal key {last!r} requires a dict, got {type(target).__name__}"
            )
        target[last] = value


__all__ = ["apply_dotted_override"]
