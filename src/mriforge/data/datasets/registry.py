"""Registry-dispatch for ``data.dataset_type`` (CLAUDE.md non-negotiable #6).

``DatasetInstantiator.create_datasets`` resolved the dataset type through a
21-branch ``if/elif`` chain -- the one component family in the repo that never
got a registry, while models, losses, metrics, strategies and (as of the
transform-registry change) transforms all dispatch through one. That is the
violation ``tests/architecture/test_dispatch_hell.py`` records.

The chain was not merely inelegant. Because the branch labels were hand-written
and the schema folded aliases *before* dispatch, ten labels were unreachable and
one canonical type (``graph_mri``) had no branch at all -- a state a registry
makes structurally impossible, because membership is checkable.

What this registry deliberately does NOT absorb
-----------------------------------------------

Two routing decisions in ``create_datasets`` are not keyed on ``dataset_type``
alone, and flattening them into name lookups would misrepresent them:

* ``manifest_roles`` -- a *predicate* on the config, evaluated only for the
  residual (``kspace`` / ``image``) types.
* ``image`` without ``source.index_path`` -- a type **plus** a condition; with
  an index path the same type routes to the FastMRI/universal loader.

Both stay as explicit, commented conditions around the lookup. A registry that
hid them behind a name would trade a visible chain for an invisible one.

Registering::

    register_dataset("cine", _create_cine, indexed=False,
                     serves="4-D cine MRI")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DATASET_REGISTRY",
    "DatasetCreator",
    "get_dataset_creator",
    "list_registered_dataset_types",
    "register_dataset",
]


@dataclass(frozen=True)
class DatasetCreator:
    """One ``dataset_type`` -> (train_ds, val_ds) construction route.

    Attributes:
        name: The ``dataset_type`` string this serves.
        fn: The creator. Called with the full indexed signature
            ``(config, train_index, val_index, train_tfm, val_tfm)`` when
            ``indexed`` is True, and with ``(config, train_tfm, val_tfm)``
            otherwise -- the two families the creators actually come in
            (5 indexed, 14 self-indexed).
        indexed: Whether the creator consumes the pre-split manifest index.
            Self-indexed datasets build their own index and must be skipped by
            the ManifestLoader pre-split -- the director derives that skip-set
            from this flag (``_self_indexed_dataset_types()``). It used to
            restate it as a literal, and 7 of the 12 self-indexed types were
            missing, which made them raise before their creator ever ran.
        serves: One-line human description, surfaced in error messages.
    """

    name: str
    fn: Callable[..., tuple[Any, Any]]
    indexed: bool
    serves: str = ""


DATASET_REGISTRY: dict[str, DatasetCreator] = {}


def register_dataset(
    name: str,
    fn: Callable[..., tuple[Any, Any]],
    *,
    indexed: bool,
    serves: str = "",
) -> DatasetCreator:
    """Register a construction route for ``name``.

    Raises:
        ValueError: If ``name`` is already registered to a different callable.
            Two creators under one name would make the resolved dataset depend
            on import order -- the failure mode this repo has hit before with
            import-time tables.
    """
    existing = DATASET_REGISTRY.get(name)
    if existing is not None and existing.fn is not fn:
        raise ValueError(
            f"dataset_type {name!r} is already registered to "
            f"{getattr(existing.fn, '__qualname__', existing.fn)}; "
            f"{getattr(fn, '__qualname__', fn)} cannot claim it."
        )
    entry = DatasetCreator(name=name, fn=fn, indexed=indexed, serves=serves)
    DATASET_REGISTRY[name] = entry
    return entry


def get_dataset_creator(name: str) -> DatasetCreator | None:
    """Return the route for ``name``, or ``None`` if it has no registered route.

    ``None`` is a legitimate answer, not an error: the residual ``kspace`` /
    ``image`` types are resolved by the conditional routes documented in the
    module docstring. The caller raises with the canonical list.
    """
    return DATASET_REGISTRY.get(name)


def list_registered_dataset_types() -> tuple[str, ...]:
    """Registered names, sorted."""
    return tuple(sorted(DATASET_REGISTRY))
