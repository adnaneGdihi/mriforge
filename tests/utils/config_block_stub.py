"""Stand-ins for the top-level config blocks, built from the REAL schemas.

The sibling :mod:`tests.utils.data_config_stub` does this for ``data:``. The
block decomposition moved leaves under ``validation:``, ``logging:`` and
``losses:`` too, and the stand-ins for those blocks were hand-written
``SimpleNamespace``\\ s carrying the pre-decomposition shape -- a shape no real
config produces. Production reads ``validation.scoring.compute``; a stub setting
``metrics=[...]`` raises ``AttributeError: 'SimpleNamespace' object has no
attribute 'scoring'`` at best, and at worst agrees with a stale reader and keeps
a dead check green (the F16 pattern: two stubs pinned two dead mechanisms).

Two properties, both inherited from ``data_config_stub``'s rules:

1. **The stub IS the real schema.** Every one of these blocks constructs bare
   (verified: ``ValidationConfigSchema()``, ``LoggingConfigSchema()``,
   ``LossConfigSchema()``, ``AccelerationConfigSchema()``), so there is nothing
   to restate and a stub cannot disagree with a live config about a default.
2. **One place to extend.** A new sub-block needs no change here at all -- the
   schema already carries it.

And one it adds:

3. **The flat->canonical routing is DERIVED from**
   :data:`spectramr.config.schemas.renames.RENAMES`, not restated. 87 pairs across
   the 5 registered blocks, every one of them the same record the loader folds
   YAML with. A hand-written table would be a second resolver that agrees until
   the next rename -- which is the whole reason these tests broke.

Usage::

    from tests.utils.config_block_stub import block_stub

    val = block_stub("validation", metrics=["psnr"])   # legacy spelling ok
    val.scoring.compute            # ['psnr']  -- routed to its canonical home
    val.schedule.interval_steps    # 1000      -- real schema default

    log = block_stub("logging", save_validation_images=True)
    log.images.save_validation     # True

Unknown kwargs are set directly, so a test can still attach something the schema
does not model. A kwarg whose legacy name IS a fold record is always routed --
it can never become a second spelling the reader does not walk.
"""

from __future__ import annotations

from typing import Any

__all__ = ["BLOCK_SCHEMAS", "block_stub", "flat_to_canonical"]

#: Block name -> ``(module, class)`` of the schema that defines it.
BLOCK_SCHEMAS: dict[str, tuple[str, str]] = {
    "validation": ("spectramr.config.schemas.validation", "ValidationConfigSchema"),
    "logging": ("spectramr.config.schemas.logging", "LoggingConfigSchema"),
    "losses": ("spectramr.config.schemas.loss", "LossConfigSchema"),
    "undersampling": (
        "spectramr.config.schemas.acceleration",
        "AccelerationConfigSchema",
    ),
    "optimization": (
        "spectramr.config.schemas.optimization",
        "OptimizationConfigSchema",
    ),
}


def _schema(block: str) -> Any:
    try:
        module, cls = BLOCK_SCHEMAS[block]
    except KeyError:
        raise KeyError(
            f"no schema registered for block {block!r}; "
            f"known: {sorted(BLOCK_SCHEMAS)}"
        ) from None
    import importlib

    return getattr(importlib.import_module(module), cls)


def flat_to_canonical(block: str) -> dict[str, tuple[str, ...]]:
    """Legacy leaf -> canonical path within ``block``, derived from ``RENAMES``.

    The path is returned whole and may be any depth: ``gradient.clip.value`` is
    ``("gradient", "clip", "value")``. Only a cross-block move is left out, and
    that one genuinely has no home in a single-block stub.

    A depth cap here is not a conservative choice, it is a silent one. Capping
    at one level dropped the three ``optimization.gradient.clip.*`` records, and
    a dropped record does not fail loudly -- it lands as a stray direct
    attribute (``optimization.gradient_clip_value``) while the reader walks
    ``optimization.gradient.clip.value`` and gets the schema default. The test
    then pins the default it never chose.
    """
    from spectramr.config.schemas.renames import RENAMES

    prefix = f"{block}."
    out: dict[str, tuple[str, ...]] = {}
    for legacy, record in RENAMES.items():
        if not legacy.startswith(prefix):
            continue
        leaf = legacy[len(prefix) :]
        if "." in leaf:
            continue
        canonical = record.canonical
        if not canonical.startswith(prefix):
            continue
        rest = tuple(canonical[len(prefix) :].split("."))
        if len(rest) >= 2:
            out[leaf] = rest
    return out


def _nested_update(obj: Any, path: tuple[str, ...], value: Any, *, block: str) -> Any:
    """Return ``obj`` with ``path`` set to ``value``, rebuilding frozen models.

    Walks down to the owning sub-block, then rebuilds back up: every level is
    frozen like a live config, so each one needs its own ``model_copy``.
    """
    head, rest = path[0], path[1:]
    current = getattr(obj, head, None)
    if current is None:
        raise AttributeError(
            f"{block}.{head} is not a sub-block on {type(obj).__name__}; "
            "RENAMES points at a path the schema does not have"
        )
    if len(rest) == 1:
        updated = current.model_copy(update={rest[0]: value})
    else:
        updated = _nested_update(current, rest, value, block=f"{block}.{head}")
    return obj.model_copy(update={head: updated})


def block_stub(block: str, **kwargs: Any) -> Any:
    """A real schema instance for ``block``, with legacy kwargs routed home.

    Sub-blocks are frozen, so updates go through ``model_copy``. That bypasses
    validation by design -- a test double may need a value the schema would
    reject, and the point here is the SHAPE the reader walks, not the value.
    """
    schema = _schema(block)
    routes = flat_to_canonical(block)

    obj = schema()
    direct: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in routes:
            obj = _nested_update(obj, routes[key], value, block=block)
        else:
            direct[key] = value

    if direct:
        obj = obj.model_copy(update=direct)
    return obj
