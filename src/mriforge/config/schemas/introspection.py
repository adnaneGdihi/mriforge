"""What models a field annotation can actually resolve to.

Every reflective consumer of the schema tree -- the key-reference generator, the
rename-table validator, the unconsumed-key census, the phantom-key check --
needs to answer one question about a field: *if I descend into this, which
model classes might I land in?* Each of them answered it independently, and each
of them answered it the same wrong way::

    nested = [
        a for a in [field.annotation, *typing.get_args(field.annotation)]
        if isinstance(a, type) and issubclass(a, BaseModel)
    ]
    model = nested[0] if nested else None

That looks exhaustive and is one level deep. It handles ``X`` and ``X | None``,
which was every field in the tree until ``training.diffusion`` became a
discriminated union in 2026-08. Its annotation is::

    Optional[Annotated[UnspecifiedParams | ColdParams | ... , FieldInfo(discriminator='type')]]

``get_args`` on that yields ``(Annotated[...], NoneType)`` -- and ``Annotated[...]``
is not a class, so ``issubclass`` never fires and the walk finds **nothing**.
Six independent introspection sites went silently blind at once: not by
erroring, but by concluding the block had no fields, which reads exactly like a
block that legitimately has none.

Two properties are worth stating because the callers depend on them:

* **Recursive, not one level.** Annotated wraps unions wrap models, and a future
  ``list[Model]`` or ``dict[str, Model]`` nests further still.
* **Order is deterministic** (depth-first, declaration order), because a caller
  that picks ``[0]`` must not have its answer depend on set iteration.

A caller that genuinely wants one model still has to decide WHICH -- picking
``[0]`` off a union means "the first variant", which for a discriminated union
is an arbitrary choice. Prefer checking all of them; :func:`nested_models`
returns every candidate so that is the easy path.
"""

from __future__ import annotations

import typing

from pydantic import BaseModel

__all__ = ["nested_models"]


def nested_models(annotation: object) -> list[type[BaseModel]]:
    """Every ``BaseModel`` subclass reachable from ``annotation``.

    Unwraps ``Optional``, ``Annotated``, unions and generic containers to any
    depth. Returns ``[]`` when the annotation resolves to no model, which is the
    honest answer for a leaf field.

    >>> from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema
    >>> ann = TrainingStrategyConfigSchema.model_fields["diffusion"].annotation
    >>> [m.__name__ for m in nested_models(ann)][:2]
    ['UnspecifiedParams', 'ColdParams']
    """
    found: list[type[BaseModel]] = []
    seen: set[int] = set()

    def _walk(node: object) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))

        if isinstance(node, type) and issubclass(node, BaseModel):
            if node not in found:
                found.append(node)
            # Deliberately do NOT descend into the model's own fields: the
            # caller is walking a path one segment at a time and will ask again
            # for the next segment. Descending here would collapse the levels
            # and let `a.b` resolve against a field that only exists on `a.c`.
            return

        for arg in typing.get_args(node):
            _walk(arg)

    _walk(annotation)
    return found
