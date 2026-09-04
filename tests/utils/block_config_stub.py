"""Config stand-ins for the decomposed ``validation:`` and ``logging:`` blocks.

The sibling of :mod:`tests.utils.data_config_stub`, for the two other blocks the
15-phase decomposition split into sub-blocks. Same hazard, same rule.

Nineteen tests were red because their hand-rolled ``SimpleNamespace`` predated
the split and still carried the leaves flat::

    AttributeError: 'types.SimpleNamespace' object has no attribute 'snapshots'   x12
    AttributeError: 'types.SimpleNamespace' object has no attribute 'scoring'     x7

That is #714, and the count is the point: a stub built inline is a **second
implementation of the schema**, so one rename breaks it in as many places as it
was copied. Two of those nineteen are the only guards for the #371/#390
wrong-channel visualization family, which means the rename did not merely turn
tests red -- it disarmed a live mechanism check while looking like fixture noise.

Unlike ``DataConfigStub`` this module does **not** restate the block list. It
reads it off the schema: any field whose annotation is itself a ``BaseModel`` is
a sub-block. A phase that adds one is picked up with no edit here, which is the
property the flat stubs lacked.

Usage::

    from tests.utils.block_config_stub import LoggingConfigStub, ValidationConfigStub

    logging_cfg = LoggingConfigStub(snapshots={"enabled": True, "interval_steps": 100})
    logging_cfg.snapshots.enabled          # True
    logging_cfg.intervals.log_interval     # schema default, not a guess

    val = ValidationConfigStub(scoring={"compute": ["psnr", "ssim"]})
    val.scoring.compute                    # ['psnr', 'ssim']
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

from spectramr.config.schemas.logging import LoggingConfigSchema
from spectramr.config.schemas.validation import ValidationConfigSchema

__all__ = [
    "BlockConfigStub",
    "LoggingConfigStub",
    "ValidationConfigStub",
    "sub_blocks",
]


def _sub_block_schemas(schema: type[BaseModel]) -> dict[str, type[BaseModel]]:
    """``{field_name: sub_schema}`` for every BaseModel-typed field.

    Derived, not listed. A hand-kept list is the thing this module exists to
    stop being: it agrees with the schema until the phase that moves something.
    """
    out: dict[str, type[BaseModel]] = {}
    for name, info in schema.model_fields.items():
        ann = info.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            out[name] = ann
    return out


def sub_blocks(schema: type[BaseModel], **override: Any) -> dict[str, Any]:
    """Every sub-block of ``schema``, instantiated at its REAL defaults.

    ``override`` takes either a constructed block or a dict of leaf overrides;
    a dict is fed to the real sub-schema, so an unknown leaf raises here rather
    than becoming an attribute no production reader looks at.
    """
    schemas = _sub_block_schemas(schema)
    blocks: dict[str, Any] = {name: cls() for name, cls in schemas.items()}
    for name, value in override.items():
        if name not in schemas:
            raise KeyError(
                f"{schema.__name__} has no sub-block {name!r}. "
                f"Available: {sorted(schemas)}"
            )
        blocks[name] = schemas[name](**value) if isinstance(value, dict) else value
    return blocks


class BlockConfigStub(SimpleNamespace):
    """Stand-in for a decomposed config block, carrying every sub-block.

    Subclasses bind ``_SCHEMA``. Keyword arguments name a sub-block and give
    either a constructed instance or a dict of leaf overrides.
    """

    _SCHEMA: type[BaseModel]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**sub_blocks(self._SCHEMA, **kwargs))


class ValidationConfigStub(BlockConfigStub):
    """Stand-in for ``config.validation`` (schedule / loader / scoring / ...).

    ``validation.metrics`` became ``validation.scoring.compute`` in the phase-10a
    decomposition. Reading it off the schema means a stub cannot keep the old
    spelling alive after the move.
    """

    _SCHEMA = ValidationConfigSchema


class LoggingConfigStub(BlockConfigStub):
    """Stand-in for ``config.logging`` (identity / sinks / snapshots / ...).

    Note ``LoggingConfigSchema`` is ``extra="ignore"`` in production, so a
    mistyped key there VANISHES rather than raising. That makes an accurate
    stand-in more load-bearing here than elsewhere: the schema will not catch
    the drift for you.
    """

    _SCHEMA = LoggingConfigSchema
