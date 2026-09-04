"""`BaseTrainingConfigSchema` is a second config root that nothing can reach.

It declared a full duplicate of the top-level blocks — `data`, `model`,
`optimization`, `logging`, `validation`, … — inherited by 20 strategy
subclasses. That made ``training.gan.data.batch_size`` look like legal,
meaningful YAML. It was neither meaningful nor, in the way anyone assumed,
legal:

* **Unreachable.** A recursive annotation walk from ``TrainingSettings`` reaches
  268 model classes, and neither this class nor any of its 20 subclasses is
  among them. No instance is ever constructed on any config path.
* **Unconstructible.** All 20 subclasses raise on ``cls()`` — measured before and
  after the field deletion, 0/20 either way.
* **Unused by the corpus.** Zero arms under ``experiments/`` declare any
  duplicated block under ``training.<paradigm>:``.

So the 17 duplicated fields were deleted (2026-08-02). This file pins the three
facts that make that safe, because each of them could silently stop being true.

**What the deletion does NOT do:** it does not stop
``training.gan.data.batch_size`` being *accepted*. That is
``TrainingStrategyConfigSchema`` being ``extra="allow"`` and
``GANSubConfigSchema`` being ``extra="ignore"`` — the key is swallowed, not
validated, and this class was never involved. Declaring the paradigm blocks
(``strategy_knobs_2026_08``) is what narrows that.
"""

from __future__ import annotations

import typing

import pytest
from pydantic import BaseModel

from spectramr.config.schemas.training.base import BaseTrainingConfigSchema
from spectramr.config.settings import TrainingSettings

#: The blocks that were duplicated. Re-adding any of them re-creates the second
#: root, so the test below names them rather than checking a count.
DELETED_DUPLICATES = frozenset(
    {
        "acquisition",
        "artifacts",
        "certification",
        "checkpoint",
        "data",
        "early_stopping",
        "ema",
        "logging",
        "loss_logging",
        "metadata",
        "metrics",
        "model",
        "optimization",
        "parallel",
        "physics",
        "training",
        "validation",
    }
)


def _reachable_models() -> set[type[BaseModel]]:
    """Every pydantic model reachable from `TrainingSettings` by annotation.

    Flattens `get_args` recursively, so `X | None`, `list[X]` and
    `dict[str, X]` are all followed — a shallower walk would report a class
    unreachable merely because it sits behind a container.
    """
    seen: set[type[BaseModel]] = set()

    def walk(model: type[BaseModel]) -> None:
        if model in seen:
            return
        seen.add(model)
        for field in model.model_fields.values():
            stack = [field.annotation]
            while stack:
                ann = stack.pop()
                stack.extend(typing.get_args(ann))
                if isinstance(ann, type) and issubclass(ann, BaseModel):
                    walk(ann)

    walk(TrainingSettings)
    return seen


class TestTheSecondRootStaysDead:
    def test_the_walk_is_not_vacuous(self) -> None:
        """A walk that reached almost nothing would pass every test below."""
        reachable = _reachable_models()
        assert len(reachable) > 200, f"only {len(reachable)} models reached"
        assert TrainingSettings in reachable

    def test_it_is_unreachable_from_training_settings(self) -> None:
        assert BaseTrainingConfigSchema not in _reachable_models(), (
            "BaseTrainingConfigSchema is now reachable from TrainingSettings. "
            "Its fields would start being constructed, so the deletion's premise "
            "no longer holds — re-audit before relying on it."
        )

    def test_no_subclass_is_reachable_either(self) -> None:
        reachable = _reachable_models()
        subs = BaseTrainingConfigSchema.__subclasses__()
        assert subs, "no subclasses found — the detector has stopped working"
        live = [c.__name__ for c in subs if c in reachable]
        assert not live, f"these subclasses are now reachable: {live}"

    @pytest.mark.parametrize("block", sorted(DELETED_DUPLICATES))
    def test_the_duplicated_block_is_gone(self, block: str) -> None:
        assert block not in BaseTrainingConfigSchema.model_fields, (
            f"`{block}` is back on BaseTrainingConfigSchema, re-creating the "
            f"second root: `training.<paradigm>.{block}` becomes legal-looking "
            f"YAML that nothing reads."
        )

    def test_no_field_shadows_a_top_level_block(self) -> None:
        """The general form: any overlap at all is a second root."""
        overlap = sorted(
            set(BaseTrainingConfigSchema.model_fields)
            & set(TrainingSettings.model_fields)
        )
        assert not overlap, f"duplicated root blocks are back: {overlap}"
