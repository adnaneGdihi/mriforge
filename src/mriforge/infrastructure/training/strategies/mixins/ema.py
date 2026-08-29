"""EMA mixin marker.

Historical note: this mixin used to expose ``_update_ema``. The body was
``pass`` and nothing called it — the actual EMA shadow update runs at
``src/pipelines/train.py`` (``pipeline.ema.update(pipeline.generator)``)
once per training step. The dead method has been removed per
``TODO/audit/05_strategies_core_mixins_builders.md`` F3.

The class is kept (empty) so ``BaseTrainingStrategy``'s MRO and any
``isinstance`` checks elsewhere remain stable.
"""


class EMAMixin:
    """Marker mixin. EMA updates live in ``pipelines/train.py``."""
