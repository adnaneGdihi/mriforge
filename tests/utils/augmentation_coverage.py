"""The single owner of "which augmentation flag is actually wired".

Two suites ask this question -- ``tests/unit/data/transforms/
test_augmentation_factory.py`` and ``tests/unit/config/
test_augmentation_schema_coverage.py``.  Before this module they answered it
two different ways, and both were wrong in the same direction:

* the coverage suite computed ``all_flags - CONSUMED_BY_FACTORY``, deriving
  "actually unconsumed" from the very ledger it was auditing -- a tautology
  that could never notice a flag becoming consumed;
* a companion check matched ``\\.enable_(\\w+)`` against the factory source,
  which scores a flag "consumed" for being *mentioned*, not for producing a
  transform.

Non-negotiable 17: elect one owner and delete the loser's enforcement, rather
than keeping a weaker checker as defence in depth.  The predicate here is
behavioural -- build the pipeline and look at what came out -- because that is
the only form that would have caught the elided factory sections it replaced.
"""

from __future__ import annotations

from collections.abc import Callable

import torchio as tio

from mriforge.config.schemas.augmentation import AugmentationConfigSchema
from mriforge.data.transforms.augmentation_factory import TorchIOAugmentationFactory

BuildFn = Callable[[AugmentationConfigSchema, str], "tio.Compose | None"]

#: Flags the factory deliberately builds no per-sample transform for.  An entry
#: is a documented decision, not an omission, and ``NOT_PER_SAMPLE`` is audited
#: for staleness by the suites that import it.
NOT_PER_SAMPLE: dict[str, str] = {
    "enable_mixup": "batch-level op; needs two samples, not a tio.Transform",
    "enable_cutmix": "batch-level op; needs two samples, not a tio.Transform",
    "enable_kspace_undersampling_augmentation": (
        "deferred to the physics lane: must route through fft_ops.fft2c "
        "(non-negotiable 2) and must be reported applied-vs-declared in the "
        "debug snapshot (non-negotiable 14)"
    ),
}


def enable_flags() -> list[str]:
    """Every ``enable_*`` knob the augmentation schema exposes."""
    return sorted(f for f in AugmentationConfigSchema.model_fields if f.startswith("enable_"))


def real_build(config: AugmentationConfigSchema, dataset_type: str):
    """The production factory, in the signature ``unwired_flags`` expects."""
    return TorchIOAugmentationFactory.build(config, dataset_type=dataset_type)


def unwired_flags(build: BuildFn = real_build, dataset_type: str = "image") -> set[str]:
    """Flags that produce no transform when they are the only one enabled.

    Parameterised by ``build`` so a planted-violation stub exercises *this*
    predicate rather than a copy of it (non-negotiable 15).
    """
    unwired: set[str] = set()
    for flag in enable_flags():
        config = AugmentationConfigSchema(enabled=True, **{flag: True})
        result = build(config, dataset_type)
        if result is None or not result.transforms:
            unwired.add(flag)
    return unwired


def build_ignoring(*ignored: str) -> BuildFn:
    """A planted-violation stub: the real factory with some flags forced off."""

    def build(config: AugmentationConfigSchema, dataset_type: str):
        patched = config.model_copy(update=dict.fromkeys(ignored, False))
        return TorchIOAugmentationFactory.build(patched, dataset_type=dataset_type)

    return build


__all__ = [
    "NOT_PER_SAMPLE",
    "build_ignoring",
    "enable_flags",
    "real_build",
    "unwired_flags",
]
