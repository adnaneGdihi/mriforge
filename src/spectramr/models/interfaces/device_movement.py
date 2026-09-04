"""The one device capability a model-layer component may depend on.

Why this exists
---------------

``models/pipelines/generation_pipeline.py`` imported ``DevicePolicy`` and two of
its factories from ``spectramr.infrastructure.services.device_policy``. That is an
upward import: the dependency order is
``infrastructure/ -> models/, domain/ -> core/, config/`` (non-negotiable 5), so
``models/`` may not reach into ``infrastructure/``.

It is worth recording HOW the violation arrived, because it was not carelessness.
The import used to read ``from ....services.device_policy import ...`` -- four
dots from ``spectramr/models/pipelines/``, which walks past the top-level package
and raises ``ImportError``. The only consumer wrapped it in
``except ImportError: generation_pipeline_module = None``, so a broken module was
indistinguishable from an absent optional dependency, in every environment
(non-negotiable 18). Repairing that import to an absolute one is what made the
upward dependency *real* rather than merely written down. The layering gate had
nothing to see while the module could not be imported at all.

Why a Protocol, and not a function-local import
-----------------------------------------------

Every grep in ``scripts/ci/check_layering.sh`` is anchored at ``^``, so moving the
import inside a function would make the gate green while leaving the dependency
exactly where it was -- the detector-blindness shape non-negotiable 15 is about
(#1183). Silencing a gate you can see through is worse than the violation,
because the next reader has no way to find it.

Why a Protocol and not a move to ``core/``
------------------------------------------

``device_policy.py`` imports only stdlib, torch and ``spectramr.core.compute_device``,
so it *could* live in ``core/`` and that would fix this violation plus one already
in the baseline. That is a relocation of a 392-line module the DI services depend
on, and a bigger decision than this repair -- left as an option, not taken here.

What the consumer actually needs
--------------------------------

``GenerationPipeline`` calls exactly one method on the objects it is handed:
``move_to_device``. Nothing else. So the seam is one method wide, and the
concrete ``DevicePolicy`` still satisfies it structurally -- no registration, no
subclassing, no change at the injection site.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["SupportsDeviceMovement"]


@runtime_checkable
class SupportsDeviceMovement(Protocol):
    """Anything that can move a tensor / module / container onto a device.

    ``infrastructure.services.device_policy.DevicePolicy`` satisfies this
    structurally; so does any test double with the same one method, which is
    what lets the model layer be exercised without constructing an
    infrastructure service.

    The signature matches ``DevicePolicy.move_to_device`` including its optional
    arguments, so a caller written against the Protocol can be handed the real
    policy and behave identically.
    """

    def move_to_device(
        self,
        data: Any,
        dtype: Any = None,
        non_blocking: bool = False,
    ) -> Any:
        """Return ``data`` on this policy's device."""
        ...
