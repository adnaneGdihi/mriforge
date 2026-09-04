"""Does the recorded model-input snapshot describe the tensor the model was fed?

Non-negotiable 14's carve-out lets a strategy that degrades its input INSIDE the
step point a reader at a second snapshot (``snapshot_model_input_tag``). Until
#1298 the enforcement checked only that the snapshot ARRIVED, so it was
satisfied by one naming the wrong tensor: ``diffusion.py`` stamped
``model_input_key = "noisy_kspace"`` unconditionally while the backbone was fed
the 16-channel ``cat([noisy_images, smaps])``. The recorded snapshot did not
contain the model input at all -- it held the 8-channel k-space half -- and
every contract test passed, because presence was the whole test.

This module is the content half, in two tiers that follow the same static /
artifact split :meth:`_snapshot_declared_model_input` already draws:

* **naming** -- ``model_input_key`` must be present and must name a key that is
  actually in the snapshot. Both are wrong before any run starts, so the caller
  RAISES (:func:`require_model_input_key`).
* **width** -- the named tensor's channel count against the backbone's declared
  input width. This one WARNS and stamps (:func:`verify_model_input`), because
  the resolution is a heuristic: a model whose first layer is neither conv nor
  linear leaves the width unresolved, and a complex tensor legitimately feeds a
  real-view conv of twice its channel count. Aborting a training run over a
  diagnostic's false positive is the worse failure.

An unresolved width is reported as ``unresolved``, never dropped -- a check that
silently declines to run is indistinguishable from one that passed (pitfall
#16), which is the exact shape of the defect this module exists to close.

Everything here is pure and shape-only: no ``.item()``, no host copy, no forward
pass. The channel-axis convention is NCHW/NCDHW, matching every other
channel-aware consumer in this package.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, NamedTuple

import torch
from torch import nn

#: Channel axis for the tensors this contract inspects (N, C, ...).
CHANNEL_DIM: Final[int] = 1

#: Verdicts. ``unresolved`` is a first-class outcome, not a silent skip.
STATUS_MATCH: Final[str] = "match"
STATUS_MISMATCH: Final[str] = "mismatch"
STATUS_UNRESOLVED: Final[str] = "unresolved"


class ModelInputVerdict(NamedTuple):
    """The declared-vs-applied comparison, in the form that reaches the artifact."""

    model_input_key: str
    declared_channels: int | None
    declared_shape: tuple[int, ...] | None
    is_complex: bool
    model_in_channels: int | None
    in_channels_source: str
    accepted_channels: tuple[int, ...]
    status: str
    detail: str

    def as_record(self) -> dict[str, Any]:
        """The nested block stamped into ``snapshot.json`` / ``snapshot.txt``."""
        return {
            "model_input_key": self.model_input_key,
            "declared": {
                "channels": self.declared_channels,
                "shape": list(self.declared_shape) if self.declared_shape else None,
                "is_complex": self.is_complex,
            },
            "applied": {
                "in_channels": self.model_in_channels,
                "resolved_from": self.in_channels_source,
                "accepted": list(self.accepted_channels),
            },
            "status": self.status,
            "detail": self.detail,
        }


def require_model_input_key(
    *,
    strategy_name: str,
    tag: str,
    tensors: Mapping[str, Any],
    model_input_key: str | None,
) -> str:
    """Return the key naming the model input, or RAISE if the naming is broken.

    Raises rather than warns because both failures are static: a class that
    never names its model input, and one that names a key it does not emit, are
    both wrong before a single batch is loaded. The message names the tag so a
    reader lands on the right emitter rather than on the base wrapper.
    """
    if not model_input_key:
        raise ValueError(
            f"{strategy_name} emits the model-input snapshot {tag!r} without "
            "naming which of its tensors the model is actually fed. The tag "
            "tells a reader 'the real model input is in here'; without a key "
            f"they must guess between {sorted(tensors)!r}, and a guess is what "
            "let #1298's mislabel stand. Pass model_input_key=<one of those "
            "keys> to save_debug_snapshot (or put it in the extra dict handed "
            "to _declare_model_input)."
        )
    if model_input_key not in tensors:
        raise ValueError(
            f"{strategy_name} names {model_input_key!r} as the model input of "
            f"snapshot {tag!r}, but that snapshot holds {sorted(tensors)!r}. "
            "The artifact points at a tensor it does not contain, so nothing "
            "in it records what the model was fed (non-negotiable 14)."
        )
    return model_input_key


def resolve_model_in_channels(module: Any) -> tuple[int | None, str]:
    """Best-effort read of the width consumed by whatever eats the model input.

    Preference order, most authoritative first:

    1. ``backbone_in_channels`` -- a WRAPPER's explicit statement of the width
       its inner backbone consumes, which is what the snapshotted
       ``model_input`` is compared against;
    2. a module-level ``in_channels`` attribute -- what the architecture itself
       says it takes, and the only source that survives an unusual first layer;
    3. the first ``Conv{1,2,3}d`` in ``modules()`` order;
    4. the first ``Linear``'s ``in_features``.

    Tier 1 exists because tiers 2-4 answer a subtly different question for a
    module that adapts channels before delegating. ``KSpaceColdDiffusionGenerator``
    sets ``in_channels = 8`` (the bare measurement its ``forward`` concat gate
    keys on) while building its backbone at 16 for the concatenated
    ``[noisy_kspace || smaps]`` stack the training path actually feeds it. Read
    as "the declared input width", 8 is correct; read as "the width the tensor
    under test must match", it is wrong, and the mismatch fired on every
    smaps-conditioned cold-diffusion arm. A wrapper that adapts channels knows
    both numbers -- this tier is where it says which one this check means.

    Returns ``(None, "unresolved")`` for anything else (a transformer patchifier,
    a functional first op, a mocked module). Fail-open by construction: this is
    a diagnostic, and every caller treats ``None`` as "cannot say", never as
    "mismatch".
    """
    if module is None:
        return None, "unresolved"

    backbone = getattr(module, "backbone_in_channels", None)
    if isinstance(backbone, int) and not isinstance(backbone, bool) and backbone > 0:
        return backbone, "module.backbone_in_channels"

    declared = getattr(module, "in_channels", None)
    if isinstance(declared, int) and not isinstance(declared, bool) and declared > 0:
        return declared, "module.in_channels"

    try:
        children = list(module.modules())
    except Exception:  # pragma: no cover - a mock without .modules()
        return None, "unresolved"

    for sub in children:
        if isinstance(sub, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return int(sub.in_channels), f"first {type(sub).__name__}"
    for sub in children:
        if isinstance(sub, nn.Linear):
            return int(sub.in_features), "first Linear.in_features"
    return None, "unresolved"


def verify_model_input(
    *,
    tensors: Mapping[str, Any],
    model_input_key: str,
    in_channels: int | None,
    in_channels_source: str,
) -> ModelInputVerdict:
    """Compare the named tensor's channel count against the backbone's width.

    A complex tensor accepts ``{C, 2C}``: half this repo's k-space backbones take
    a real view of a complex input, doubling the channel axis, and the snapshot
    records the tensor as the strategy holds it -- before that view is taken.
    Accepting only ``C`` would make every complex-input arm report a mismatch,
    and a check that cries wolf on a whole paradigm gets muted rather than read.

    Takes the ALREADY-RESOLVED width rather than the module: the caller runs on
    every training step and :func:`resolve_model_in_channels` walks ``modules()``,
    so the resolution is cached there once (non-negotiable 9) and this function
    stays pure over its arguments.
    """
    tensor = tensors.get(model_input_key)
    source = in_channels_source

    if not isinstance(tensor, torch.Tensor) or tensor.dim() <= CHANNEL_DIM:
        return ModelInputVerdict(
            model_input_key=model_input_key,
            declared_channels=None,
            declared_shape=tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else None,
            is_complex=False,
            model_in_channels=in_channels,
            in_channels_source=source,
            accepted_channels=(),
            status=STATUS_UNRESOLVED,
            detail=(
                f"{model_input_key!r} has no channel axis to compare "
                "(not a tensor, or fewer dims than NCHW)"
            ),
        )

    channels = int(tensor.shape[CHANNEL_DIM])
    is_complex = bool(tensor.is_complex())
    accepted = (channels, 2 * channels) if is_complex else (channels,)

    if in_channels is None:
        return ModelInputVerdict(
            model_input_key=model_input_key,
            declared_channels=channels,
            declared_shape=tuple(tensor.shape),
            is_complex=is_complex,
            model_in_channels=None,
            in_channels_source=source,
            accepted_channels=accepted,
            status=STATUS_UNRESOLVED,
            detail=(
                "the backbone declares no in_channels and its first layer is "
                "neither Conv nor Linear, so the width could not be read"
            ),
        )

    if in_channels in accepted:
        return ModelInputVerdict(
            model_input_key=model_input_key,
            declared_channels=channels,
            declared_shape=tuple(tensor.shape),
            is_complex=is_complex,
            model_in_channels=in_channels,
            in_channels_source=source,
            accepted_channels=accepted,
            status=STATUS_MATCH,
            detail=(
                f"{model_input_key!r} carries {channels} channels"
                + (" (complex; a real view doubles it)" if is_complex else "")
                + f" and the backbone takes {in_channels}"
            ),
        )

    return ModelInputVerdict(
        model_input_key=model_input_key,
        declared_channels=channels,
        declared_shape=tuple(tensor.shape),
        is_complex=is_complex,
        model_in_channels=in_channels,
        in_channels_source=source,
        accepted_channels=accepted,
        status=STATUS_MISMATCH,
        detail=(
            f"{model_input_key!r} carries {channels} channels "
            f"(accepting {list(accepted)}) but the backbone takes {in_channels}, "
            f"read from {source}. Either the key names the wrong tensor -- the "
            "snapshot then records something the model never saw -- or the "
            "wiring feeds the backbone a width it did not declare."
        ),
    )


__all__ = [
    "CHANNEL_DIM",
    "STATUS_MATCH",
    "STATUS_MISMATCH",
    "STATUS_UNRESOLVED",
    "ModelInputVerdict",
    "require_model_input_key",
    "resolve_model_in_channels",
    "verify_model_input",
]
