r"""Shared contrast-conditioning helper for field-aware FiLM generators.

The mrixfields corpus mixes three contrasts (T1w / T2w / T2-FLAIR) at every
field strength, and the dataset always emits a per-sample ``contrast_id``
(see ``data.datasets.mrixfields_dataset._CONTRAST_INDEX``). A field-FiLM
generator becomes *contrast-aware* by widening its :class:`FieldFiLMBlock`
``sequence_dim`` by ``num_contrasts`` and concatenating a contrast one-hot
into the FiLM ``sequence_features`` vector — the field strength itself stays
the separate leading scalar of :class:`FieldFiLMBlock` (the ``1 +`` in its
input ``Linear``).

This module is the single home for that construction so every generator
enforces the same two always-on invariants (CLAUDE.md):

* **#15 (wire the knob):** ``use_contrast_conditioning=True`` with no
  ``contrast_id`` in the batch **raises** — a wired knob never silently
  degrades to unconditional mode-averaging.
* **#9 (no silent fallback):** an out-of-range contrast id **raises** (via
  :func:`torch.nn.functional.one_hot`), never a silent clamp. The one-hot is
  built without a ``.item()`` host sync so it is training-loop safe
  (``performance.md``).

Two shapes of base sequence are supported:

* **No intrinsic sequence features** (``base_seq_dim == 0``): the field
  scalar is the only conditioning. Off-mode uses a single zero placeholder
  (``FieldFiLMBlock`` requires ``sequence_dim >= 1``); on-mode uses the
  contrast one-hot alone. This reproduces ``FieldVelocityUNet``'s original
  ``seq_dim = num_contrasts if use_contrast_conditioning else 1``.
* **Real base features** (``base_seq_dim >= 1``, e.g. a diffusion time
  embedding or a source/target field pair): the base is PRESERVED and the
  contrast one-hot is APPENDED, so contrast conditioning never clobbers the
  existing conditioning channel.
"""

from __future__ import annotations

import torch
from torch.nn.functional import one_hot


def contrast_sequence_dim(
    base_seq_dim: int,
    num_contrasts: int,
    enabled: bool,
) -> int:
    """FiLM ``sequence_dim`` after optionally appending a contrast one-hot.

    Args:
        base_seq_dim: Width of the model's intrinsic FiLM sequence features
            (``0`` when the model conditions on the field scalar only).
        num_contrasts: Number of contrasts in the one-hot (>= 2 when enabled).
        enabled: Whether contrast conditioning is on.

    Returns:
        The ``sequence_dim`` to pass to :class:`FieldFiLMBlock` (always >= 1).
    """
    if base_seq_dim < 0:
        raise ValueError(f"base_seq_dim must be >= 0; got {base_seq_dim}")
    if enabled:
        if num_contrasts < 2:
            raise ValueError(f"contrast conditioning needs num_contrasts >= 2; got {num_contrasts}")
        return base_seq_dim + num_contrasts
    return base_seq_dim if base_seq_dim > 0 else 1


def build_contrast_sequence(
    base_seq: torch.Tensor | None,
    contrast_id: torch.Tensor | None,
    num_contrasts: int,
    enabled: bool,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the ``sequence_features`` tensor for :class:`FieldFiLMBlock`.

    Args:
        base_seq: The model's intrinsic sequence features ``[B, base_seq_dim]``
            or ``None`` when the model has no base features (field-scalar only).
        contrast_id: Per-sample contrast index ``[B]`` (long). Required when
            ``enabled``; ignored otherwise.
        num_contrasts: One-hot width.
        enabled: Whether contrast conditioning is on.
        batch_size: Batch size ``B`` — used only to size the off-mode zero
            placeholder when ``base_seq is None``.
        device / dtype: Target device/dtype for a constructed tensor.

    Returns:
        ``[B, sequence_dim]`` matching :func:`contrast_sequence_dim`.

    Raises:
        ValueError: ``enabled`` but ``contrast_id is None`` (#15).
        RuntimeError: an out-of-range / negative id (#9, from ``one_hot``).
    """
    if not enabled:
        if base_seq is not None:
            return base_seq
        return torch.zeros(batch_size, 1, device=device, dtype=dtype)

    if contrast_id is None:
        raise ValueError(
            "contrast conditioning is enabled but the batch carries no "
            "'contrast_id'. The mrixfields dataset emits contrast_id — thread "
            "it through the strategy to the model forward (CLAUDE.md #15)."
        )
    # one_hot raises on negative / >= num_contrasts ids (no silent clamp, #9)
    # and avoids a host sync (no .item()) in the forward path.
    cid = contrast_id.reshape(-1).long()
    onehot = one_hot(cid, num_contrasts).to(device=device, dtype=dtype)
    if base_seq is None:
        return onehot
    return torch.cat([base_seq, onehot], dim=-1)


__all__ = ["build_contrast_sequence", "contrast_sequence_dim"]
