"""Device-resident memoisation of the deterministic k-space mask cascade.

Why this exists
---------------
:meth:`KSpaceMaskGenerator.generate_batch_masks` used to resolve a batch of
masks by pulling the timestep tensor to the host::

    timestep_list = timesteps.detach().to("cpu", non_blocking=False).tolist()

That is a blocking device-to-host copy, so it drains every kernel queued ahead
of it. A Scalene profile of ``experiment_11_attention_none`` (300 iterations,
V100-PCIE-32GB) charged **24.24 % of the whole run — ~225 s — to that single
line**, while the line that actually *builds* a mask cost 0.8 s. The cost was
never mask construction; it was ~92 synchronisation points, one per
reverse-diffusion step of a three-rung validation cascade.

Why memoising is safe
---------------------
On the fixed-seed branch a mask is a **pure function** of
``(seed, pattern, shape, acceleration_factor, enforce_nested, timestep)``:

* every RNG site in ``infrastructure/physics/sampling.py`` builds a *fresh*
  local ``torch.Generator`` seeded from ``self.seed`` plus a fixed offset, so no
  RNG state is carried between calls;
* ``VariableDensityKSpaceAccelerator`` already memoises its priority ranking as
  timestep-invariant, and derives each timestep's mask by truncating it.

The table is therefore built by calling the *existing* per-timestep resolver
once per timestep, which makes parity true **by construction** rather than by
assertion — this cache cannot drift from the code it caches, because it is that
code (non-negotiable 17: one owner per invariant).

What is deliberately NOT cached
-------------------------------
``KSpaceUndersamplingProcess._generate_batch_masks_dynamic`` — the
``enable_dynamic_mask`` *training* path — mutates the inner accelerator's seed
per sample so each sample draws a different pattern. Its masks are not a
function of the timestep alone and must never be served from here. That path
does not call :meth:`generate_batch_masks` at all, and the seed and
``enforce_nested`` it temporarily mutates are both part of the cache key, so a
table built during its window can never be mistaken for a fixed-seed one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


class MaskTableCache:
    """Cache of ``[T, 1, H, W]`` mask cascades, keyed by everything that moves.

    One instance is owned per :class:`KSpaceMaskGenerator`; entries are keyed so
    that any change which alters mask *content* yields a different table rather
    than a stale hit.
    """

    def __init__(self) -> None:
        self._tables: dict[tuple[Any, ...], torch.Tensor] = {}

    @staticmethod
    def build_key(
        acceleration_type: str,
        image_shape: tuple[int, int],
        acceleration_factor: int,
        device: torch.device,
        accelerator: Any,
    ) -> tuple[Any, ...]:
        """Key covering every input that changes mask content.

        The seed is read off the **inner** accelerator, not the
        ``ColdDiffusionAccelerator`` wrapper: the wrapper exposes ``seed`` as a
        read-only property while the concrete accelerator it wraps carries the
        settable one, so reading the wrapper would key every table on the same
        vacuous value and serve a stale cascade after a seed change.

        ``acceleration_type`` is the RESOLVED name, not the spelling the caller
        wrote. 31 accepted pattern names collapse onto 19 canonical types, 9 of
        which have more than one spelling (``random`` / ``cartesian_random`` /
        ``random_cartesian``), and ``KSpaceMaskGenerator._get_accelerator``
        already caches one accelerator instance per resolved type -- so two
        spellings share an accelerator and therefore produce identical masks.
        Keying on the raw spelling built and held a second, bit-identical table
        for the same cascade. Merging them is safe for exactly that reason and
        for no weaker one: if accelerators were per-spelling, two spellings could
        carry divergent kwargs and one table would serve the wrong mask.
        """
        inner = getattr(accelerator, "accelerator", None)
        holder = accelerator if inner is None else inner
        return (
            str(acceleration_type),
            int(image_shape[0]),
            int(image_shape[1]),
            int(acceleration_factor),
            str(device),
            getattr(holder, "seed", None),
            bool(getattr(accelerator, "enforce_nested", False)),
        )

    def table_for(
        self,
        key: tuple[Any, ...],
        num_timesteps: int,
        build_one: Callable[[int], torch.Tensor],
    ) -> torch.Tensor:
        """Return the cascade for ``key``, building it once on first request.

        ``build_one`` is the caller's existing single-timestep resolver; it is
        invoked exactly ``num_timesteps`` times per key and never again. Masks
        are stacked verbatim, so dtype, device and the rank-3 ``(1, H, W)``
        per-timestep contract are inherited rather than re-declared.
        """
        table = self._tables.get(key)
        if table is None:
            table = torch.stack([build_one(t) for t in range(num_timesteps)], dim=0)
            self._tables[key] = table
        return table

    def clear(self) -> None:
        """Drop every cached cascade (used by tests and on device changes)."""
        self._tables.clear()

    def __len__(self) -> int:
        """Number of distinct cascades currently held."""
        return len(self._tables)
