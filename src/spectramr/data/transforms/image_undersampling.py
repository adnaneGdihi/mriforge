"""Retrospective k-space undersampling of a fully-sampled magnitude image.

The bridge for image-domain reconstruction arms whose data source is a
fully-sampled NIfTI corpus (MRIxFields2026) but whose task is *undersampled*
reconstruction (exp_c1 federated VF). A k-space-acceleration arm expects an
aliased ``input`` paired with a full ``target``; an image dataset gives only the
full image. This transform synthesises the aliased input on the fly:

    full magnitude image --fft2c--> synthetic k-space --Cartesian mask-->
    undersampled k-space --ifft2c--> |·| --> aliased ``input``

``target`` is left untouched (the full image), so a paired ``tio.Subject``
(input + target) emerges as the (degraded, clean) pair the recon loss needs.

Uses the physics SSOT exclusively — ``fft2c``/``ifft2c`` (centred, ortho) and
``KSpaceMaskGenerator`` — never a raw ``torch.fft`` or a hand-rolled mask
(CLAUDE.md non-negotiables #2, #6). Magnitude images carry no phase, so the
synthesised k-space is Hermitian-symmetric; this is the standard, documented
caveat of retrospective-undersampling demonstrations (m4raw_dataset.py says the
same of ``fft2c(|rss|)``).
"""

from __future__ import annotations

import torch
import torchio as tio

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.sampling import MaskGenerator, MaskType


class RetrospectiveImageUndersampling(tio.Transform):
    """Replace ``input`` with its retrospectively-undersampled magnitude.

    Args:
        acceleration: undersampling factor R (PE lines kept ≈ 1/R + ACS).
        center_fraction: fraction of central phase-encode lines always kept (ACS).
        mask_type: a Cartesian :class:`MaskType` value/name (default
            ``cartesian_lines`` — deterministic equispaced PE undersampling with
            a fully-sampled readout, so the degradation is reproducible without a
            per-sample seed).
        image_key / target_key: which subject keys to degrade / preserve.
    """

    def __init__(
        self,
        acceleration: float = 4.0,
        center_fraction: float = 0.08,
        mask_type: str | MaskType = MaskType.CARTESIAN_LINES,
        image_key: str = "input",
        target_key: str = "target",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if acceleration < 1.0:
            raise ValueError(f"acceleration must be >= 1, got {acceleration}")
        if not 0.0 < center_fraction <= 1.0:
            raise ValueError(f"center_fraction must be in (0, 1], got {center_fraction}")
        self.acceleration = float(acceleration)
        self.center_fraction = float(center_fraction)
        self.mask_type = MaskType(mask_type) if isinstance(mask_type, str) else mask_type
        self.image_key = image_key
        self.target_key = target_key
        self._mask_gen = MaskGenerator()
        # Per-shape mask cache. ``MaskGenerator`` is deterministic (no
        # per-sample seed), so for fixed ``(mask_type, h, w, acceleration,
        # center_fraction)`` the mask is byte-identical every call — recomputing
        # it in every ``__getitem__`` (in the worker) is pure waste. A plain
        # dict keyed by ``(h, w)`` is pickle-safe for forkserver workers.
        self._mask_cache: dict[tuple[int, int], torch.Tensor] = {}

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        if self.image_key not in subject:
            # No silent degradation to identity — an arm that asks for
            # undersampling but emits no ``input`` is mis-wired (pitfall #15).
            raise KeyError(
                f"RetrospectiveImageUndersampling: subject has no {self.image_key!r} "
                f"key to undersample (keys: {sorted(subject.keys())})"
            )
        if self.acceleration == 1.0:
            return subject  # explicit pass-through for the no-acceleration case

        data = subject[self.image_key].data  # tio layout [C, H, W, D], real magnitude
        _c, h, w, _d = data.shape
        # [C, H, W, D] -> [D, C, H, W] so D rides as the FFT batch axis.
        batched = data.permute(3, 0, 1, 2).contiguous()
        kspace = fft2c(batched)  # real coerced to complex internally; centred/ortho

        mask = self._mask_cache.get((h, w))
        if mask is None:
            mask = self._mask_gen.generate_mask(
                self.mask_type,
                (h, w),
                acceleration_factor=self.acceleration,
                center_fraction=self.center_fraction,
            )
            self._mask_cache[(h, w)] = mask
        mask = mask.to(kspace.device)
        # [H, W] -> [1, 1, H, W] broadcast over (D-batch, C).
        undersampled_k = kspace * mask.view(1, 1, h, w)

        aliased = ifft2c(undersampled_k).abs()  # magnitude image, real
        out = aliased.permute(1, 2, 3, 0).contiguous()  # back to [C, H, W, D]
        subject[self.image_key].set_data(out.to(data.dtype))
        return subject
