"""Render a ULF acquisition from quantitative maps, via the forward operator.

This is the production consumer of
:class:`~spectramr.infrastructure.physics.ulf_forward_operator.DifferentiableULFForwardOperator`
(non-negotiable 16, issue #1708): the operator existed, was complete and was
imported by nothing, so its five stages had never run under ``train``.

**How this differs from its sibling** (non-negotiable 17 — the difference is
stated in the registered name, not left to a docstring):
:class:`~spectramr.data.transforms.synthetic_ulf_simulator.SimulateULFFromHF`
degrades an *image* with four empirical stages (B0 shift, Maxwell, T2*-blur,
Rician). This one starts from *quantitative maps* and renders through the
electrodynamic operator, so the ULF image it produces is a function of tissue
parameters rather than of a high-field picture of them. Both write ``input``;
they are not interchangeable and neither replaces the other.

Supplier: ``dataset_type: quantitative``
(:class:`~spectramr.data.datasets.quantitative_dataset.QuantitativeDataset`),
which emits per-map ``t1`` / ``t2`` / ``pd`` :class:`tio.ScalarImage` keys by
name.

**Tier 1 only.** The released cross-field volumes are co-registered onto a
common grid, which has already corrected out the gradient non-linearity and
off-resonance the operator's stages 2-3 model — re-applying them would distort
an already-undistorted volume. So ``enable_gnl`` and ``enable_maxwell`` default
to ``False`` and the Cartesian branch is used (``trajectory=None``). GNL,
Maxwell and the NUFFT branch need raw k-space and are Tier 2.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch
import torchio as tio

from spectramr.data.transforms.registry import register_transform
from spectramr.infrastructure.physics.fft_ops import ifft2c
from spectramr.infrastructure.physics.ulf_forward_operator import (
    DifferentiableULFForwardOperator,
)

logger = logging.getLogger(__name__)

# T1 in milliseconds is order 10^2-10^3 for brain tissue at any field in range
# (~230 ms fat at 0.064 T to ~4500 ms CSF at 7 T). In *seconds* the same tissue
# reads 0.23-4.5, and the operator's exponentials would then be evaluated at
# roughly TR/T1 = 500/1.0 instead of 500/1000 -- a plausible-looking image that
# is wrong by three orders of magnitude, with nothing raised. The band is
# deliberately wide: it separates seconds from milliseconds, and is not a
# physiological plausibility check.
_T1_MS_MIN = 20.0
_T1_MS_MAX = 20_000.0


# ``requires``/``produces`` name the DEFAULT keys. Both sides are constructor
# kwargs (rho_map/t1_map/t2_map, output_image/target_image), so an arm that
# renames them makes this declaration stale. That is a registry-wide limit --
# a static tuple cannot express a configurable key -- recorded on #1714, which
# is the issue proposing these fields get a production reader.
@register_transform(
    "simulate_ulf_from_qmaps",
    requires=("t1", "t2", "pd"),
    produces=("input", "target"),
)
class SimulateULFFromQMaps(tio.Transform):
    """Render a ULF image from (PD, T1, T2) maps through the forward operator.

    Writes the rendered ULF magnitude image to ``output_image``. When
    ``render_target`` is True it also renders the *same maps* at ``b0_source``
    into ``target_image``, giving a physically paired ULF/HF couple from one
    set of tissue parameters -- which is the cross-field hypothesis stated
    directly, rather than two images related by an image-to-image map.

    Args:
        b0_source: field strength in Tesla at which the maps were measured.
            **Required, with no default**: the operator's parameters are named
            ``t1_3t``/``t2_3t``, and feeding maps estimated at another field
            would mislabel the reference and mis-transport them silently.
        b0_target: field strength in Tesla to render at (0.064 for Hyperfine
            SWOOP, 0.3 for M4Raw).
        rho_map / t1_map / t2_map: subject keys holding the maps.
        output_image: key to write the rendered ULF image under.
        render_target: also render at ``b0_source`` into ``target_image``.
        target_image: key for the ``render_target`` render.
        beta / gamma_t2: power-law dispersion exponents, passed through.
        tr / te: sequence timings in **milliseconds**.
        enable_noise / enable_emi: the two stochastic stages. Left on: the
            field-dependent noise term is most of what makes a ULF render
            differ from a high-field one.
        enable_gnl / enable_maxwell: Tier-2 stages, off by default (see the
            module docstring). ``enable_gnl`` has no effect without
            coefficients, which this transform does not supply.
        include / exclude / copy / kwargs: standard ``tio.Transform`` args.

    Raises:
        ValueError: if a declared map key is absent from the subject, or if T1
            does not read as milliseconds.

    Note:
        The ``requires=`` argument above is **declarative only** -- nothing in
        ``src/`` reads :attr:`TransformRegistration.requires` today, so the
        enforcing check is the explicit one in :meth:`apply_transform`. Do not
        rely on ``requires`` alone in a new transform until that is wired.
    """

    def __init__(
        self,
        b0_source: float,
        b0_target: float = 0.064,
        rho_map: str = "pd",
        t1_map: str = "t1",
        t2_map: str = "t2",
        output_image: str = "input",
        render_target: bool = True,
        target_image: str = "target",
        beta: float = 0.35,
        gamma_t2: float = 0.05,
        tr: float = 500.0,
        te: float = 14.0,
        enable_noise: bool = True,
        enable_emi: bool = True,
        enable_gnl: bool = False,
        enable_maxwell: bool = False,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        copy: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(include=include, exclude=exclude, copy=copy, **kwargs)
        for _name, _value in (("b0_source", b0_source), ("b0_target", b0_target)):
            if _value <= 0:
                raise ValueError(f"{_name} must be > 0 Tesla; got {_value!r}")
        self.b0_source = float(b0_source)
        self.b0_target = float(b0_target)
        self.rho_map = rho_map
        self.t1_map = t1_map
        self.t2_map = t2_map
        self.output_image = output_image
        self.render_target = bool(render_target)
        self.target_image = target_image
        self.beta = float(beta)
        self.gamma_t2 = float(gamma_t2)
        self.tr = float(tr)
        self.te = float(te)
        self.enable_noise = bool(enable_noise)
        self.enable_emi = bool(enable_emi)
        self.enable_gnl = bool(enable_gnl)
        self.enable_maxwell = bool(enable_maxwell)
        # Keyed by (b0_target, H, W): the operator holds no per-sample state,
        # so one instance serves every subject of a given geometry. Rebuilding
        # it per call would allocate the SH basis and noise buffers per item.
        self._operators: dict[tuple[float, int, int], DifferentiableULFForwardOperator] = {}

    def _operator(self, b0_target: float, height: int, width: int):
        key = (b0_target, height, width)
        if key not in self._operators:
            self._operators[key] = DifferentiableULFForwardOperator(
                b0_target=b0_target,
                b0_source=self.b0_source,
                beta=self.beta,
                gamma_t2=self.gamma_t2,
                tr=self.tr,
                te=self.te,
                image_shape=(height, width),
                enable_emi=self.enable_emi,
                enable_gnl=self.enable_gnl,
                enable_maxwell=self.enable_maxwell,
                enable_noise=self.enable_noise,
            )
        return self._operators[key]

    def _render(
        self, rho: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, b0_target: float
    ) -> torch.Tensor:
        """Render one [C, X, Y, Z] map triple to a [C, X, Y, Z] magnitude image.

        TorchIO stores images as ``[C, W, H, D]``; the operator is 2-D and takes
        ``[B, 1, H, W]``. The depth axis becomes the batch, so every slice is
        rendered independently -- which is what the operator models.
        """
        # [C, X, Y, Z] -> [Z, C, X, Y]
        rho_b, t1_b, t2_b = (m.permute(3, 0, 1, 2) for m in (rho, t1, t2))
        operator = self._operator(b0_target, rho_b.shape[-2], rho_b.shape[-1])
        kspace = operator(rho_b, t1_b, t2_b, trajectory=None, gnl_coefficients=None)
        image = ifft2c(kspace).abs()
        # [Z, C, X, Y] -> [C, X, Y, Z]
        return image.permute(1, 2, 3, 0)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        missing = [k for k in (self.rho_map, self.t1_map, self.t2_map) if k not in subject]
        if missing:
            raise ValueError(
                f"{type(self).__name__} renders from quantitative maps and the "
                f"subject is missing {missing}. It requires "
                f"'{self.rho_map}', '{self.t1_map}' and '{self.t2_map}' -- which "
                "dataset_type: quantitative emits by name. Available keys: "
                f"{sorted(subject.keys())}."
            )

        rho = subject[self.rho_map].data.float()
        t1 = subject[self.t1_map].data.float()
        t2 = subject[self.t2_map].data.float()

        # Unit guard. quantitative_config.units='normalized' rescales every map
        # to [0, 1], which silently turns T1 into a unitless fraction; the
        # render then evaluates exp(-TR/T1) at TR/T1 ~ 500 and saturates.
        positive = t1[t1 > 0]
        if positive.numel() > 0:
            median_t1 = positive.median().item()
            if not _T1_MS_MIN <= median_t1 <= _T1_MS_MAX:
                raise ValueError(
                    f"{type(self).__name__} expects '{self.t1_map}' in "
                    f"milliseconds; its median is {median_t1:.4g}, outside "
                    f"[{_T1_MS_MIN}, {_T1_MS_MAX}]. A median below 10 means "
                    "seconds (multiply by 1000); a median at or below 1 means "
                    "the maps were normalized -- set quantitative.units to "
                    "'physical', because the operator's exponentials are not "
                    "scale-free and a normalized map renders a plausible but "
                    "wrong image."
                )

        ulf = self._render(rho, t1, t2, self.b0_target)
        subject[self.output_image] = tio.ScalarImage(tensor=ulf)
        logger.info(
            "[TRANSFORM] %s: rendered %r at %.4g T from (%s, %s, %s), shape=%s",
            type(self).__name__,
            self.output_image,
            self.b0_target,
            self.rho_map,
            self.t1_map,
            self.t2_map,
            tuple(ulf.shape),
        )

        if self.render_target:
            hf = self._render(rho, t1, t2, self.b0_source)
            subject[self.target_image] = tio.ScalarImage(tensor=hf)
            logger.info(
                "[TRANSFORM] %s: rendered %r at %.4g T (paired source render)",
                type(self).__name__,
                self.target_image,
                self.b0_source,
            )

        return subject


__all__ = ["SimulateULFFromQMaps"]
