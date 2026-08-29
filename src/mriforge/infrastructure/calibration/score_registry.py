"""Registry of split-conformal non-conformity scores.

Scores registered here are dispatched from YAML via
``certification.conformal.score_fn``. The registry is a thin
dispatch layer over the score *functions* already in
:mod:`mriforge.infrastructure.calibration.scores`; it exists so YAMLs
can opt into the marker-augmented :func:`vf_residual` score without
touching strategy code.

Usage::

    from mriforge.infrastructure.calibration.score_registry import (
        register_calibration_score, resolve_calibration_score,
    )

    @register_calibration_score("absolute_residual")
    def absolute_residual(pred, target):
        return (pred - target).abs()

    score_fn = resolve_calibration_score("absolute_residual")

Scores must be callables of signature
``(prediction, target, **kwargs) -> tensor``. The registry forwards
the YAML-provided ``score_kwargs`` (e.g. ``marker_basis_path``) into
the call site so stateless scores can adapt to per-experiment context.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

import torch

logger = logging.getLogger(__name__)


class CalibrationScoreFn(Protocol):
    """A non-conformity score: maps ``(pred, target)`` to a per-element score tensor."""

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor: ...


CALIBRATION_SCORE_REGISTRY: dict[str, CalibrationScoreFn] = {}


def register_calibration_score(name: str) -> Callable[[CalibrationScoreFn], CalibrationScoreFn]:
    """Decorator: register ``name`` → score-callable in the global registry.

    Raises:
        ValueError: if ``name`` is already registered under a different
            callable. Re-registering the *same* callable (e.g. via
            module reload) is silently allowed.
    """

    def decorator(fn: CalibrationScoreFn) -> CalibrationScoreFn:
        existing = CALIBRATION_SCORE_REGISTRY.get(name)
        if existing is not None and existing is not fn:
            raise ValueError(
                f"Calibration score '{name}' already registered to "
                f"{existing!r}; refusing to overwrite with {fn!r}."
            )
        CALIBRATION_SCORE_REGISTRY[name] = fn
        logger.debug("Registered calibration score: %s", name)
        return fn

    return decorator


def resolve_calibration_score(name: str) -> CalibrationScoreFn:
    """Look up a score function by name. Raises if unknown."""
    if name not in CALIBRATION_SCORE_REGISTRY:
        raise KeyError(
            f"Calibration score '{name}' is not registered. "
            f"Available: {sorted(CALIBRATION_SCORE_REGISTRY)}."
        )
    return CALIBRATION_SCORE_REGISTRY[name]


def list_calibration_scores() -> list[str]:
    return sorted(CALIBRATION_SCORE_REGISTRY)


# Register the existing stateless scores so the registry covers the
# pre-existing dispatch keys.
from mriforge.infrastructure.calibration.scores import (  # noqa: E402, I001  (intentional late import)
    absolute_residual as _abs_residual,
    kappa_residual as _kappa_residual,
    null_energy as _null_energy,
    physics_residual as _physics_residual,
    quantile_regression as _quantile_regression,
)


@register_calibration_score("absolute_residual")
def _absolute_residual(prediction: torch.Tensor, target: torch.Tensor, **_: Any) -> torch.Tensor:
    return _abs_residual(prediction, target)


@register_calibration_score("pixel_residual")
def _pixel_residual(prediction: torch.Tensor, target: torch.Tensor, **_: Any) -> torch.Tensor:
    return _abs_residual(prediction, target)


@register_calibration_score("quantile_regression")
def _qr(prediction: torch.Tensor, target: torch.Tensor, **_: Any) -> torch.Tensor:
    # quantile_regression expects (lo, hi, target); when called through
    # this registry the prediction is expected to be a (2, ...) stack of
    # (lower, upper) quantiles. Callers that don't want this shape should
    # use the un-registered scores function directly.
    if prediction.ndim < 1 or prediction.shape[0] != 2:
        raise ValueError(
            "quantile_regression score expects prediction of shape "
            "(2, ...) with the leading dim ordering (lower, upper)."
        )
    return _quantile_regression(prediction[0], prediction[1], target)


@register_calibration_score("cqr")
def _cqr_alias(prediction: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return _qr(prediction, target, **kwargs)


@register_calibration_score("physics_residual")
def _physics_residual_score(
    prediction: torch.Tensor, target: torch.Tensor, **kwargs: Any
) -> torch.Tensor:
    """PR-CC physics-consistency residual (re-render observed contrast).

    ``prediction`` is the estimated tissue-parameter map ``(rho, T1, T2)``;
    ``target`` the observed image. The acquisition vector and contrast label
    arrive through ``score_kwargs`` (``acquisition``, ``contrast_type``).
    """
    return _physics_residual(prediction, target, **kwargs)


@register_calibration_score("kappa_residual")
def _kappa_residual_score(
    prediction: torch.Tensor, target: torch.Tensor, **kwargs: Any
) -> torch.Tensor:
    """Cold-diffusion trust score κ (C7): per-image relative residual against
    the MEASURED k-space on the observed support.

    ``target`` here is the acquired measurement (label-free), NOT ground
    truth; the sampling ``mask`` arrives through ``score_kwargs``.
    """
    return _kappa_residual(prediction, target, **kwargs)


@register_calibration_score("null_energy")
def _null_energy_score(
    prediction: torch.Tensor, target: torch.Tensor, **kwargs: Any
) -> torch.Tensor:
    """Cold-diffusion trust score η^null (C7): per-image invented null-band
    energy fraction. No-reference — ``target`` is accepted for protocol
    compatibility and ignored; ``mask`` (mandatory) arrives via
    ``score_kwargs``.
    """
    del target
    return _null_energy(prediction, **kwargs)


__all__ = [
    "CALIBRATION_SCORE_REGISTRY",
    "CalibrationScoreFn",
    "list_calibration_scores",
    "register_calibration_score",
    "resolve_calibration_score",
]
