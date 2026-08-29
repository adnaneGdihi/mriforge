"""Tests for Generation3D orchestration request validation (WS-3).

Regression: ``validate_request`` rendered the offending parameter type via
``param_type.__name__``, which raised ``AttributeError`` for the
``guidance_scale`` requirement whose type is a tuple ``(int, float)`` — so a
bad ``guidance_scale`` crashed instead of being rejected.

The service ``__init__`` builds heavy pipelines; the validation path needs only
``self.logger`` and ``self._pipelines``, so we bypass ``__init__`` with
``__new__`` and attach the minimum.
"""

from __future__ import annotations

import logging

import torch

import mriforge.domain.services.generation_3d_orchestration_service as gen3d_module
from mriforge.domain.services.generation_3d_orchestration_service import (
    Generation3DOrchestrationService,
    GenerationMode,
    GenerationRequest,
)


def test_service_module_flags_dormant_status() -> None:
    """The service's docstring warns it is dormant (its pipeline backend is
    dead code), so a reader does not mistake it for a live execution path.

    See ``TODO/backlog_dead_code_pipelines_audit_2026_06_12.md``.
    """
    doc = gen3d_module.__doc__ or ""
    assert "DORMANT" in doc
    assert "generation_pipeline" in doc


def _service() -> Generation3DOrchestrationService:
    svc = Generation3DOrchestrationService.__new__(Generation3DOrchestrationService)
    svc.logger = logging.getLogger("test.gen3d")  # type: ignore[attr-defined]
    svc._pipelines = {GenerationMode.TRELLIS_GAUSSIAN: object()}  # type: ignore[attr-defined]
    return svc


def _request(params: dict[str, object]) -> GenerationRequest:
    return GenerationRequest(
        input_data=torch.zeros(1, 1, 4, 4),  # 4D as validate_request requires
        mode=GenerationMode.TRELLIS_GAUSSIAN,
        parameters=params,
    )


def test_guidance_scale_requirement_is_tuple() -> None:
    reqs = _service().get_mode_requirements(GenerationMode.TRELLIS_GAUSSIAN)
    assert reqs["guidance_scale"] == (int, float)


def test_bad_guidance_scale_rejected_without_crashing() -> None:
    # Must return False (rejected) rather than raise AttributeError on
    # (int, float).__name__ when rendering the warning.
    assert _service().validate_request(_request({"guidance_scale": "not-a-number"})) is False


def test_valid_guidance_scale_accepted() -> None:
    assert _service().validate_request(_request({"guidance_scale": 7.5})) is True
