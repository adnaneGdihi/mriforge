"""Unit tests for training orchestration services."""


class TestModelManagementService:
    """Test model management service."""

    def test_service_exists(self):
        """Test model management service exists."""
        from mriforge.infrastructure.services.model_management_service import (
            ModelManagementService,
        )

        assert hasattr(ModelManagementService, "__init__")


class TestModelCompilationService:
    """Test model compilation service."""

    def test_service_exists(self):
        """Test model compilation service exists."""
        from mriforge.infrastructure.services.model_compilation_service import (
            ModelCompilationService,
        )

        assert hasattr(ModelCompilationService, "__init__")

    def test_service_construction_is_deprecated(self):
        """2026-07-01: the service is dormant (removed from DI, no training
        consumer); constructing it must warn and point at the live
        ModelBuilder.compile() path."""
        from unittest.mock import MagicMock

        import pytest

        from mriforge.infrastructure.services.model_compilation_service import (
            ModelCompilationService,
        )

        with pytest.warns(DeprecationWarning, match="ModelBuilder.compile"):
            ModelCompilationService(logging_service=MagicMock())


class TestPipelineExecutionService:
    """Test pipeline execution service."""

    def test_service_exists(self):
        """Test pipeline execution service exists."""
        from mriforge.infrastructure.services.pipeline_execution_service import (
            PipelineExecutionService,
        )

        assert hasattr(PipelineExecutionService, "__init__")


class TestProfilingServices:
    """Test profiling services."""

    def test_profiling_service_exists(self):
        """Test profiling service exists."""
        from mriforge.infrastructure.services.profiling_service import ProfilingService

        assert hasattr(ProfilingService, "__init__")


# TestValidationService removed with the module (#710). Its single assertion was
# `hasattr(ValidationService, "__init__")` -- true of every class in Python, so it
# passed regardless of whether the service worked. The service itself fabricated
# its result: `run_validation` hardcoded `validation_loss: 0.0` and
# `get_best_validation_result` then sorted by that constant.


