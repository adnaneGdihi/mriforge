"""Validation and Error Handling Mixin for Training Strategies.

Provides configuration validation and training-time error handling.
Follows Single Responsibility Principle (SKILL_ARCHITECTURE.md).
"""

from typing import TYPE_CHECKING, Any

import torch

# Import CUDA OOM error with fallback for CPU-only environments
try:
    from torch.cuda import OutOfMemoryError as CudaOutOfMemoryError
except (ImportError, AttributeError):  # pragma: no cover

    class CudaOutOfMemoryError(RuntimeError):
        """Sentinel error used when CUDA OOM class is unavailable."""


if TYPE_CHECKING:
    from mriforge.domain.interfaces.service_interfaces import ILoggingService


class ValidationMixin:
    """Mixin for configuration validation and error handling.

    Encapsulates:
    - Config validation and verification
    - Error handling for training-time failures
    - CUDA memory recovery attempts
    - Feature flag logging

    Used by: All training strategies for consistent error handling
    """

    logging_service: "ILoggingService"
    state: Any
    _telemetry_logged: bool = False

    def _attempt_cuda_recovery(self, error: Exception) -> None:
        """Attempt to reclaim CUDA memory after an out-of-memory error.

        Only attempts recovery if:
        - Error is CudaOutOfMemoryError
        - CUDA is available
        - Logging service is available

        Args:
            error: The exception that was raised

        Side Effects:
            - Logs warning message if recovery attempted
            - Clears CUDA cache (implicit via PyTorch)
        """
        if not isinstance(error, CudaOutOfMemoryError):
            return

        if not torch.cuda.is_available():
            return

        if hasattr(self, "logging_service"):
            self.logging_service.log_warning(
                "CUDA OOM error encountered. Consider using gradient checkpointing or reducing batch size.",
                model_type=self.state.model_type,
            )

    def _handle_train_step_failure(
        self,
        error: Exception,
        *,
        epoch: int | None,
        message_prefix: str,
    ) -> None:
        """Handle train-step failures with fail-fast behavior.

        Implements fail-fast principle:
        - Log error details
        - Re-raise exception for caller to handle

        Args:
            error: The exception that occurred
            epoch: Current epoch (for logging context)
            message_prefix: Prefix for error message

        Raises:
            Immediately re-raises the input error

        Side Effects:
            - Logs error message with context
        """
        message = f"{message_prefix}: {error}"

        if hasattr(self, "logging_service"):
            self.logging_service.log_error(
                message,
                model_type=self.state.model_type,
                epoch=epoch,
            )

        # Fail-fast: re-raise the exception
        raise error

    def _verify_strategy_config(
        self,
        *,
        expected_modes: tuple[str, ...] | None = None,
    ) -> None:
        """Legacy method for training_mode validation - now deprecated.

        This method is preserved for backward compatibility but no longer
        performs validation. Override in subclasses if custom validation needed.

        Args:
            expected_modes: (Unused) Expected training mode identifiers
        """
        pass

    def _log_config_features(self, logger: Any) -> None:
        """Log active configuration feature flags once per strategy.

        Collects and logs feature flags from config.collect_feature_flags()
        method. Logs only once per strategy instance to avoid spam.

        Args:
            logger: Logger instance (usually ILoggingService)

        Side Effects:
            - Logs feature flags to logger (first call only)
            - Sets _telemetry_logged=True to prevent duplicate logs
        """
        if self._telemetry_logged:
            return

        collector = self.config.collect_feature_flags
        if not callable(collector):
            return

        try:
            feature_flags = collector()
        except Exception as exc:  # pragma: no cover - defensive
            logger.log_warning(
                f"Failed to collect feature flags for telemetry: {exc}",
                model_type=self.state.model_type,
            )
            return

        logger.log_info(
            f"Active config features: {feature_flags}",
            model_type=self.state.model_type,
        )
        self._telemetry_logged = True


__all__ = ["ValidationMixin"]
