"""Service Audit Module - Validate DI Container Service Registration

This module provides comprehensive audit functionality for the DI container,
verifying that all required services are properly registered at startup.

Usage:
    from mriforge.infrastructure.di.service_audit import audit_services

    container = build_container(config)
    result = audit_services(container)

    if not result.is_healthy:
        print(f"Missing services: {result.missing_services}")
        raise ConfigurationError("Service audit failed")
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from mriforge.domain.interfaces.service_interfaces import (
    ICheckpointService,
    IDeviceService,
    ILoggingService,
)
from mriforge.infrastructure.services import IMetricsService

logger = logging.getLogger(__name__)


@dataclass
class ServiceRegistration:
    """Record of a single service registration."""

    name: str
    interface: Any
    registered: bool
    implementation_type: str | None = None
    error_message: str | None = None

    def __str__(self) -> str:
        """__str__.

        Returns:
            str: Description.
        """
        status = "✅ REGISTERED" if self.registered else "❌ MISSING"
        impl = f" ({self.implementation_type})" if self.implementation_type else ""
        return f"{status} {self.name}{impl}"


@dataclass
class ServiceAuditResult:
    """Complete audit result for DI container."""

    timestamp: str = ""
    total_services: int = 0
    registered_services: int = 0
    missing_services: int = 0
    registrations: list[ServiceRegistration] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        """Return True if all required services are registered."""
        # Only check missing_services (required), ignore optional service errors
        return self.missing_services == 0

    @property
    def health_status(self) -> str:
        """Return human-readable health status."""
        if self.is_healthy:
            return "🟢 HEALTHY"
        elif self.missing_services > 0:
            return f"🟡 DEGRADED ({self.missing_services} missing)"
        else:
            return "🔴 UNHEALTHY"

    def summary(self) -> str:
        """Return summary of audit results."""
        lines = [
            "=" * 80,
            "DI Container Service Audit Report",
            "=" * 80,
            f"Status: {self.health_status}",
            f"Services: {self.registered_services}/{self.total_services} registered",
            "",
        ]

        if self.registrations:
            lines.append("Service Registrations:")
            for reg in self.registrations:
                lines.append(f"  {reg}")
            lines.append("")

        if self.errors:
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  ⚠️  {error}")
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)


def audit_services(container: Any) -> ServiceAuditResult:
    """Audit DI container for required service registration.

    Args:
        container: DI container instance to audit

    Returns:
        ServiceAuditResult containing audit status and details

    Raises:
        ValueError: If container is None or invalid
    """
    from datetime import datetime

    if container is None:
        raise ValueError("Container cannot be None")

    result = ServiceAuditResult(timestamp=datetime.now().isoformat())

    # Required services audited at startup. The list mirrors what
    # ``bootstrap.build_container`` actually registers — see audit
    # 15 F2: ``IManifestService`` / ``IModelManagementService`` /
    # ``IErrorHandlingService`` / ``IModelCompilationService`` /
    # ``IProfilingService`` used to live here but had no consumer in
    # the training path, so their registrations were removed.
    required_services = [
        ("IDeviceService", IDeviceService, True),  # (name, interface, required)
        ("ILoggingService", ILoggingService, True),
        ("ICheckpointService", ICheckpointService, True),
        ("IMetricsService", IMetricsService, False),  # Optional - not always needed
    ]

    result.total_services = len(required_services)

    # Check each service
    for service_name, service_interface, is_required in required_services:
        try:
            service = container.resolve(service_interface)
            result.registered_services += 1
            result.registrations.append(
                ServiceRegistration(
                    name=service_name,
                    interface=service_interface,
                    registered=True,
                    implementation_type=type(service).__name__,
                )
            )
        except Exception as e:
            error_msg = str(e)
            if is_required:
                result.missing_services += 1
                result.errors.append(
                    f"Required service {service_name} not registered: {error_msg}",
                )

            result.registrations.append(
                ServiceRegistration(
                    name=service_name,
                    interface=service_interface,
                    registered=False,
                    error_message=error_msg if is_required else None,
                )
            )

    return result


def verify_startup_services(container: Any, fail_on_missing: bool = True) -> bool:
    """Verify services are available at startup.

    Args:
        container: DI container to verify
        fail_on_missing: If True, raise ConfigurationError on missing required services

    Returns:
        True if healthy, False otherwise (or raises if fail_on_missing=True)

    Raises:
        ConfigurationError: If required services missing and fail_on_missing=True
    """
    result = audit_services(container)

    # Log summary (if logging available)
    try:
        logging_service = container.resolve(ILoggingService)
        logging_service.log_info(result.summary(), level="info")
    except Exception:
        # DI logging service not available; fall back to the stdlib logger.
        logger.info(result.summary())

    if not result.is_healthy and fail_on_missing:
        from mriforge.domain.exceptions import ConfigurationError

        raise ConfigurationError(
            f"Service audit failed: {result.missing_services} required services missing",
        )

    return result.is_healthy
