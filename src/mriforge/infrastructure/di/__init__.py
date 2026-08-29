#!/usr/bin/env python3
"""Dependency Injection Module

This module provides dependency injection (DI) container for managing service
registration and resolution throughout the application.

Key exports:
- DIContainer: Main DI container class
- ServiceResolutionError: Raised when a service cannot be wired (registration
  or type-hint miss); provider/constructor failures propagate as-is
- get_global_container: Get the global singleton container instance
- register_service: Register a service in the global container
- resolve_service: Resolve a service from the global container
"""

from mriforge.infrastructure.di.di_container import (
    DIContainer,
    ServiceResolutionError,
    get_global_container,
    register_service,
    resolve_service,
)

__all__ = [
    "DIContainer",
    "ServiceResolutionError",
    "get_global_container",
    "register_service",
    "resolve_service",
]
