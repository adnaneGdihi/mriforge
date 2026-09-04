"""Declarative adapters: the *only* way to bridge a capability mismatch.

Adapters fire only when explicitly declared in the YAML's ``adapters:``
block — never auto-applied. The audit ``check_data_model_compatibility``
verifies that any chain declared by the user actually bridges the gap
it claims to bridge. Per CLAUDE.md item #9, silent rescue is forbidden.

See ``docs/superpowers/specs/2026-05-05-experiment-spec-card-and-adapters-design.md``.
"""

# Force registration of concrete adapters by importing the modules
# that declare them. New adapter modules MUST be added here; the
# registry only knows about adapters that have been imported.
from spectramr.data.adapters import channels as _channels  # noqa: F401
from spectramr.data.adapters import fourier as _fourier  # noqa: F401
from spectramr.data.adapters import spatial as _spatial  # noqa: F401
from spectramr.data.adapters.registry import (
    ADAPTER_REGISTRY,
    Adapter,
    get_adapter,
    get_adapter_capabilities,
    list_adapters,
    register_adapter,
)

__all__ = [
    "ADAPTER_REGISTRY",
    "Adapter",
    "get_adapter",
    "get_adapter_capabilities",
    "list_adapters",
    "register_adapter",
]
