__all__ = ["sota_registry"]

"""Implementations Package
======================

This package contains all concrete implementations following SOLID principles.
Each implementation has a single responsibility and implements appropriate interfaces.
"""

# NOTE: ``sota_registry`` is intentionally NOT imported eagerly here.
# Historically importing it pulled the entire SOTA model catalogue
# (timm / torchio / torchmetrics / torchvision / wandb), which dominated
# CLI cold-start (~6 s on ``mriforge --help``). Since 2026-07-02 it is a
# deprecated no-op shim: model registration is owned by
# ``populate_model_registry()`` (``init_registry.py``) via per-class
# ``@register_model`` decorators + the pkgutil walk. The PEP-562
# ``__getattr__`` below keeps ``mriforge.models.sota_registry`` resolvable
# on demand for any legacy caller that references it as an attribute.


def __getattr__(name: str):
    """Lazily resolve ``sota_registry`` without eager heavy imports."""
    if name == "sota_registry":
        import mriforge.models.sota_registry as _sota_registry

        return _sota_registry
    raise AttributeError(f"module 'mriforge.models' has no attribute {name!r}")
