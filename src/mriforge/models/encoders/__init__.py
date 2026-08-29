"""Encoder model implementations.

This package holds encoder architectures that register themselves into
``MODEL_REGISTRY`` via ``@register_model`` decorators. ``init_registry``
walks this package at boot so the decorators fire — without this
``__init__.py`` the directory was a namespace package and
``pkgutil.walk_packages`` skipped it, leaving ``kan_encoder`` and
``medical_dino`` registrations dead. See TODO/audit/10_models_blocks_layers.md F2.
"""
