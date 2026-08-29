"""DEPRECATED no-op shim — SOTA registrations moved to decorators (2026-07-02).

This module used to import ~43 model classes at module top and fire ~41
dynamic ``register_model(name, mode)(Class)`` calls covering the SOTA
innovation roster (phases 2-8 of the roadmap). That registration style had
two structural defects:

* it was invisible to the static ``@register_model`` source scans
  (``tests/unit/models/test_registry_integrity.py``), so mode typos and
  duplicate names could not be caught at the source level;
* a single swallowed ``ImportError`` on this module silently dropped the
  ENTIRE catalog (the failure mode ``test_registry_census.py`` now pins).

Every registration was therefore migrated to a per-class
``@register_model(name=..., training_mode=...)`` decorator on the model class
itself, fired by the ``populate_model_registry()`` pkgutil walk
(``mriforge/models/init_registry.py`` — see the dated 2026-07-02 walk-list
block). The one-class-many-names aliases (``vision_mamba`` → ``d2_mamba``)
live in the alias section of the same file. ``metal_diffusion`` →
``cold_diffusion_process`` was there too until 2026-08-12, when it was retired:
the name promised a metal-artefact degradation that an alias cannot set, and no
arm declared it. ``DiffSEQ``/``ThermPINN`` moved from
``infrastructure/physics/`` to ``mriforge/models/physics/`` as part of the
same pass.

The module is kept importable (and ``register_sota_models`` callable) only
because external callers and tests referenced it by name; both are inert.
Do NOT add registrations here — decorate the class and ensure its package is
in the ``populate_model_registry`` walk list instead.
"""


def register_sota_models() -> None:
    """No-op. SOTA models register via their own ``@register_model`` decorators.

    Kept for import compatibility. The authoritative registration path is
    ``mriforge.models.init_registry.populate_model_registry()``; the roster is
    pinned by ``tests/unit/models/test_registry_census.py``.
    """
