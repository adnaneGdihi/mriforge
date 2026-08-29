"""Plugin-discovery configuration schema (out-of-tree component registration).

Defines the top-level ``plugins`` block consumed at registry population
(:func:`mriforge.plugins.discover_plugins`) so a user can register custom
models / losses / metrics / strategies that live **outside** the
``mriforge`` source tree — in their own pip-installed package or a standalone
script — and have them resolve by name.

Before this schema existed, :class:`~mriforge.config.settings.TrainingSettings`
carried no ``plugins`` field, so a YAML ``plugins:`` block was rejected by
``extra="forbid"``. The block is the YAML-facing peer of the
``MRIFORGE_PLUGINS`` env var: both name dotted import paths to import (which
fires the ``@register_*`` decorators) before the in-tree discovery walk.

The block follows the ``optimization``/``logging`` precedent of
``extra="forbid"``: a misspelled knob must raise at load time rather than
silently no-op (pitfall #15 — every advertised knob is read + validated).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PluginsConfigSchema(BaseModel):
    """Out-of-tree plugin import paths.

    Attributes:
        enabled: Master switch. Discovery layers (entry-points + the
            ``MRIFORGE_PLUGINS`` env var) run regardless; this flag gates the
            ``paths`` list specifically, so a config can carry a documented
            plugin list that is toggled off without deleting it.
        paths: Dotted module paths imported before the in-tree registry walk
            (e.g. ``["mypkg.models.my_unet", "mypkg.losses.my_loss"]``). Each
            import fires that module's ``@register_*`` decorators. An
            unimportable path is surfaced (never silently skipped).

    No ``hooks`` field: a ``plugins.hooks`` block was advertised earlier but had
    **zero consumers** (nothing read ``settings.plugins.hooks``), so it was an
    unwired knob — pitfall #15's "every advertised knob is read + validated".
    ``extra="forbid"`` now rejects ``plugins.hooks`` outright; if hook
    configuration is wanted later, add the field back *together with* the
    strategy/validator that reads, validates, and stamps it into provenance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    enabled: bool = Field(
        default=False,
        description="Enable importing the dotted module paths in ``paths``.",
    )
    paths: list[str] = Field(
        default_factory=list,
        description=(
            "Dotted module paths imported at registry population so their "
            "``@register_*`` decorators fire (out-of-tree components)."
        ),
    )
