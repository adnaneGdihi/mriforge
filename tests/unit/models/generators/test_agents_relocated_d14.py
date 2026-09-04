"""Regression test for the D14 move.

``RLAcquisitionAgent`` and ``SurgeryLoopAgent`` live in
``src/models/generators/`` (canonical home for ``nn.Module +
IGenerator`` classes per CLAUDE.md pitfall #12). They self-register
via ``@register_model`` and no longer need a special factory entry
that imports across the application/models boundary.

The ``src/spectramr/application/agents/`` compatibility shim (which re-exported
the canonical classes so the old import path kept resolving) was removed
2026-07-02: no YAML config or production code imported it, and the canonical
classes self-register via ``@register_model``. This test now pins the canonical
registration and the absence of a factory cross-import into
``spectramr.application.agents``.
"""

from __future__ import annotations

from pathlib import Path


def test_agents_self_register_via_decorator() -> None:
    """The agents should register via ``@register_model`` at import
    time, with no help from the factory's domain-specific block.
    """
    from spectramr.models.generators.rl_acquisition_agent import (  # noqa: F401
        RLAcquisitionAgent,
    )
    from spectramr.models.generators.surgery_loop_agent import (  # noqa: F401
        SurgeryLoopAgent,
    )
    from spectramr.models.registry import MODEL_REGISTRY

    assert "rl_acquisition_agent" in MODEL_REGISTRY
    assert "surgery_loop_agent" in MODEL_REGISTRY


def test_model_factory_no_longer_has_special_agent_entries() -> None:
    # Path corrected 2026-05-24: src→spectramr refactor moved the package
    # under src/spectramr/; the stale src/models/ path FileNotFoundError'd and
    # silently disabled this assertion. See smoke_audit_20260524.md.
    src = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "spectramr"
        / "models"
        / "factories"
        / "model_factory.py"
    ).read_text()
    # The old tuple-list entries must be gone.
    assert '"spectramr.application.agents"' not in src, (
        "model_factory.py still imports agents from spectramr.application.agents. "
        "They should self-register via @register_model from "
        "src/spectramr/models/generators/ — D14."
    )
