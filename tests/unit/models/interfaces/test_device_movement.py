"""The model layer depends on a Protocol, not on an infrastructure service.

``models/pipelines/generation_pipeline.py`` imported ``DevicePolicy`` and two
factories from ``spectramr.infrastructure.services.device_policy`` -- an upward
import (``infrastructure/ -> models/``, never the reverse; non-negotiable 5).

Worth recording how it arrived, because the repair is what exposed it: the
import previously read ``from ....services.device_policy import ...``, which
walks past the top-level package and raises ``ImportError``, and the only
consumer swallowed that as ``generation_pipeline_module = None`` -- so a broken
module was indistinguishable from an absent optional dependency (non-negotiable
18). ``check_layering.sh`` had nothing to see while the module could not import
at all.

These tests pin the seam itself rather than the grep: a gate anchored at ``^``
(as all of ``check_layering.sh`` is) would go green again for a function-local
import that changed nothing, which is the detector-blindness shape
non-negotiable 15 is about (#1183).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from spectramr.models.interfaces.device_movement import SupportsDeviceMovement

_PIPELINE = (
    Path(__file__).resolve().parents[4]
    / "src/spectramr/models/pipelines/generation_pipeline.py"
)


def test_the_real_device_policy_satisfies_the_protocol():
    """Structural, so the concrete policy needs no registration or subclassing.

    If this fails the seam is a fiction: the composition root would be injecting
    something the model layer cannot actually use.
    """
    torch = pytest.importorskip("torch")
    assert torch is not None
    from spectramr.infrastructure.services.device_policy import create_cpu_device_policy

    assert isinstance(create_cpu_device_policy(), SupportsDeviceMovement)


def test_a_minimal_double_satisfies_the_protocol():
    """One method wide -- which is what lets the model layer be tested alone."""

    class _Double:
        def move_to_device(self, data, dtype=None, non_blocking=False):
            return data

    assert isinstance(_Double(), SupportsDeviceMovement)


def test_an_object_without_the_method_does_not():
    """The Protocol must still be able to say no (NN15: prove the negative)."""

    class _NotAPolicy:
        pass

    assert not isinstance(_NotAPolicy(), SupportsDeviceMovement)


def _imported_modules(path: Path) -> set[str]:
    """Every module named by an import ANYWHERE in the file, not just at ``^``.

    Walks the AST rather than grepping, so a function-local import counts. That
    is the whole point: ``check_layering.sh`` anchors every pattern at ``^`` and
    would not see one.
    """
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
    return found


def test_generation_pipeline_imports_no_infrastructure_at_any_depth():
    """Top-of-file AND function-local, because the gate only sees the first."""
    offenders = sorted(
        m for m in _imported_modules(_PIPELINE) if m.startswith("spectramr.infrastructure")
    )
    assert not offenders, (
        "models/ may not import infrastructure/ (non-negotiable 5); found: " + ", ".join(offenders)
    )


def test_the_pipeline_refuses_rather_than_building_its_own_policy():
    """No injected policy -> raise, not a silently-resolved default.

    The old fallback was ``get_default_device_policy()``, which resolves a
    device by itself -- a second device resolver reached without the caller
    asking (non-negotiable 9b), and a silent substitution (non-negotiable 3).
    """
    pytest.importorskip("torch")
    from spectramr.models.pipelines.generation_pipeline import TrellisPipeline

    p = TrellisPipeline.__new__(TrellisPipeline)
    p._device_policy = None
    p._cpu_policy = None
    with pytest.raises(ValueError, match="device_policy"):
        _ = p.device_policy
    with pytest.raises(ValueError, match="cpu_policy"):
        p._ensure_cpu(object())
