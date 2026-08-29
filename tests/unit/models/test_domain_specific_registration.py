"""Regression tests for the per-class domain-specific registration block.

Pin the post-audit-02-F2 fix.

Previously, ``ModelFactory._register_extended_models`` had a single
``try/except ImportError`` wrapping 16 generator registrations. When
``RLAcquisitionAgent`` / ``SurgeryLoopAgent`` failed to import (they
lived in ``src/application/agents/``, the factory tried to import them
from ``src/models/generators/``), the *entire* block was silently
skipped — all 16 mappings disappeared. ``stubs.py`` then aliased the
missing agent names to ``MODEL_REGISTRY["rl_mri"]`` (an unrelated
ActiveScanner class), turning the silent fallback into an identity
collision (CLAUDE.md pitfall #9).

The fix:
- Per-class registration: each model has its own try/except.
- 2026-05-15 D14 follow-up: agents moved into
  ``src/models/generators/`` (their canonical home — they are
  ``nn.Module + IGenerator``). They now self-register via
  ``@register_model`` and no longer need a special factory entry.
- Misleading rl_mri aliases for the agent names dropped from stubs.py.
"""

from __future__ import annotations

import warnings

import pytest


@pytest.fixture(scope="module")
def factory_registry():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    from mriforge.models.factories.model_factory import ModelFactory

    return ModelFactory()._registry


DOMAIN_SPECIFIC_GENERATORS = (
    "disentangled_mri",
    "disentangled_vae",
    "nesvor",
    "padnet",
    "diffdeur",
    "qspace_translation",
    "digital_twin_simulator",
    "reflexion_reconstruction",
    "continual_learning_unet",
    "spiking_unet",
    "symbolic_regression",
    "uncertainty_wrapper",
    "rl_acquisition_agent",
    "surgery_loop_agent",
    "split_client_encoder",
    "split_server_decoder",
)


def test_all_domain_specific_models_register(factory_registry) -> None:
    missing = [n for n in DOMAIN_SPECIFIC_GENERATORS if not factory_registry.has_generator(n)]
    assert not missing, (
        f"Domain-specific models failed to register: {missing}. "
        "Audit-02-F2 cause: the giant try/except in "
        "ModelFactory._register_extended_models swallowed every "
        "registration when one import failed. Each mapping must have "
        "its own try block."
    )


def test_agents_resolve_to_their_canonical_home(factory_registry) -> None:
    """``rl_acquisition_agent`` and ``surgery_loop_agent`` must point at
    the classes in ``src/models/generators/`` (their canonical home
    after the 2026-05-15 D14 move), NOT at the unrelated
    ``ActiveScanner`` they used to alias to via stubs.py.
    """
    rl_cls = factory_registry.get_generator_class("rl_acquisition_agent")
    surgery_cls = factory_registry.get_generator_class("surgery_loop_agent")
    assert rl_cls.__module__ == "mriforge.models.generators.rl_acquisition_agent", (
        f"rl_acquisition_agent should resolve to the canonical agent in "
        f"src/mriforge/models/generators/; got {rl_cls.__module__}.{rl_cls.__name__}."
    )
    assert surgery_cls.__module__ == "mriforge.models.generators.surgery_loop_agent", (
        f"surgery_loop_agent should resolve to the canonical agent in "
        f"src/mriforge/models/generators/; got {surgery_cls.__module__}.{surgery_cls.__name__}."
    )


def test_stubs_no_longer_aliases_agents_to_rl_mri() -> None:
    """The misleading alias must stay gone."""
    import re
    from pathlib import Path

    import mriforge

    # Derived from the package, not a path literal. The 2026-05 refactor moved
    # the tree to ``src/mriforge/`` and every hardcoded ``parents[N] / "src"``
    # silently pointed at a directory that does not exist -- 5 of cluster job
    # 8004252's failures, all reading as FileNotFoundError rather than as the
    # stale constant they were. ``mriforge.__file__`` cannot go stale on a move.

    pkg = Path(mriforge.__file__).resolve().parent
    stubs = (pkg / "models" / "stubs.py").read_text()
    # The alias being banned is the literal `setdefault("<agent>", MODEL_REGISTRY["rl_mri"])`
    banned = re.compile(
        r'setdefault\(\s*"(rl_acquisition_agent|surgery_loop_agent)"\s*,\s*MODEL_REGISTRY\["rl_mri"\]'
    )
    assert not banned.search(stubs), (
        "stubs.py reintroduced the agent → rl_mri silent alias. "
        "These names must resolve to their canonical classes via "
        "ModelFactory, not silently fall back to ActiveScanner."
    )
