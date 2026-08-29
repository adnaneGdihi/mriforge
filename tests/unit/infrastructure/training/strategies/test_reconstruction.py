"""Tests for ReconstructionTrainingStrategy's loss-folding seam.

Scoped deliberately: the strategy's forward/backward path is exercised end-to-end by
the Tier-2 audit probe and by the per-paradigm strategy tests. What is pinned here is
the ``_INLINE_MANAGED_EXTRA`` seam — the Open-Closed hook a subclass uses to declare
which registered losses it computes inline, so that declaring those losses on
``losses.image_losses`` (the only surface ``LossScheduleController`` can resolve a
curriculum rule's base weight from) does not silently double-count them.
"""

from __future__ import annotations

from mriforge.infrastructure.training.strategies.loss_folding import (
    _INLINE_MANAGED,
    inline_managed_with,
)
from mriforge.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)


def test_base_strategy_declares_no_extra_inline_terms() -> None:
    """The base recon strategy folds everything it does not compute itself."""
    assert ReconstructionTrainingStrategy._INLINE_MANAGED_EXTRA == ()


def test_base_skip_set_is_exactly_the_universal_fidelity_terms() -> None:
    skip = inline_managed_with(*ReconstructionTrainingStrategy._INLINE_MANAGED_EXTRA)
    assert skip == _INLINE_MANAGED
    assert skip == frozenset({"l1", "l2"})


def test_subclasses_widen_rather_than_replace_the_skip_set() -> None:
    """A subclass must never be able to un-skip the inline l1/l2 placeholders."""
    from mriforge.infrastructure.training.strategies.bloch_synth_strategy import (
        BlochSynthesisStrategy,
    )

    assert issubclass(BlochSynthesisStrategy, ReconstructionTrainingStrategy)
    widened = inline_managed_with(*BlochSynthesisStrategy._INLINE_MANAGED_EXTRA)
    assert _INLINE_MANAGED <= widened
    assert widened > _INLINE_MANAGED


def test_recoverability_vib_inherits_the_base_skip_set() -> None:
    """A subclass that adds no inline registered loss keeps the default behaviour."""
    from mriforge.infrastructure.training.strategies.recoverability_vib_strategy import (
        RecoverabilityVIBStrategy,
    )

    assert RecoverabilityVIBStrategy._INLINE_MANAGED_EXTRA == ()


def test_multislice_flag_is_read_without_a_hasattr_fallback() -> None:
    """`multislice_enabled` was undeclared, so the `hasattr` guard was always
    False and `is_multislice` could never be True — the flag both docstrings
    advertise as replacing shape heuristics never fired (pitfall #16)."""
    import inspect

    from mriforge.config.schemas.data import DataConfigSchema
    from mriforge.infrastructure.training.strategies import reconstruction

    assert DataConfigSchema().multislice_enabled is False
    assert DataConfigSchema(multislice_enabled=True).multislice_enabled is True
    src = inspect.getsource(reconstruction)
    assert 'hasattr(config.data, "multislice_enabled")' not in src
    assert "multislice_enabled = config.data.multislice_enabled" in src
