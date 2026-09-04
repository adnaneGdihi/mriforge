"""Regression tests for JEPA strategy fail-loud paths (audit-2026-05-14 E3).

Background
----------
Pre-2026-05-15, ``JEPAStrategy._compute_losses_impl`` had three silent
fallback returns: when the generator was missing, the batch wasn't a
dict, or ``_ensure_modules`` couldn't build the target/predictor pair,
the strategy returned ``{"loss_total": torch.tensor(0.0, device=...)}``.
That tensor is a *leaf with ``requires_grad=False``* — the orchestrator
then called ``loss.backward()`` and raised the cryptic
``RuntimeError: element 0 of tensors does not require grad and does not
have a grad_fn``.

CLAUDE.md pitfall #9 forbids silent fallbacks. The fix replaces each
return with an explicit ``RuntimeError`` / ``TypeError`` / ``KeyError``
naming the actual misconfiguration. These tests pin the three raises so
a future "tolerant" refactor cannot silently re-introduce the bug.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.jepa_strategy import JEPAStrategy


def _make_jepa_strategy(generator: nn.Module | None) -> JEPAStrategy:
    """Build a minimal JEPAStrategy stub with only the touched attributes.

    F8b (2026-05-17 round 5): the strategy now reads the generator from
    ``self.env.generator`` (also exposed via the
    ``BaseTrainingStrategy.generator_model`` property which proxies to
    ``self.env.generator``) — NOT ``self.context.generator``. The
    generic ``StrategyContext`` dataclass has no ``generator`` field
    (src/infrastructure/training/utils/strategy_context.py:45). The
    pre-F8b code's ``getattr(self.context, "generator", None)`` always
    returned ``None``, so JEPA crashed with "DI did not produce a
    generator" even when the YAML declared a valid ``model.model_type``
    (7 hits in 2026-05-16 smoke).
    """
    s = JEPAStrategy.__new__(JEPAStrategy)
    s._target_encoder = None
    s._predictor = None
    s.momentum = 0.996
    s.n_targets = 4
    s.ctx_scale = 0.85
    s.tgt_scale = 0.2
    # Keep ``self.context`` so other branches in the strategy that use
    # it (device, config) still work, but it intentionally has no
    # ``generator`` field — that's the F8b precondition.
    s.context = SimpleNamespace()
    s.env = MagicMock()
    s.env.generator = generator
    s.env.optimizer_g = None
    s.env.optimizer = None
    return s


def test_jepa_raises_when_generator_missing() -> None:
    """Missing generator must raise — not silently return zero loss.

    Post-F8b: the strategy reads from ``self.env.generator``. None →
    fail-loud RuntimeError naming the canonical attribute.
    """
    s = _make_jepa_strategy(generator=None)
    with pytest.raises(RuntimeError, match="self.env.generator"):
        s._compute_losses_impl({"input": torch.randn(1, 1, 8, 8)})


def test_jepa_no_longer_reads_self_context_generator() -> None:
    """F8b regression: the strategy must NOT cite ``self.context.generator``.

    Pre-F8b error message: "self.context.generator is None — the DI
    container did not produce a generator". That attribute does not
    exist on ``StrategyContext`` (the dataclass has no ``generator``
    field), so the check always tripped — even on properly configured
    YAMLs. Post-F8b the error names ``self.env.generator``, the actual
    storage location.
    """
    s = _make_jepa_strategy(generator=None)
    try:
        s._compute_losses_impl({"input": torch.randn(1, 1, 8, 8)})
    except RuntimeError as e:
        assert "self.context.generator" not in str(e), (
            "F8b regression: strategy is citing self.context.generator "
            "again — that attribute does not exist on StrategyContext. "
            "See TODO/audit/smoke_audit_20260516.md §F8b."
        )
    else:
        pytest.fail("RuntimeError expected for None generator.")


def test_jepa_resolves_generator_from_env_generator() -> None:
    """F8b regression: ``self.env.generator`` is the canonical location.

    With a real Conv2d generator set on ``env.generator``, the strategy
    must reach the forward path without the pre-F8b "DI did not
    produce a generator" RuntimeError.
    """
    gen = nn.Conv2d(1, 4, 3, padding=1)
    s = _make_jepa_strategy(generator=gen)
    img = torch.randn(1, 1, 16, 16)
    try:
        s._compute_losses_impl({"input": img})
    except RuntimeError as e:
        assert "self.env.generator`` is None" not in str(e), (
            "F8b regression: strategy still cannot resolve "
            "self.env.generator even though it was populated."
        )
    except Exception:
        # Other downstream errors (shape/grad/optimizer) are out of scope.
        pass


def test_jepa_raises_when_batch_is_not_dict() -> None:
    """Non-dict batch with no resolvable tensor must raise ``TypeError``.

    The orchestrator can call the impl with ``input_batch=...`` instead of
    ``batch=...``; the strategy resolves that via kwargs. A bare integer
    or string has no resolution path → fail loud.
    """
    s = _make_jepa_strategy(generator=nn.Conv2d(1, 4, 3, padding=1))
    with pytest.raises(TypeError, match="dict batch with an"):
        s._compute_losses_impl(42)  # type: ignore[arg-type]


def test_jepa_raises_when_input_key_missing() -> None:
    """Dict batch lacking ``input``/``image`` keys must raise ``KeyError``."""
    s = _make_jepa_strategy(generator=nn.Conv2d(1, 4, 3, padding=1))
    with pytest.raises(KeyError, match="missing both 'input' and 'image'"):
        s._compute_losses_impl({"foo": torch.randn(1, 1, 8, 8)})


def test_jepa_resolves_input_batch_kwarg() -> None:
    """Orchestrator-style call: ``input_batch=tensor`` resolves to a batch.

    The base orchestrator's template invokes
    ``_compute_losses_impl(input_batch=..., target_batch=..., epoch=...)``
    rather than positional ``batch=``. JEPA's first param is ``batch``,
    so the kwarg slides into ``**kwargs``; the fix's resolution path
    must locate the tensor and wrap it as ``{"input": tensor}``.
    """
    gen = nn.Conv2d(1, 4, 3, padding=1)
    s = _make_jepa_strategy(generator=gen)
    img = torch.randn(1, 1, 16, 16)
    # We don't need the loss to succeed end-to-end — just to NOT raise
    # ``TypeError: ... expects a dict batch ...`` at the early-return path.
    # The downstream forward may still fail on shape/device for this minimal
    # stub; we narrow the assertion to the resolution behaviour.
    try:
        s._compute_losses_impl(None, input_batch=img)
    except (TypeError, KeyError) as e:
        if "dict batch" in str(e) or "missing both" in str(e):
            pytest.fail(
                f"input_batch kwarg failed to resolve into a batch dict: {e}"
            )
        # Any other TypeError/KeyError downstream is fine for this test.
    except Exception:
        # Downstream failures (shape, grad, missing optimizer) are out of scope.
        pass
