"""``get_model_factory`` with ``config=None``, and the deleted unknown-model policy.

F18 (smoke_audit_20260521 round 10): ``experiment_118_symbolic_regression``
crashed at pipeline build with ``'NoneType' object has no attribute
'unknown_model_policy'`` because the policy-resolution step dereferenced a
``config`` documented as optional. The policy itself was deleted on 2026-09-03
(#1338): every creation path raises ``ModelCreationError`` on a name the
registry does not hold, the fallback handler returned ``None`` on every call,
and the stored policy had no reader, so the "warn" fallback advertised a
degradation that never existed. What stays pinned here: ``config=None`` is a
legitimate input, the ``strict`` flag that only selected the policy is refused,
and an unknown model type raises rather than degrading.
"""

from __future__ import annotations

import pytest


def test_get_model_factory_handles_none_config() -> None:
    from spectramr.models.factories.model_factory import get_model_factory

    assert get_model_factory() is not None
    assert get_model_factory(config=None) is not None


def test_get_model_factory_refuses_the_deleted_strict_flag() -> None:
    """``strict`` selected the deleted policy and nothing else; the flag is gone with it."""
    from spectramr.models.factories.model_factory import get_model_factory

    with pytest.raises(TypeError):
        get_model_factory(config=None, strict=True)


def test_get_model_factory_handles_object_config_without_field() -> None:
    from spectramr.models.factories.model_factory import get_model_factory

    class _Bare:
        pass

    assert get_model_factory(config=_Bare()) is not None


def test_factory_no_longer_accepts_the_deleted_policy() -> None:
    from spectramr.models.factories.model_factory import ModelFactory

    with pytest.raises(TypeError):
        ModelFactory(unknown_model_policy="warn_fallback")
    assert not hasattr(ModelFactory(), "_unknown_model_policy")
    assert not hasattr(ModelFactory(), "_fallback_handler")


def test_unknown_generator_type_raises_instead_of_degrading() -> None:
    """The planted violation for the deleted fallback: no policy softens this."""
    from spectramr.models.factories.model_factory import ModelCreationError, get_model_factory

    with pytest.raises(ModelCreationError, match="not registered"):
        get_model_factory().create_generator("no_such_model_type_2026_09_03")
