"""Regression test for F18 (smoke_audit_20260521 round 10) —
``get_model_factory`` must handle ``config=None`` cleanly.

Root cause: ``experiment_118_symbolic_regression`` crashed at
training-pipeline-build with
``'NoneType' object has no attribute 'unknown_model_policy'``.

The function's signature already documents ``config: Any | None = None``
as a legitimate input, but the dereference at the policy-resolution
step (``config.unknown_model_policy``) and the strict_kwargs lookup
(``config.strict_model_kwargs``) both assumed config is non-None.
F18 adds explicit None branches with the historical-default values
(``"warn_fallback"`` policy, ``False`` strict_kwargs).

This test pins:
1. ``get_model_factory()`` (no args) doesn't crash.
2. ``get_model_factory(config=None)`` returns a factory.
3. ``get_model_factory(config=None, strict=True)`` still respects strict.
4. The returned factory uses the documented default policy.
"""

from __future__ import annotations

# No DeprecationWarning suppression needed any more: the warning moved off
# ``ModelFactory.__init__`` onto ``ModelFactory.create_model``, and
# ``get_model_factory`` only constructs. Removing the filter is the point --
# with it in place, a deprecation reappearing on construction would be
# invisible here, and pyproject's promote-to-error could not gate this path.


def test_get_model_factory_handles_none_config() -> None:
    """The function must not crash with ``config=None`` (the
    documented default). Pre-fix behavior raised
    ``AttributeError: 'NoneType' object has no attribute
    'unknown_model_policy'``."""
    from mriforge.models.factories.model_factory import get_model_factory

    # Pre-fix this raised AttributeError. Post-fix returns a factory.
    factory = get_model_factory()  # all defaults, config=None
    assert factory is not None
    # The factory must carry the documented default policy.
    assert factory._unknown_model_policy == "warn_fallback", (
        "Default policy must be 'warn_fallback' (historical loose "
        "behavior). See TODO/audit/smoke_audit_20260521.md §F18."
    )


def test_get_model_factory_handles_explicit_none() -> None:
    """``config=None`` explicit must also be safe."""
    from mriforge.models.factories.model_factory import get_model_factory

    factory = get_model_factory(config=None)
    assert factory is not None


def test_get_model_factory_strict_override_still_works_with_none_config() -> None:
    """When ``strict=True`` is passed alongside ``config=None``, the
    strict override must take precedence over the None-default
    policy."""
    from mriforge.models.factories.model_factory import get_model_factory

    factory = get_model_factory(config=None, strict=True)
    assert factory._unknown_model_policy == "strict"


def test_get_model_factory_handles_object_config_without_field() -> None:
    """If config is a plain object lacking ``unknown_model_policy``,
    the function must still fall back to ``warn_fallback`` instead
    of raising AttributeError. The pre-fix ``config.unknown_model_policy``
    raised; the post-fix ``getattr(config, ..., 'warn_fallback')``
    handles it."""
    from types import SimpleNamespace

    from mriforge.models.factories.model_factory import get_model_factory

    # SimpleNamespace without the attribute
    cfg = SimpleNamespace()
    factory = get_model_factory(config=cfg)
    assert factory._unknown_model_policy == "warn_fallback"
