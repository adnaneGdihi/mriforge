"""Model registry contract suite.

One parametric test per registered model key (all_models()).

Default-lane assertions (cheap — pure dict access):
- The key exists in MODEL_REGISTRY.
- Each registry entry has the documented fields: "class", "mode", "capabilities".
- "class" is a Python type (class).
- "mode" is a non-empty string.
- "capabilities" is a ModelCapabilities instance.
- No capabilities field that IS set to a non-None value contains an
  obviously invalid value (e.g. empty spatial_dims tuple).

``@pytest.mark.slow`` assertions (instantiation + forward-shape probe):
- :func:`tests.utils.minimal_builders.build_minimal_model` constructs a
  minimal runnable instance via a three-step ladder: (a) zero-arg, (b)
  signature introspection with capability-derived values, (c) explicit
  per-model ``_OVERRIDES`` dict.
- A forward pass with :func:`~tests.utils.minimal_builders.phantom_for`
  (domain-correct synthetic input) runs under ``torch.no_grad()`` and the
  output passes :func:`~tests.utils.minimal_builders.expected_output_ok`.

Two-mode gate behaviour
-----------------------
``CONTRACT_STRICT=0`` (default, unset):
    Any build or forward failure is *xfailed* (first-run discovery mode).
    The allowlist ``tests/contracts/_known_unbuildable.json`` is unused.
    Run the suite once on the cluster to discover which models genuinely
    can't be built, then populate the allowlist and flip to strict mode.

``CONTRACT_STRICT=1``:
    * Models listed in ``_known_unbuildable.json`` are xfailed (tracked debt).
    * Models **not** in the list that fail to build are HARD FAILs — they
      are regressions that must be fixed before merging.
    This mode is the steady-state gate on the cluster after the allowlist
    has been seeded from a first-run discovery pass.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from mriforge.models.capabilities import ModelCapabilities
from mriforge.models.registry import MODEL_REGISTRY
from tests.utils.minimal_builders import build_minimal_model, expected_output_ok, phantom_for
from tests.utils.registry_iterators import all_models

# ---------------------------------------------------------------------------
# Documented required field names for a MODEL_REGISTRY entry
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = ("class", "mode", "capabilities")

# ---------------------------------------------------------------------------
# Allowlist: known-unbuildable models (technical debt tracker)
# ---------------------------------------------------------------------------

_ALLOWLIST_PATH = Path(__file__).parent / "_known_unbuildable.json"

def _load_known_unbuildable() -> frozenset[str]:
    """Return the set of model names currently in the unbuildable allowlist."""
    try:
        data = json.loads(_ALLOWLIST_PATH.read_text())
        return frozenset(data.get("unbuildable", []))
    except Exception:
        return frozenset()


_KNOWN_UNBUILDABLE: frozenset[str] = _load_known_unbuildable()
_CONTRACT_STRICT: bool = os.environ.get("CONTRACT_STRICT", "0").strip() == "1"


# ---------------------------------------------------------------------------
# Torch-module ownership (derived, never transcribed)
# ---------------------------------------------------------------------------

def _owned_module_attrs(instance: Any) -> list[str]:
    """Attribute names on ``instance`` that hold an :class:`torch.nn.Module`.

    This is the discriminator between the two populations of registry entry that
    are not themselves ``nn.Module`` -- see
    :func:`test_model_instantiation_and_forward_shape`. It is read off the built
    instance rather than transcribed into a committed list, because a hand-list
    is a second source of truth that drifts the moment a class gains or loses a
    base (the ``test_metric_registry`` precedent, where the skip must read
    ``MetricsRegistry.needs(key)`` and never a hand-list).
    """
    return sorted(
        n for n, v in vars(instance).items() if isinstance(v, torch.nn.Module)
    )


# ---------------------------------------------------------------------------
# Default-lane: metadata well-formedness
# ---------------------------------------------------------------------------

@pytest.mark.registry_contract
@pytest.mark.parametrize("name", all_models(), ids=lambda n: n)
def test_model_registry_entry_has_required_fields(name: str) -> None:
    """Every MODEL_REGISTRY entry must contain the documented required keys."""
    entry = MODEL_REGISTRY.get(name)
    assert entry is not None, (
        f"Model '{name}' in all_models() but MODEL_REGISTRY.get('{name}') is None"
    )
    for field in _REQUIRED_FIELDS:
        assert field in entry, (
            f"Model '{name}' registry entry missing required field '{field}'. "
            f"Present keys: {sorted(entry.keys())}"
        )


@pytest.mark.registry_contract
@pytest.mark.parametrize("name", all_models(), ids=lambda n: n)
def test_model_registry_class_field_is_a_type(name: str) -> None:
    """The 'class' field of each registry entry must be a Python class."""
    entry = MODEL_REGISTRY[name]
    cls = entry.get("class")
    assert isinstance(cls, type), (
        f"Model '{name}' registry 'class' is {cls!r} (type={type(cls).__name__}), "
        f"expected a Python class."
    )


@pytest.mark.registry_contract
@pytest.mark.parametrize("name", all_models(), ids=lambda n: n)
def test_model_registry_mode_is_nonempty_string(name: str) -> None:
    """The 'mode' field of each registry entry must be a non-empty string."""
    entry = MODEL_REGISTRY[name]
    mode = entry.get("mode")
    assert isinstance(mode, str) and mode, (
        f"Model '{name}' registry 'mode' is {mode!r}, expected a non-empty string."
    )


@pytest.mark.registry_contract
@pytest.mark.parametrize("name", all_models(), ids=lambda n: n)
def test_model_registry_capabilities_type(name: str) -> None:
    """The 'capabilities' field must be a ModelCapabilities instance."""
    entry = MODEL_REGISTRY[name]
    caps = entry.get("capabilities")
    assert isinstance(caps, ModelCapabilities), (
        f"Model '{name}' registry 'capabilities' is {caps!r} "
        f"(type={type(caps).__name__}), expected ModelCapabilities."
    )


@pytest.mark.registry_contract
@pytest.mark.parametrize("name", all_models(), ids=lambda n: n)
def test_model_registry_capabilities_valid_values(name: str) -> None:
    """When capability fields are annotated, their values must be valid.

    Checks:
    - spatial_dims, if set, is a non-empty tuple of positive ints.
    - input_domain / output_domain, if set, are valid Domain literals or
      tuples thereof.
    - accepts_complex / expects_real_imag_interleaved / requires_paired_data,
      if set, are booleans.
    """
    from mriforge.models.capabilities import Domain

    entry = MODEL_REGISTRY[name]
    caps: ModelCapabilities = entry["capabilities"]

    # spatial_dims
    if caps.spatial_dims is not None:
        assert isinstance(caps.spatial_dims, tuple) and len(caps.spatial_dims) > 0, (
            f"Model '{name}' spatial_dims={caps.spatial_dims!r} must be a non-empty tuple."
        )
        for dim in caps.spatial_dims:
            assert isinstance(dim, int) and dim > 0, (
                f"Model '{name}' spatial_dims contains invalid value {dim!r}."
            )

    # domain fields — accept a single literal string or tuple of strings;
    # we do not hard-validate the literal set because Domain is a Literal
    # type alias (checked at type-check time, not runtime).
    for attr in ("input_domain", "output_domain"):
        val = getattr(caps, attr)
        if val is not None:
            valid = isinstance(val, (str, tuple))
            assert valid, (
                f"Model '{name}' {attr}={val!r} is neither a Domain string nor a tuple."
            )

    # boolean flags
    for attr in ("accepts_complex", "expects_real_imag_interleaved", "requires_paired_data"):
        val = getattr(caps, attr)
        if val is not None:
            assert isinstance(val, bool), (
                f"Model '{name}' {attr}={val!r} is not a bool."
            )


# ---------------------------------------------------------------------------
# Slow: capabilities-driven instantiation + forward-pass shape check
# ---------------------------------------------------------------------------

@pytest.mark.registry_contract
@pytest.mark.slow
@pytest.mark.parametrize("name", all_models(), ids=lambda n: n)
def test_model_instantiation_and_forward_shape(name: str) -> None:
    """Model must be constructible and produce a valid forward output.

    Uses :func:`~tests.utils.minimal_builders.build_minimal_model` (three-step
    ladder: zero-arg → introspection → explicit override) to build the model,
    then runs a synthetic forward pass under ``torch.no_grad()`` with a
    domain-correct input from :func:`~tests.utils.minimal_builders.phantom_for`.

    Gate behaviour is controlled by the ``CONTRACT_STRICT`` env var:

    ``CONTRACT_STRICT=0`` (default):
        Build or forward failures are *xfailed* with the failure reason.
        Run once on the cluster to discover the real unbuildable set, then
        populate ``tests/contracts/_known_unbuildable.json`` and flip to
        strict mode.

    ``CONTRACT_STRICT=1``:
        * Models in ``_known_unbuildable.json`` → xfail (debt, not regression).
        * Models NOT in the list that fail → HARD FAIL (regression).
        Add new overrides to
        :data:`tests.utils.minimal_builders._OVERRIDES` and remove the name
        from the allowlist to shrink the debt set over time.
    """
    entry = MODEL_REGISTRY[name]
    caps = entry.get("capabilities")

    # --- Build ---
    model, build_status = build_minimal_model(name)

    if model is None:
        reason = (
            f"Model '{name}' ({entry['class'].__qualname__}) could not be "
            f"built by build_minimal_model: {build_status}"
        )
        if _CONTRACT_STRICT and name not in _KNOWN_UNBUILDABLE:
            # Regression: model used to be buildable (or was never in the
            # allowlist), and now it can't be constructed.
            pytest.fail(reason + "\n[CONTRACT_STRICT=1] Add an override to "
                        "tests/utils/minimal_builders._OVERRIDES or add this "
                        "name to tests/contracts/_known_unbuildable.json.")
        else:
            pytest.xfail(reason)
        return  # unreachable; satisfies type-checker

    # --- Torch-module contract ---
    # An entry that is not an nn.Module cannot be `.eval()`-ed, and the two
    # reasons it might not be have OPPOSITE correct outcomes. So the predicate
    # is "does it OWN a network", not "is it one":
    #
    #   owns no nn.Module -> a process/schedule carrying buffers, no learnable
    #       state. Instantiate -> eval -> forward-shape does not apply to it.
    #       SKIP, permanently and by derivation.
    #   owns an nn.Module -> a network wrapped in a plain object. It has no
    #       parameters(), to(), state_dict() or eval(), so it cannot be
    #       optimised, moved to a device, or checkpointed. Real debt (#801).
    #       XFAIL, so it is recorded rather than escaping as an AttributeError.
    #
    # Before this gate `model.eval()` sat outside the guarded forward below, so
    # this whole population hard-failed even in discovery mode (CONTRACT_STRICT
    # is set nowhere) -- 9 of cluster job 8004252's failures.
    if not isinstance(model, torch.nn.Module):
        owned = _owned_module_attrs(model)
        del model
        gc.collect()
        if owned:
            pytest.xfail(
                f"Model '{name}' ({entry['class'].__qualname__}) is not an "
                f"nn.Module but owns one via {owned}: its parameters are "
                f"invisible to the optimiser, .to(device) and state_dict, so it "
                f"cannot train or checkpoint. Tracked debt (#801)."
            )
        pytest.skip(
            f"Model '{name}' ({entry['class'].__qualname__}) owns no nn.Module "
            f"-- it carries buffers/schedules, not learnable state -- so the "
            f"forward-shape contract does not apply."
        )

    # --- Forward pass ---
    x = phantom_for(caps)
    try:
        model.eval()
        with torch.no_grad():
            out = model(x)
    except Exception as exc:
        del model
        gc.collect()
        fwd_reason = (
            f"Model '{name}' forward raised {type(exc).__name__}: "
            f"{str(exc)[:120]}"
        )
        if _CONTRACT_STRICT and name not in _KNOWN_UNBUILDABLE:
            pytest.fail(fwd_reason + "\n[CONTRACT_STRICT=1] Fix the forward "
                        "pass or add the model to "
                        "tests/contracts/_known_unbuildable.json.")
        else:
            pytest.xfail(fwd_reason)
        return

    del model
    gc.collect()

    # --- Output sanity check ---
    # Routed through the SAME two-mode gate as build and forward above. It was
    # the one step of three that hard-failed in discovery mode, which is why
    # `medgs` -- which builds and forwards fine, then returns a flat [1024]
    # vector -- was a red test rather than recorded debt. Discovery mode exists
    # to enumerate; only CONTRACT_STRICT=1 is the gate that blocks.
    if not expected_output_ok(out, caps):
        shape_or_value = (
            f"shape={out.shape}" if isinstance(out, torch.Tensor) else str(out)[:80]
        )
        out_reason = (
            f"Model '{name}' forward output failed sanity check. "
            f"Type={type(out).__name__}, value={shape_or_value}"
        )
        if _CONTRACT_STRICT and name not in _KNOWN_UNBUILDABLE:
            pytest.fail(
                out_reason + "\n[CONTRACT_STRICT=1] Fix the forward output or "
                "add the model to tests/contracts/_known_unbuildable.json."
            )
        pytest.xfail(out_reason)


# ---------------------------------------------------------------------------
# Self-test: the ownership predicate must keep discriminating
# ---------------------------------------------------------------------------

@pytest.mark.registry_contract
def test_owned_module_attrs_separates_wrappers_from_processes() -> None:
    """The predicate must return attrs for a wrapper and nothing for a process.

    Guards the failure mode that would make the gate go quiet rather than red:
    if :func:`_owned_module_attrs` ever returned ``[]`` for everything -- a
    ``__slots__`` class where ``vars()`` is empty, a rename of the attribute
    holding the network -- every entry in the debt population would silently
    become a permanent SKIP instead of a tracked XFAIL, and #801 would look
    drained while nothing had been fixed. "Skipped" and "passed" are the same
    boolean to a reader scanning a summary line.
    """

    class _Wrapper:  # the shape of KANGenerator / RealESRGANDiscriminator
        def __init__(self) -> None:
            self.backbone = torch.nn.Conv2d(1, 1, 1)
            self.use_sigmoid = True

    class _Process:  # the shape of LaplaceDiffusion: buffers, no network
        def __init__(self) -> None:
            self.betas = torch.linspace(1e-4, 0.02, 8)
            self.timesteps = 8

    assert _owned_module_attrs(_Wrapper()) == ["backbone"]
    assert _owned_module_attrs(_Process()) == []
