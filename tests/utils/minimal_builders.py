"""Capabilities-driven minimal model builder for contract tests.

Provides three functions used by :mod:`tests.contracts.test_model_registry`
to replace the hollow zero-arg-only construction attempt:

* :func:`build_minimal_model` — constructs a minimal but runnable instance
  of any registered model, falling back through a three-step ladder.
* :func:`phantom_for` — synthesises a domain-correct input tensor that
  matches the model's declared :class:`~mriforge.models.capabilities.ModelCapabilities`.
* :func:`expected_output_ok` — loose sanity check on a forward-pass output
  (Tensor or container holding one) with positive spatial dims.

Build-strategy ladder
---------------------
For each registered model class the builder tries, in order:

1. **Zero-arg** — ``cls()`` with no arguments.
2. **Signature introspection** — inspect ``cls.__init__``, fill required
   positional params (no default) with sensible values derived from:

   * ``in_channels`` / ``out_channels`` / ``channels`` / ``dim`` /
     ``hidden_dim`` / ``hidden_channels`` / ``features`` / ``base_channels`` /
     ``num_channels`` → integer inferred from capabilities channel spec (1 or 2)
   * ``spatial_dims`` / ``ndim`` → inferred from capabilities (2 or 3)
   * ``spatial_shape`` / ``input_shape`` → small tuple ``(32,32)`` or ``(8,8,8)``
   * ``in_caps`` / ``out_caps`` → small integer (4)
   * ``in_dim`` / ``out_dim`` / ``capsule_dim`` → small integer (8)
   * ``num_capsules`` → 4
   * ``num_variables`` → 4
   * ``basis_dim`` / ``time_emb_dim`` → 16
   * ``denoising_model`` / ``base_model`` / ``base_generator`` / ``model`` → a
     zero-weight 1-layer ``nn.Linear(1, 1)`` stand-in
   * ``generator_factory`` → a callable that returns ``nn.Linear(1, 1)``
   * ``num_scanners`` → 2
   * ``in_ch`` / ``out_ch`` → same as channels
   * ``growth`` / ``growth_rate`` → 16
   * ``ch`` / ``mid_ch`` / ``bottleneck_ch`` → channels value (1 or 2)
   * ``routing_iters`` and other ints → small default (3 or 4)
   * ``in_features`` / ``out_features`` → 16
   * ``gate_ch`` / ``skip_ch`` / ``inter_ch`` → channels value

3. **Explicit overrides** — ``_OVERRIDES`` dict maps model name or class
   qualname (case-sensitive) to a kwargs dict that is passed directly to
   the constructor, bypassing introspection.  Used for models whose
   signatures need precise values (e.g. ``CSMNOOperator`` requires a
   non-trivial ``spatial_shape``).

Returns ``(instance, "ok")`` on success or ``(None, reason_str)`` on
failure at every step.

Two-mode allowlist gate
-----------------------
This module does NOT enforce the allowlist — that is the responsibility of
the calling test.  See :mod:`tests.contracts.test_model_registry` for the
``CONTRACT_STRICT`` gate.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import torch
import torch.nn as nn

from mriforge.models.capabilities import ModelCapabilities
from mriforge.models.init_registry import populate_model_registry
from mriforge.models.registry import MODEL_REGISTRY

# Trigger full registry population so that _BackboneAlias models can
# resolve their backbone (configurable_unet) at construction time.
populate_model_registry()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Param-name → minimal filler value table
# ---------------------------------------------------------------------------
# These are heuristic defaults applied when a required positional param
# (no default) matches a known name pattern.  Channel values are 1 or 2
# depending on whether the model expects complex / interleaved input.

_INT_SMALL = 4      # capsule counts, routing iters, num_scanners, etc.
_DIM_SMALL = 8      # capsule_dim, in_dim, out_dim, inter_ch, gate_ch
_EMB_SMALL = 16     # time_emb_dim, basis_dim, in_features, out_features
_GROWTH = 16        # growth / growth_rate for dense blocks

# Param names that should receive the channels value (int from capabilities)
_CHANNEL_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "in_channels",
        "out_channels",
        "channels",
        "dim",
        "hidden_dim",
        "hidden_channels",
        "base_channels",
        "num_channels",
        "in_ch",
        "out_ch",
        "ch",
        "mid_ch",
        "bottleneck_ch",
        "num_features",
        "feature_dim",
        "feature_channels",
    }
)

# Param names that receive spatial_dims int (2 or 3)
_SDIM_PARAM_NAMES: frozenset[str] = frozenset({"spatial_dims", "ndim"})

# Param names that receive a module stand-in (nn.Linear)
_MODULE_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "denoising_model",
        "base_model",
        "base_generator",
        "model",
        "artifact_simulator",
        "base_dataset",
    }
)

# Param names that receive a tuple spatial shape
_SHAPE_PARAM_NAMES: frozenset[str] = frozenset({"spatial_shape", "input_shape"})

# Other specific param → (value | callable) mappings (static)
_STATIC_PARAM_MAP: dict[str, Any] = {
    "num_capsules": _INT_SMALL,
    "num_variables": _INT_SMALL,
    "num_scanners": 2,
    "num_domains": 2,
    "in_caps": _INT_SMALL,
    "out_caps": _INT_SMALL,
    "in_dim": _DIM_SMALL,
    "out_dim": _DIM_SMALL,
    "capsule_dim": _DIM_SMALL,
    "inter_ch": _DIM_SMALL,
    "gate_ch": _DIM_SMALL,
    "skip_ch": _DIM_SMALL,
    "time_emb_dim": _EMB_SMALL,
    "basis_dim": _EMB_SMALL,
    "in_features": _EMB_SMALL,
    "out_features": _EMB_SMALL,
    "growth": _GROWTH,
    "growth_rate": _GROWTH,
    "routing_iters": 3,
    "latent_dim": 8,
    "aug_policy": None,          # type: ignore[assignment]  ignored if None triggers errors
    "augmentation_policy": None,
}


# ---------------------------------------------------------------------------
# Per-model explicit overrides
# Maps model registry name → kwargs dict passed directly to cls().
# ---------------------------------------------------------------------------

_OVERRIDES: dict[str, dict[str, Any]] = {
    # CSMNOOperator, HSMNOOperator, CMNOOperator, etc. require spatial_shape
    "cs_mno_operator": {"in_channels": 1, "out_channels": 1, "spatial_shape": (32, 32)},
    "hs_mno_operator": {"in_channels": 1, "out_channels": 1, "spatial_shape": (32, 32)},
    "c_mno_operator":  {"in_channels": 1, "out_channels": 1, "spatial_shape": (32, 32)},
    "s_mno_operator":  {"in_channels": 1, "out_channels": 1, "spatial_shape": (32, 32)},
    "me_mno_operator": {"in_channels": 1, "out_channels": 1, "spatial_shape": (32, 32)},
    "g_mno_operator":  {
        "in_channels": 1, "out_channels": 1,
        # GMNOOperator requires exactly 3-D spatial_shape
        "spatial_shape": (8, 8, 8),
    },
    # Models that wrap a mandatory base model / base generator
    "adversarial_purification": {"base_model": nn.Linear(1, 1)},
    "laplace_diffusion_generator": {"denoising_model": nn.Linear(1, 1)},
    "rician_diffusion": {"denoising_model": nn.Linear(1, 1)},
    "scanner_adaptive_generator": {"base_model": nn.Linear(1, 1)},
    "mc_dropout": {"base_generator": nn.Linear(1, 1)},
    "deep_ensemble": {"generator_factory": lambda: nn.Linear(1, 1)},
    # Wrapper models that require an inner model + latent_dim
    "latent_input_wrapper": {"model": nn.Linear(1, 1), "latent_dim": 8},
    # Domain adaptation models with required int args
    "domain_discriminator": {"in_channels": 1},
    "cross_scanner_adaptation": {"in_channels": 1, "num_scanners": 2},
    "multi_domain_extractor": {"in_channels": 1},
    # complex_unet expects required in/out channels
    "complex_unet": {"in_channels": 1, "out_channels": 1},
    # mamba variants with required channels
    "mamba_unet": {"in_channels": 1, "out_channels": 1},
    "mamba_unet_v2": {"in_channels": 1, "out_channels": 1},
}


# ---------------------------------------------------------------------------
# Helper: minimal stand-in module
# ---------------------------------------------------------------------------

def _stub_module() -> nn.Module:
    """Return a trivial 1-layer module usable as a positional-arg stand-in."""
    return nn.Linear(1, 1)


# ---------------------------------------------------------------------------
# Helper: infer sensible channel count from capabilities
# ---------------------------------------------------------------------------

def _channels_from_caps(caps: ModelCapabilities | None) -> int:
    """Return 1 or 2 depending on declared complex expectations."""
    if caps is None:
        return 1
    if caps.expects_real_imag_interleaved:
        return 2
    if caps.accepts_complex:
        return 1  # complex64 single channel
    return 1


def _sdims_from_caps(caps: ModelCapabilities | None) -> int:
    """Return 2 or 3 depending on declared spatial dims."""
    if caps is None:
        return 2
    if caps.spatial_dims and 3 in caps.spatial_dims and 2 not in caps.spatial_dims:
        return 3
    return 2


def _shape_from_caps(caps: ModelCapabilities | None) -> tuple[int, ...]:
    """Return a small spatial shape consistent with the model's declared dims."""
    if _sdims_from_caps(caps) == 3:
        return (8, 8, 8)
    return (32, 32)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def _fill_required_param(
    name: str,
    annotation: Any,
    caps: ModelCapabilities | None,
) -> Any:
    """Return a sensible minimal value for a required constructor param."""
    ch = _channels_from_caps(caps)
    sdims = _sdims_from_caps(caps)
    shape = _shape_from_caps(caps)

    if name in _CHANNEL_PARAM_NAMES:
        return ch
    if name in _SDIM_PARAM_NAMES:
        return sdims
    if name in _SHAPE_PARAM_NAMES:
        return shape
    if name in _MODULE_PARAM_NAMES:
        return _stub_module()
    if name == "generator_factory":
        return lambda: _stub_module()
    if name in _STATIC_PARAM_MAP:
        return _STATIC_PARAM_MAP[name]
    # Fallback: if annotated as int or no annotation, use small int
    if annotation is inspect.Parameter.empty or annotation is int:
        return _INT_SMALL
    # nn.Module stand-in for any Module-typed arg
    try:
        if issubclass(annotation, nn.Module):
            return _stub_module()
    except TypeError:
        pass
    # Last resort: small int
    return _INT_SMALL


def build_minimal_model(name: str) -> tuple[nn.Module | None, str]:
    """Construct a minimal runnable instance of a registered model.

    Three-step build ladder:

    1. Zero-arg construction: ``cls()``.
    2. Signature introspection: fill required params with heuristic values
       derived from declared capabilities.
    3. Explicit override: look up ``_OVERRIDES[name]`` and pass kwargs directly.

    Args:
        name: Key in ``MODEL_REGISTRY``.

    Returns:
        ``(instance, "ok")`` on success; ``(None, reason_str)`` on failure.
        The reason string starts with ``"zero_arg:"``, ``"introspect:"``, or
        ``"override:"`` to indicate which step was attempted last.
    """
    entry = MODEL_REGISTRY.get(name)
    if entry is None:
        return None, f"not_registered:{name}"

    cls: type = entry["class"]
    caps: ModelCapabilities | None = entry.get("capabilities")

    # --- Step 1: zero-arg ---
    try:
        instance = cls()
        return instance, "ok"
    except TypeError:
        pass  # needs args — continue to step 2
    except Exception as exc:
        last_err = f"zero_arg:{type(exc).__name__}:{str(exc)[:80]}"
        # Don't stop — try steps 2 and 3 anyway; sometimes zero-arg raises a
        # non-TypeError due to internal logic but the model is fine with args.

    # --- Step 3 (priority): check explicit override first ---
    if name in _OVERRIDES:
        override_kwargs = dict(_OVERRIDES[name])
        try:
            instance = cls(**override_kwargs)
            return instance, "ok"
        except Exception as exc:
            return None, f"override:{type(exc).__name__}:{str(exc)[:120]}"

    # --- Step 2: signature introspection ---
    try:
        sig = inspect.signature(cls.__init__)
        kwargs: dict[str, Any] = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if param.default is inspect.Parameter.empty:
                # Required param — fill it
                value = _fill_required_param(param_name, param.annotation, caps)
                kwargs[param_name] = value

        instance = cls(**kwargs)
        return instance, "ok"
    except Exception as exc:
        return None, f"introspect:{type(exc).__name__}:{str(exc)[:120]}"


# ---------------------------------------------------------------------------
# Phantom tensor factory
# ---------------------------------------------------------------------------

def phantom_for(caps: ModelCapabilities | None) -> torch.Tensor:
    """Build a domain-correct synthetic input tensor.

    Shape convention:

    * 2-D: ``[1, C, 32, 32]``
    * 3-D: ``[1, C, 16, 16, 16]``

    Where ``C`` is:

    * 1 — real-valued image / latent / pde_grid / kspace (single channel)
    * 2 — real/imag interleaved (``expects_real_imag_interleaved=True``)
    * 1 with dtype ``torch.complex64`` — ``accepts_complex=True``

    Args:
        caps: Declared capabilities of the model, or ``None`` for defaults.

    Returns:
        A CPU ``torch.Tensor`` of the appropriate shape and dtype.
    """
    if caps is None:
        return torch.randn(1, 1, 64, 64)

    sdims = _sdims_from_caps(caps)

    if sdims == 3:
        spatial = (16, 16, 16)
    else:
        spatial = (32, 32)  # type: ignore[assignment]

    if caps.accepts_complex and not caps.expects_real_imag_interleaved:
        # complex64 single channel
        ch = 1
        real = torch.randn(1, ch, *spatial)
        imag = torch.randn(1, ch, *spatial)
        return torch.complex(real, imag)

    if caps.expects_real_imag_interleaved:
        ch = 2
    else:
        ch = 1

    return torch.randn(1, ch, *spatial)


# ---------------------------------------------------------------------------
# Output sanity check
# ---------------------------------------------------------------------------

def _find_first_tensor(obj: Any) -> torch.Tensor | None:
    """Recursively find the first Tensor in obj (Tensor, tuple, list, dict)."""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            t = _find_first_tensor(v)
            if t is not None:
                return t
    if isinstance(obj, (list, tuple)):
        for v in obj:
            t = _find_first_tensor(v)
            if t is not None:
                return t
    return None


def expected_output_ok(out: Any, caps: ModelCapabilities | None) -> bool:  # noqa: ARG001
    """Loose sanity check that a forward output looks plausible.

    Accepts ``torch.Tensor``, ``tuple``, ``list``, or ``dict`` containing at
    least one Tensor.  The check is intentionally lenient:

    * Must contain at least one Tensor.
    * That Tensor must have ``ndim >= 2`` (batch dim + at least one spatial).
    * All spatial dimensions must be positive.
    * The ``caps`` parameter is accepted for future tightening but unused now
      because many models legitimately change channel count.

    Args:
        out: Forward-pass return value.
        caps: Model capabilities (reserved for future strict checks).

    Returns:
        ``True`` if the output passes the sanity check, ``False`` otherwise.
    """
    t = _find_first_tensor(out)
    if t is None:
        return False
    if t.ndim < 2:
        return False
    if any(s <= 0 for s in t.shape):
        return False
    return True
