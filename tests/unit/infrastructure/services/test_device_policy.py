"""Tests for ``DevicePolicy`` and convenience factories.

Targets ``spectramr.infrastructure.services.device_policy``. Centralized
device-resolution policy that replaces scattered ``.to(device)`` calls.
We exercise the CPU + AUTO branches plus the data-movement helpers —
CUDA-specific paths are GPU-marked and skipped on CPU runners.

The AUTO branch is asserted against the **accelerated-run contract**
(non-negotiable 9b, ``docs/accelerated_run_contract.rst``): with no accelerator
and no explicit opt-in it RAISES. A test that accepts "cpu or cuda" here is
asserting the silent fallback that contract exists to abolish.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectramr.core.compute_device import AcceleratorRequiredError
from spectramr.infrastructure.services.device_policy import (
    DevicePolicy,
    DevicePolicyConfig,
    DevicePolicyType,
    create_auto_device_policy,
    create_cpu_device_policy,
    create_cuda_device_policy,
    create_specific_device_policy,
    get_default_device_policy,
    move_to_device_auto,
    set_default_device_policy,
)

# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


def test_default_construction_uses_auto() -> None:
    """No config → AUTO policy."""
    policy = DevicePolicy()
    assert policy.config.policy_type == DevicePolicyType.AUTO


def test_specific_policy_requires_device_string() -> None:
    """SPECIFIC policy without ``specific_device`` → ``ValueError``."""
    cfg = DevicePolicyConfig(
        policy_type=DevicePolicyType.SPECIFIC, specific_device=None
    )
    with pytest.raises(ValueError, match="specific_device must be provided"):
        DevicePolicy(cfg)


def test_memory_threshold_only_supported_with_cuda() -> None:
    """``memory_threshold_mb`` requires CUDA — raises on CPU-only."""
    if torch.cuda.is_available():
        pytest.skip("CUDA available — skipping CPU-only check")
    cfg = DevicePolicyConfig(
        policy_type=DevicePolicyType.AUTO, memory_threshold_mb=1024
    )
    with pytest.raises(ValueError, match="Memory threshold only supported"):
        DevicePolicy(cfg)


# ---------------------------------------------------------------------------
# CPU policy
# ---------------------------------------------------------------------------


def test_cpu_policy_returns_cpu() -> None:
    """CPU policy always returns CPU."""
    policy = create_cpu_device_policy()
    device = policy.get_device()
    assert device.type == "cpu"


def test_cpu_policy_caches_device() -> None:
    """``get_device`` is cached — repeated calls return the same instance."""
    policy = create_cpu_device_policy()
    d1 = policy.get_device()
    d2 = policy.get_device()
    assert d1 == d2


# ---------------------------------------------------------------------------
# CUDA policy with fallback
# ---------------------------------------------------------------------------


def test_cuda_policy_falls_back_to_cpu_when_unavailable() -> None:
    """When CUDA is unavailable but ``fallback_to_cpu=True``, returns CPU."""
    if torch.cuda.is_available():
        pytest.skip("CUDA available — skipping fallback test")
    policy = create_cuda_device_policy(fallback_to_cpu=True)
    assert policy.get_device().type == "cpu"


def test_cuda_policy_no_fallback_raises_when_unavailable() -> None:
    """``fallback_to_cpu=False`` raises ``RuntimeError`` when CUDA is unavailable."""
    if torch.cuda.is_available():
        pytest.skip("CUDA available")
    policy = create_cuda_device_policy(fallback_to_cpu=False)
    with pytest.raises(RuntimeError, match="CUDA requested but not available"):
        policy.get_device()


# ---------------------------------------------------------------------------
# AUTO policy
# ---------------------------------------------------------------------------


def test_auto_policy_honours_the_accelerated_run_contract() -> None:
    """AUTO gives CUDA, or RAISES — it does not quietly settle for CPU.

    This asserted ``device.type in {"cpu", "cuda"}``, the pre-2026 behaviour: it
    passed on a GPU box by returning cuda and failed on a CPU node with
    ``AcceleratorRequiredError`` — exactly where non-negotiable 9b is meant to
    bite. Satisfying the old assertion by turning on ``fallback_to_cpu`` would
    have been the worse repair: that IS the silent CPU fallback 9b abolished,
    the one that let a GPU-less node run ~100x slower and still report success.

    Both arms are pinned, so the test means the same thing on either host.
    """
    policy = create_auto_device_policy()

    if torch.cuda.is_available():
        assert policy.get_device().type == "cuda"
        return

    with pytest.raises(AcceleratorRequiredError):
        policy.get_device()


def test_auto_policy_cpu_opt_in_is_explicit() -> None:
    """The user may still dictate CPU — but only by asking for it."""
    if torch.cuda.is_available():
        pytest.skip("CUDA available — the opt-in branch is unreachable")

    cfg = DevicePolicyConfig(policy_type=DevicePolicyType.AUTO, fallback_to_cpu=True)
    assert DevicePolicy(cfg).get_device().type == "cpu"


# ---------------------------------------------------------------------------
# SPECIFIC policy
# ---------------------------------------------------------------------------


def test_specific_policy_cpu_device() -> None:
    """``cpu`` specific device works without CUDA."""
    policy = create_specific_device_policy("cpu")
    assert policy.get_device().type == "cpu"


def test_specific_cuda_on_a_cuda_less_host_raises() -> None:
    """An explicit ``cuda:0`` with no CUDA is a broken environment, not a preference.

    This asserted the opposite — "Falls back by default since cfg.fallback_to_cpu
    defaults True" — and that comment is simply out of date: the default is
    ``False``, flipped when the accelerated-run contract landed. 9b is explicit
    that an explicit ``cuda`` request on a CUDA-less host raises
    *unconditionally*.

    Not in the job-8000966 log; found by re-running this file under
    ``CUDA_VISIBLE_DEVICES=""``, which reproduces the CPU array on a GPU box.
    """
    if torch.cuda.is_available():
        pytest.skip("CUDA available — the unavailable branch is unreachable")

    assert DevicePolicyConfig(policy_type=DevicePolicyType.CPU).fallback_to_cpu is False

    with pytest.raises(RuntimeError, match="CUDA"):
        create_specific_device_policy("cuda:0").get_device()


def test_specific_cuda_falls_back_only_on_the_explicit_opt_in() -> None:
    """``fallback_to_cpu=True`` is the caller dictating CPU, which is allowed."""
    if torch.cuda.is_available():
        pytest.skip("CUDA available — the unavailable branch is unreachable")

    cfg = DevicePolicyConfig(
        policy_type=DevicePolicyType.SPECIFIC,
        specific_device="cuda:0",
        fallback_to_cpu=True,
    )
    assert DevicePolicy(cfg).get_device().type == "cpu"


# ---------------------------------------------------------------------------
# move_to_device — data movement
# ---------------------------------------------------------------------------


def test_move_tensor_to_cpu() -> None:
    """Tensor is moved to the policy device."""
    policy = create_cpu_device_policy()
    t = torch.randn(4)
    out = policy.move_to_device(t)
    assert out.device.type == "cpu"


def test_move_none_returns_none() -> None:
    """``None`` is returned untouched."""
    policy = create_cpu_device_policy()
    assert policy.move_to_device(None) is None


def test_move_list_recurses() -> None:
    """Lists are recursively moved."""
    policy = create_cpu_device_policy()
    out = policy.move_to_device([torch.zeros(2), torch.ones(3)])
    for t in out:
        assert t.device.type == "cpu"


def test_move_tuple_preserves_type() -> None:
    """Tuple input → tuple output."""
    policy = create_cpu_device_policy()
    out = policy.move_to_device((torch.zeros(2), torch.ones(3)))
    assert isinstance(out, tuple)


def test_move_dict_recurses() -> None:
    """Dicts are recursively moved (preserving keys)."""
    policy = create_cpu_device_policy()
    inp = {"a": torch.zeros(2), "b": torch.ones(3)}
    out = policy.move_to_device(inp)
    assert {"a", "b"} == out.keys()
    for v in out.values():
        assert v.device.type == "cpu"


def test_move_model_moves_params_and_buffers() -> None:
    """Modules: parameters and buffers all end on the target device."""
    model = nn.Linear(4, 4)
    model.register_buffer("scratch", torch.zeros(4))
    policy = create_cpu_device_policy()
    out = policy.move_to_device(model)
    for p in out.parameters():
        assert p.device.type == "cpu"
    for b in out.buffers():
        assert b.device.type == "cpu"


def test_move_non_movable_returned_as_is() -> None:
    """A plain int passes through unchanged."""
    policy = create_cpu_device_policy()
    assert policy.move_to_device(42) == 42


def test_move_with_dtype_conversion() -> None:
    """``dtype`` argument converts tensor dtype."""
    policy = create_cpu_device_policy()
    t = torch.zeros(4, dtype=torch.float32)
    out = policy.move_to_device(t, dtype=torch.float64)
    assert out.dtype == torch.float64


# ---------------------------------------------------------------------------
# is_device_available
# ---------------------------------------------------------------------------


def test_cpu_always_available() -> None:
    """``is_device_available(torch.device('cpu')) == True``."""
    policy = create_cpu_device_policy()
    assert policy.is_device_available(torch.device("cpu")) is True


def test_default_check_uses_policy_device() -> None:
    """Without ``device``, checks the policy device."""
    policy = create_cpu_device_policy()
    assert policy.is_device_available() is True


# ---------------------------------------------------------------------------
# get_device_info
# ---------------------------------------------------------------------------


def test_device_info_cpu_keys() -> None:
    """Device info dict has ``device``, ``type``, ``policy``."""
    policy = create_cpu_device_policy()
    info = policy.get_device_info()
    assert {"device", "type", "policy"} <= info.keys()
    assert info["type"] == "cpu"


def test_clear_cache_idempotent() -> None:
    """``clear_cache`` is safe to call multiple times."""
    policy = create_cpu_device_policy()
    _ = policy.get_device()
    policy.clear_cache()
    policy.clear_cache()  # no exception
    # After clearing, get_device works again.
    assert policy.get_device().type == "cpu"


# ---------------------------------------------------------------------------
# Default global policy
# ---------------------------------------------------------------------------


def test_default_policy_is_auto() -> None:
    """Default global policy is AUTO."""
    policy = get_default_device_policy()
    assert isinstance(policy, DevicePolicy)


def test_set_default_policy_persists() -> None:
    """``set_default_device_policy`` overrides the singleton."""
    custom = create_cpu_device_policy()
    set_default_device_policy(custom)
    assert get_default_device_policy() is custom
    # Restore: AUTO will be re-created lazily next time the module loads,
    # but for this session reset to AUTO directly.
    set_default_device_policy(create_auto_device_policy())


def test_move_to_device_auto_uses_default_policy() -> None:
    """``move_to_device_auto`` delegates to the default policy."""
    set_default_device_policy(create_cpu_device_policy())
    out = move_to_device_auto(torch.zeros(4))
    assert out.device.type == "cpu"
