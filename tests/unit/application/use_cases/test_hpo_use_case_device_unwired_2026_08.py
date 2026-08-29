"""``HPORequest.device``: honest type, and a characterization pin for #1389.

D01#4 flipped ``hpo_parser``'s ``--device`` default from ``"cuda"`` to ``None``.
Tracing the value before flipping it turned up that it goes nowhere:
``HPOCoordinator.execute_hpo`` binds ``device`` and never reads it, and
``subprocess_training_objective`` has no ``device`` parameter at all, so every
trial inherits the base YAML's device. Filed as #1389 and deliberately NOT
fixed in the same PR -- wiring it means changing what the trial YAML contains,
which is a behaviour change on the cluster path, not a launch-surface fix.

``test_device_is_not_yet_forwarded`` fails the moment somebody wires it. That is
the intent: at that point delete it and assert the forwarding instead.
"""

import dataclasses
import inspect

from mriforge.application.use_cases.hpo_use_case import HPORequest
from mriforge.infrastructure.coordination.hpo_coordinator import HPOCoordinator


def test_request_device_defaults_to_none():
    """``None`` = "not requested here". A ``"cuda"`` default cannot be told
    apart from a user asking for CUDA, and only the latter may override
    ``FORCE_CPU``."""
    field = {f.name: f for f in dataclasses.fields(HPORequest)}["device"]
    assert field.default is None
    assert HPORequest(config_path="x.yaml", model_types=["unet"]).device is None


def test_request_device_accepts_an_explicit_value():
    assert HPORequest(config_path="x.yaml", model_types=["unet"], device="cpu").device == "cpu"


def test_coordinator_device_defaults_to_none():
    param = inspect.signature(HPOCoordinator.execute_hpo).parameters["device"]
    assert param.default is None


def test_device_is_not_yet_forwarded():
    """Characterization pin for #1389 — see the module docstring."""
    source = inspect.getsource(HPOCoordinator.execute_hpo)
    body = source[source.index('"""', source.index('"""') + 3) + 3 :]
    assert "device" not in body, (
        "HPOCoordinator.execute_hpo now uses `device` — #1389 looks fixed. "
        "Delete this test and replace it with one asserting the device reaches "
        "the written trial config."
    )
