"""AMP must autocast on the device the run actually resolved.

``MixedPrecisionIntegrationHelper`` used to derive its own device from
``torch.cuda.is_available()``. That made it a second device resolver
(non-negotiable 9b/17): on a CUDA box where the user opted into CPU it
autocast on ``cuda`` while the tensors sat on ``cpu``, and on a multi-GPU node
it named ``cuda`` while the run was pinned elsewhere.
"""

from __future__ import annotations

import ast
import pathlib

import torch

from spectramr.infrastructure.training.mixed_precision import (
    MixedPrecisionIntegrationHelper,
)

_SOURCE = pathlib.Path(
    "src/spectramr/infrastructure/training/mixed_precision.py"
)


def _config(**over: object):
    from spectramr.infrastructure.training.mixed_precision import MixedPrecisionConfig

    return MixedPrecisionConfig(**over)


def test_injected_device_decides_the_autocast_device_type() -> None:
    helper = MixedPrecisionIntegrationHelper(_config(), torch.device("cpu"))
    assert helper.device_type == "cpu"


def test_a_string_device_is_accepted_and_normalised() -> None:
    helper = MixedPrecisionIntegrationHelper(_config(), "cpu")
    assert helper.device_type == "cpu"


def test_indexed_device_keeps_its_type_not_its_index() -> None:
    """``cuda:1`` is still ``cuda`` for autocast, but must not become ``cuda:0``."""
    helper = MixedPrecisionIntegrationHelper(_config(), torch.device("cuda", 1))
    assert helper.device_type == "cuda"


def test_the_forbidden_availability_ternary_is_gone() -> None:
    """The literal expression non-negotiable 9b forbids, checked at the source.

    A behavioural test cannot see this on a CPU-only host: the old code and the
    new one both yield ``cpu`` here. Only reading the source distinguishes them.
    """
    tree = ast.parse(_SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            rendered = ast.unparse(node)
            assert "cuda.is_available" not in rendered, (
                f"forbidden device ternary reintroduced: {rendered}"
            )
