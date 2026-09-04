"""Phase 0 #2 — the Tier-2 forward probe must report an unresolvable
``model_type`` as a clean instantiation failure, never a raw traceback.

This locks the behaviour the deletion-ledger plan asked for: a config that
slips an unregistered/aspirational model name past Tier-1 still cannot reach
the trainer — the probe converts it into a structured ``ProbeResult``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.validation.forward_probe import (  # noqa: E402
    synthetic_forward_probe,
)
from tests.utils.data_config_stub import DataConfigStub  # noqa: E402


def _config(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            model_type=model_type,
            in_channels=1,
            out_channels=1,
            model_kwargs={},
        ),
        data=DataConfigStub(patch_size=[16, 16], batch_size=1),
    )


@pytest.mark.registry_contract
def test_unresolvable_model_is_clean_failure() -> None:
    res = synthetic_forward_probe(
        _config("__definitely_not_a_registered_model__"),
        device="cpu",
        backward=False,
        use_phantom=False,
    )
    assert res.passed is False
    assert res.category == "instantiation"
    assert res.severity == "error"
    # The fix hint must point the user at registration.
    assert "model.model_type" in (res.yaml_keys or [])
