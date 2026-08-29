"""CheckpointService ``strategy_state`` round-trip (2026-06, section R).

The director is the canonical save path; CheckpointService is the director-failure
fallback (and the model-management utility). This test proves the ``strategy_state``
kwarg round-trips through the service for BOTH formats — natively for ``pth`` and via
the ``_pickle_strategy_state`` blob (mirroring ``rng_state``) for ``safetensors`` — so
the fallback path doesn't silently drop strategy-owned learnable state.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mriforge.config.schemas.checkpoint import CheckpointConfigSchema
from mriforge.infrastructure.services.checkpoint_service import CheckpointService


def _strategy_state() -> dict:
    sfc = nn.Linear(3, 3)
    with torch.no_grad():
        sfc.weight.add_(1.5)
    return {
        "modules": {"sfc": sfc.state_dict()},
        "params": {"diffusion": torch.tensor(0.05)},
    }


@pytest.mark.parametrize("fmt", ["pth", "safetensors"])
def test_strategy_state_round_trips(tmp_path, fmt) -> None:
    config = CheckpointConfigSchema(
        checkpoint_dir=str(tmp_path),
        save_interval=10,
        keep_last_n=3,
        format=fmt,
    )
    service = CheckpointService(config=config)

    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sstate = _strategy_state()

    path = service.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=2,
        loss=0.5,
        step=50,
        strategy_state=sstate,
        file_path=str(tmp_path / f"ckpt.{fmt}"),
    )

    loaded = service.load_checkpoint(
        model=nn.Linear(4, 4),
        optimizer=torch.optim.SGD(nn.Linear(4, 4).parameters(), lr=0.1),
        file_path=path,
    )
    assert loaded is not None
    assert "strategy_state" in loaded, f"strategy_state dropped for format={fmt}"

    got = loaded["strategy_state"]
    assert torch.allclose(got["params"]["diffusion"], torch.tensor(0.05))
    for k, v in sstate["modules"]["sfc"].items():
        assert torch.allclose(got["modules"]["sfc"][k], v)


def test_no_strategy_state_kwarg_omits_key(tmp_path) -> None:
    # Backward compatibility: callers that don't pass strategy_state produce a
    # checkpoint without the key (no empty/garbage entry).
    config = CheckpointConfigSchema(
        checkpoint_dir=str(tmp_path), save_interval=10, keep_last_n=3, format="pth"
    )
    service = CheckpointService(config=config)
    model = nn.Linear(4, 4)
    path = service.save_checkpoint(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        epoch=0,
        loss=0.0,
        step=0,
        file_path=str(tmp_path / "ckpt.pth"),
    )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert "strategy_state" not in blob
