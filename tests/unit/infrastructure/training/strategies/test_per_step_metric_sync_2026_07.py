"""Regression: per-step closures must NOT do a GPU→CPU sync per metric key.

``domain_adaptation`` / ``gan`` / ``disentangled`` stored their step metrics with
a per-key ``v.item()`` / ``float(v.detach())`` inside the optimizer closure —
one blocking host transfer PER loss component EVERY step (training-loop perf rule:
no ``.item()`` in the hot loop). They now store DETACHED TENSORS and convert to
float once, at the throttled ``get_last_metrics`` reporting boundary (a single
fused sync for disentangled).
"""

from __future__ import annotations

import inspect

import pytest

from spectramr.infrastructure.training.strategies import domain_adaptation as _da_mod
from spectramr.infrastructure.training.strategies import gan as _gan_mod
from spectramr.infrastructure.training.strategies.disentangled_strategy import (
    DisentangledTrainingStrategy,
)


def test_disentangled_get_last_metrics_stays_on_device_and_passes_through():
    """The fourth implementation of a four-way contract (#707).

    This asserted a FUSED conversion here, which was already far better than a
    sync per key -- but `training_loop` calls `get_last_metrics` on every
    iteration, outside the `log_interval` gate, so a fused transfer per step is
    still a sync per step. `base`, `gan` and `mixins.adversarial` had already
    stopped converting; this strategy was the one left disagreeing with its
    siblings about what the method returns.

    Non-tensor entries still pass through unchanged -- ``loss_output.to_dict()``
    may carry non-numeric fields, and that half of the contract is unchanged.
    """
    import torch

    strat = DisentangledTrainingStrategy.__new__(DisentangledTrainingStrategy)
    strat._last_step_metrics = {
        "g_total_loss": torch.tensor(1.5),
        "d_total_loss": torch.tensor(0.25),
        "note": "not-a-number",  # non-numeric must pass through, not raise
    }
    out = strat.get_last_metrics()
    assert isinstance(out["g_total_loss"], torch.Tensor)
    assert float(out["g_total_loss"]) == pytest.approx(1.5)
    assert float(out["d_total_loss"]) == pytest.approx(0.25)
    assert out["note"] == "not-a-number"


def _closure_sources(module) -> str:
    """Return the source of every train-step / closure method in the module."""
    chunks = []
    for _name, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ != module.__name__:
            continue
        for meth_name, meth in cls.__dict__.items():
            if callable(meth) and (
                "closure" in meth_name
                or "train" in meth_name
                or "generator_step" in meth_name
                or "discriminator_step" in meth_name
            ):
                try:
                    chunks.append(inspect.getsource(meth))
                except (OSError, TypeError):
                    pass
    return "\n".join(chunks)


@pytest.mark.parametrize(
    "module", [_da_mod, _gan_mod], ids=["domain_adaptation", "gan"]
)
def test_closures_store_detached_tensors_not_item(module) -> None:
    """The metric-store lines in the closures must not call ``.item()`` or
    ``float(...detach())`` (per-key host syncs)."""
    src = _closure_sources(module)
    assert "_last_step_metrics" in src, "sanity: found the metric-store closures"

    # Look for the specific anti-patterns feeding _last_step_metrics.
    for anti in (".item()", "float(v.detach())", "float(d_total.detach())"):
        # Allow .item() elsewhere; only flag it in a _last_step_metrics store line.
        for line in src.splitlines():
            if "_last_step_metrics" in line and anti in line:
                raise AssertionError(
                    f"{module.__name__}: per-step host sync `{anti}` in a "
                    f"_last_step_metrics store: {line.strip()}"
                )
