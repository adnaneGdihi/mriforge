"""Regression test: ``build_container`` must NOT fall back to CPU.

History. Audit-15-F8 made the implicit CPU fallback *audible* (a warning) — but
audible is not enough. A cluster job whose GPU allocation failed still ran to
completion on CPU at ~100x slowdown, burning its wall-clock budget while
reporting success, because nobody reads a warning in a 40-hour sbatch log.

The accelerated-run contract replaces the warning with a raise: heavy pipelines
(train / infer / validate / hpo / ablation / probe) get an accelerator or they
stop. ``bootstrap.py`` now delegates every device decision to the SSOT policy in
:mod:`spectramr.core.compute_device`.

The two fallbacks removed here are what made ``DeviceManager``'s own (correct)
CUDA guard unreachable: bootstrap flattened ``"auto"`` to ``"cpu"`` *before*
``DeviceManager`` was constructed, so the guard never fired on the live path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

BOOTSTRAP_SRC = (
    Path(__file__).resolve().parents[2] / "src" / "spectramr" / "bootstrap.py"
).read_text()


def test_bootstrap_no_longer_silently_falls_back_to_cpu() -> None:
    """The literal fallback assignments must be gone from the live path."""
    assert 'device_arg = "cpu"' not in BOOTSTRAP_SRC, (
        "bootstrap.py still hard-assigns CPU. Heavy pipelines must raise "
        "AcceleratorRequiredError instead — see spectramr.core.compute_device."
    )
    assert '"cuda" if torch.cuda.is_available() else "cpu"' not in BOOTSTRAP_SRC, (
        "bootstrap.py still flattens 'auto' to CPU itself. That is what made "
        "DeviceManager's CUDA guard unreachable; delegate to resolve_torch_device."
    )


def test_bootstrap_delegates_to_the_compute_device_ssot() -> None:
    assert "resolve_torch_device" in BOOTSTRAP_SRC, (
        "bootstrap.py must resolve the device through the SSOT policy so the "
        "accelerated-run contract is enforced on the DI path."
    )


def test_no_accelerator_raises_for_heavy_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'auto' + no GPU + heavy pipeline → raise, not a silent CPU run."""
    import torch

    from spectramr.core.compute_device import (
        AcceleratorRequiredError,
        resolve_torch_device,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("FORCE_CPU", raising=False)

    with pytest.raises(AcceleratorRequiredError):
        resolve_torch_device("auto", pipeline="train")


def test_explicit_cpu_still_permitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The user-dictated escape hatch survives: ``device: cpu`` is honoured."""
    import torch

    from spectramr.core.compute_device import resolve_torch_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    decision = resolve_torch_device("cpu", pipeline="train")
    assert decision.device == "cpu"
    assert decision.cpu_opt_in is True
