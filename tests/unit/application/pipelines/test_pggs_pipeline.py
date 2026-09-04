"""``PGGSReconstructionPipeline`` resolves its device through the shared policy."""

from __future__ import annotations

from types import SimpleNamespace

import torch


class _Stub(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)


def test_default_device_comes_from_the_resolver(monkeypatch) -> None:
    """Planted violation: a hard-coded ``cuda if available else cpu`` default
    never asks the policy; the spy must be reached."""
    from spectramr.application.pipelines import pggs_pipeline
    from spectramr.core import compute_device

    seen: list[tuple] = []

    def _spy(requested, *, pipeline, source="unspecified"):
        seen.append((requested, pipeline, source))
        return SimpleNamespace(device="cpu", accelerated=False, source="test", cpu_opt_in=True)

    monkeypatch.setattr(compute_device, "resolve_torch_device", _spy)
    pipeline = pggs_pipeline.PGGSReconstructionPipeline(_Stub())
    assert seen == [(None, "pggs", "PGGSReconstructionPipeline")]
    assert pipeline.device == torch.device("cpu")


def test_an_explicit_device_is_honoured(monkeypatch) -> None:
    from spectramr.application.pipelines import pggs_pipeline
    from spectramr.core import compute_device

    monkeypatch.setattr(
        compute_device,
        "resolve_torch_device",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("resolver must not run")),
    )
    pipeline = pggs_pipeline.PGGSReconstructionPipeline(_Stub(), device=torch.device("cpu"))
    assert pipeline.device == torch.device("cpu")
