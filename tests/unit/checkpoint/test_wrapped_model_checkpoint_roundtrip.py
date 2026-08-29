"""A checkpoint written by a WRAPPED model must load into a BARE one.

This is the regression suite for #619 F3, and it is an end-to-end seam test on
purpose: ``test_module_utils.py`` proves the helper strips prefixes, which is a
different claim from "the checkpoint a compiled run writes is loadable by
inference". The bug lived in the gap between those two.

The failure it guards is unusually quiet. Every inference/evaluation path builds
a **bare** model and loads with ``strict=True``, so a compiled or DDP run's
checkpoint raised there — but the same file loaded with ``strict=False``
(``campaign_evaluator``, warm-start, distillation) matches **nothing**, loads
**nothing**, and reports success. A silently-random model then produces metrics
that look like a bad arm rather than a broken load.

Parametrised over both checkpoint formats because the safetensors branch
flattens the state dict into ``model_state_dict_<key>``: a compiled model there
produced ``model_state_dict__orig_mod.conv.weight``, so the container prefix and
the wrapper prefix had to be stripped in sequence.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from mriforge.config.schemas.checkpoint import CheckpointConfigSchema  # noqa: E402
from mriforge.infrastructure.services.checkpoint_service import (  # noqa: E402
    CheckpointService,
)


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, padding=1)
        self.head = nn.Linear(4, 2)

    def forward(self, x):  # pragma: no cover - never called
        return self.head(self.conv(x).flatten(1))


def _service(tmp_path, fmt: str) -> CheckpointService:
    return CheckpointService(
        CheckpointConfigSchema(
            checkpoint_dir=str(tmp_path), format=fmt, save_interval=1
        )
    )


def _distinctive(model: nn.Module) -> nn.Module:
    """Make the weights recognisable so a no-op load cannot pass silently.

    With default init, a "loaded" model and a freshly-built one can look close
    enough that an equality check on a single element passes by luck. Filling
    with a constant no initialiser produces makes a failed load unmistakable.
    """
    with torch.no_grad():
        model.conv.weight.fill_(0.375)
        model.conv.bias.fill_(-0.125)
        model.head.weight.fill_(0.75)
    return model


WRAPPERS = {
    "compiled": lambda m: torch.compile(m),
    "data_parallel": lambda m: nn.DataParallel(m),
    "data_parallel_of_compiled": lambda m: nn.DataParallel(torch.compile(m)),
}


@pytest.mark.unit
@pytest.mark.parametrize("fmt", ["pth", "safetensors"])
@pytest.mark.parametrize("wrapper", list(WRAPPERS), ids=list(WRAPPERS))
def test_wrapped_save_loads_into_a_bare_model(tmp_path, fmt: str, wrapper: str) -> None:
    service = _service(tmp_path, fmt)
    source = _distinctive(_Tiny())
    wrapped = WRAPPERS[wrapper](source)

    path = service.save_checkpoint(
        model=wrapped,
        optimizer=torch.optim.SGD(source.parameters(), lr=0.1),
        epoch=1,
        loss=0.5,
        step=10,
    )

    # The bare model every inference path builds.
    target = _Tiny()
    service.load_checkpoint(target, None, str(path))

    # Compare on CPU: ``nn.DataParallel.__init__`` moves the wrapped module to
    # ``device_ids[0]``, so ``source`` may now live on cuda:0 while ``target``
    # (freshly built, as inference builds it) is on CPU. That is a property of
    # the wrapper, not of the checkpoint.
    assert torch.allclose(target.conv.weight, source.conv.weight.cpu())
    assert torch.allclose(target.conv.bias, source.conv.bias.cpu())
    assert torch.allclose(target.head.weight, source.head.weight.cpu())


@pytest.mark.unit
@pytest.mark.parametrize("fmt", ["pth", "safetensors"])
def test_saved_keys_carry_no_wrapper_prefix(tmp_path, fmt: str) -> None:
    """Assert the artefact, not just the round-trip.

    The round-trip test above would also pass if the save side kept writing
    prefixed keys and the load side merely tolerated them. It must not: a
    checkpoint is read by tools outside this repo's load path (``mriforge
    export``, the campaign evaluator's ``strict=False`` load, manual
    ``torch.load`` inspection), and those cannot all be taught about prefixes.
    """
    service = _service(tmp_path, fmt)
    path = service.save_checkpoint(
        model=torch.compile(_distinctive(_Tiny())),
        optimizer=None,
        epoch=1,
        loss=0.5,
        step=1,
    )

    if fmt == "pth":
        keys = set(
            torch.load(str(path), map_location="cpu", weights_only=False)[
                "model_state_dict"
            ]
        )
    else:
        from safetensors.torch import load_file

        keys = {
            k[len("model_state_dict_") :]
            for k in load_file(str(path))
            if k.startswith("model_state_dict_")
        }

    assert keys, "no model keys were persisted at all"
    for prefix in ("_orig_mod.", "module.", "_fsdp_wrapped_module."):
        offenders = sorted(k for k in keys if k.startswith(prefix))
        assert not offenders, f"checkpoint still carries {prefix!r} keys: {offenders}"
    assert keys == set(_Tiny().state_dict())


@pytest.mark.unit
def test_a_legacy_prefixed_checkpoint_still_loads(tmp_path) -> None:
    """Checkpoints written BEFORE the save-side fix must remain loadable.

    Every compiled or DDP run that already happened produced prefixed keys. If
    the load side only handled clean keys, fixing the save side would strand all
    of them — a fix that breaks the artefacts it exists to make usable.
    """
    service = _service(tmp_path, "pth")
    source = _distinctive(_Tiny())

    legacy = tmp_path / "legacy.pth"
    torch.save(
        {
            "epoch": 1,
            "step": 1,
            "model_state_dict": {
                f"_orig_mod.{k}": v for k, v in source.state_dict().items()
            },
        },
        legacy,
    )

    target = _Tiny()
    service.load_checkpoint(target, None, str(legacy))
    assert torch.allclose(target.conv.weight, source.conv.weight)


@pytest.mark.unit
def test_ema_state_is_saved_unprefixed(tmp_path) -> None:
    """``ModelEma`` keeps its shadow under ``.module``, so an EMA state dict was
    ``module.``-prefixed on EVERY run, single-GPU and eager included. That is why
    the GAN inference strategy's ``use_ema`` path could never have worked."""
    from mriforge.infrastructure.optimization.ema import ModelEma

    service = _service(tmp_path, "pth")
    source = _distinctive(_Tiny())
    ema = ModelEma(source, decay=0.9)

    assert any(
        k.startswith("module.") for k in ema.state_dict()
    ), "precondition: ModelEma is expected to prefix its keys"

    path = service.save_checkpoint(
        model=source,
        optimizer=None,
        epoch=1,
        loss=0.5,
        step=1,
        ema_state_dict=ema.state_dict(),
    )
    stored = torch.load(str(path), map_location="cpu", weights_only=False)

    target = _Tiny()
    ema_target = ModelEma(target, decay=0.9)
    service.load_checkpoint(target, None, str(path), ema_model=ema_target)

    assert "ema_state_dict" in stored
    assert torch.allclose(ema_target.module.conv.weight, ema.module.conv.weight)
