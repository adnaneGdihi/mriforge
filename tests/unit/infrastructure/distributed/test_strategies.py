"""The five strategy plugins, and the ordering constraints they encode.

Wrap ordering is not a style choice here: FSDP in Stage B or DeepSpeed in
Stage A would build, run, and produce wrong numbers rather than raising.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from spectramr.infrastructure.distributed.checkpoint_adapters import (
    DeepSpeedCheckpointAdapter,
    DefaultCheckpointAdapter,
    FSDPCheckpointAdapter,
)
from spectramr.infrastructure.distributed.strategies import (
    DataParallelStrategy,
    DDPStrategy,
    DeepSpeedStrategy,
    FSDPStrategy,
    NoParallelStrategy,
)


def _ctx(*, save_consolidated_best: bool = True, world_size: int = 1):
    return SimpleNamespace(
        config=None,
        device="cpu",
        world_size=world_size,
        parallel=SimpleNamespace(
            deepspeed=SimpleNamespace(save_consolidated_best=save_consolidated_best),
            sync_batch_norm=False,
        ),
    )


class TestCheckpointAdapterPerStrategy:
    @pytest.mark.parametrize(
        ("plugin", "expected"),
        [
            (NoParallelStrategy(), DefaultCheckpointAdapter),
            (DataParallelStrategy(), DefaultCheckpointAdapter),
            (DDPStrategy(), DefaultCheckpointAdapter),
            (FSDPStrategy(), FSDPCheckpointAdapter),
            (DeepSpeedStrategy(), DeepSpeedCheckpointAdapter),
        ],
        ids=lambda v: getattr(v, "name", getattr(v, "__name__", str(v))),
    )
    def test_each_supplies_the_right_adapter(self, plugin, expected) -> None:
        assert isinstance(plugin.checkpoint_adapter(_ctx()), expected)

    def test_replicating_strategies_do_not_need_all_ranks(self) -> None:
        """DP and DDP replicate the model, so rank 0's state_dict IS the model
        and gating the write on rank 0 is correct for them."""
        for plugin in (DataParallelStrategy(), DDPStrategy(), NoParallelStrategy()):
            assert plugin.checkpoint_adapter(_ctx()).requires_all_ranks is False

    def test_sharding_strategies_do(self) -> None:
        for plugin in (FSDPStrategy(), DeepSpeedStrategy()):
            assert plugin.checkpoint_adapter(_ctx()).requires_all_ranks is True


class TestSaveConsolidatedBestIsRead:
    """The knob is read HERE and nowhere else (pitfall #15).

    ``build_deepspeed_config`` correctly omits it -- it is a spectramr concern,
    not a DeepSpeed key -- so anything reconstructing the adapter from the
    rendered ds_config in provenance would silently see the default and the
    knob would be validated, stamped, and inert.
    """

    def test_true_propagates(self) -> None:
        adapter = DeepSpeedStrategy().checkpoint_adapter(
            _ctx(save_consolidated_best=True)
        )
        assert adapter.save_consolidated_best is True

    def test_false_propagates(self) -> None:
        adapter = DeepSpeedStrategy().checkpoint_adapter(
            _ctx(save_consolidated_best=False)
        )
        assert adapter.save_consolidated_best is False

    def test_it_is_not_a_deepspeed_config_key(self) -> None:
        """If it ever became one, provenance would carry it and the reader
        above could quietly move -- back to reading a default."""
        from spectramr.infrastructure.distributed.deepspeed_backend.config_builder import (
            DERIVED_KEYS,
        )

        assert "save_consolidated_best" not in DERIVED_KEYS


class TestWrapOrderingIsEncoded:
    def test_fsdp_wraps_in_stage_a(self) -> None:
        """FSDP.__init__ re-points parameter storage; an optimizer built first
        gets shard-shaped gradients against full-shape moment buffers."""
        import inspect

        source = inspect.getsource(FSDPStrategy.prepare_models)
        assert "maybe_wrap_with_fsdp" in source

    def test_deepspeed_does_nothing_in_stage_a(self) -> None:
        """initialize() CONSUMES an already-built optimizer, so it cannot run
        before OptimizationBuilder."""
        models = {"generator": object()}
        assert DeepSpeedStrategy().prepare_models(models, _ctx()) is models

    def test_ddp_does_not_wrap_in_stage_a(self) -> None:
        """DDP could wrap early (it preserves parameter identity) but then
        count_parameters and the optimizer builder would see the wrapper."""
        import inspect

        source = inspect.getsource(DDPStrategy.prepare_models)
        assert "DistributedDataParallel" not in source


# --------------------------------------------------------------------------
# `requires_process_group` — the flag `cli.profile_preflight` reads to refuse a
# single-process profiling run before it costs twelve minutes.
# --------------------------------------------------------------------------


def test_requires_process_group_values():
    """`dp` is the one that matters: DataParallel needs NO process group.

    A caller spelling this rule as `strategy != "none"` would block it, which is
    why the flag exists instead of that expression.
    """
    from spectramr.infrastructure.distributed.strategies import (
        DataParallelStrategy,
        DDPStrategy,
        DeepSpeedStrategy,
        FSDPStrategy,
        NoParallelStrategy,
    )

    assert NoParallelStrategy.requires_process_group is False
    assert DataParallelStrategy.requires_process_group is False
    assert DDPStrategy.requires_process_group is True
    assert FSDPStrategy.requires_process_group is True
    assert DeepSpeedStrategy.requires_process_group is True


def test_the_flag_agrees_with_the_require_process_group_call_sites():
    """The drift lock (non-negotiable 17): the flag is not a second declaration.

    Read off the AST rather than asserted as a literal list, so a backend added
    tomorrow that calls `_require_process_group` without setting the flag — or
    sets it without calling — fails here instead of silently profiling into a
    twelve-minute RuntimeError.
    """
    src = Path("src/spectramr/infrastructure/distributed/strategies.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))

    callers, flagged = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_require_process_group"
            ):
                callers.add(node.name)
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "requires_process_group"
                    for t in stmt.targets
                )
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is True
            ):
                flagged.add(node.name)

    assert callers, "AST found no _require_process_group call sites — test is blind"
    assert callers == flagged, (
        f"drift: classes calling _require_process_group={sorted(callers)} but "
        f"classes setting requires_process_group=True={sorted(flagged)}"
    )
