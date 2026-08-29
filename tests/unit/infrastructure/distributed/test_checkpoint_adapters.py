"""The three checkpoint adapters, and the one flag that prevents a hung job.

The bug these exist to prevent has no exception and no log line: rank 0 enters
a collective (FSDP's state_dict all-gather, DeepSpeed's save_checkpoint) that
ranks 1..N never enter, because the training loop gated the whole checkpoint
block on ``is_main_rank()``. Every assertion here is CPU-runnable; the actual
multi-rank behaviour needs a cluster.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mriforge.infrastructure.distributed.checkpoint_adapters import (
    DEEPSPEED_TAG_PREFIX,
    DeepSpeedCheckpointAdapter,
    DefaultCheckpointAdapter,
    FSDPCheckpointAdapter,
    IParallelCheckpointAdapter,
    resolve_checkpoint_adapter,
)
from mriforge.infrastructure.distributed.strategy_registry import (
    ParallelRuntime,
    list_parallel_strategies,
    resolve_parallel_strategy,
)

ALL_ADAPTERS = (
    DefaultCheckpointAdapter(),
    FSDPCheckpointAdapter(),
    DeepSpeedCheckpointAdapter(),
)


class _FakeEngine:
    """Duck-types the two DeepSpeed engine methods the adapter calls."""

    def __init__(self, *, consolidate_ok: bool = True, explode: bool = False) -> None:
        self.saved: list[tuple[str, str]] = []
        self.loaded: list[tuple[str, str]] = []
        self.consolidated: list[tuple[str, str]] = []
        self._consolidate_ok = consolidate_ok
        self._explode = explode

    def save_checkpoint(self, path, tag, client_state=None):
        self.saved.append((str(path), tag))

    def load_checkpoint(self, path, tag):
        self.loaded.append((str(path), tag))
        return str(path), {"epoch": 3, "global_step": 300}

    def save_16bit_model(self, directory, filename):
        if self._explode:
            raise RuntimeError("gather failed")
        self.consolidated.append((str(directory), filename))
        (Path(directory) / filename).write_text("weights")
        return self._consolidate_ok


class TestProtocolConformance:
    @pytest.mark.parametrize("adapter", ALL_ADAPTERS, ids=lambda a: a.name)
    def test_satisfies_the_protocol(self, adapter) -> None:
        assert isinstance(adapter, IParallelCheckpointAdapter)

    @pytest.mark.parametrize("adapter", ALL_ADAPTERS, ids=lambda a: a.name)
    def test_declares_both_capability_flags(self, adapter) -> None:
        """Read as booleans by the loop and the director; a missing one would
        be ``None`` and read as False, i.e. silently back to the hang."""
        assert isinstance(adapter.requires_all_ranks, bool)
        assert isinstance(adapter.writes_native_artifact, bool)

    @pytest.mark.parametrize("adapter", ALL_ADAPTERS, ids=lambda a: a.name)
    def test_gather_context_is_reentrant_on_a_plain_module(self, adapter) -> None:
        """Every adapter must tolerate a module it did not wrap: the director
        calls this unconditionally on generator AND discriminator."""
        import torch.nn as nn

        with adapter.gather_full_state_dict(nn.Linear(2, 2)):
            pass


class TestRequiresAllRanks:
    """The flag that decides who *participates*, distinct from who *writes*."""

    def test_default_is_rank0_only(self) -> None:
        assert DefaultCheckpointAdapter().requires_all_ranks is False

    @pytest.mark.parametrize(
        "adapter", [FSDPCheckpointAdapter(), DeepSpeedCheckpointAdapter()]
    )
    def test_sharded_strategies_need_every_rank(self, adapter) -> None:
        """This is THE assertion. If either flips to False, a multi-rank run
        hangs at the first checkpoint with no error -- SLURM kills it at
        walltime and the log's last line is a normal training iteration."""
        assert adapter.requires_all_ranks is True

    def test_only_deepspeed_writes_its_own_artifact(self) -> None:
        assert DeepSpeedCheckpointAdapter().writes_native_artifact is True
        assert FSDPCheckpointAdapter().writes_native_artifact is False
        assert DefaultCheckpointAdapter().writes_native_artifact is False


class TestResolveFromRuntime:
    def test_no_runtime_is_the_default_adapter(self) -> None:
        assert isinstance(resolve_checkpoint_adapter(None), DefaultCheckpointAdapter)

    def test_runtime_without_an_adapter_defaults(self) -> None:
        """The scripting API builds a TrainingEnvironment by hand."""
        runtime = ParallelRuntime(strategy="none")
        assert isinstance(resolve_checkpoint_adapter(runtime), DefaultCheckpointAdapter)

    def test_the_runtime_carries_the_plugin_s_adapter(self) -> None:
        runtime = ParallelRuntime(
            strategy="fsdp", checkpoint_adapter=FSDPCheckpointAdapter()
        )
        assert isinstance(resolve_checkpoint_adapter(runtime), FSDPCheckpointAdapter)

    def test_checkpoints_require_all_ranks_tracks_the_adapter(self) -> None:
        assert ParallelRuntime(strategy="none").checkpoints_require_all_ranks is False
        assert (
            ParallelRuntime(
                strategy="deepspeed", checkpoint_adapter=DeepSpeedCheckpointAdapter()
            ).checkpoints_require_all_ranks
            is True
        )


class TestEveryStrategySuppliesOne:
    """A plugin returning ``None`` would read as "rank 0 only" -- the hang."""

    @pytest.mark.parametrize("name", list_parallel_strategies())
    def test_plugin_builds_an_adapter(self, name: str) -> None:
        from types import SimpleNamespace

        plugin = resolve_parallel_strategy(name)
        ctx = SimpleNamespace(
            config=None,
            device="cpu",
            parallel=SimpleNamespace(
                deepspeed=SimpleNamespace(save_consolidated_best=True)
            ),
        )
        adapter = plugin.checkpoint_adapter(ctx)
        assert adapter is not None
        assert isinstance(adapter, IParallelCheckpointAdapter)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("none", False),
            ("dp", False),
            ("ddp", False),
            ("fsdp", True),
            ("deepspeed", True),
        ],
    )
    def test_collective_strategies_are_exactly_the_sharding_ones(
        self, name: str, expected: bool
    ) -> None:
        """DP and DDP REPLICATE rather than shard, so rank 0's state_dict is
        the whole model and gating on rank 0 is correct for them."""
        from types import SimpleNamespace

        plugin = resolve_parallel_strategy(name)
        ctx = SimpleNamespace(
            config=None,
            device="cpu",
            parallel=SimpleNamespace(
                deepspeed=SimpleNamespace(save_consolidated_best=True)
            ),
        )
        assert plugin.checkpoint_adapter(ctx).requires_all_ranks is expected


class TestDeepSpeedNativeArtifacts:
    def test_writes_a_tag_directory_per_engine(self, tmp_path: Path) -> None:
        engine = _FakeEngine()
        adapter = DeepSpeedCheckpointAdapter()
        adapter.save_native(
            models={"generator": engine},
            checkpoint_dir=tmp_path,
            tag="best",
            client_state={"epoch": 1},
        )
        assert engine.saved == [
            (str(tmp_path / DEEPSPEED_TAG_PREFIX / "generator"), "best")
        ]

    def test_consolidated_best_is_published_for_non_deepspeed_consumers(
        self, tmp_path: Path
    ) -> None:
        """discover_best_checkpoint / campaign eval / `mriforge infer` all open a
        FILE. Without this the run is resume-only."""
        engine = _FakeEngine()
        path = DeepSpeedCheckpointAdapter().save_native(
            models={"generator": engine},
            checkpoint_dir=tmp_path,
            tag="best",
            client_state={},
        )
        assert path == tmp_path / "checkpoint_best.pt"
        assert path.exists()
        assert engine.consolidated == [(str(tmp_path), "checkpoint_best.pt")]

    def test_disabling_consolidation_still_writes_the_shards(
        self, tmp_path: Path
    ) -> None:
        engine = _FakeEngine()
        path = DeepSpeedCheckpointAdapter(save_consolidated_best=False).save_native(
            models={"generator": engine},
            checkpoint_dir=tmp_path,
            tag="best",
            client_state={},
        )
        assert path is None
        assert engine.saved, "resume artifact must be written regardless"
        assert engine.consolidated == []

    def test_a_failed_consolidation_does_not_lose_the_shards(
        self, tmp_path: Path
    ) -> None:
        """Resume must survive a consolidation failure -- only the eval
        artifact is missing, and the error names that consequence."""
        engine = _FakeEngine(explode=True)
        path = DeepSpeedCheckpointAdapter().save_native(
            models={"generator": engine},
            checkpoint_dir=tmp_path,
            tag="best",
            client_state={},
        )
        assert path is None
        assert engine.saved

    def test_no_engine_means_nothing_to_do(self, tmp_path: Path) -> None:
        """A DeepSpeed runtime whose models were never wrapped (audit/probe)."""
        import torch.nn as nn

        assert (
            DeepSpeedCheckpointAdapter().save_native(
                models={"generator": nn.Linear(2, 2)},
                checkpoint_dir=tmp_path,
                tag="best",
                client_state={},
            )
            is None
        )

    def test_load_returns_the_client_state(self, tmp_path: Path) -> None:
        engine = _FakeEngine()
        state = DeepSpeedCheckpointAdapter().load_native(
            models={"generator": engine}, checkpoint_dir=tmp_path, tag="best"
        )
        assert state == {"epoch": 3, "global_step": 300}
        assert engine.loaded

    def test_engines_are_duck_typed_not_isinstance_checked(self) -> None:
        """So this module never imports deepspeed and stays testable without it."""
        import mriforge.infrastructure.distributed.checkpoint_adapters as mod

        assert "import deepspeed" not in Path(mod.__file__).read_text()


class TestDefaultAdapterIsInert:
    def test_save_and_load_are_no_ops(self, tmp_path: Path) -> None:
        adapter = DefaultCheckpointAdapter()
        assert (
            adapter.save_native(
                models={}, checkpoint_dir=tmp_path, tag="t", client_state={}
            )
            is None
        )
        assert adapter.load_native(models={}, checkpoint_dir=tmp_path, tag="t") is None


class _MissingTagEngine(_FakeEngine):
    """The real engine reports a missing tag by RETURNING ``(None, None)``.

    It does not raise, which is what made the miss swallowable.
    """

    def load_checkpoint(self, path, tag):
        self.loaded.append((str(path), tag))
        return None, None


class TestLoadNativeDoesNotSwallowAMiss:
    """A restore that loaded nothing must not look like one that loaded everything.

    ``load_from`` now treats a successful native load as authoritative and stops
    parsing the file beside the tag directory. If a miss came back as an empty
    dict, the run would log "restored best weights" while still holding the
    weights it started with -- the silent fallback non-negotiable 3 forbids.
    """

    def test_a_missing_tag_raises(self, tmp_path) -> None:
        adapter = DeepSpeedCheckpointAdapter()
        with pytest.raises(FileNotFoundError, match="no checkpoint for tag"):
            adapter.load_native(
                models={"generator": _MissingTagEngine()},
                checkpoint_dir=tmp_path,
                tag="best",
            )

    def test_the_error_names_the_tag_and_the_directory(self, tmp_path) -> None:
        """A cluster operator reads this message, not the traceback."""
        adapter = DeepSpeedCheckpointAdapter()
        with pytest.raises(FileNotFoundError) as excinfo:
            adapter.load_native(
                models={"generator": _MissingTagEngine()},
                checkpoint_dir=tmp_path,
                tag="best",
            )
        message = str(excinfo.value)
        assert "'best'" in message
        assert str(tmp_path / DEEPSPEED_TAG_PREFIX / "generator") in message

    def test_no_engines_still_returns_none_rather_than_raising(self, tmp_path) -> None:
        """Distinct from a miss: there was nothing to load INTO.

        ``load_from`` discriminates on this to decide whether it can fall back
        to the generic payload, so it must stay a return value.
        """
        import torch.nn as nn

        assert (
            DeepSpeedCheckpointAdapter().load_native(
                models={"generator": nn.Linear(2, 2)},
                checkpoint_dir=tmp_path,
                tag="best",
            )
            is None
        )

    def test_a_hit_with_no_client_state_is_an_empty_dict_not_none(
        self, tmp_path
    ) -> None:
        """``{}`` means loaded-but-no-metadata and must not read as failure."""

        class _NoClientState(_FakeEngine):
            def load_checkpoint(self, path, tag):
                return str(path), {}

        assert (
            DeepSpeedCheckpointAdapter().load_native(
                models={"generator": _NoClientState()},
                checkpoint_dir=tmp_path,
                tag="best",
            )
            == {}
        )
