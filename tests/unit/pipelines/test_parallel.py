"""``pipelines.parallel`` rank helpers.

``is_rank_zero`` predates ``fsdp``/``deepspeed`` being reachable from YAML. It
tested ``strategy != "ddp"`` and returned True for everything else, so under
FSDP or DeepSpeed EVERY rank believed it was rank 0. It is live at ``train.py``
gating the TensorBoard writer, so all N ranks opened a writer on the same event
directory.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mriforge.pipelines.parallel import is_rank_zero, resolve_data_rank


def _config(strategy: str | None) -> SimpleNamespace:
    if strategy is None:
        return SimpleNamespace(parallel=None)
    return SimpleNamespace(parallel=SimpleNamespace(strategy=strategy))


class TestIsRankZero:
    def test_no_parallel_block_is_rank_zero(self) -> None:
        assert is_rank_zero(_config(None)) is True
        assert is_rank_zero(SimpleNamespace()) is True

    @pytest.mark.parametrize("strategy", ["none", "dp", "NONE", "Dp"])
    def test_single_process_strategies_are_always_rank_zero(
        self, strategy: str
    ) -> None:
        """``none`` and ``dp`` run in ONE process (DataParallel uses threads),
        so there is no rank to be non-zero."""
        assert is_rank_zero(_config(strategy)) is True

    @pytest.mark.parametrize("strategy", ["ddp", "fsdp", "deepspeed"])
    def test_process_group_strategies_consult_torch_distributed(
        self, strategy: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The regression: fsdp/deepspeed short-circuited to True here, so the
        rank check never ran and every rank reported rank 0."""
        import mriforge.pipelines.parallel as parallel_mod

        monkeypatch.setattr(
            parallel_mod.torch.distributed, "is_initialized", lambda: True
        )
        monkeypatch.setattr(parallel_mod.torch.distributed, "get_rank", lambda: 3)

        assert is_rank_zero(_config(strategy)) is False

    @pytest.mark.parametrize("strategy", ["ddp", "fsdp", "deepspeed"])
    def test_process_group_strategies_are_rank_zero_on_rank_zero(
        self, strategy: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mriforge.pipelines.parallel as parallel_mod

        monkeypatch.setattr(
            parallel_mod.torch.distributed, "is_initialized", lambda: True
        )
        monkeypatch.setattr(parallel_mod.torch.distributed, "get_rank", lambda: 0)

        assert is_rank_zero(_config(strategy)) is True

    def test_uninitialised_process_group_is_rank_zero(self) -> None:
        """Declared fsdp but launched without torchrun: one process, so it is
        rank 0 -- the declaration alone must not suppress its own logging."""
        assert is_rank_zero(_config("fsdp")) is True

    def test_the_strategy_set_covers_every_process_group_strategy(self) -> None:
        """Pins the set against the schema so a newly-added sharded strategy
        cannot silently inherit the 'everyone is rank 0' answer."""
        from mriforge.pipelines.parallel import _PROCESS_GROUP_STRATEGIES

        assert _PROCESS_GROUP_STRATEGIES == {"ddp", "fsdp", "deepspeed"}


class TestResolveDataRank:
    """Issue #1124: seeding the data stream needs the rank itself, not the
    boolean collapse of it. Same guards as ``is_rank_zero`` by construction."""

    def test_no_parallel_block_is_rank_zero(self) -> None:
        assert resolve_data_rank(_config(None)) == 0
        assert resolve_data_rank(SimpleNamespace()) == 0

    @pytest.mark.parametrize("strategy", ["none", "dp", "NONE", "Dp"])
    def test_single_process_strategies_are_rank_zero(self, strategy: str) -> None:
        """``dp`` is DataParallel threads inside ONE process: no rank to offset
        by, and offsetting would reseed the single stream for nothing."""
        assert resolve_data_rank(_config(strategy)) == 0

    def test_uninitialised_process_group_is_rank_zero(self) -> None:
        """Declared ddp but launched without torchrun — one process."""
        assert resolve_data_rank(_config("ddp")) == 0

    @pytest.mark.parametrize("strategy", ["ddp", "fsdp", "deepspeed"])
    @pytest.mark.parametrize("rank", [0, 1, 3])
    def test_initialised_process_group_returns_the_real_rank(
        self, strategy: str, rank: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mriforge.pipelines.parallel as parallel_mod

        monkeypatch.setattr(parallel_mod.torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(parallel_mod.torch.distributed, "get_rank", lambda: rank)

        assert resolve_data_rank(_config(strategy)) == rank

    @pytest.mark.parametrize("strategy", ["ddp", "fsdp", "deepspeed", "dp", "none"])
    def test_agrees_with_is_rank_zero_on_every_strategy(
        self, strategy: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two helpers must never disagree about who rank 0 is. Pins them
        together so a guard added to one cannot be forgotten in the other."""
        import mriforge.pipelines.parallel as parallel_mod

        monkeypatch.setattr(parallel_mod.torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(parallel_mod.torch.distributed, "get_rank", lambda: 2)

        cfg = _config(strategy)
        assert (resolve_data_rank(cfg) == 0) is is_rank_zero(cfg)
