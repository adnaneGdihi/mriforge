"""How each parallel strategy wants its checkpoints written.

Three facts differ per strategy, and getting any of them wrong is silent:

1. **Who calls.** FSDP's ``state_dict()`` and DeepSpeed's ``save_checkpoint()``
   are *collectives*. The training loop gates every checkpoint on
   ``RankUtility.is_main_rank()``, so rank 0 would enter an all-gather that
   ranks 1..N never enter: the job HANGS with no error and no log line, and
   SLURM eventually kills it at walltime. That is the most expensive possible
   failure mode -- far worse than a crash -- which is why
   :attr:`requires_all_ranks` exists and why the loop derives ONE predicate
   from it rather than editing each call site.

2. **What ``state_dict()`` returns.** Under FSDP the default is the local
   *shard*. Saving that produces a file with the right key names, the right
   dtypes, and 1/N of the values -- so it loads into a bare model without
   complaint and reconstructs a model that was never trained.

3. **What the artifact IS.** DeepSpeed writes a tag *directory* of per-rank
   shards, not a file. Every consumer in this repo (``discover_best_checkpoint``,
   the campaign orchestrator, ``spectramr infer``) expects a single file, so the
   backend additionally publishes a consolidated one.

``rank0_only=True`` controls who *receives* the gathered tensors, never who
*participates* -- a distinction that reads as an optimisation and is actually
the deadlock.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Generator, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "DeepSpeedCheckpointAdapter",
    "DefaultCheckpointAdapter",
    "FSDPCheckpointAdapter",
    "IParallelCheckpointAdapter",
    "resolve_checkpoint_adapter",
]

#: Subdirectory name for DeepSpeed's sharded tag directories.
DEEPSPEED_TAG_PREFIX = "deepspeed"


@runtime_checkable
class IParallelCheckpointAdapter(Protocol):
    """The checkpoint contract one parallel strategy imposes."""

    name: str

    #: True when save/load must be entered by EVERY rank because the strategy's
    #: state-gathering is collective. The training loop ORs this with
    #: ``is_main_rank()``; the adapter itself decides who touches the disk.
    requires_all_ranks: bool

    #: True when the strategy writes its own artifact and the generic
    #: ``torch.save`` payload would be shard-shaped nonsense.
    writes_native_artifact: bool

    def gather_full_state_dict(self, model: Any) -> Any:
        """Context manager active while ``model.state_dict()`` is read."""
        ...

    def save_native(
        self,
        *,
        models: Mapping[str, Any],
        checkpoint_dir: Path,
        tag: str,
        client_state: Mapping[str, Any],
    ) -> Path | None:
        """Write the strategy-native artifact. ``None`` = nothing extra to write."""
        ...

    def load_native(
        self, *, models: Mapping[str, Any], checkpoint_dir: Path, tag: str
    ) -> dict[str, Any] | None:
        """Restore from the strategy-native artifact, returning its client state."""
        ...


class DefaultCheckpointAdapter:
    """No wrapper, DataParallel, DDP: ``state_dict()`` is already complete.

    DP and DDP replicate rather than shard, so rank 0's copy IS the model.
    Prefix stripping is handled by :mod:`spectramr.core.module_utils` at the
    save site and is not this adapter's concern.
    """

    name = "default"
    requires_all_ranks = False
    writes_native_artifact = False

    @contextlib.contextmanager
    def gather_full_state_dict(self, model: Any) -> Generator[None]:
        yield

    def save_native(self, **_kwargs: Any) -> Path | None:
        return None

    def load_native(self, **_kwargs: Any) -> dict[str, Any] | None:
        return None


class FSDPCheckpointAdapter:
    """Gather the full state dict onto rank 0 before it is read.

    Without the context manager, ``state_dict()`` returns this rank's flat
    shard. The resulting file has plausible key names and 1/N of the weights,
    so ``load_state_dict(..., strict=True)`` on a bare model raises a *shape*
    error at world_size>1 -- but at world_size==1 (``no_shard``, or a
    single-GPU debug run) it loads cleanly, which is how this class of bug
    survives local testing.

    ``offload_to_cpu=True`` is not a nicety: the gathered dict is the FULL
    model on one device, on top of that rank's shard and its optimizer state.
    """

    name = "fsdp"
    #: state_dict() under FSDP all-gathers -- every rank must call it.
    requires_all_ranks = True
    writes_native_artifact = False

    @contextlib.contextmanager
    def gather_full_state_dict(self, model: Any) -> Generator[None]:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if not isinstance(model, FSDP):
            # Nested/partially-wrapped trees: the context manager is a no-op on
            # a module that is not itself an FSDP root, and entering it would
            # raise rather than degrade.
            yield
            return
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            yield

    def save_native(self, **_kwargs: Any) -> Path | None:
        return None

    def load_native(self, **_kwargs: Any) -> dict[str, Any] | None:
        return None


class DeepSpeedCheckpointAdapter:
    """A sharded tag directory for resume, plus a consolidated file for everyone else.

    Both artifacts are deliberate. ``engine.save_checkpoint`` writes per-rank
    ZeRO shards -- the only thing that can restore optimizer state at stage 2/3
    -- but it is a *directory*, and every consumer in this repo
    (``discover_best_checkpoint``, campaign evaluation, ``spectramr infer``)
    opens a file. Publishing only the tag directory would mean a run that
    trains for a week and then cannot be evaluated by any existing tool.
    """

    name = "deepspeed"
    #: save_checkpoint/load_checkpoint are collective.
    requires_all_ranks = True
    writes_native_artifact = True

    def __init__(self, *, save_consolidated_best: bool = True) -> None:
        self.save_consolidated_best = save_consolidated_best

    @contextlib.contextmanager
    def gather_full_state_dict(self, model: Any) -> Generator[None]:
        # The engine owns gathering; nothing to arrange around state_dict().
        yield

    @staticmethod
    def _engines(models: Mapping[str, Any]) -> dict[str, Any]:
        """The values that are DeepSpeed engines, by model key.

        Duck-typed on ``save_checkpoint`` rather than ``isinstance`` so this
        module never imports deepspeed -- keeping it importable, and testable,
        on a host without the extra.
        """
        return {
            key: model
            for key, model in models.items()
            if model is not None and hasattr(model, "save_checkpoint")
        }

    def save_native(
        self,
        *,
        models: Mapping[str, Any],
        checkpoint_dir: Path,
        tag: str,
        client_state: Mapping[str, Any],
    ) -> Path | None:
        engines = self._engines(models)
        if not engines:
            return None

        tag_root = checkpoint_dir / DEEPSPEED_TAG_PREFIX
        for key, engine in engines.items():
            # COLLECTIVE: every rank calls, DeepSpeed decides who writes what.
            engine.save_checkpoint(str(tag_root / key), tag=tag, client_state=dict(client_state))
        logger.info("[DeepSpeed] wrote sharded checkpoint tag %r under %s", tag, tag_root)

        if not self.save_consolidated_best:
            return None
        return self._save_consolidated(engines, checkpoint_dir, tag)

    def _save_consolidated(
        self, engines: Mapping[str, Any], checkpoint_dir: Path, tag: str
    ) -> Path | None:
        """Single-file fp16/fp32 weights so non-DeepSpeed consumers keep working.

        ``save_16bit_model`` is itself collective at stage 3 (it all-gathers the
        partitioned parameters), so it is called on every rank; it no-ops off
        rank 0. Only the generator is consolidated -- the discriminator is not
        an inference artifact.
        """
        engine = engines.get("generator") or next(iter(engines.values()))
        filename = f"checkpoint_{tag}.pt" if tag != "best" else "checkpoint_best.pt"
        try:
            ok = engine.save_16bit_model(str(checkpoint_dir), filename)
        except Exception as exc:
            # A failed consolidation must not lose the sharded checkpoint that
            # DID land -- resume still works, only the eval artifact is missing.
            logger.error(
                "[DeepSpeed] consolidated save failed (%s). The sharded tag %r "
                "was written and resume still works, but discover_best_checkpoint "
                "and `spectramr infer` will not find a file for this run.",
                exc,
                tag,
            )
            return None
        if not ok:
            return None
        return checkpoint_dir / filename

    def load_native(
        self, *, models: Mapping[str, Any], checkpoint_dir: Path, tag: str
    ) -> dict[str, Any] | None:
        engines = self._engines(models)
        if not engines:
            return None
        tag_root = checkpoint_dir / DEEPSPEED_TAG_PREFIX
        client_state: dict[str, Any] = {}
        for key, engine in engines.items():
            # COLLECTIVE, same as the save side.
            load_path, state = engine.load_checkpoint(str(tag_root / key), tag=tag)
            if load_path is None:
                # DeepSpeed reports a miss by RETURNING (None, None), not by
                # raising. Swallowing that yields an empty dict indistinguishable
                # from "loaded, but this tag carried no client state" -- and the
                # caller's next move is to treat the sharded load as authoritative
                # and stop parsing the file beside it. The run would then report a
                # successful restore while still holding the weights it started
                # with, which is the silent-fallback shape non-negotiable 3 exists
                # to prevent.
                raise FileNotFoundError(
                    f"[{self.name}] no checkpoint for tag {tag!r} under "
                    f"{tag_root / key}. engine.load_checkpoint returned no path, "
                    "which is how DeepSpeed reports a missing or incomplete tag "
                    "directory."
                )
            if state:
                client_state.update(state)
        return client_state


def resolve_checkpoint_adapter(parallel: Any) -> IParallelCheckpointAdapter:
    """The adapter carried by a resolved :class:`ParallelRuntime` (or ``None``).

    The *plugin* builds its own adapter during ``adopt`` -- it is the only thing
    that knows how the model was wrapped and what the arm declared -- and this
    just reads it back, defaulting for the single-process and scripting-API
    paths where there is no runtime at all.

    Deriving the adapter here instead (by branching on ``parallel.strategy``)
    was the first shape of this function, and it could not see
    ``deepspeed.save_consolidated_best``: that knob is a spectramr concern, not a
    DeepSpeed one, so ``build_deepspeed_config`` correctly omits it from the
    rendered ds_config that provenance carries. The knob would have read as
    always-true -- wired, validated, stamped, and inert (pitfall #15).
    """
    if parallel is None:
        return DefaultCheckpointAdapter()
    adapter = getattr(parallel, "checkpoint_adapter", None)
    if adapter is None:
        return DefaultCheckpointAdapter()
    return adapter
