"""Render the DeepSpeed ``ds_config`` dict from ``TrainingSettings``.

**The schema is the SSOT; this dict is a derived artifact.** There is
deliberately no ``deepspeed.config_path`` field, and that is the load-bearing
design decision in this whole backend.

A hand-edited ``ds_config.json`` is exactly the object ``.claude/rules/data.md``
bans -- a cluster-critical file with no committed generator. It is worse than a
data manifest, because a manifest that fails to sync produces a loud
``Index file not found``, whereas a ds_config that disagrees with the YAML
produces a **successful run of the wrong experiment**: DeepSpeed will happily
accept a ``train_micro_batch_size_per_gpu`` that contradicts ``data.batch_size``,
and nothing anywhere compares them.

So four groups of keys are **not declarable** in ``DeepSpeedConfigSchema`` and
are always sourced from their existing SSOT:

======================================  ==========================================
ds_config key                           comes from
======================================  ==========================================
``train_micro_batch_size_per_gpu``      ``data.batch_size``
``gradient_accumulation_steps``         ``optimization.gradient.accumulation_steps``
``gradient_clipping``                   ``optimization.gradient.clip.value``
``fp16.enabled`` / ``bf16.enabled``     ``resolve_amp_precision(...)``
``optimizer`` / ``scheduler``           **absent** -- the engine adopts the objects
                                        ``OptimizationBuilder`` already built
======================================  ==========================================

Letting DeepSpeed construct the optimizer from its own dict would fork the
optimizer vocabulary away from TTUR (``discriminator_learning_rate``), the LR
multipliers, and ``resolve_scheduler_spec`` -- and ``provenance["lr_schedule"]``
would then describe a scheduler nothing ever stepped.

Pure: no torch import, no engine, no device. That is what lets the golden-dict
tests assert *which* knobs reached DeepSpeed on a machine with no GPU.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DERIVED_KEYS", "build_deepspeed_config"]

#: Keys this builder owns. Named so a test can assert the schema does not also
#: declare them -- two homes for one number is how a ds_config drifts.
DERIVED_KEYS: frozenset[str] = frozenset(
    {
        "train_micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
        "gradient_clipping",
        "fp16",
        "bf16",
        "optimizer",
        "scheduler",
    }
)


def _zero_block(ds: Any) -> dict[str, Any]:
    """The ``zero_optimization`` sub-dict."""
    block: dict[str, Any] = {
        "stage": ds.zero_stage,
        "contiguous_gradients": ds.contiguous_gradients,
        "overlap_comm": ds.overlap_comm,
        "reduce_bucket_size": ds.reduce_bucket_size,
        "allgather_bucket_size": ds.allgather_bucket_size,
        "round_robin_gradients": ds.round_robin_gradients,
    }
    if ds.offload_optimizer != "none":
        block["offload_optimizer"] = {"device": ds.offload_optimizer}
        if ds.offload_optimizer == "nvme":
            block["offload_optimizer"]["nvme_path"] = ds.nvme_path
    if ds.offload_param != "none":
        block["offload_param"] = {"device": ds.offload_param}
        if ds.offload_param == "nvme":
            block["offload_param"]["nvme_path"] = ds.nvme_path
    if ds.zero_stage == 3:
        # Required for save_16bit_model() to produce the consolidated
        # checkpoint that inference and campaign evaluation read.
        block["stage3_gather_16bit_weights_on_model_save"] = (
            ds.stage3_gather_16bit_weights_on_model_save
        )
    zenflow = _zenflow_block(ds)
    if zenflow is not None:
        # Nested under zero_optimization, NOT top-level: DeepSpeed reads it as
        # `self._config.zero_config.zenflow`.
        block["zenflow"] = zenflow
    return block


def _zenflow_block(ds: Any) -> dict[str, Any] | None:
    """The ``zero_optimization.zenflow`` sub-dict, or ``None`` when disabled.

    ``None`` rather than ``{"enabled": False}`` is load-bearing: DeepSpeed's
    ``configure_zenflow`` branches on ``zenflow_config == None`` to decide
    whether ZenFlow runs at all, and its ``ZenFlowConfig`` has no ``enabled``
    field. Emitting an empty dict would construct a ZenFlowConfig with all
    defaults and turn ZenFlow ON.
    """
    zf = getattr(ds, "zenflow", None)
    if zf is None or not zf.enabled:
        return None
    return {
        "topk_ratio": zf.topk_ratio,
        "select_strategy": zf.select_strategy,
        "select_interval": zf.select_interval,
        "update_interval": zf.update_interval,
        "overlap_step": zf.overlap_step,
        "offload": zf.offload,
        "auto_ratio": zf.auto_ratio,
        "full_warm_up_rounds": zf.full_warm_up_rounds,
        "pt_reserved_cores_perc": zf.pt_reserved_cores_perc,
    }


def _compile_block(ds: Any) -> dict[str, Any] | None:
    """The top-level ``compile`` sub-dict, or ``None`` when DeepCompile is off.

    Rendering this is only HALF the wiring -- ``engine.compile()`` must also be
    called, or the engine logs one rank-0 line and runs eagerly. See
    ``DeepSpeedStepPolicy`` / ``DeepSpeedStrategy.adopt``.
    """
    compile_cfg = getattr(ds, "compile", None)
    if compile_cfg is None or not compile_cfg.enabled:
        return None
    block: dict[str, Any] = {
        "deepcompile": True,
        "free_activation": compile_cfg.free_activation,
        "offload_activation": compile_cfg.offload_activation,
        "offload_opt_states": compile_cfg.offload_opt_states,
        "offload_parameters": compile_cfg.offload_parameters,
        "double_buffer": compile_cfg.double_buffer,
        "symmetric_memory": compile_cfg.symmetric_memory,
        "debug_log": compile_cfg.debug_log,
    }
    if compile_cfg.passes:
        block["passes"] = list(compile_cfg.passes)
    return block


def build_deepspeed_config(settings: Any) -> dict[str, Any]:
    """Render the ds_config for *settings*.

    Raises:
        ValueError: if the arm's precision cannot be expressed to DeepSpeed.
    """
    from spectramr.infrastructure.training.mixed_precision import resolve_amp_precision

    parallel = settings.parallel
    ds = parallel.deepspeed
    optimization = settings.optimization

    # Returns ``(enabled, precision)`` -- note the order; `amp_dtype: float32`
    # resolves to enabled=False, which is why the fp16/bf16 blocks below are
    # gated on `amp_enabled` and not merely on the precision string.
    amp_enabled, precision = resolve_amp_precision(
        optimization.precision.enabled,
        optimization.precision.dtype,
    )

    config: dict[str, Any] = {
        # NOT declarable in the schema -- sourced from data.batch_size so the
        # ds_config cannot claim a batch size the dataloader is not producing.
        "train_micro_batch_size_per_gpu": int(settings.data.loader.batch_size),
        "gradient_accumulation_steps": int(optimization.gradient.accumulation_steps or 1),
        "steps_per_print": ds.steps_per_print,
        "zero_optimization": _zero_block(ds),
        # No "optimizer"/"scheduler": the engine adopts the objects
        # OptimizationBuilder built, so TTUR and the LR-schedule SSOT keep
        # describing what actually ran.
    }

    clip = optimization.gradient.clip.value
    if optimization.gradient.clip.enabled and clip:
        config["gradient_clipping"] = float(clip)

    if amp_enabled and precision == "fp16":
        config["fp16"] = {"enabled": True}
    elif amp_enabled and precision == "bf16":
        config["bf16"] = {"enabled": True}

    compile_block = _compile_block(ds)
    if compile_block is not None:
        config["compile"] = compile_block

    # ZenFlow OVERWRITES gradient_accumulation_steps with its update_interval
    # (configure_zenflow's last statement). Render the value the engine will
    # actually use, so the ds_config on disk is not contradicted the moment the
    # engine starts; check_zenflow_accumulation_conflict errors when the
    # declared value and update_interval disagree in the first place.
    #
    # ONLY for an integer update_interval. With "auto" the engine sets
    # update_interval = 0, and rendering a 0 here is NOT harmless: DeepSpeed's
    # own parser asserts "Train batch size: 0 has to be greater than 0" and
    # deepspeed.initialize dies. It tolerates 0 only POST-parse, where
    # configure_zenflow assigns it. So on the "auto" path we leave the declared
    # value alone -- the audit has already forced it to 1.
    zenflow = getattr(ds, "zenflow", None)
    if zenflow is not None and zenflow.enabled and isinstance(zenflow.update_interval, int):
        config["gradient_accumulation_steps"] = zenflow.update_interval

    return config
