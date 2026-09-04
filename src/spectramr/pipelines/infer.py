"""Unified Inference Pipeline - Config-Driven Single Entry Point.

This module provides a configuration-aware inference pipeline that:
1. Uses the training config to understand model architecture
2. Uses the config to understand data preprocessing (normalization, channel layout)
3. Uses the config to determine the training paradigm for proper inference strategy
4. Ensures consistent data flow between training and inference

Single Source of Truth: TrainingSettings.from_yaml()
"""

import logging
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from spectramr.config.settings import TrainingSettings
from spectramr.core.compute_device import resolve_torch_device
from spectramr.core.module_utils import resolve_state_dict, unwrap_model
from spectramr.domain.exceptions import (
    ConfigurationError,
    DataCorruptionError,
    DimensionMismatchError,
    SpectraMRError,
)
from spectramr.infrastructure.inference.inference_factory import InferenceStrategyFactory
from spectramr.infrastructure.reporting.inference_artifacts import (
    STATUS_COMPUTED,
    InferenceEvaluator,
    write_inference_run_summary,
)
from spectramr.infrastructure.reporting.run_hook import maybe_run_reporting

logger = logging.getLogger(__name__)


def _resolve_inference_paradigm(config: Any) -> str:
    """Resolve the inference paradigm from the SSOT strategy detector.

    Defers to the SAME authority the forward-pass dispatch uses
    (:class:`InferenceStrategyFactory` → ``StrategyDetector``), so the
    logged/recorded paradigm cannot diverge from the strategy actually built.
    The previous inline substring ``if/elif`` in ``run_inference_pipeline`` was a
    weaker *second* classifier that silently recorded ``"unknown"`` for any
    strategy it didn't recognise — a misleading provenance label (pitfall
    #9/#18).

    Multi-stage is the ONE exception: the detector has no ``"multi"`` rule, so
    the multi-stage checkpoint-loading branch is reachable ONLY via the declared
    ``strategy_class`` name — that gate stays a name check.
    """
    training = getattr(config, "training", None)
    strategy_class = getattr(training, "strategy_class", None) if training else None
    if strategy_class and "multi" in strategy_class.split(".")[-1].lower():
        return "multi"
    return InferenceStrategyFactory._infer_strategy_type(config.model_dump())


def resolve_inference_settings(
    config_path: Path | str | None,
    checkpoint_path: Path | str | None,
    *,
    from_yaml: bool = False,
) -> tuple[TrainingSettings, dict[str, Any]]:
    """The settings an inference run uses, and where they came from.

    Precedence (#1379, the corpus review's T0.13): the run's own
    ``resolved_config.json`` beside the checkpoint wins, because it holds what
    the training run declared after every override; the training YAML is read
    only when ``from_yaml`` is set or no artifact exists. When both exist and
    the YAML disagrees with the artifact, the top-level blocks that differ are
    logged at WARNING and recorded in the source, so a predict run cannot
    silently score under a config the checkpoint never trained under.

    Returns ``(settings, source)`` where ``source`` is
    ``{"kind": "resolved_config" | "yaml", "path": ..., ...}``.
    """
    from spectramr.infrastructure.validation.resolved_config_artifact import (
        has_declared_block,
        resolved_config_beside,
        settings_from_resolved_config,
    )

    artifact = resolved_config_beside(checkpoint_path)
    # An artifact written before the ``_declared`` block cannot rebuild the
    # settings; every run directory on the cluster is in that state today, so
    # the documented ``infer --config ...`` command must keep working there:
    # the YAML is used and the artifact's age is reported, never a refusal.
    predates = artifact is not None and not has_declared_block(artifact)
    if from_yaml or artifact is None or predates:
        if config_path is None or not Path(config_path).exists():
            raise FileNotFoundError(
                "inference needs its settings: pass --config <training YAML>, or run against "
                "a checkpoint whose run directory holds a resolved_config.json with a "
                "`_declared` block"
                + (
                    f" ({artifact} predates that block, written before 2026-09-03)"
                    if predates
                    else (
                        f" (none beside {checkpoint_path})" if checkpoint_path is not None else ""
                    )
                )
                + ("; --from-yaml was set" if from_yaml else "")
                + "."
            )
        settings = TrainingSettings.from_yaml(str(config_path))
        source: dict[str, Any] = {
            "kind": "yaml",
            "path": str(config_path),
            "resolved_config": str(artifact) if artifact is not None else None,
        }
        if predates and not from_yaml:
            source["resolved_config_predates_declared"] = True
            logger.warning(
                "[config] %s predates the `_declared` block (written before 2026-09-03) and cannot "
                "rebuild the run's settings; inference uses %s. Re-running training rewrites the "
                "artifact with the block.",
                artifact,
                config_path,
            )
        return settings, source
    settings = settings_from_resolved_config(artifact)
    source = {"kind": "resolved_config", "path": str(artifact), "yaml": None}
    if config_path is not None and Path(config_path).exists():
        source["yaml"] = str(config_path)
        from_artifact = settings.model_dump(mode="json")
        from_file = TrainingSettings.from_yaml(str(config_path)).model_dump(mode="json")
        diverging = sorted(
            key
            for key in set(from_artifact) | set(from_file)
            if from_artifact.get(key) != from_file.get(key)
        )
        if diverging:
            source["diverging_blocks"] = diverging
            logger.warning(
                "[config] %s and %s disagree in %s; inference uses the run's resolved config. "
                "Pass --from-yaml to score under the YAML instead.",
                artifact,
                config_path,
                diverging,
            )
    return settings, source


def _describe_inference_source(
    config_path: Path | str | None, checkpoint_path: Path | str | None, from_yaml: bool
) -> dict[str, Any]:
    """The ``source`` :func:`resolve_inference_settings` would report, without loading."""
    from spectramr.infrastructure.validation.resolved_config_artifact import (
        has_declared_block,
        resolved_config_beside,
    )

    artifact = resolved_config_beside(checkpoint_path)
    predates = artifact is not None and not has_declared_block(artifact)
    if from_yaml or artifact is None or predates:
        source: dict[str, Any] = {
            "kind": "yaml",
            "path": str(config_path),
            "resolved_config": str(artifact) if artifact else None,
        }
        if predates and not from_yaml:
            source["resolved_config_predates_declared"] = True
        return source
    return {
        "kind": "resolved_config",
        "path": str(artifact),
        "yaml": str(config_path) if config_path else None,
    }


def run_inference_pipeline(
    config_path: Path | None,
    checkpoint_path: Path,
    input_path: Path | None,
    output_path: Path,
    device: str | None = "cuda",
    batch_size: int | None = None,
    from_manifest_test_split: bool = False,
    from_yaml: bool = False,
    settings: TrainingSettings | None = None,
) -> dict[str, Any]:
    """Unified inference pipeline using config-driven setup.

    SSOT Principle: All configuration comes from training YAML.
    Data preprocessing, model architecture, and strategy selection
    are derived from the training configuration.

    Args:
        config_path: Path to the training YAML. Optional once the checkpoint's run
            directory holds ``resolved_config.json`` (see
            :func:`resolve_inference_settings`); required with ``from_yaml``.
        checkpoint_path: Path to saved model checkpoint
        input_path: Path to input file or directory (NIfTI, NPY, HDF5)
        output_path: Path to output directory
        device: Device to run on ('auto' | 'cuda' | 'cuda:N' | 'cpu'). Inference
            is a heavy pipeline: with no accelerator and no explicit CPU opt-in
            this RAISES ``AcceleratorRequiredError`` rather than degrading to
            CPU (see :mod:`spectramr.core.compute_device`).
        batch_size: Override batch size (uses config default if None)
        from_manifest_test_split: Resolve the input roster from the paired v4
            manifest's held-out test split (``split_hint == "test"``, the
            unpaired-ULF cohort) instead of walking ``input_path``. Requires
            ``data.source.paired_manifest_path``. Opt-in on purpose: making
            this a fallback for an empty glob would let a mistyped
            ``input_path`` silently switch data sources (pitfall #9).

    Returns:
        dict with output paths, metrics, and processing summary

    Raises:
        FileNotFoundError: If config or checkpoint not found
        ValueError: If config is invalid or incomplete
        AcceleratorRequiredError: If no accelerator is available and CPU was not
            explicitly requested.
    """
    # The bare ``torch.device(device)`` this replaces neither honoured "auto"
    # (torch.device("auto") is a TypeError) nor validated CUDA availability —
    # it happily built a cuda device object on a GPU-less host and only failed
    # later, deep in a ``.to(device)``. Route through the SSOT contract.
    decision = resolve_torch_device(device, pipeline="infer", source="caller")
    device_obj = torch.device(decision.device)
    logger.info(
        "Starting unified inference on %s (accelerated=%s)",
        decision.device,
        decision.accelerated,
    )

    # =====================================================================
    # PHASE 1: Load Configuration (SSOT - Single Source of Truth)
    # =====================================================================
    if settings is None:
        config, config_source = resolve_inference_settings(
            config_path, checkpoint_path, from_yaml=from_yaml
        )
    else:
        # The preamble (``main.begin_inference_run``) resolved them once already;
        # describing the source again costs a file-existence check, not a load.
        config = settings
        config_source = _describe_inference_source(config_path, checkpoint_path, from_yaml)
    logger.info(
        "[config] inference settings from %s (%s)", config_source["path"], config_source["kind"]
    )

    # Workflow maturity gate — a STUB regime cannot run inference either.
    # EVAL_ONLY regimes are allowed to predict (only training is blocked).
    from spectramr.domain.workflows import enforce_pipeline_maturity_for_config

    enforce_pipeline_maturity_for_config(config, "predict")

    paradigm = _resolve_inference_paradigm(config)
    logger.info(f"Config loaded: model={config.model.model_type}, paradigm={paradigm}")

    # =====================================================================
    # PHASE 2: Create Model/Strategy and Load Checkpoint
    # =====================================================================
    if paradigm == "multi":
        logger.info("Using MultiTrainingStrategy for multi-stage inference")
        from spectramr.infrastructure.inference.multi_inference_strategy import (
            MultiInferenceStrategy,
        )
        from spectramr.infrastructure.training.strategies.pipeline_strategy import (
            MultiTrainingStrategy,
        )

        multi_strategy = MultiTrainingStrategy(env=config, device=device_obj)
        multi_strategy.load_checkpoint(str(checkpoint_path))
        multi_strategy.eval_mode()

        # We dummy-out the model variable for later phases
        model = torch.nn.Identity()
        if hasattr(multi_strategy, "multi_stages") and multi_strategy.multi_stages:
            # Provide an internal module for shape/channel extraction
            model = next(iter(multi_strategy.multi_stages.values()))

        strategy = MultiInferenceStrategy(multi_strategy, device_obj, config.model_dump())
        logger.info(f"Using inference strategy: {strategy.__class__.__name__}")
    else:
        logger.info(f"Creating model: {config.model.model_type}")
        # Build through the SAME canonical builder training uses (the `fit.py`
        # precedent for a pipeline importing ModelBuilder).
        #
        # `ModelFactory.create_model(config.model)` -- what this replaces -- was
        # handed a bare ModelConfigSchema and so took the branch injecting only
        # in/out_channels plus the two diffusion blocks. It never injected
        # `acceleration_config`, dropped on all 58 kspace_filling arms, every
        # one of which declares `undersampling:`; and never injected
        # `kspace_log_scaled`, whose absence makes the 12 arms setting
        # `output_kspace_clip_ratio` raise at construction (#1306).
        #
        # ModelBuilder's injections are contract-gated against the generator's
        # own __init__ signature, so this cannot hand a model a kwarg it does
        # not accept, and its SSOT reconciliations raise on a config that
        # disagrees with itself. GeneratorBuilder places the module on
        # `device_obj` as it builds, so no `.to()` is needed here.
        from spectramr.infrastructure.training.builders.model_builder import ModelBuilder

        model = ModelBuilder(config, device_obj).build_generator().validate().build()["generator"]
        logger.info("Model created on %s via ModelBuilder", device_obj)

        # Load weights
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device_obj)
            # ``model`` here is freshly built and bare, so a checkpoint written
            # by a compiled / DDP / FSDP run carries "_orig_mod."/"module."
            # prefixes -- and CheckpointDirector wraps the parameters under
            # "generator", an envelope this reader knew nothing about. The old
            # `model_state_dict`-else-whole-dict branch therefore handed the
            # entire envelope (epoch, optimizer, config, ...) to
            # load_state_dict and failed with 'generator' (#1310).
            model.load_state_dict(
                resolve_state_dict(
                    checkpoint, model.state_dict().keys(), source=str(checkpoint_path)
                )
            )
            logger.info("Checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

        # Sub-phase 2.10: enforce train↔infer transform parity when the
        # YAML declares ``data.modes.infer.strict_train_parity=true``.
        # CheckpointService writes the training-time signature under key
        # 'transform_signature' (Phase 2 amendments). When the flag is
        # true, we recompute the val-mode signature from the current YAML
        # and compare; on divergence we refuse to proceed.
        from spectramr.data.transforms.signature import enforce_train_infer_parity

        _checkpoint_signature: str | None = None
        if isinstance(checkpoint, dict):
            _checkpoint_signature = checkpoint.get("transform_signature")
        _infer_sig = enforce_train_infer_parity(config, _checkpoint_signature)
        logger.info(
            "[STRICT-PARITY] Transform signature %s… — inference proceeds.",
            _infer_sig[:16],
        )

        # =====================================================================
        # PHASE 3: Create Inference Strategy (Training Paradigm)
        # =====================================================================
        logger.info(f"Creating inference strategy for paradigm: {paradigm}")
        strategy = InferenceStrategyFactory.create(model, device_obj, config.model_dump())
        logger.info(f"Using inference strategy: {strategy.__class__.__name__}")

    # =====================================================================
    # PHASE 4: Prepare Output and Collect Inputs
    # =====================================================================
    logger.info(f"Building inference data loader from {input_path}")
    if batch_size is None:
        batch_size = config.data.loader.batch_size

    # Collect input files. Two mutually exclusive rosters: the manifest's
    # held-out test split (explicit opt-in) or a walk of ``input_path``.
    if from_manifest_test_split:
        from spectramr.data.metadata.test_split_resolver import (
            resolve_manifest_test_paths,
        )

        if not config.data.source.paired_manifest_path:
            raise ValueError(
                "from_manifest_test_split=True requires "
                "data.source.paired_manifest_path to be set; the config "
                "declares no paired manifest, so there is no test split to "
                "resolve."
            )
        input_files = resolve_manifest_test_paths(config.data)
        logger.info(
            "Test-split inference: %d unpaired-ULF record(s) from %s",
            len(input_files),
            config.data.source.paired_manifest_path,
        )
        if not input_files:
            raise ValueError(
                "from_manifest_test_split=True resolved 0 records from "
                f"{config.data.source.paired_manifest_path}. The manifest "
                "declares no split_hint=='test' entries, so the held-out "
                "cohort is empty."
            )
    else:
        if input_path is None:
            raise ValueError(
                "input_path is required unless from_manifest_test_split=True "
                "selects the manifest's held-out test cohort instead."
            )
        input_files = _collect_input_files(Path(input_path))
        logger.info(f"Found {len(input_files)} input file(s)")

        if not input_files:
            raise ValueError(f"No input files found in {input_path}")

    # =====================================================================
    # PHASE 5: Execute Inference
    # =====================================================================
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")

    results = {
        "config_used": config_source["path"],
        "config_source": config_source,
        "checkpoint_used": str(checkpoint_path),
        "paradigm": paradigm,
        "model_type": config.model.model_type,
        "outputs": [],
        "num_processed": 0,
        "failures": [],
    }
    # Stamped again after the loop with the counts; declared here so a run that
    # aborts before its first file still says what the knob was.
    results["data_consistency_at_predict"] = strategy.predict_dc_provenance()

    # Evaluation is set up before the loop so the declared set is resolved once,
    # and so a metric the arm names but this path cannot compute is recorded for
    # EVERY case rather than discovered per file.
    evaluator = InferenceEvaluator(_declared_metrics(config), device=str(device_obj))
    started_at = time.monotonic()

    model.eval()
    with torch.no_grad():
        _process_all_files(
            input_files,
            output_path,
            model,
            strategy,
            device_obj,
            config,
            batch_size,
            results,
            evaluator,
        )

    # =====================================================================
    # PHASE 6: Report Artifacts
    # =====================================================================
    # This is what makes `report` usable after `infer`/`predict`. The verb used
    # to write output tensors and stop, while its own docstring above promised
    # "output paths, metrics, and processing summary" -- so the otherwise fully
    # artifact-driven report verb had nothing to consume.
    #
    # ORDERING IS LOAD-BEARING, and it is the same mistake `train.py` made until
    # this PR: the artifacts must be on disk BEFORE the reporting hook draws,
    # or the figures that read them soft-skip and the run yields a different
    # figure set depending on whether `report` was re-run by hand afterwards.
    # Whether, and by whom, predictions were projected onto the measurement.
    results["data_consistency_at_predict"] = strategy.predict_dc_provenance()
    artifacts = evaluator.write(output_path)
    write_inference_run_summary(
        output_path,
        model=model,
        duration_sec=time.monotonic() - started_at,
        effective_batch=batch_size,
        extra={
            "pipeline": "infer",
            "config_used": config_source["path"],
            "config_source": config_source,
            "checkpoint_used": str(checkpoint_path),
            "paradigm": paradigm,
            "model_type": config.model.model_type,
            "num_processed": results["num_processed"],
            "num_failed": len(results["failures"]),
            "seed": config.run.seed,
            "data_consistency_at_predict": results["data_consistency_at_predict"],
        },
    )
    outcomes = evaluator.outcomes()
    results["metrics"] = {o.name: o.values for o in outcomes if o.status == STATUS_COMPUTED}
    results["metrics_skipped"] = {o.name: o.reason for o in outcomes if o.status != STATUS_COMPUTED}
    results["artifacts"] = {k: str(v) for k, v in artifacts.items()}

    maybe_run_reporting(config, run_dir=output_path, logger_=logger)

    logger.info(
        f"Inference complete. Processed {results['num_processed']} file(s). "
        f"Outputs saved to {output_path}"
    )
    return results


def _declared_metrics(config: TrainingSettings) -> list[str]:
    """The metric names this arm declares, resolved the way TRAINING resolves them.

    Deliberately not a second resolver. ``metrics.compute`` wins outright when
    present and the legacy ``compute_*`` flags are the fallback, with the same
    strict/lenient split (an explicit name raises, a dangling flag warns) -- and
    the only way to be sure of that is to call the code training calls. A private
    method is an awkward thing for a pipeline to reach for, but the alternative is
    a fourth implementation of a selection rule that already exists in three
    places, and a report whose metric set silently disagrees with the run it
    describes is exactly the divergence this change set exists to close.
    ``pipelines/training_loop.py`` reaches the same class the same way.
    """
    from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
        MetricsMixin,
    )

    return MetricsMixin._extract_metrics_from_config(MetricsMixin(), config.metrics)


def _process_all_files(
    input_files: list[Path],
    output_path: Path,
    model: torch.nn.Module,
    strategy: Any,
    device_obj: torch.device,
    config: TrainingSettings,
    batch_size: int,
    results: dict[str, Any],
    evaluator: Any = None,
) -> None:
    """Run every input through the model, classifying failures by blast radius.

    Extracted from :func:`run_inference_pipeline` so the classification below is
    reachable by a test. The loop it replaces caught bare ``Exception`` for every
    input, logged it, and continued -- so a run in which *every* file failed
    still returned normally, logged "Processed 0 file(s)", and exited 0.

    The distinction that matters is not the severity of a fault but its blast
    radius:

    * A **per-file** fault (an unreadable input) says nothing about the next
      file. Continuing is correct for a batch verb -- but the failure is recorded
      in ``results["failures"]``, never merely logged.
    * A **run-invariant** fault is a property of ``(config, checkpoint, model)``
      and therefore recurs identically on every remaining input. The channel
      contract in :func:`_adapt_channels` and the normalization gate in
      :func:`_normalize_like_training` are both of this kind. Catching them
      per-file turns that strictness into a facade (pitfall #16): N identical
      errors in the log and a zero exit status with nothing written.
    """
    for input_file in tqdm(input_files, desc="Processing files"):
        try:
            _process_single_file(
                input_file,
                output_path,
                model,
                strategy,
                device_obj,
                config,
                batch_size,
                results,
                evaluator,
            )
        except DataCorruptionError as e:
            logger.error("Failed to process %s: %s", input_file.name, e)
            results["failures"].append({"file": str(input_file), "error": str(e)})
        except SpectraMRError:
            # Run-invariant -- see the docstring. Abort on the first one rather
            # than repeating it once per remaining file.
            raise
        except Exception as e:
            # Unclassified: treated as per-file so one unexpected fault cannot
            # kill a long batch, but recorded like any other failure.
            logger.error("Failed to process %s: %s", input_file.name, e)
            results["failures"].append({"file": str(input_file), "error": str(e)})

    if results["failures"] and results["num_processed"] == 0:
        raise RuntimeError(
            f"Inference produced no output: all {len(results['failures'])} "
            f"input file(s) failed. First error: "
            f"{results['failures'][0]['error']}"
        )
    if results["failures"]:
        logger.warning(
            "Inference completed with %d failure(s) out of %d file(s); they are "
            "recorded in results['failures'].",
            len(results["failures"]),
            len(input_files),
        )


def _collect_input_files(input_path: Path) -> list[Path]:
    """Collect all valid input files."""
    input_files = []

    if input_path.is_file():
        if input_path.suffix == ".txt":
            # File manifest
            logger.info(f"Reading file manifest from {input_path}")
            with open(input_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    p = Path(line)
                    if not p.is_absolute():
                        p = input_path.parent / line
                    if p.exists():
                        input_files.append(p)
                    else:
                        logger.warning(f"File not found: {line}")
        else:
            # Single file
            input_files = [input_path]
    else:
        # Directory - collect all data files
        for suffix in ["*.nii.gz", "*.nii", "*.npy", "*.h5", "*.hdf5"]:
            input_files.extend(input_path.glob(suffix))

    return sorted(set(input_files))  # Remove duplicates


def _process_single_file(
    input_file: Path,
    output_path: Path,
    model: torch.nn.Module,
    strategy: Any,
    device_obj: torch.device,
    config: TrainingSettings,
    batch_size: int,
    results: dict,
    evaluator: Any = None,
) -> None:
    """Process a single input file using config-aligned preprocessing."""
    logger.info(f"Processing: {input_file.name}")

    # Load input
    tensor = _load_input(input_file, device_obj)
    logger.info(f"Loaded input: shape={tensor.shape}, dtype={tensor.dtype}")

    # Preprocess
    tensor = _preprocess_tensor(tensor, config, device_obj, model)
    logger.info(f"After preprocessing: shape={tensor.shape}")

    # What the predict-time projection needs, when the arm asks for it.
    dc_inputs = _predict_dc_inputs(strategy, input_file, tensor, config)

    # Batch processing
    num_samples = tensor.shape[0]
    outputs_list = []

    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch_tensor = tensor[i:batch_end]

        with torch.no_grad():
            batch_output = strategy.infer_single(
                batch_tensor, **_dc_batch_kwargs(dc_inputs, batch_tensor, i, batch_end)
            )

        if isinstance(batch_output, tuple):
            batch_output = batch_output[0]

        outputs_list.append(batch_output.detach().cpu())

    # Concatenate all batches
    output = torch.cat(outputs_list, dim=0)

    # Save output
    _save_output(output, input_file, output_path, config, device_obj)

    if evaluator is not None:
        # No ``target=`` argument, and that is the finding rather than an
        # oversight: ``_load_input`` reads one file straight off disk instead of
        # going through ``DataPipelineDirector``, so nothing on this path ever
        # constructs the reference that ``data.target_mode`` describes. Every
        # full-reference metric the arm declares is therefore recorded as
        # skipped-with-reason instead of silently missing. Supply a target here
        # and they compute unchanged.
        evaluator.observe(case_id=input_file.stem, prediction=output)

    results["outputs"].append(str(output_path / f"{input_file.stem}_output.npy"))
    results["num_processed"] += 1


def _predict_dc_inputs(
    strategy: Any, input_file: Path, tensor: torch.Tensor, config: TrainingSettings
) -> dict[str, Any] | None:
    """The mask and the measurement for ``physics.data_consistency.apply_at_predict``.

    ``None`` when the knob is off, so the off state changes nothing about the
    call the strategy receives. When it is on, this attaches what this path can
    honestly supply and says so per file:

    * the **mask** from the input's ``mask`` HDF5 dataset (an explicit input
      contract, the fastMRI test-set shape; no dataset or writer in the data
      layer produces one today), read through the io SSOT's strict reader so
      an absent key is ``None`` and never another dataset;
    * the **measurement**, which is the preprocessed input itself on a k-space
      route (``data.dataset_type`` serves k-space): the model and the projection
      then see the same tensor on the same scale by construction. On an image
      route there is no measurement to attach.

    Nothing is raised here. ``PredictDataConsistency.project`` is the one owner
    of "knob on requires both", and it raises naming the knob; this log line is
    what puts the file name above that traceback.
    """
    if getattr(strategy, "predict_dc", None) is None:
        return None
    from spectramr.data.io_strategies import load_h5_dataset_if_present
    from spectramr.data.signal_domain import is_kspace_dataset_type

    mask = load_h5_dataset_if_present(input_file, "mask")
    measurement_is_input = is_kspace_dataset_type(config.data.dataset_type)
    logger.info(
        "[PREDICT-DC] %s: mask=%s, measurement=%s (data.dataset_type=%r)",
        input_file.name,
        "absent" if mask is None else tuple(mask.shape),
        tuple(tensor.shape) if measurement_is_input else "absent (image route)",
        config.data.dataset_type,
    )
    return {
        "mask": mask,
        "measurement_is_input": measurement_is_input,
        "num_samples": tensor.shape[0],
    }


def _dc_batch_kwargs(
    dc_inputs: dict[str, Any] | None, batch_tensor: torch.Tensor, start: int, end: int
) -> dict[str, Any]:
    """Per-batch kwargs for ``infer_single``: empty when the knob is off.

    A mask whose leading entry count equals the file's sample count is sliced
    with the batch (one plane per slice); any other layout is passed whole and
    broadcast by the projection.
    """
    if dc_inputs is None:
        return {}
    mask = dc_inputs["mask"]
    if mask is not None and mask.ndim >= 3 and mask.shape[0] == dc_inputs["num_samples"]:
        mask = mask[start:end]
    return {
        "mask": mask,
        "measured_kspace": batch_tensor if dc_inputs["measurement_is_input"] else None,
    }


def _load_input(input_file: Path, device_obj: torch.device) -> torch.Tensor:
    """Load input from various formats (NIfTI, NPY, HDF5) via the data-layer SSOT.

    File→tensor conversion MUST go through ``spectramr.data.io_strategies`` — the
    pipeline layer never calls ``h5py.File`` / ``nib.load`` / ``np.load``
    directly (CLAUDE.md pitfall #11). The old inline reader had already drifted
    from the canonical one (no ``.nii.gz`` compound-suffix handling, a different
    H5 key order, a broken ``.float()``-then-complex path); routing through the
    SSOT keeps future fixes (dtype coercion, key precedence) reaching inference.
    """
    from spectramr.data.io_strategies import load_tensor_from_file

    tensor = load_tensor_from_file(
        input_file,
        h5_keys=("kspace", "reconstruction_rss", "reconstruction_esc", "data"),
    )
    # Real-valued models consume float32; keep complex k-space complex (the old
    # code's ``.float()`` on a complex array raised before it could be used).
    if not torch.is_complex(tensor):
        tensor = tensor.float()
    return tensor


def _preprocess_tensor(
    tensor: torch.Tensor,
    config: TrainingSettings,
    device_obj: torch.device,
    model: torch.nn.Module,
) -> torch.Tensor:
    """Preprocess using config-aligned settings."""
    tensor = _reshape_to_batch(tensor)

    tensor = _normalize_like_training(tensor, config)

    model_channels = _get_model_channels(model)
    tensor = _adapt_channels(tensor, model_channels, config)

    tensor = tensor.to(device_obj)
    return tensor


def _reshape_to_batch(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape input to (Batch, Channels, Height, Width) layout."""
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        dims = tensor.shape
        if dims[0] <= 3:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.unsqueeze(1)
    elif tensor.ndim == 4:
        if tensor.shape[-1] <= 3:
            tensor = tensor.permute(2, 3, 0, 1)

    return tensor


def _normalize_like_training(tensor: torch.Tensor, config: TrainingSettings) -> torch.Tensor:
    """Apply exactly the image normalization the TRAINING chain applies.

    Asks the SAME resolver the transform builder asks
    (:func:`~spectramr.data.transforms.normalization.resolve_image_normalization`)
    and then defers to the same SSOT dispatcher
    (:func:`~spectramr.data.transforms.normalization.normalize_tensor`) that
    ``ImageNormalizationTransform`` calls. Inference cannot reuse the transform
    object itself -- that one takes a ``tio.Subject`` and this path holds a bare
    tensor -- but it can and must reuse the *resolver* and the *dispatcher*, so
    there is no second place where image intensity is decided. This function
    used to mirror the builder's step 5 "decision-for-decision" with a copy of
    its own (the k-space gate, the ``robust_percentile`` fold, the spec call);
    a mirror is a second owner, and the resolver replaced both copies.

    What the mirror itself replaced was a local reimplementation, and it
    disagreed with training on the two things that matter:

    * **Whether to normalize at all.** The builder gates the whole step on
      ``if not normalize_kspace`` -- an arm that normalizes k-space gets *no*
      image normalization, because image normalizers clamp and shift values and
      that destroys complex k-space (negative values, phase). The old code had
      no such gate. Across ``experiments/inprogress/kspace_filling`` this is not
      a corner case: 58/58 arms set ``enable_kspace_normalization: true`` and 46
      of them declare ``normalization_type: robust_percentile``, so training
      skipped image normalization on every one of them while predict applied a
      window to 46 -- and every metric or figure compared across the two verbs
      was comparing differently-scaled tensors.
    * **Which arithmetic.** The old code clamped to ``[p_low, p_high]`` and
      affinely rescaled into a hardcoded ``[0, 1]``. ``normalize_percentile``
      divides by a magnitude quantile and clamps only when the resolved config
      asks for it -- phase-preserving by construction. It also took its upper
      bound from ``data.processing.rescale_percentiles``, a field that **no
      training code reads**: the builder still forwards it (``:876``) into a
      dataclass slot left behind when ``RescaleIntensity`` was replaced by
      ``ImageNormalizationSpec``, and nothing consumes it from there. Predict
      was the sole live reader of a knob with no effect on training, and its
      lower bound was not read even there -- ``p_low`` was hardcoded to 0.5 %
      regardless of what the pair declared.

    Args:
        tensor: Batched input tensor, already in ``(B, C, H, W)`` layout.
        config: The SSOT settings object for the arm.

    Returns:
        The tensor, normalized exactly as training would normalize it.

    Raises:
        ConfigurationError: On an unrecognised ``normalization_type``, via the
            resolver -- never a silent degrade to "none" (non-negotiable 3). The
            resolver raises a bare ``ValueError``, which is *not* a
            ``SpectraMRError``; it is retyped here so :func:`_process_all_files`
            classifies it by its real blast radius (run-invariant, so abort on
            the first file) instead of treating it as a per-file fault and
            repeating it once per input.
    """
    from spectramr.data.transforms.normalization import (
        ImageNormalizationTransform,
        normalize_tensor,
        resolve_image_normalization,
    )

    processing = config.data.processing
    try:
        spec = resolve_image_normalization(
            normalization_type=processing.normalization_type,
            dataset_type=config.data.dataset_type,
            normalization_kwargs=processing.normalization_kwargs,
            kspace_normalization_enabled=processing.enable_kspace_normalization,
        )
    except ValueError as e:
        # Retyped, not re-worded -- the message is the resolver's, verbatim.
        # A bare ValueError is not a SpectraMRError, so the per-file handler in
        # `_process_all_files` would swallow this one and repeat it once per
        # input file. An unrecognised normalization_type is a property of the
        # config, identical on every file, so it belongs to the class that
        # aborts on the first one.
        raise ConfigurationError(str(e)) from e
    if spec is None or not spec.enabled:
        # None: k-space normalization took precedence (the resolver logged it).
        return tensor

    # Kept in step with the transform: PERCENTILE divides by a magnitude
    # quantile and is phase-safe, while ZSCORE/MINMAX would take a complex
    # mean/min/max. Read off the transform rather than restated here, so the
    # two cannot drift.
    if tensor.is_complex() and spec.config.strategy in (
        ImageNormalizationTransform._MAGNITUDE_ONLY
    ):
        tensor = tensor.abs()

    return normalize_tensor(tensor, spec.config)


def _get_model_channels(model: torch.nn.Module) -> int:
    """Extract input channel count from model."""
    # This loop was the most complete unwrap in the tree (it was one of only two
    # sites that knew about ``_orig_mod``); it has been promoted to
    # ``spectramr.core.module_utils.unwrap_model``, which additionally handles FSDP
    # and activation-checkpointing wrappers.
    unwrapped = unwrap_model(model)

    if hasattr(unwrapped, "in_channels"):
        return unwrapped.in_channels
    elif hasattr(unwrapped, "config") and hasattr(unwrapped.config, "in_channels"):
        return unwrapped.config.in_channels

    return 1


def _adapt_channels(
    tensor: torch.Tensor,
    target_channels: int,
    config: TrainingSettings,
) -> torch.Tensor:
    """Verify the loaded tensor already has the channel layout the model expects.

    This used to *adapt*, via three branches that all reported success at INFO
    while changing the data (pitfall #16). Each one was wrong in its own way:

    * **Zero-padding up.** Fabricated channels the acquisition never contained
      and fed them to the model as though they were measured. A generator whose
      first layer weights those channels non-trivially then produced output
      derived from invented data, with nothing downstream marking it.
    * **``narrow()`` down.** Silently discarded every channel past the first
      ``target_channels`` -- for multi-echo, multi-contrast or multi-coil input
      that is throwing away most of the acquisition and reporting a clean run.
    * **RSS coil combination.** The one branch that looked like physics, and the
      most misleading of the three. It reinterpreted the channel axis as
      *coil-major interleaved* (``view(B, n_coils, 2, H, W)``), but this repo
      packs complex data as **stacked halves** -- ``torch.stack([real, imag])``
      in ``m4raw_dataset.py:402``, ``preprocessed_dataset.py:803``,
      ``io_strategies.py:257`` and ``physics_sync.py:164`` -- so on the m4raw
      path it paired the real part of one coil with the imaginary part of
      another before combining. It also duplicated a *declared* capability:
      ``data.coils.processing_mode='rss'`` is specified as "returns 2-ch
      real/imag of the RSS-combined k-space (applies IFFT-RSS-FFT for k-space
      inputs)", and the data pipeline has already applied it by the time the
      tensor arrives here. On an arm that declared it, this branch was a second
      application -- the same double-normalization shape as #571 and #760.

    So there is no adaptation left to do that would be honest. A channel-count
    mismatch here means the checkpoint, the config and the input file disagree,
    and the fix belongs in whichever of the three is wrong -- not in a reshape
    at the last moment before the forward pass.

    Args:
        tensor: Batched input tensor in ``(B, C, H, W)`` layout.
        target_channels: Channel count the constructed model accepts.
        config: The SSOT settings object, used to name the declared knobs that
            determine the expected count.

    Returns:
        ``tensor`` unchanged, when the counts already agree.

    Raises:
        ValueError: When they do not (non-negotiable 3 -- no silent fallbacks).
    """
    current_channels = tensor.shape[1]
    if current_channels == target_channels:
        return tensor

    coils = config.data.coils
    raise DimensionMismatchError(
        f"Input has {current_channels} channel(s) but the model built from this "
        f"config accepts {target_channels}. Inference will not reshape the "
        f"acquisition to fit: padding up fabricates channels that were never "
        f"measured, narrowing down discards them, and combining coils here "
        f"would re-apply a step the data pipeline already owns.\n"
        f"  declared model.in_channels      : {target_channels}\n"
        f"  declared data.coils.processing_mode: {coils.processing_mode!r}\n"
        f"  declared data.coils.num_virtual_coils: {coils.num_virtual_coils}\n"
        f"Resolve it at the source: coil handling belongs to "
        f"data.coils.processing_mode (use 'rss' for 2-channel real/imag of the "
        f"RSS-combined k-space, or 'svd' with num_virtual_coils for "
        f"in_channels = 2 * num_virtual_coils), and the checkpoint must come "
        f"from a run of THIS architecture."
    )


def _save_output(
    output: torch.Tensor,
    input_file: Path,
    output_path: Path,
    config: TrainingSettings,
    device_obj: torch.device,
) -> None:
    """Save output via the config-driven OutputWriter (Phase 4d / Phase 7).

    The writer:
    - splits multi-channel outputs by per-channel name/scale when
      ``data.modes.infer.output.multi_channel.enabled=true``
    - restores cine frame order when ``...temporal.enabled=true``
    - dispatches to nifti / h5 / npy based on the YAML format

    When no ``modes.infer.output`` block is configured, the writer
    defaults to legacy ``.npy`` writes preserving pre-Phase-4d behavior.
    """
    from spectramr.data.writers.output_writer import OutputWriter

    # k-space → magnitude conversion (legacy behavior, unchanged).
    if output.shape[1] % 2 == 0 and output.ndim == 4 and not output.is_complex():
        from spectramr.infrastructure.physics.fft_ops import ifft2c

        logger.info("Converting k-space output to magnitude image")
        B, C, H, W = output.shape
        o_reshaped = output.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
        complex_output = torch.view_as_complex(o_reshaped).permute(0, 3, 1, 2)
        image_output = ifft2c(complex_output)
        output = torch.sqrt(torch.sum(image_output.abs() ** 2, dim=1, keepdim=True))
        m_min, m_max = output.min(), output.max()
        output = (output - m_min) / (m_max - m_min + 1e-8)

    # Pull the writer config off the resolved infer-mode block. The
    # mode dispatcher (``data_config.resolve_mode("infer")``) returns
    # the populated ModeConfigSchema; its ``.output`` attr is either
    # a ModeOutputSchema or None (legacy/unset YAMLs).
    data_config = config.data if hasattr(config, "data") else config
    try:
        infer_mode = data_config.resolve_mode("infer")
    except Exception:
        infer_mode = None
    writer_cfg = getattr(infer_mode, "output", None) if infer_mode is not None else None

    if writer_cfg is None:
        # Legacy fallback: minimal default writer (.npy + legacy filename).
        from spectramr.config.schemas.data import ModeOutputSchema

        writer_cfg = ModeOutputSchema(
            format="npy",
            filename_template="{file_id}_output",
        )

    writer = OutputWriter(config=writer_cfg, output_dir=output_path)
    # Strip the batch dim so the writer sees (C, ...) — its multi_channel
    # split expects channel-first. Strategies always emit B=1 at inference.
    tensor = output[0] if output.ndim >= 4 and output.shape[0] == 1 else output
    paths = writer.write(
        tensor=tensor,
        subject_id=input_file.stem,
        file_id=input_file.stem,
    )
    for p in paths:
        logger.info(f"Saved output: {p}")
