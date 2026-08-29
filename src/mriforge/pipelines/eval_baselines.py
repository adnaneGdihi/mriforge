"""Evaluation runner for the MRIxFields2026 pretrained baselines (Task 6).

Orchestrates the faithful 3-stage reproduction of the original challenge
pipeline (inference -> [SynthSeg] -> metrics), driven by our validation manifest
for pairing and scored exclusively through the framework metric registry (the
metric SSOT).  The runner is the only place that stitches together

    discovery -> eval-task expansion -> generator load -> paired-volume iteration
    -> per-slice inference + brain mask -> registry metrics -> aggregate.

Layer note: ``pipelines/`` may import ``infrastructure/``, ``data/`` and
``core/`` (deps point inward).  All NIfTI / checkpoint I/O lives in the consumed
data-layer loader and the generator loader; this module only orchestrates.

Anti-facade obligations honoured here:
    - #15 (every knob wired): ``segmenter`` / ``task3_pairs`` are validated and
      raise on an unknown value; ``seed``, ``metrics``, ``segmenter``,
      ``task3_pairs``, the checkpoint path and generator dims are all stamped
      into each result's ``provenance``.
    - #18 (metric <-> claim): a metric is only ever reported if the run computed
      it.  Segmentation metrics (``synthseg_dice`` / ``volume_consistency``) are
      OMITTED (never faked) when no segmenter is injected.
    - #16 (no inert mechanism): even ``dry_run`` genuinely resolves the generator
      and runs one synthetic forward through the load-bearing ``[0,1] <-> [-1,1]``
      round-trip.

Metric layout (load-bearing — resolves the plan's 5-D ambiguity):
    Voxel metrics are computed on a **batch of 2-D axial slices** shaped
    ``[D, 1, H, W]`` (``volume_to_slice_batch``).  This makes ``ssim`` /
    ``lpips_alex`` aggregate as a per-axial-slice mean (matching the official
    ``Evaluation/evaluate.py``) and ``nrmse_l2`` a single global L2 ratio over
    all voxels (matching the official full-volume nRMSE).

    Segmentation metrics (``synthseg_dice`` / ``volume_consistency``) instead run
    on the **full 3-D volume** shaped ``[1, 1, D, H, W]`` (``volume_to_3d_tensor``)
    so the segmenter produces one 3-D label map and Dice-14 / volume-consistency are
    computed once over it — the official volume-level metric, NOT a per-slice mean
    (metric<->claim, #18).
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from mriforge.core.metrics.computer import create_validation_metrics_computer
from mriforge.core.metrics.quantitative.segmentation import get_segmenter
from mriforge.data.builders.mrixfields_baseline_eval import (
    iter_paired_volumes,
    load_manifest_records,
)
from mriforge.infrastructure.evaluation.mrixfields_baselines.discovery import (
    build_eval_tasks,
    discover_baselines,
)
from mriforge.infrastructure.evaluation.mrixfields_baselines.generator_loader import (
    load_baseline_generator,
)
from mriforge.infrastructure.evaluation.mrixfields_baselines.volume_inference import (
    apply_brain_mask,
    predict_volume,
)

logger = logging.getLogger(__name__)

#: Metrics that require a segmentation of pred + target.  They are only computed
#: when a segmenter is injected; otherwise they are omitted (never faked, #18).
_SEG_METRICS: frozenset[str] = frozenset(
    {"synthseg_dice", "synthseg_dice_risk", "volume_consistency"}
)
#: Accepted values for the ``segmenter`` knob (``None`` = omit seg metrics).
_VALID_SEGMENTERS: frozenset[str] = frozenset({"synthseg", "label_dice"})
#: Accepted values for the ``task3_pairs`` knob.
_VALID_TASK3_PAIRS: frozenset[str] = frozenset({"all", "task1_task2", "to7t"})
#: Accepted values for the ``pairing`` knob (how source/target volumes are matched).
_VALID_PAIRING: frozenset[str] = frozenset({"ordinal", "subject_id"})
#: Stamped into provenance so downstream can tell how a number was produced.
_METRIC_IMPL: str = "framework_registry"


@dataclass
class BaselineResult:
    """Aggregated metrics for one ``EvalTask`` (one baseline specialization)."""

    name: str
    task: int
    method: str
    source_field: float
    target_field: float
    contrast: str
    metrics_mean: dict[str, float]
    metrics_std: dict[str, float]
    n_subjects: int
    provenance: dict = field(default_factory=dict)


def volume_to_slice_batch(
    vol: np.ndarray,
    *,
    slice_axis: int = 2,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build a ``[D, 1, H, W]`` batch of axial slices from a ``[H, W, D]`` volume.

    Moving the slice axis to the batch dimension is the load-bearing layout
    choice: it makes per-slice metrics (``ssim`` / ``lpips_alex``) aggregate as a
    mean over axial slices, while a whole-tensor metric (``nrmse_l2``) becomes a
    single global ratio over all voxels.  Both match ``Evaluation/evaluate.py``.

    Args:
        vol: Volume of shape ``[H, W, D]`` (or any 3-D layout whose ``slice_axis``
            indexes the axial dimension).
        slice_axis: Axis to move to the batch dimension (default ``2``).
        device: Device for the returned tensor.

    Returns:
        ``torch.Tensor`` of shape ``[D, 1, H, W]`` (``float32``).
    """
    arr = np.moveaxis(np.asarray(vol, dtype=np.float32), slice_axis, 0)  # [D, H, W]
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(1)  # [D, 1, H, W]
    return tensor.to(device)


def volume_to_3d_tensor(
    vol: np.ndarray,
    *,
    slice_axis: int = 2,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build a single ``[1, 1, D, H, W]`` 3-D tensor from a ``[H, W, D]`` volume.

    Segmentation metrics (``synthseg_dice`` / ``volume_consistency``) must score the
    **whole 3-D label map at once** — the official challenge Dice-14 / volume
    consistency are volume-level, not per-slice.  Feeding the ``[D, 1, H, W]`` axial
    batch would instead segment each slice independently (batch dim = D) and average
    a per-slice Dice, which is NOT the challenge metric (metric<->claim, #18).  This
    helper keeps the whole volume in one ``batch=1`` sample so the segmenter produces
    one 3-D label map and the metric is computed once over it.

    Args:
        vol: Volume of shape ``[H, W, D]`` (``slice_axis`` indexes the axial dim).
        slice_axis: Axis moved to the depth position (default ``2``).
        device: Device for the returned tensor.

    Returns:
        ``torch.Tensor`` of shape ``[1, 1, D, H, W]`` (``float32``).
    """
    arr = np.moveaxis(np.asarray(vol, dtype=np.float32), slice_axis, 0)  # [D, H, W]
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)  # [1, 1, D, H, W]


class _SynthSegDeviceAdapter:
    """Transparent device-transfer wrapper for TorchScript SynthSeg models.

    When SynthSeg runs on a different device than the main inference device
    (e.g., CPU to avoid CUDA OOM at 192³), this adapter moves the input to
    the model's device and the output back to the input's original device.
    ``eval()`` and ``to()`` delegate to the wrapped model.
    """

    def __init__(self, model: object, model_device: torch.device) -> None:
        self._model = model
        self._model_device = model_device

    def eval(self) -> _SynthSegDeviceAdapter:
        if hasattr(self._model, "eval"):
            self._model.eval()  # type: ignore[operator]
        return self

    def to(self, device: object) -> _SynthSegDeviceAdapter:
        return self

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        original_device = image.device
        out: torch.Tensor = self._model(image.to(self._model_device))  # type: ignore[operator]
        return out.to(original_device)


class _KerasSynthSegShim:
    """Numpy-bridge shim that wraps a Keras SynthSeg model as a torch callable.

    SynthSeg ships as a Keras/HDF5 model that operates on NumPy arrays in
    channel-last order.  This shim converts between the framework's PyTorch
    channel-first convention and Keras channel-last so ``SynthSegBackend.segment``
    can call ``self._model(image)`` unchanged.

    Dimension mapping (3-D case, the normal eval path):
        framework in:  ``[B, 1, D, H, W]`` torch float32
        Keras in:      ``[B, H, W, D, 1]`` numpy float32
        Keras out:     ``[B, H, W, D, n_classes]`` numpy float32 (softmax probs)
        framework out: ``[B, n_classes, D, H, W]`` torch float32

    2-D inputs (``[B, 1, H, W]``) are also handled for completeness.

    ``eval()`` and ``to()`` are no-ops so the ``hasattr`` guards in
    ``_resolve_synthseg_model`` do not crash.
    """

    def __init__(self, keras_model: object) -> None:
        self._keras_model = keras_model

    def eval(self) -> _KerasSynthSegShim:
        return self

    def to(self, device: object) -> _KerasSynthSegShim:
        return self

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] != 1:
            raise ValueError(
                f"SynthSeg expects single-channel input; got shape {tuple(image.shape)}"
            )
        # [B, 1, *spatial] -> [B, *spatial]
        arr = image[:, 0].detach().cpu().float().numpy()

        if arr.ndim == 4:
            # 3-D: [B, D, H, W] -> [B, H, W, D, 1]
            arr = arr.transpose(0, 2, 3, 1)[..., np.newaxis]
        elif arr.ndim == 3:
            # 2-D: [B, H, W] -> [B, H, W, 1]
            arr = arr[..., np.newaxis]
        else:
            raise ValueError(f"Unexpected number of spatial dims: {arr.ndim - 1}")

        probs = self._keras_model(arr)  # type: ignore[operator]
        if hasattr(probs, "numpy"):
            probs = probs.numpy()

        if probs.ndim == 5:
            # 3-D: [B, H, W, D, n_classes] -> [B, n_classes, D, H, W]
            probs_t = torch.from_numpy(np.ascontiguousarray(probs.transpose(0, 4, 3, 1, 2)))
        elif probs.ndim == 4:
            # 2-D: [B, H, W, n_classes] -> [B, n_classes, H, W]
            probs_t = torch.from_numpy(np.ascontiguousarray(probs.transpose(0, 3, 1, 2)))
        else:
            raise ValueError(f"Unexpected SynthSeg output ndim: {probs.ndim}")

        return probs_t.to(image.device)


def _resolve_synthseg_model(
    synthseg_model: str | object | None,
    device: torch.device,
    synthseg_device: torch.device | None = None,
) -> object:
    """Resolve the SynthSeg segmentation model for the ``synthseg`` backend.

    ``SynthSegBackend`` wraps an already-loaded callable that maps an image tensor
    to per-voxel class logits.  This resolver accepts:

    - A ``.h5`` / ``.hdf5`` / ``.keras`` path: loaded via ``keras.models.load_model``
      and wrapped in :class:`_KerasSynthSegShim` (numpy-bridge, channel-last↔first).
      Raises ``RuntimeError`` with install instructions when ``keras`` is absent.
    - A ``.pt`` / ``.pth`` / other path: first tried as a TorchScript archive via
      ``torch.jit.load``; falls back to ``torch.load`` for serialised state-dicts.
    - An already-loaded callable: passed through unchanged (test injection).

    When ``synthseg_device`` differs from ``device`` (e.g., CPU while inference runs
    on CUDA), the model is loaded to ``synthseg_device`` and wrapped in a
    :class:`_SynthSegDeviceAdapter` that handles the input→synthseg_device→output
    round-trip transparently.  This prevents CUDA OOM at 192³ (a 5-level 3-D U-Net
    forward pass allocates ~5 GiB of skip-connection activations).

    ``None`` fails loud (#15).

    Args:
        synthseg_model: Path to a loadable model, or an already-loaded callable.
        device: Main inference device (used when ``synthseg_device`` is ``None``).
        synthseg_device: Override device for SynthSeg.  Defaults to ``device``
            when ``None``.

    Returns:
        The resolved (loaded, ``eval``-mode) segmentation model or shim.

    Raises:
        ValueError: When ``synthseg_model`` is ``None``.
        RuntimeError: When a Keras path is given but ``keras`` is not installed.
    """
    if synthseg_model is None:
        raise ValueError(
            "segmenter='synthseg' requires a segmentation model, but none was "
            "provided. Pass --synthseg-model <path> (a Keras .h5 model or a "
            "TorchScript .pt model mapping an image tensor to per-voxel class "
            "logits); the label_dice proxy is only a local smoke stand-in."
        )
    seg_dev = synthseg_device if synthseg_device is not None else device
    if isinstance(synthseg_model, (str, Path)):
        path = Path(synthseg_model)
        if path.suffix.lower() in {".h5", ".hdf5", ".keras"}:
            try:
                import keras  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError(
                    f"Loading a Keras/HDF5 SynthSeg model ({path.name}) requires "
                    "'keras' and 'tensorflow' to be installed "
                    "(pip install keras tensorflow).  The framework venv does not "
                    "include them — install into the venv or convert the model to "
                    "TorchScript first."
                ) from exc
            keras_model = keras.models.load_model(str(path), compile=False)
            return _KerasSynthSegShim(keras_model)
        # TorchScript archives (.pt produced by convert_synthseg_h5_to_pt.py) must
        # be loaded with torch.jit.load; falling back to torch.load silences a
        # PyTorch UserWarning and avoids the zip-dispatch overhead.
        try:
            model: object = torch.jit.load(str(path), map_location=seg_dev)
        except RuntimeError:
            model = torch.load(str(path), map_location=seg_dev, weights_only=False)
    else:
        model = synthseg_model  # already-loaded callable (test injection)
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "to"):
        model = model.to(seg_dev)
    # Wrap in a device adapter when SynthSeg runs on a different device than
    # the main pipeline (prevents CUDA OOM when seg_dev="cpu", device="cuda").
    if seg_dev != device:
        model = _SynthSegDeviceAdapter(model, seg_dev)
    return model


def _chunked_voxel_compute(
    computer: object,
    pred: torch.Tensor,
    target: torch.Tensor,
    chunk_size: int,
    **kwargs: object,
) -> dict[str, float]:
    """Compute voxel metrics over chunks of the slice batch and return the mean.

    Prevents CUDA OOM when perceptual metrics (LPIPS / AlexNet) receive a full
    [D, 1, H, W] batch where D and H, W are both large (e.g. D=H=W=192 would
    push ~1.5 GiB of AlexNet activations onto an already-loaded GPU).

    For per-slice-mean metrics (SSIM, LPIPS) the result is identical to the
    full-batch computation.  For global-ratio NRMSE it is an approximation
    (mean of per-chunk ratios ≈ global ratio when signal is uniform across
    slices — negligible error for typical brain MRI).

    Args:
        computer: Metric computer with a ``compute(pred, target, **kwargs)``
            method returning ``dict[str, float]``.
        pred: Slice batch ``[N, C, H, W]``.
        target: Slice batch ``[N, C, H, W]``.
        chunk_size: Slices per chunk.  ``<= 0`` disables chunking.
        **kwargs: Forwarded to ``computer.compute`` (e.g. ``segmenter=``).

    Returns:
        Dict of metric name → mean across chunks.
    """
    if chunk_size <= 0 or pred.shape[0] <= chunk_size:
        return dict(computer.compute(pred, target, **kwargs))  # type: ignore[union-attr]
    chunks_pred = pred.split(chunk_size, dim=0)
    chunks_target = target.split(chunk_size, dim=0)
    per_chunk: list[dict[str, float]] = [
        dict(computer.compute(cp, ct, **kwargs))  # type: ignore[union-attr]
        for cp, ct in zip(chunks_pred, chunks_target)
    ]
    keys = per_chunk[0].keys()
    return {k: float(np.mean([row[k] for row in per_chunk if k in row])) for k in keys}


def _aggregate(
    per_subject: list[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Mean / std of each metric across subjects (finite values only).

    A metric whose values are all non-finite aggregates to ``NaN`` (not a
    finite-but-fake ``0.0``) — consistent with the "NaN means not computed"
    contract enforced by the metrics computer.
    """
    if not per_subject:
        return {}, {}
    keys = list(per_subject[0].keys())
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in keys:
        vals = np.asarray([d[key] for d in per_subject if key in d], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            mean[key] = float("nan")
            std[key] = float("nan")
        else:
            mean[key] = float(np.mean(finite))
            std[key] = float(np.std(finite))
    return mean, std


def _write_outputs(out_dir: Path, results: list[BaselineResult]) -> None:
    """Write ``baseline_results.json`` (+ ``.csv``), one CSV row per EvalTask."""
    (out_dir / "baseline_results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, default=str)
    )

    metric_names = sorted({k for r in results for k in r.metrics_mean})
    fieldnames = [
        "name",
        "task",
        "method",
        "source_field",
        "target_field",
        "contrast",
        "n_subjects",
    ]
    for m in metric_names:
        fieldnames += [f"{m}_mean", f"{m}_std"]

    with open(out_dir / "baseline_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row: dict[str, object] = {
                "name": r.name,
                "task": r.task,
                "method": r.method,
                "source_field": r.source_field,
                "target_field": r.target_field,
                "contrast": r.contrast,
                "n_subjects": r.n_subjects,
            }
            for m in metric_names:
                row[f"{m}_mean"] = r.metrics_mean.get(m, "")
                row[f"{m}_std"] = r.metrics_std.get(m, "")
            writer.writerow(row)


def run_baseline_evaluation(
    baselines_root: str | Path,
    manifest_path: str | Path,
    data_root: str | Path,
    out_dir: str | Path,
    *,
    metrics: tuple[str, ...] = ("nrmse_l2", "ssim", "lpips_alex"),
    segmenter: str | None = None,
    synthseg_model: str | object | None = None,
    synthseg_device: str = "cpu",
    seed: int = 0,
    device: str = "cuda",
    contrasts: tuple[str, ...] = ("T1w", "T2w", "T2FLAIR"),
    fields: tuple[float, ...] = (0.1, 1.5, 3.0, 5.0, 7.0),
    task3_pairs: str = "all",
    pairing: str = "ordinal",
    metric_slice_chunk: int = 8,
    dry_run: bool = False,
    resnet_kwargs: dict | None = None,
    stargan_kwargs: dict | None = None,
) -> list[BaselineResult]:
    """Evaluate the pretrained MRIxFields2026 baselines and write a results table.

    Args:
        baselines_root: Root of the discovered ``baselines/`` checkpoint tree.
        manifest_path: Validation manifest JSON (``{"records": [...]}``).  Unused
            in ``dry_run`` (no real volumes are read).
        data_root: Root under which manifest ``relative_path`` values resolve
            (the ``ChallengeData/`` tree).
        out_dir: Directory to write ``baseline_results.{json,csv}`` into.
        metrics: Registry metric names to score.  Seg metrics
            (``synthseg_dice`` / ``volume_consistency``) are only computed when a
            ``segmenter`` is injected; otherwise omitted (never faked).
        segmenter: ``None`` | ``"synthseg"`` | ``"label_dice"``.  Injected into
            the seg-metric computations.  ``"label_dice"`` is a **local smoke proxy**
            for ``synthseg_dice`` only — it emits intensity-quantile bins, not the 14
            FreeSurfer DGM labels, so requesting ``volume_consistency`` with it raises
            (use ``"synthseg"`` for real Dice-14 / volume-consistency).
        synthseg_model: Required when ``segmenter == "synthseg"``.  A path to a
            torch-loadable / TorchScript segmentation model, or an already-loaded
            callable (test injection).  ``None`` with ``segmenter == "synthseg"``
            raises (#15 — the advertised backend must be wired to a real model).
        synthseg_device: Device for SynthSeg inference (default ``"cpu"``).  A
            5-level 3-D U-Net at 192³ allocates ~5 GiB of skip activations, which
            OOMs on a shared GPU.  Run on CPU to keep the GPU budget for the
            baseline generators; inputs are moved to this device and outputs moved
            back automatically via :class:`_SynthSegDeviceAdapter`.
        seed: RNG seed for the StarGAN style latent (stamped into provenance).
        device: ``"cuda"`` or ``"cpu"``.  Falls back to CPU if CUDA is requested
            but unavailable.
        contrasts: Contrast keys for task3 eval-task expansion.
        fields: Field strengths for task3 eval-task expansion.
        task3_pairs: ``"all"`` | ``"task1_task2"`` | ``"to7t"`` — the task3
            cross-field pairing protocol.
        pairing: ``"ordinal"`` (default) | ``"subject_id"`` — how source/target
            volumes are matched.  ``"ordinal"`` pairs by rank-within-field, which is
            required for the travelling-volunteer val manifest (its ``subject_id``
            numbering is per-field, so id-matching yields zero pairs).  See
            :func:`~mriforge.data.builders.mrixfields_baseline_eval.iter_paired_volumes`.
        metric_slice_chunk: Slice batch is split into chunks of this size before
            being passed to the voxel-metric computer (default 8).  Prevents CUDA
            OOM when perceptual metrics such as LPIPS run AlexNet on all D slices
            at once (D=192 → ~1.5 GiB extra VRAM on top of the generator).  For
            per-slice-mean metrics (SSIM, LPIPS) chunking is exact; for global-ratio
            NRMSE it is a minor approximation (negligible when signal is roughly
            uniform across slices).  Set to 0 to disable chunking.
        dry_run: If ``True``, skip real-volume metrics: discover, expand tasks,
            load each generator and run one synthetic forward (wiring proof), and
            emit stub results with ``provenance["dry_run"] = True``.
        resnet_kwargs: Test-only override for ResNet generator dims.
        stargan_kwargs: Test-only override for StarGAN generator dims.

    Returns:
        One ``BaselineResult`` per ``EvalTask``.

    Raises:
        ValueError: On an unknown ``segmenter`` / ``task3_pairs`` / ``pairing``
            value; when ``segmenter == "synthseg"`` without a ``synthseg_model``;
            when ``segmenter == "label_dice"`` is paired with ``volume_consistency``;
            when discovery finds zero checkpoints; or when a non-``dry_run`` run
            scores **zero** subjects across every task (likely a pairing / data_root /
            manifest mismatch).
    """
    # -- validate the exposed knobs (fail-loud, pitfall #15) ------------------
    if segmenter is not None and segmenter not in _VALID_SEGMENTERS:
        raise ValueError(
            f"unknown segmenter {segmenter!r}; expected None or one of {sorted(_VALID_SEGMENTERS)}"
        )
    if task3_pairs not in _VALID_TASK3_PAIRS:
        raise ValueError(
            f"unknown task3_pairs {task3_pairs!r}; expected one of {sorted(_VALID_TASK3_PAIRS)}"
        )
    if pairing not in _VALID_PAIRING:
        raise ValueError(f"unknown pairing {pairing!r}; expected one of {sorted(_VALID_PAIRING)}")
    # I3: the label_dice proxy emits intensity-quantile bins, never the 14 FreeSurfer
    # DGM labels — so volume_consistency would score a constant ~1.0 regardless of the
    # prediction (a #20 measurement-independent facade). Fail loud rather than report it.
    if segmenter == "label_dice" and "volume_consistency" in metrics:
        raise ValueError(
            "the label_dice proxy cannot express the 14 FreeSurfer DGM labels that "
            "volume_consistency measures (it emits only intensity-quantile bins), so "
            "the score would be measurement-independent; use --segmenter synthseg for "
            "real volume_consistency. label_dice is a smoke proxy for synthseg_dice only."
        )

    metrics = tuple(metrics)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        dev = torch.device("cpu")

    # -- split voxel vs segmentation metrics; build the segmenter if requested -
    voxel_metrics = [m for m in metrics if m not in _SEG_METRICS]
    requested_seg = [m for m in metrics if m in _SEG_METRICS]
    seg_obj = None
    active_seg: list[str] = []
    if segmenter == "synthseg":
        # I1: wire the injected SynthSeg model (fail loud if absent, #15).
        seg_dev = torch.device(synthseg_device)
        model = _resolve_synthseg_model(synthseg_model, dev, synthseg_device=seg_dev)
        seg_obj = get_segmenter("synthseg", model=model)
        active_seg = requested_seg
    elif segmenter == "label_dice":
        seg_obj = get_segmenter("label_dice")
        active_seg = requested_seg
    elif requested_seg:
        logger.warning(
            "seg metrics %s requested but no segmenter injected — omitting (never faked, #18)",
            requested_seg,
        )
    active_metrics = voxel_metrics + active_seg

    # -- discovery + eval-task expansion --------------------------------------
    specs = discover_baselines(Path(baselines_root))
    tasks = build_eval_tasks(specs, contrasts=contrasts, fields=fields, task3_pairs=task3_pairs)
    logger.info(
        "discovered %d baseline spec(s) -> %d eval task(s) [dry_run=%s]",
        len(specs),
        len(tasks),
        dry_run,
    )

    base_provenance: dict[str, object] = {
        "seed": int(seed),
        "metric_impl": _METRIC_IMPL,
        "metrics": list(metrics),
        "active_metrics": list(active_metrics),
        "segmenter": segmenter,
        "synthseg_model": (
            str(synthseg_model)
            if isinstance(synthseg_model, (str, Path))
            else ("<injected>" if synthseg_model is not None else None)
        ),
        "task3_pairs": task3_pairs,
        "pairing": pairing,
        "device": str(dev),
        "synthseg_device": synthseg_device if segmenter == "synthseg" else None,
        "metric_slice_chunk": metric_slice_chunk,
    }

    records = None if dry_run else load_manifest_records(manifest_path)

    # Metric computers are created ONCE before the task loop so perceptual models
    # (LPIPS/AlexNet) are only loaded once and never compete with generator VRAM.
    # All metric computation runs on CPU regardless of `dev`: generators run on CUDA
    # for fast inference; MetricsComputer.compute() calls `.to(self.device)` so the
    # CUDA→CPU transfer is handled at compute time, not at the call site here.
    voxel_computer = (
        create_validation_metrics_computer(list(voxel_metrics), device="cpu", domain="image")
        if voxel_metrics
        else None
    )
    seg_computer = (
        create_validation_metrics_computer(list(active_seg), device="cpu", domain="image")
        if active_seg
        else None
    )

    # Eagerly warm-up metric instances BEFORE the task loop so that perceptual
    # models (LPIPS/AlexNet, ~480 MB peak during init) are loaded when the
    # process footprint is minimal.  Without this, LPIPS is lazily instantiated
    # on the first compute() call — which happens while the first generator
    # checkpoint already occupies RAM, causing a peak-RAM kill on shared nodes.
    if not dry_run and voxel_computer is not None:
        logger.info("Pre-warming metric instances (forces lazy models to load now) …")
        # 64×64 is the minimum spatial size for LPIPS/AlexNet (11×11 conv with
        # padding=2 needs padded input ≥ 11 → raw spatial ≥ 9; 64 is safe).
        # Using a 4×4 dummy causes AlexNet's first conv to fail (padded 8×8 <
        # 11×11 kernel), so the warm-up would succeed at loading the model but
        # log a spurious WARNING and leave the compute path un-exercised.
        _warm = torch.ones(2, 1, 64, 64)
        try:
            voxel_computer.compute(_warm, _warm)
        except Exception as _exc:  # SSIM/NRMSE may complain about tiny dummy — that's fine
            logger.debug("Metric warm-up non-fatal error (safe to ignore): %s", _exc)

    results: list[BaselineResult] = []

    for task in tasks:
        loaded = load_baseline_generator(
            task.spec.method,
            task.spec.checkpoint,
            dev,
            seed=int(seed),
            target_domain=task.target_domain,
            resnet_kwargs=resnet_kwargs,
            stargan_kwargs=stargan_kwargs,
        )
        prov = {
            **base_provenance,
            "checkpoint": str(task.spec.checkpoint),
            "generator_meta": dict(loaded.meta),
            "target_domain": task.target_domain,
        }

        if dry_run:
            # Prove the generator resolves AND the [0,1]<->[-1,1] round-trip runs
            # on a synthetic slice — no cluster data touched (anti-facade #16).
            probe = np.zeros((8, 8, 2), dtype=np.float32)
            predict_volume(loaded, probe, slice_axis=2, device=dev)
            prov["dry_run"] = True
            results.append(
                BaselineResult(
                    name=task.label,
                    task=task.spec.task,
                    method=task.spec.method,
                    source_field=task.source_field,
                    target_field=task.target_field,
                    contrast=task.contrast,
                    metrics_mean={},
                    metrics_std={},
                    n_subjects=0,
                    provenance=prov,
                )
            )
            logger.info("[dry-run] %s: generator + synthetic forward OK", task.label)
            continue

        # Reached only on the non-dry-run path, where records was loaded (narrow None).
        assert records is not None
        # Voxel metrics score the [D, 1, H, W] axial-slice batch; segmentation
        # metrics score the FULL 3-D volume [1, 1, D, H, W] so Dice-14 / volume
        # consistency are computed once over the whole label map (#18), not
        # mislabelled per-slice.
        # Both computers are created BEFORE the loop (see above) so LPIPS/AlexNet
        # is never re-initialised per task and never loaded to the generator's GPU.
        assert voxel_computer is not None or not voxel_metrics, "voxel_computer not built"

        per_subject: list[dict[str, float]] = []
        for pv in iter_paired_volumes(
            records,
            data_root,
            source_field=task.source_field,
            target_field=task.target_field,
            contrast=task.contrast,
            pairing=pairing,
        ):
            pred = predict_volume(loaded, pv.source, slice_axis=2, device=dev)
            pred = apply_brain_mask(pred, pv.source)
            pred_batch = volume_to_slice_batch(pred, device=dev)
            target_batch = volume_to_slice_batch(pv.target, device=dev)
            row: dict[str, float] = (
                _chunked_voxel_compute(
                    voxel_computer, pred_batch, target_batch, chunk_size=metric_slice_chunk
                )
                if voxel_computer is not None
                else {}
            )
            if seg_computer is not None:
                pred_3d = volume_to_3d_tensor(pred, device=dev)
                target_3d = volume_to_3d_tensor(pv.target, device=dev)
                row.update(
                    _chunked_voxel_compute(
                        seg_computer,
                        pred_3d,
                        target_3d,
                        chunk_size=0,  # seg metrics run on full 3-D volume, never chunked
                        segmenter=seg_obj,
                    )
                )
            per_subject.append(row)

        if not per_subject:
            # No pairs for this task. Warn here; the cross-task zero-subjects guard
            # below fails loud if EVERY task is empty (never a silent all-zero, C2).
            logger.warning(
                "%s: 0 subject(s) scored (no paired volumes for pairing=%r "
                "source=%sT target=%sT contrast=%s)",
                task.label,
                pairing,
                task.source_field,
                task.target_field,
                task.contrast,
            )

        mean, std = _aggregate(per_subject)
        logger.info(
            "%s: %d subject(s) scored -> %s",
            task.label,
            len(per_subject),
            {k: round(v, 4) for k, v in mean.items()},
        )
        results.append(
            BaselineResult(
                name=task.label,
                task=task.spec.task,
                method=task.spec.method,
                source_field=task.source_field,
                target_field=task.target_field,
                contrast=task.contrast,
                metrics_mean=mean,
                metrics_std=std,
                n_subjects=len(per_subject),
                provenance=prov,
            )
        )
        # Free the generator tensors so the next task starts with a clean CUDA
        # budget.  empty_cache() is banned in the training loop (performance) but
        # is appropriate here — evaluation is throughput-limited by I/O, not GPU.
        del loaded
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # C2: a non-dry-run that scored ZERO subjects across every task is almost
    # always a pairing / data_root / manifest mismatch (e.g. subject_id pairing on
    # the travelling-volunteer val manifest, whose ids are per-field and never
    # recur across fields). Fail loud rather than silently emit an all-zero table.
    if not dry_run and sum(r.n_subjects for r in results) == 0:
        raise ValueError(
            "baseline evaluation scored 0 subjects across all "
            f"{len(results)} task(s). Likely causes: pairing={pairing!r} produced no "
            "pairs (the val manifest numbers subject_id per-field, so 'ordinal' is "
            "required — 'subject_id' yields zero); a wrong data_root "
            f"({data_root!r}); or a manifest ({manifest_path!r}) whose contrasts / "
            "field strengths / split do not match the discovered checkpoints."
        )

    _write_outputs(out_dir, results)
    return results
