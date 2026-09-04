"""Unit tests for the MRIxFields2026 baseline evaluation runner (Task 6).

Covers:
- ``dry_run=True``: one ``BaselineResult`` per ``EvalTask`` (count matches
  ``build_eval_tasks``), provenance carries ``dry_run: True``; no real-volume
  metrics computed (empty ``metrics_mean``).  The generator is genuinely
  resolved + one synthetic forward runs (anti-facade #16).
- Full mini-run (CPU, tiny ResNet via ``resnet_kwargs``): ``metrics_mean``
  carries the requested registry metrics; ``n_subjects >= 1``; provenance stamps
  seed + metric list + ``metric_impl="framework_registry"``; seg metrics absent
  without a segmenter, present with one.
- Fail-loud: unknown ``segmenter`` / ``task3_pairs`` → ``ValueError``.
- Metric-layout pin: the computer's ``ssim`` over a ``[D,1,H,W]`` axial-slice
  batch equals the manual mean of the per-slice 2-D ``ssim`` (aggregation axis).
- Determinism: two ``seed=0`` runs give identical ``metrics_mean``.

All tests run on CPU with tiny models — no GPU / no cluster data required.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from spectramr.models.registry import get_model_class

_RESNET_TINY = {"ngf": 8, "n_blocks": 1}
_SPLIT = "Validating_prospective"


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _save_resnet_ckpt(path):
    """Save a tiny ResNet generator in the ``{"model": {"netG.<k>": v}}`` layout."""
    import spectramr.infrastructure.evaluation.mrixfields_baselines.generator_loader  # noqa: F401

    gen = get_model_class("cyclegan_generator")(input_nc=1, output_nc=1, **_RESNET_TINY)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": {f"netG.{k}": v for k, v in gen.state_dict().items()}}, str(path)
    )


def _build_baselines_tree(tmp_path, methods=("cut", "cyclegan")):
    """Build a ``baselines/task1_0.1T_to_7T_T1W/{methods}/...`` tree."""
    root = tmp_path / "baselines"
    for method in methods:
        ckpt = (
            root
            / "task1_0.1T_to_7T_T1W"
            / method
            / "pro_pretrained"
            / "weights"
            / "checkpoint_epoch100.pth"
        )
        _save_resnet_ckpt(ckpt)
    return root


def _write_nifti(path, data):
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4))
    img.header.set_zooms((1.0, 1.0, 1.0))
    nib.save(img, str(path))


def _build_manifest_and_data(tmp_path, *, n_subjects=1, shape=(16, 16, 4)):
    """Write source (0.1T) + target (7T) T1w volumes and a manifest for them."""
    data_root = tmp_path / "ChallengeData"
    records = []
    rng = np.random.default_rng(1234)
    for i in range(n_subjects):
        sid = f"s{i:04d}"
        sfx = f"{i:04d}"
        src_rel = f"{_SPLIT}/T1W/0.1T/P_T1W_0.1T_{sfx}.nii.gz"
        tgt_rel = f"{_SPLIT}/T1W/7T/P_T1W_7T_{sfx}.nii.gz"
        _write_nifti(data_root / src_rel, rng.random(shape))
        _write_nifti(data_root / tgt_rel, rng.random(shape))
        records.append(
            {
                "relative_path": src_rel,
                "field_strength": 0.1,
                "contrast": "T1w",
                "subject_id": sid,
            }
        )
        records.append(
            {
                "relative_path": tgt_rel,
                "field_strength": 7.0,
                "contrast": "T1w",
                "subject_id": sid,
            }
        )
    manifest_path = tmp_path / "mrixfields2026_val.json"
    manifest_path.write_text(json.dumps({"records": records}))
    return manifest_path, data_root


def _build_cross_field_manifest(
    tmp_path, *, src_ids=("0001", "0002"), tgt_ids=("0016", "0017"), shape=(16, 16, 4)
):
    """Manifest with DISTINCT per-field subject ids (as the val manifest has).

    Source records live at 0.1T with ``src_ids``; target records at 7T with
    ``tgt_ids``.  ``subject_id`` never coincides across fields, so ``subject_id``
    pairing yields zero pairs while ``ordinal`` pairs by rank.
    """
    data_root = tmp_path / "ChallengeData"
    records = []
    rng = np.random.default_rng(7)
    for sid in src_ids:
        rel = f"{_SPLIT}/T1W/0.1T/P_T1W_0.1T_{sid}.nii.gz"
        _write_nifti(data_root / rel, rng.random(shape))
        records.append(
            {"relative_path": rel, "field_strength": 0.1, "contrast": "T1w",
             "subject_id": sid}
        )
    for tid in tgt_ids:
        rel = f"{_SPLIT}/T1W/7T/P_T1W_7T_{tid}.nii.gz"
        _write_nifti(data_root / rel, rng.random(shape))
        records.append(
            {"relative_path": rel, "field_strength": 7.0, "contrast": "T1w",
             "subject_id": tid}
        )
    manifest_path = tmp_path / "mrixfields2026_val.json"
    manifest_path.write_text(json.dumps({"records": records}))
    return manifest_path, data_root


class _ShapeRecordingSeg:
    """Stub SynthSeg model that records the shape of every image it segments.

    Returns all-zero per-voxel logits (``[B, 14, *spatial]``); the wiring test only
    cares that the segmenter saw the FULL 3-D volume (``[1,1,D,H,W]``), not the
    ``[D,1,H,W]`` axial-slice batch.
    """

    def __init__(self):
        self.shapes: list[tuple[int, ...]] = []

    def __call__(self, image):
        self.shapes.append(tuple(image.shape))
        b = image.shape[0]
        spatial = image.shape[2:]
        return torch.zeros(b, 14, *spatial)

    def eval(self):
        return self

    def to(self, _device):
        return self


# ---------------------------------------------------------------------------
# dry_run
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dry_run_returns_one_result_per_task(tmp_path):
    from spectramr.infrastructure.evaluation.mrixfields_baselines.discovery import (
        build_eval_tasks,
        discover_baselines,
    )
    from spectramr.pipelines.eval_baselines import BaselineResult, run_baseline_evaluation

    root = _build_baselines_tree(tmp_path, methods=("cut", "cyclegan"))
    manifest_path, data_root = _build_manifest_and_data(tmp_path)

    results = run_baseline_evaluation(
        root,
        manifest_path,
        data_root,
        tmp_path / "out",
        metrics=("nrmse_l2", "ssim"),
        device="cpu",
        dry_run=True,
        resnet_kwargs=_RESNET_TINY,
    )

    n_tasks = len(
        build_eval_tasks(
            discover_baselines(root),
            contrasts=("T1w", "T2w", "T2FLAIR"),
            fields=(0.1, 1.5, 3.0, 5.0, 7.0),
            task3_pairs="all",
        )
    )
    assert n_tasks == 2
    assert len(results) == n_tasks
    assert all(isinstance(r, BaselineResult) for r in results)
    for r in results:
        assert r.provenance["dry_run"] is True
        assert r.metrics_mean == {}  # no real-volume metrics in dry-run
        assert r.provenance["metric_impl"] == "framework_registry"
        assert "checkpoint" in r.provenance
        assert "generator_meta" in r.provenance  # generator genuinely resolved


@pytest.mark.unit
def test_dry_run_writes_results_json(tmp_path):
    from spectramr.pipelines.eval_baselines import run_baseline_evaluation

    root = _build_baselines_tree(tmp_path, methods=("cut",))
    manifest_path, data_root = _build_manifest_and_data(tmp_path)
    out = tmp_path / "out"

    run_baseline_evaluation(
        root,
        manifest_path,
        data_root,
        out,
        metrics=("nrmse_l2", "ssim"),
        device="cpu",
        dry_run=True,
        resnet_kwargs=_RESNET_TINY,
    )
    assert (out / "baseline_results.json").exists()
    assert (out / "baseline_results.csv").exists()


# ---------------------------------------------------------------------------
# full mini-run
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_full_run_computes_registry_metrics_no_segmenter(tmp_path):
    from spectramr.pipelines.eval_baselines import run_baseline_evaluation

    root = _build_baselines_tree(tmp_path, methods=("cut",))
    manifest_path, data_root = _build_manifest_and_data(tmp_path, n_subjects=2)

    results = run_baseline_evaluation(
        root,
        manifest_path,
        data_root,
        tmp_path / "out",
        metrics=("nrmse_l2", "ssim"),
        segmenter=None,
        seed=0,
        device="cpu",
        resnet_kwargs=_RESNET_TINY,
    )

    assert len(results) == 1
    r = results[0]
    assert r.task == 1
    assert r.method == "cut"
    assert r.source_field == 0.1
    assert r.target_field == 7.0
    assert r.contrast == "T1w"
    assert r.n_subjects == 2
    # requested metrics present, seg metrics absent (no segmenter)
    assert set(r.metrics_mean) == {"nrmse_l2", "ssim"}
    assert "synthseg_dice" not in r.metrics_mean
    assert "volume_consistency" not in r.metrics_mean
    assert np.isfinite(r.metrics_mean["nrmse_l2"])
    assert np.isfinite(r.metrics_mean["ssim"])
    assert set(r.metrics_std) == {"nrmse_l2", "ssim"}
    # provenance stamping (#15)
    assert r.provenance["seed"] == 0
    assert r.provenance["metric_impl"] == "framework_registry"
    assert r.provenance["metrics"] == ["nrmse_l2", "ssim"]
    assert r.provenance["segmenter"] is None
    assert r.provenance["task3_pairs"] == "all"
    assert "checkpoint" in r.provenance
    assert "generator_meta" in r.provenance


@pytest.mark.unit
def test_full_run_with_segmenter_includes_synthseg_dice(tmp_path):
    from spectramr.pipelines.eval_baselines import run_baseline_evaluation

    root = _build_baselines_tree(tmp_path, methods=("cut",))
    manifest_path, data_root = _build_manifest_and_data(tmp_path)

    results = run_baseline_evaluation(
        root,
        manifest_path,
        data_root,
        tmp_path / "out",
        metrics=("nrmse_l2", "ssim", "synthseg_dice"),
        segmenter="label_dice",
        seed=0,
        device="cpu",
        resnet_kwargs=_RESNET_TINY,
    )
    r = results[0]
    assert "synthseg_dice" in r.metrics_mean
    assert np.isfinite(r.metrics_mean["synthseg_dice"])
    assert r.provenance["segmenter"] == "label_dice"


@pytest.mark.unit
def test_seg_metric_omitted_when_no_segmenter(tmp_path):
    """A requested seg metric is silently OMITTED (never faked) without a segmenter."""
    from spectramr.pipelines.eval_baselines import run_baseline_evaluation

    root = _build_baselines_tree(tmp_path, methods=("cut",))
    manifest_path, data_root = _build_manifest_and_data(tmp_path)

    results = run_baseline_evaluation(
        root,
        manifest_path,
        data_root,
        tmp_path / "out",
        metrics=("nrmse_l2", "synthseg_dice"),
        segmenter=None,
        seed=0,
        device="cpu",
        resnet_kwargs=_RESNET_TINY,
    )
    r = results[0]
    assert "nrmse_l2" in r.metrics_mean
    assert "synthseg_dice" not in r.metrics_mean  # omitted, not faked


# ---------------------------------------------------------------------------
# fail-loud
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unknown_segmenter_raises(tmp_path):
    from spectramr.pipelines.eval_baselines import run_baseline_evaluation

    root = _build_baselines_tree(tmp_path, methods=("cut",))
    manifest_path, data_root = _build_manifest_and_data(tmp_path)
    with pytest.raises(ValueError, match="segmenter"):
        run_baseline_evaluation(
            root,
            manifest_path,
            data_root,
            tmp_path / "out",
            segmenter="not_a_segmenter",
            device="cpu",
            resnet_kwargs=_RESNET_TINY,
        )


@pytest.mark.unit
def test_unknown_task3_pairs_raises(tmp_path):
    from spectramr.pipelines.eval_baselines import run_baseline_evaluation

    root = _build_baselines_tree(tmp_path, methods=("cut",))
    manifest_path, data_root = _build_manifest_and_data(tmp_path)
    with pytest.raises(ValueError, match="task3_pairs"):
        run_baseline_evaluation(
            root,
            manifest_path,
            data_root,
            tmp_path / "out",
            task3_pairs="bogus",
            device="cpu",
            resnet_kwargs=_RESNET_TINY,
        )


# ---------------------------------------------------------------------------
# metric-layout pin: ssim aggregation axis == per-axial-slice mean
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ssim_aggregation_is_per_axial_slice_mean():
    """A ``[D,1,H,W]`` axial-slice batch makes the computer's ssim a per-slice mean.

    This pins the aggregation axis so a subtle 5-D-vs-slice-batch mistake cannot
    silently poison every reported SSIM.
    """
    from spectramr.core.metrics.computer import create_validation_metrics_computer
    from spectramr.core.metrics.registry import get_metric
    from spectramr.pipelines.eval_baselines import volume_to_slice_batch

    rng = np.random.default_rng(0)
    h, w, d = 16, 16, 5
    pred_vol = rng.random((h, w, d)).astype(np.float32)
    tgt_vol = np.clip(pred_vol + 0.1 * rng.random((h, w, d)), 0.0, 1.0).astype(
        np.float32
    )

    pred_b = volume_to_slice_batch(pred_vol)
    tgt_b = volume_to_slice_batch(tgt_vol)
    assert pred_b.shape == (d, 1, h, w)

    computer = create_validation_metrics_computer(
        ["ssim"], device="cpu", domain="image", data_range=1.0
    )
    batched = computer.compute(pred_b, tgt_b)["ssim"]

    ssim = get_metric("ssim", device="cpu")
    per_slice = [
        float(ssim(pred_b[k : k + 1], tgt_b[k : k + 1], data_range=1.0))
        for k in range(d)
    ]
    manual = float(np.mean(per_slice))
    assert abs(batched - manual) < 1e-5


@pytest.mark.unit
def test_nrmse_l2_is_global_volume_ratio():
    """nrmse_l2 over a ``[D,1,H,W]`` batch is a single global L2 ratio (full-volume)."""
    from spectramr.core.metrics.computer import create_validation_metrics_computer
    from spectramr.pipelines.eval_baselines import volume_to_slice_batch

    rng = np.random.default_rng(3)
    h, w, d = 12, 12, 4
    pred_vol = rng.random((h, w, d)).astype(np.float32)
    tgt_vol = rng.random((h, w, d)).astype(np.float32)
    pred_b = volume_to_slice_batch(pred_vol)
    tgt_b = volume_to_slice_batch(tgt_vol)

    computer = create_validation_metrics_computer(
        ["nrmse_l2"], device="cpu", domain="image"
    )
    got = computer.compute(pred_b, tgt_b)["nrmse_l2"]

    manual = float(
        np.linalg.norm(pred_vol.ravel() - tgt_vol.ravel())
        / np.linalg.norm(tgt_vol.ravel())
    )
    assert abs(got - manual) < 1e-5


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_two_seed0_runs_identical_metrics(tmp_path):
    from spectramr.pipelines.eval_baselines import run_baseline_evaluation

    root = _build_baselines_tree(tmp_path, methods=("cut",))
    manifest_path, data_root = _build_manifest_and_data(tmp_path, n_subjects=2)

    def _run(out_name):
        return run_baseline_evaluation(
            root,
            manifest_path,
            data_root,
            tmp_path / out_name,
            metrics=("nrmse_l2", "ssim"),
            seed=0,
            device="cpu",
            resnet_kwargs=_RESNET_TINY,
        )

    r1 = _run("out1")
    r2 = _run("out2")
    assert r1[0].metrics_mean == r2[0].metrics_mean
    assert r1[0].metrics_std == r2[0].metrics_std


# ---------------------------------------------------------------------------
# C1 / C2 — pairing default + fail-loud on zero subjects
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ordinal_pairing_scores_cross_field_distinct_ids(tmp_path):
    """C1/C2: ordinal (default) pairs per-field-distinct ids -> n_subjects > 0."""
    from spectramr.pipelines.eval_baselines import run_baseline_evaluation

    root = _build_baselines_tree(tmp_path, methods=("cut",))
    manifest_path, data_root = _build_cross_field_manifest(tmp_path)

    results = run_baseline_evaluation(
        root,
        manifest_path,
        data_root,
        tmp_path / "out",
        metrics=("nrmse_l2", "ssim"),
        device="cpu",
        resnet_kwargs=_RESNET_TINY,
    )
    assert results[0].n_subjects == 2  # {0001,0002} <-> {0016,0017}
    assert results[0].provenance["pairing"] == "ordinal"
    assert np.isfinite(results[0].metrics_mean["nrmse_l2"])


@pytest.mark.unit
def test_zero_subjects_fails_loud(tmp_path):
    """C2: subject_id pairing on per-field-distinct ids scores 0 -> ValueError."""
    from spectramr.pipelines.eval_baselines import run_baseline_evaluation

    root = _build_baselines_tree(tmp_path, methods=("cut",))
    manifest_path, data_root = _build_cross_field_manifest(tmp_path)

    with pytest.raises(ValueError, match="0 subjects"):
        run_baseline_evaluation(
            root,
            manifest_path,
            data_root,
            tmp_path / "out",
            metrics=("nrmse_l2", "ssim"),
            pairing="subject_id",  # yields no pairs on per-field ids
            device="cpu",
            resnet_kwargs=_RESNET_TINY,
        )


@pytest.mark.unit
def test_unknown_pairing_raises(tmp_path):
    from spectramr.pipelines.eval_baselines import run_baseline_evaluation

    root = _build_baselines_tree(tmp_path, methods=("cut",))
    manifest_path, data_root = _build_manifest_and_data(tmp_path)
    with pytest.raises(ValueError, match="pairing"):
        run_baseline_evaluation(
            root,
            manifest_path,
            data_root,
            tmp_path / "out",
            pairing="bogus",
            device="cpu",
            resnet_kwargs=_RESNET_TINY,
        )


# ---------------------------------------------------------------------------
# I1 — SynthSeg model-injection seam
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_synthseg_without_model_raises(tmp_path):
    """I1: segmenter='synthseg' with no model fails loud, naming --synthseg-model."""
    from spectramr.pipelines.eval_baselines import run_baseline_evaluation

    root = _build_baselines_tree(tmp_path, methods=("cut",))
    manifest_path, data_root = _build_manifest_and_data(tmp_path)
    with pytest.raises(ValueError, match="synthseg-model"):
        run_baseline_evaluation(
            root,
            manifest_path,
            data_root,
            tmp_path / "out",
            metrics=("nrmse_l2", "synthseg_dice"),
            segmenter="synthseg",  # no synthseg_model provided
            device="cpu",
            resnet_kwargs=_RESNET_TINY,
        )


@pytest.mark.unit
def test_synthseg_model_stamped_in_provenance(tmp_path):
    """I1/#15: an injected synthseg model is recorded in provenance."""
    from spectramr.pipelines.eval_baselines import run_baseline_evaluation

    root = _build_baselines_tree(tmp_path, methods=("cut",))
    manifest_path, data_root = _build_manifest_and_data(tmp_path)
    results = run_baseline_evaluation(
        root,
        manifest_path,
        data_root,
        tmp_path / "out",
        metrics=("nrmse_l2", "synthseg_dice"),
        segmenter="synthseg",
        synthseg_model=_ShapeRecordingSeg(),
        device="cpu",
        resnet_kwargs=_RESNET_TINY,
    )
    assert results[0].provenance["segmenter"] == "synthseg"
    assert results[0].provenance["synthseg_model"] == "<injected>"


# ---------------------------------------------------------------------------
# I2 — seg metrics run on the FULL 3-D volume, not the per-slice batch
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_seg_metric_runs_on_3d_volume(tmp_path):
    """I2: the segmenter sees the [1,1,D,H,W] 3-D volume, NOT the [D,1,H,W] batch."""
    from spectramr.pipelines.eval_baselines import run_baseline_evaluation

    root = _build_baselines_tree(tmp_path, methods=("cut",))
    manifest_path, data_root = _build_manifest_and_data(tmp_path, shape=(16, 16, 4))

    seg_model = _ShapeRecordingSeg()
    results = run_baseline_evaluation(
        root,
        manifest_path,
        data_root,
        tmp_path / "out",
        metrics=("nrmse_l2", "synthseg_dice"),
        segmenter="synthseg",
        synthseg_model=seg_model,
        device="cpu",
        resnet_kwargs=_RESNET_TINY,
    )
    assert "synthseg_dice" in results[0].metrics_mean
    assert seg_model.shapes, "segmenter was never called"
    # Every segmented tensor is the whole 3-D volume: 5-D with batch dim 1
    # (the per-slice batch would be [D,1,H,W] with batch dim D=4).
    for shp in seg_model.shapes:
        assert len(shp) == 5, f"expected 3-D volume [1,1,D,H,W], got {shp}"
        assert shp[0] == 1, f"expected single-batch 3-D volume, got batch {shp[0]}"


# ---------------------------------------------------------------------------
# I3 — label_dice + volume_consistency facade is rejected
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_label_dice_with_volume_consistency_raises(tmp_path):
    """I3: label_dice cannot express the 14 DGM labels volume_consistency needs."""
    from spectramr.pipelines.eval_baselines import run_baseline_evaluation

    root = _build_baselines_tree(tmp_path, methods=("cut",))
    manifest_path, data_root = _build_manifest_and_data(tmp_path)
    with pytest.raises(ValueError, match="volume_consistency"):
        run_baseline_evaluation(
            root,
            manifest_path,
            data_root,
            tmp_path / "out",
            metrics=("nrmse_l2", "volume_consistency"),
            segmenter="label_dice",
            device="cpu",
            resnet_kwargs=_RESNET_TINY,
        )


@pytest.mark.unit
def test_label_dice_with_synthseg_dice_ok(tmp_path):
    """I3: label_dice remains a valid smoke proxy for synthseg_dice (not volume)."""
    from spectramr.pipelines.eval_baselines import run_baseline_evaluation

    root = _build_baselines_tree(tmp_path, methods=("cut",))
    manifest_path, data_root = _build_manifest_and_data(tmp_path)
    results = run_baseline_evaluation(
        root,
        manifest_path,
        data_root,
        tmp_path / "out",
        metrics=("nrmse_l2", "synthseg_dice"),
        segmenter="label_dice",
        device="cpu",
        resnet_kwargs=_RESNET_TINY,
    )
    assert "synthseg_dice" in results[0].metrics_mean
    assert np.isfinite(results[0].metrics_mean["synthseg_dice"])


# ---------------------------------------------------------------------------
# I4 — _KerasSynthSegShim numpy-bridge (3-D and 2-D)
# ---------------------------------------------------------------------------


def _make_fake_keras_model(n_classes: int = 14):
    """Return a callable that mimics a Keras SynthSeg forward pass.

    Accepts numpy arrays in Keras channel-last convention and returns
    uniform softmax-like probabilities of the correct shape.
    """

    def _forward(arr):
        # arr: [B, H, W, D, 1] (3-D) or [B, H, W, 1] (2-D)
        if arr.ndim == 5:
            b, h, w, d, _ = arr.shape
            return np.ones((b, h, w, d, n_classes), dtype=np.float32) / n_classes
        b, h, w, _ = arr.shape
        return np.ones((b, h, w, n_classes), dtype=np.float32) / n_classes

    return _forward


@pytest.mark.unit
def test_keras_shim_3d_shape_round_trip():
    """I4: shim maps [B,1,D,H,W] -> [B,n_classes,D,H,W] for the 3-D eval path."""
    from spectramr.pipelines.eval_baselines import _KerasSynthSegShim

    shim = _KerasSynthSegShim(_make_fake_keras_model(n_classes=14))
    inp = torch.rand(1, 1, 4, 8, 8)
    out = shim(inp)
    assert out.shape == (1, 14, 4, 8, 8)
    assert out.dtype == torch.float32


@pytest.mark.unit
def test_keras_shim_2d_shape_round_trip():
    """I4: shim maps [B,1,H,W] -> [B,n_classes,H,W] for 2-D slice inputs."""
    from spectramr.pipelines.eval_baselines import _KerasSynthSegShim

    shim = _KerasSynthSegShim(_make_fake_keras_model(n_classes=14))
    inp = torch.rand(2, 1, 16, 16)
    out = shim(inp)
    assert out.shape == (2, 14, 16, 16)
    assert out.dtype == torch.float32


@pytest.mark.unit
def test_keras_shim_multichannel_raises():
    """I4: shim raises immediately on multi-channel input (SynthSeg is magnitude-only)."""
    from spectramr.pipelines.eval_baselines import _KerasSynthSegShim

    shim = _KerasSynthSegShim(_make_fake_keras_model())
    with pytest.raises(ValueError, match="single-channel"):
        shim(torch.rand(1, 2, 4, 8, 8))


@pytest.mark.unit
def test_keras_shim_argmax_gives_label_map():
    """I4: SynthSegBackend.segment(shim) produces an integer label map [B,D,H,W]."""
    from spectramr.core.metrics.quantitative.segmentation import SynthSegBackend
    from spectramr.pipelines.eval_baselines import _KerasSynthSegShim

    shim = _KerasSynthSegShim(_make_fake_keras_model(n_classes=14))
    backend = SynthSegBackend(n_classes=14, model=shim)
    inp = torch.rand(1, 1, 4, 8, 8)
    labels = backend.segment(inp)
    assert labels.shape == (1, 4, 8, 8)
    assert labels.dtype == torch.long


# ---------------------------------------------------------------------------
# I5 — _resolve_synthseg_model .h5 routing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_h5_without_keras_raises(tmp_path):
    """I5: a .h5 path without keras installed raises RuntimeError naming install cmd."""
    import sys
    import unittest.mock as mock

    from spectramr.pipelines.eval_baselines import _resolve_synthseg_model

    fake_h5 = tmp_path / "synthseg.h5"
    fake_h5.write_bytes(b"")  # existence doesn't matter; keras import fails first

    # Force keras to appear absent even if somehow installed.
    with mock.patch.dict(sys.modules, {"keras": None}):
        with pytest.raises(RuntimeError, match="keras"):
            _resolve_synthseg_model(str(fake_h5), torch.device("cpu"))


@pytest.mark.unit
def test_resolve_h5_with_keras_returns_shim(tmp_path):
    """I5: a .h5 path with keras available wraps the loaded model in _KerasSynthSegShim."""
    import sys
    import types
    import unittest.mock as mock

    from spectramr.pipelines.eval_baselines import _KerasSynthSegShim, _resolve_synthseg_model

    fake_h5 = tmp_path / "synthseg.h5"
    fake_h5.write_bytes(b"")

    fake_keras_model = _make_fake_keras_model(n_classes=14)
    fake_keras = types.ModuleType("keras")
    fake_keras.models = mock.MagicMock()  # type: ignore[attr-defined]
    fake_keras.models.load_model = mock.MagicMock(return_value=fake_keras_model)  # type: ignore[attr-defined]

    with mock.patch.dict(sys.modules, {"keras": fake_keras}):
        result = _resolve_synthseg_model(str(fake_h5), torch.device("cpu"))

    assert isinstance(result, _KerasSynthSegShim)


@pytest.mark.unit
def test_resolve_none_raises():
    """I5: None model fails loud with a message naming --synthseg-model."""
    from spectramr.pipelines.eval_baselines import _resolve_synthseg_model

    with pytest.raises(ValueError, match="synthseg-model"):
        _resolve_synthseg_model(None, torch.device("cpu"))


# I6 — _SynthSegDeviceAdapter and synthseg_device parameter
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_synthseg_device_adapter_round_trips_device():
    """Adapter moves input to model device and output back to input device."""
    from spectramr.pipelines.eval_baselines import _SynthSegDeviceAdapter

    cpu = torch.device("cpu")

    class _Echo(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2.0

    adapter = _SynthSegDeviceAdapter(_Echo(), cpu)
    inp = torch.ones(1, 1, 4, 4, 4)  # cpu input
    out = adapter(inp)
    assert out.device == inp.device
    assert torch.allclose(out, torch.ones_like(inp) * 2.0)


@pytest.mark.unit
def test_synthseg_device_adapter_eval_delegates():
    """adapter.eval() calls eval() on the wrapped model and returns self."""
    import unittest.mock as mock

    from spectramr.pipelines.eval_baselines import _SynthSegDeviceAdapter

    inner = mock.MagicMock()
    adapter = _SynthSegDeviceAdapter(inner, torch.device("cpu"))
    result = adapter.eval()
    inner.eval.assert_called_once()
    assert result is adapter


@pytest.mark.unit
def test_resolve_pt_uses_jit_load(tmp_path):
    """_resolve_synthseg_model uses torch.jit.load for .pt files (TorchScript)."""
    import unittest.mock as mock

    from spectramr.pipelines.eval_baselines import _resolve_synthseg_model

    # Build a minimal real TorchScript model so torch.jit.load succeeds.
    model = torch.jit.trace(torch.nn.Identity(), torch.zeros(1))
    pt_path = tmp_path / "synthseg.pt"
    torch.jit.save(model, str(pt_path))

    with mock.patch("torch.jit.load", wraps=torch.jit.load) as spy:
        _resolve_synthseg_model(str(pt_path), torch.device("cpu"))
    spy.assert_called_once()


@pytest.mark.unit
def test_run_baseline_evaluation_synthseg_device_in_provenance(tmp_path):
    """synthseg_device is stamped into each result's provenance when segmenter=synthseg."""
    from spectramr.pipelines.eval_baselines import run_baseline_evaluation

    root = _build_baselines_tree(tmp_path, methods=("cut",))
    manifest_path, data_root = _build_manifest_and_data(tmp_path)
    results = run_baseline_evaluation(
        root,
        manifest_path,
        data_root,
        tmp_path / "out",
        metrics=("nrmse_l2", "synthseg_dice"),
        segmenter="synthseg",
        synthseg_model=_ShapeRecordingSeg(),
        synthseg_device="cpu",
        device="cpu",
        resnet_kwargs=_RESNET_TINY,
    )
    assert results
    assert results[0].provenance.get("synthseg_device") == "cpu"


@pytest.mark.unit
def test_chunked_voxel_compute_matches_full_for_ssim():
    """_chunked_voxel_compute with chunk_size=1 gives the same result as chunk_size=0."""
    from unittest.mock import MagicMock

    from spectramr.pipelines.eval_baselines import _chunked_voxel_compute

    pred = torch.rand(4, 1, 8, 8)
    target = torch.rand(4, 1, 8, 8)

    call_log: list[torch.Tensor] = []

    class _RecordingComputer:
        def compute(self, p: torch.Tensor, t: torch.Tensor) -> dict[str, float]:
            call_log.append(p)
            return {"ssim": float((p - t).abs().mean())}

    computer = _RecordingComputer()
    result_chunked = _chunked_voxel_compute(computer, pred, target, chunk_size=2)
    result_full = _chunked_voxel_compute(computer, pred, target, chunk_size=0)

    # Chunked was called twice (4 slices / chunk 2 = 2 calls), full once.
    assert len(call_log) == 3
    assert call_log[0].shape[0] == 2  # first chunk

    # Mean of two equal-size chunks equals full-batch mean for a linear-mean metric.
    assert abs(result_chunked["ssim"] - result_full["ssim"]) < 1e-5


@pytest.mark.unit
def test_metric_slice_chunk_stamped_in_provenance(tmp_path):
    """metric_slice_chunk is written into provenance (#15 — every knob stamped)."""
    from spectramr.pipelines.eval_baselines import run_baseline_evaluation

    root = _build_baselines_tree(tmp_path, methods=("cut",))
    manifest_path, data_root = _build_manifest_and_data(tmp_path)
    results = run_baseline_evaluation(
        root,
        manifest_path,
        data_root,
        tmp_path / "out",
        metrics=("nrmse_l2",),
        metric_slice_chunk=4,
        device="cpu",
        resnet_kwargs=_RESNET_TINY,
    )
    assert results
    assert results[0].provenance.get("metric_slice_chunk") == 4


def test_prewarm_dummy_size_does_not_trigger_metric_warning(tmp_path, caplog):
    """Pre-warm dummy (64x64) is large enough for all metrics; no spurious WARNING.

    Regression: the 4x4 dummy used in the original implementation padded to 8x8,
    which is smaller than AlexNet's 11x11 first conv and caused a
    ``Failed to compute metric 'lpips_alex'`` WARNING during warm-up.  The
    warm-up appeared to succeed (LPIPS loaded) but the compute path was never
    actually exercised (a warm-up that always crashes is no warm-up at all).

    With 64x64, both SSIM and NRMSE forward-pass cleanly during warm-up.
    """
    import logging

    from spectramr.pipelines.eval_baselines import run_baseline_evaluation

    root = _build_baselines_tree(tmp_path, methods=("cut",))
    manifest_path, data_root = _build_manifest_and_data(tmp_path)
    with caplog.at_level(logging.WARNING, logger="spectramr.core.metrics.computer"):
        run_baseline_evaluation(
            root,
            manifest_path,
            data_root,
            tmp_path / "out",
            metrics=("nrmse_l2", "ssim"),
            device="cpu",
            resnet_kwargs=_RESNET_TINY,
        )
    # No "Failed to compute metric" during the pre-warm phase
    prewarm_warns = [r for r in caplog.records if "Failed to compute metric" in r.message]
    assert not prewarm_warns, f"Spurious pre-warm WARNING(s): {prewarm_warns}"
