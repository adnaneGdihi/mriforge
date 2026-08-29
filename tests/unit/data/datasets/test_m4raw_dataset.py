"""Tests for :class:`mriforge.data.datasets.m4raw_dataset.M4RawRepetitionDataset`.

Regression coverage for DC-1: ``coil_processing_mode`` must be validated at
``__init__`` time. An unrecognised mode string (e.g. a YAML typo) used to
silently fall through to RSS-combine — a NN#3 / pitfall #9 silent fallback.
The fix raises ``ValueError`` for any mode not in ``_VALID_COIL_MODES`` and
makes the RSS guard fire only for the explicit ``"rss"`` mode.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import h5py
import numpy as np
import pytest

import torch

import mriforge.data.datasets.m4raw_dataset as m4raw_mod
from mriforge.data.datasets.m4raw_dataset import (
    _VALID_COIL_MODES,
    _VALID_TARGET_MODES,
    M4RawRepetitionDataset,
    _MIN_REPS_FOR_LOO,
    _average_reps,
    _clamp_worker_threads_once,
    _read_kspace_shape,
)

# ── NEX repetition averaging: complex_mean (legacy) vs phase_aligned_mean (fix) ──
# M4Raw reps are separate acquisitions with global phase drift, so plain complex
# k-space averaging CANCELS signal (SNR falls below a single rep). Aligning each
# rep's global phase to rep0 before averaging recovers the coherent sqrt(N) gain.


def test_valid_target_modes_is_the_advertised_set() -> None:
    assert _VALID_TARGET_MODES == frozenset({"complex_mean", "phase_aligned_mean"})


def test_average_reps_complex_mean_is_plain_mean() -> None:
    torch.manual_seed(0)
    reps = [torch.randn(1, 2, 4, 4, dtype=torch.complex64) for _ in range(3)]
    assert torch.allclose(
        _average_reps(reps, "complex_mean"), torch.stack(reps).mean(0)
    )


def test_average_reps_phase_aligned_recovers_signal_under_global_drift() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 2, 8, 8, dtype=torch.complex64)
    reps = [x, x * torch.exp(torch.tensor(2.0j)), x * torch.exp(torch.tensor(-1.3j))]
    aligned = _average_reps(reps, "phase_aligned_mean")
    assert torch.allclose(
        aligned.abs(), x.abs(), atol=1e-4
    )  # coherent, no cancellation
    cancelled = _average_reps(reps, "complex_mean")
    assert cancelled.abs().sum() < aligned.abs().sum()  # plain mean cancels signal


def test_average_reps_single_rep_passthrough() -> None:
    r = torch.randn(1, 2, 4, 4, dtype=torch.complex64)
    assert torch.allclose(_average_reps([r], "phase_aligned_mean"), r)


def test_average_reps_unknown_mode_raises() -> None:
    r = torch.randn(1, 2, 4, 4, dtype=torch.complex64)
    with pytest.raises(ValueError, match="target_mode"):
        _average_reps([r, r], "bogus")


def test_average_reps_exclude_index_drops_the_input_rep() -> None:
    """Leave-one-out target averages only the retained reps."""
    torch.manual_seed(0)
    reps = [torch.randn(1, 2, 4, 4, dtype=torch.complex64) for _ in range(3)]
    loo = _average_reps(reps, "complex_mean", exclude_index=0)
    expected = torch.stack(reps[1:]).mean(0)  # reps 1,2 only
    assert torch.allclose(loo, expected)


def test_average_reps_exclude_index_none_is_all_reps() -> None:
    """exclude_index=None (default) is byte-identical to the legacy average."""
    torch.manual_seed(1)
    reps = [torch.randn(1, 2, 4, 4, dtype=torch.complex64) for _ in range(3)]
    assert torch.allclose(
        _average_reps(reps, "complex_mean", exclude_index=None),
        _average_reps(reps, "complex_mean"),
    )


def test_average_reps_exclude_leaving_one_rep_passes_through() -> None:
    """Two reps, exclude one → the single retained rep (no empty mean)."""
    torch.manual_seed(2)
    reps = [torch.randn(1, 2, 4, 4, dtype=torch.complex64) for _ in range(2)]
    assert torch.allclose(
        _average_reps(reps, "phase_aligned_mean", exclude_index=0), reps[1]
    )


def test_average_reps_phase_alignment_anchors_to_first_retained_rep() -> None:
    """With rep 0 excluded, phase is anchored to rep 1, not the dropped rep."""
    torch.manual_seed(3)
    x = torch.randn(1, 2, 8, 8, dtype=torch.complex64)
    reps = [x * torch.exp(torch.tensor(2.7j)), x, x * torch.exp(torch.tensor(-0.9j))]
    loo = _average_reps(reps, "phase_aligned_mean", exclude_index=0)
    # reps 1,2 phase-aligned to rep 1 (== x) recover the coherent magnitude
    assert torch.allclose(loo.abs(), x.abs(), atol=1e-4)


def test_valid_coil_modes_is_the_advertised_set() -> None:
    """The module-level frozenset pins the four accepted coil modes."""
    assert _VALID_COIL_MODES == frozenset({"none", "rss", "svd", "flatten"})


def test_clamp_worker_threads_runs_once(monkeypatch) -> None:
    """Perf regression (2026-07-02): the process-global
    ``torch.set_num_threads(1)`` must fire once per process, not on every
    ``_load_item``. The one-time guard collapses N calls to a single
    ``set_num_threads``."""
    calls = {"n": 0}

    def _counting_set(_n):
        calls["n"] += 1

    monkeypatch.setattr(m4raw_mod.torch, "set_num_threads", _counting_set)
    monkeypatch.setattr(m4raw_mod, "_NUM_THREADS_CLAMPED", False)

    for _ in range(5):
        _clamp_worker_threads_once()

    assert calls["n"] == 1, f"set_num_threads ran {calls['n']}x, expected 1"


def test_getcwd_not_called_when_debug_disabled(monkeypatch) -> None:
    """The verbose per-item path logging (``os.getcwd()`` + reprs) must be
    gated behind ``logger.isEnabledFor(DEBUG)`` — it ran unconditionally on
    every load before. Source-level pin so we don't need a full fixture."""
    import inspect

    src = inspect.getsource(M4RawRepetitionDataset._load_item)
    # No unconditional getcwd at the function top; both getcwd sites are inside
    # a guard (DEBUG block or the corruption-error branch).
    assert "cwd = os.getcwd()" not in src
    assert "isEnabledFor(logging.DEBUG)" in src


def test_unknown_coil_processing_mode_raises() -> None:
    """A mis-typed / unsupported mode must raise, never silently RSS-combine."""
    with pytest.raises(ValueError, match="coil_processing_mode"):
        M4RawRepetitionDataset(h5_files=[], coil_processing_mode="bad_mode")


@pytest.mark.parametrize("bad_mode", ["rss_kspace", "magnitude", "RSS", "auto"])
def test_unknown_coil_processing_mode_variants_raise(bad_mode: str) -> None:
    """Common typos/aliases that used to fall through to RSS now raise."""
    with pytest.raises(ValueError):
        M4RawRepetitionDataset(h5_files=[], coil_processing_mode=bad_mode)


@pytest.mark.parametrize("mode", ["none", "rss", "flatten"])
def test_valid_coil_processing_mode_does_not_raise(mode: str) -> None:
    """Every IMPLEMENTED mode constructs cleanly (empty file list = empty dataset).

    ``svd`` is deliberately excluded here -- it is recognized but inert on this
    path, so it must raise (see ``test_svd_coil_processing_mode_raises``).
    """
    ds = M4RawRepetitionDataset(h5_files=[], coil_processing_mode=mode)
    assert ds.coil_processing_mode == mode


def test_svd_coil_processing_mode_raises() -> None:
    """svd is validated but INERT on the M4Raw repetition path (only ``rss`` is
    acted on), so accepting it silently trains on uncompressed multi-coil k-space
    -- CLAUDE.md pitfall #16 (facade) / #15b. It must fail loud; the wired svd
    path is UniversalMRIDataset (``dataset_type: kspace``)."""
    with pytest.raises(NotImplementedError, match="svd"):
        M4RawRepetitionDataset(h5_files=[], coil_processing_mode="svd")


# ──────────────────────────────────────────────────────────────────────────
# F1 (smoke 2026-06-16): header-only shape index → queue-build OOM avoidance
#
# ``TorchIOQueueBuilder._filter_patch_compatible_subjects`` has a FAST PATH that
# reads ``dataset.index`` (a list of dict records each carrying a ``shape``) so
# the spatial-extent check needs no voxel load. M4Raw exposed no ``index`` and
# its groups had no ``shape``, so the filter fell through to the slow
# ``for subj in dataset`` probe that materialises the WHOLE corpus
# (1024 vols x 38 MB ~= 39 GB) → host OOM at queue-build time, before step 1.
# See [[project_queue_build_oom_patch_filter]].
# ──────────────────────────────────────────────────────────────────────────


def _write_kspace_h5(path, shape, *, complex_dtype: bool) -> None:
    """Write a minimal M4Raw-style H5 with a ``kspace`` dataset of *shape*."""
    with h5py.File(str(path), "w") as f:
        if complex_dtype:
            f.create_dataset("kspace", data=np.zeros(shape, dtype=np.complex64))
        else:
            # real storage with trailing real/imag axis of size 2
            f.create_dataset("kspace", data=np.zeros((*shape, 2), dtype=np.float32))


def test_read_kspace_shape_complex_returns_full_shape(tmp_path) -> None:
    """A complex-dtype kspace returns its shape verbatim (H/W trailing)."""
    p = tmp_path / "scan.h5"
    _write_kspace_h5(p, (3, 4, 32, 24), complex_dtype=True)
    assert _read_kspace_shape(p) == (3, 4, 32, 24)


def test_read_kspace_shape_real_trailing_2_drops_the_2(tmp_path) -> None:
    """Real storage ``(..., 2)`` is the complex layout; the trailing 2 drops."""
    p = tmp_path / "scan_real.h5"
    _write_kspace_h5(p, (3, 4, 32, 24), complex_dtype=False)  # stored (3,4,32,24,2)
    # the effective complex k-space shape has H/W trailing, no real/imag axis
    assert _read_kspace_shape(p) == (3, 4, 32, 24)


def test_read_kspace_shape_missing_file_returns_none(tmp_path) -> None:
    assert _read_kspace_shape(tmp_path / "nope.h5") is None


def test_read_kspace_shape_no_kspace_key_returns_none(tmp_path) -> None:
    p = tmp_path / "empty.h5"
    with h5py.File(str(p), "w") as f:
        f.create_dataset("other", data=np.zeros((2, 2)))
    assert _read_kspace_shape(p) is None


def _build_single_contrast_ds(tmp_path, *, hw=(32, 24)):
    """Two-rep single-contrast M4Raw dataset over real on-disk H5 files."""
    h, w = hw
    files = []
    for rep in ("01", "02"):
        p = tmp_path / f"2022_T1{rep}.h5"
        _write_kspace_h5(p, (2, 4, h, w), complex_dtype=True)
        files.append(p)
    return M4RawRepetitionDataset(
        h5_files=files, single_contrast=True, coil_processing_mode="none"
    )


def test_index_property_aliases_groups_and_carries_shape(tmp_path) -> None:
    """``ds.index`` exposes the group records and each has an H/W-trailing shape."""
    ds = _build_single_contrast_ds(tmp_path, hw=(32, 24))
    idx = ds.index
    assert isinstance(idx, list) and len(idx) == len(ds) == 1
    rec = idx[0]
    assert isinstance(rec, dict) and len(rec["shape"]) >= 2
    assert rec["shape"][-2] == 32 and rec["shape"][-1] == 24


def test_index_setter_prunes_the_real_indexing_source(tmp_path) -> None:
    """Assigning ``ds.index`` rebinds ``_groups`` so ``len`` / ``__getitem__`` shrink."""
    ds = _build_single_contrast_ds(tmp_path)
    ds2 = _build_single_contrast_ds(tmp_path)
    combined = list(ds.index) + list(ds2.index)
    ds.index = combined
    assert len(ds) == 2
    ds.index = combined[:1]
    assert len(ds) == 1


# ── provenance_counts: the units a user actually checks ───────────────────
# The user's question was "the training folder has 1024 files but provenance
# says 768 -- is 25 % of my data missing?". It was not: 1024 files collapse to
# 384 (patient, contrast) groups, x4 samples_per_volume = 1536 patches, /2
# batch_size (drop_last) = 768 batches. Four different units, one number on the
# record. This hook puts the other three there too.


def test_provenance_counts_separates_files_from_groups(tmp_path) -> None:
    """Files on disk and groups in the index are different counts.

    The fixture is two repetitions of ONE contrast, so two files collapse to a
    single group -- the exact ratio that makes a batch-derived number look like
    data loss when compared against a file listing.
    """
    ds = _build_single_contrast_ds(tmp_path)
    counts = ds.provenance_counts()
    assert counts["files"] == 2
    assert counts["groups"] == len(ds) == 1
    assert counts["patients"] == 1
    assert counts["per_contrast"] == {"T1": 1}


def test_provenance_counts_dedupes_files_across_source_and_target(tmp_path) -> None:
    """A federated-pair record holds real files in BOTH halves; both are counted.

    ``_attach_shape_metadata``'s ``rec.get("source") or rec.get("target")`` idiom
    takes only the first non-empty list. Copying it here would report 2 files for
    this record instead of 3, undercounting precisely the federated corpus. Dedup
    is by path, so a file cited on both sides still counts once.
    """
    ds = _build_single_contrast_ds(tmp_path)
    ds.index = [
        {
            "source": ["/d/p1_T1.h5", "/d/shared.h5"],
            "target": ["/d/p1_T2.h5", "/d/shared.h5"],
            "patient_id": "p1",
            "target_contrast": "T2",
        }
    ]
    counts = ds.provenance_counts()
    assert counts["files"] == 3, "4 references, one shared path"
    assert counts["groups"] == 1
    assert counts["patients"] == 1
    assert counts["per_contrast"] == {"T2": 1}, "federated records key off target_contrast"


def test_provenance_counts_patients_is_not_the_group_count(tmp_path) -> None:
    """``patients`` must stay distinguishable from ``groups``.

    torchio's ``Queue.num_subjects`` returns ``len(subjects_dataset)`` -- groups,
    384 for this corpus -- while the patient count is 128. Reporting either under
    the name ``subjects`` would re-create the ambiguity this hook removes, which
    is why neither key is called that.
    """
    ds = _build_single_contrast_ds(tmp_path)
    ds.index = [
        {"paths": ["/d/a_T1.h5"], "patient_id": "p1", "contrast": "T1"},
        {"paths": ["/d/a_T2.h5"], "patient_id": "p1", "contrast": "T2"},
        {"paths": ["/d/b_T1.h5"], "patient_id": "p2", "contrast": "T1"},
    ]
    counts = ds.provenance_counts()
    assert counts["groups"] == 3
    assert counts["patients"] == 2
    assert counts["per_contrast"] == {"T1": 2, "T2": 1}, "groups per contrast, not files"


# ── files_per_contrast: the 3/3/2 repetition asymmetry (#1392) ─────────────
# ``per_contrast`` counts groups, so M4Raw's uneven repetition budget is
# invisible on the record: 30 subjects per contrast reads {T1:30, T2:30,
# FLAIR:30} whether FLAIR ships 2 repetitions or 3. The NEX target is an average
# over those repetitions, so the difference is not bookkeeping -- it changes what
# the target IS, per contrast.


def test_files_per_contrast_exposes_the_repetition_asymmetry(tmp_path) -> None:
    """The uniform ``per_contrast`` and the uneven file count coexist.

    This is the finding #1392 reports. Both keys are asserted in ONE test on
    purpose: the defect is not that either number is wrong, it is that the only
    number published was the one that cannot show the asymmetry. A test that
    checked ``files_per_contrast`` alone would pass just as well against a build
    that had quietly redefined ``per_contrast`` to mean files.
    """
    ds = _build_single_contrast_ds(tmp_path)
    ds.index = [
        {
            "paths": ["/d/p1_T101.h5", "/d/p1_T102.h5", "/d/p1_T103.h5"],
            "patient_id": "p1",
            "contrast": "T1",
        },
        {
            "paths": ["/d/p1_T201.h5", "/d/p1_T202.h5", "/d/p1_T203.h5"],
            "patient_id": "p1",
            "contrast": "T2",
        },
        # FLAIR ships 2 repetitions, not 3 (#1172).
        {
            "paths": ["/d/p1_FLAIR01.h5", "/d/p1_FLAIR02.h5"],
            "patient_id": "p1",
            "contrast": "FLAIR",
        },
    ]
    counts = ds.provenance_counts()
    assert counts["per_contrast"] == {"FLAIR": 1, "T1": 1, "T2": 1}, (
        "groups per contrast is uniform -- it cannot show the asymmetry"
    )
    assert counts["files_per_contrast"] == {"FLAIR": 2, "T1": 3, "T2": 3}, (
        "files per contrast is where the 3/3/2 split becomes visible"
    )
    assert counts["files"] == 8
    assert sum(counts["files_per_contrast"].values()) == counts["files"]


def test_files_per_contrast_counts_a_shared_federated_source_once(tmp_path) -> None:
    """One T1 group paired against two targets is still one T1 group of files.

    ``_build_federated_pairs`` pairs the patient's single T1 group against every
    other contrast, so a per-record counter would report the T1 repetitions once
    per pair -- here 4 instead of 2, inflating exactly the contrast that is
    reused. Accumulating into sets is what makes the count a property of the
    corpus rather than of the pairing.

    It also pins the attribution SIDE: the source files must land under
    ``source_contrast``. Keying the whole record off ``target_contrast`` -- the
    obvious reading, and what ``per_contrast`` does -- would file every T1
    repetition under T2 and FLAIR.
    """
    ds = _build_single_contrast_ds(tmp_path)
    t1 = ["/d/p1_T101.h5", "/d/p1_T102.h5"]
    ds.index = [
        {
            "source": list(t1),
            "target": ["/d/p1_T201.h5", "/d/p1_T202.h5"],
            "patient_id": "p1",
            "target_contrast": "T2",
            "source_contrast": "T1",
        },
        {
            "source": list(t1),
            "target": ["/d/p1_FLAIR01.h5"],
            "patient_id": "p1",
            "target_contrast": "FLAIR",
            "source_contrast": "T1",
        },
    ]
    counts = ds.provenance_counts()
    assert counts["files_per_contrast"] == {"FLAIR": 1, "T1": 2, "T2": 2}
    assert counts["per_contrast"] == {"FLAIR": 1, "T2": 1}, (
        "groups still key off target_contrast -- unchanged by this hook"
    )
    assert sum(counts["files_per_contrast"].values()) == counts["files"] == 5


def test_files_per_contrast_omits_a_record_naming_no_contrast(tmp_path) -> None:
    """An unattributable file is left OUT, and the sum invariant reports it.

    Folding it into ``UNKNOWN``, or into whichever contrast the record last
    mentioned, would produce a total that still equals ``files`` -- a
    plausible-looking number carrying a guess. Omitting it instead makes
    ``sum(files_per_contrast) < files`` the signal that something was
    unattributable (non-negotiable 3: absent is a state to report).
    """
    ds = _build_single_contrast_ds(tmp_path)
    ds.index = [
        {"paths": ["/d/p1_T101.h5"], "patient_id": "p1", "contrast": "T1"},
        {"paths": ["/d/p2_mystery.h5"], "patient_id": "p2"},
    ]
    counts = ds.provenance_counts()
    assert counts["files_per_contrast"] == {"T1": 1}
    assert counts["files"] == 2
    assert sum(counts["files_per_contrast"].values()) < counts["files"], (
        "the shortfall is the report"
    )


def test_federated_pairs_stamp_their_source_contrast(tmp_path) -> None:
    """The builder declares the source contrast; nothing downstream re-parses it.

    ``_index_stats`` attributes source files by READING ``source_contrast``. If
    the builder stopped stamping it, that attribution would silently drop every
    T1 file rather than fail, so the key is pinned at its origin.
    """
    from mriforge.data.datasets.m4raw_dataset import M4RawRepetitionDataset

    groups = [
        [tmp_path / "p1_T101.h5", tmp_path / "p1_T102.h5"],
        [tmp_path / "p1_T201.h5"],
        [tmp_path / "p1_FLAIR01.h5"],
    ]
    pairs = M4RawRepetitionDataset._build_federated_pairs(groups)
    assert pairs, "T1 present, so pairs must be produced"
    assert {p["source_contrast"] for p in pairs} == {"T1"}
    assert {p["target_contrast"] for p in pairs} == {"T2", "FLAIR"}


def test_provenance_counts_opens_no_voxel_data(tmp_path, monkeypatch) -> None:
    """Counting is metadata-only -- it must never touch an HDF5 file.

    ``describe_dataloader`` calls this during startup provenance, and a count
    that opened 1024 volumes to answer "how many files?" would be the same
    full-corpus materialisation the queue-filter fast path exists to avoid
    (non-negotiable 9).
    """
    ds = _build_single_contrast_ds(tmp_path)

    def _boom(*a, **k):
        raise AssertionError("provenance_counts must not open voxel data")

    monkeypatch.setattr(h5py, "File", _boom)
    assert ds.provenance_counts()["files"] == 2


def test_queue_filter_uses_fast_path_without_materializing(
    tmp_path, monkeypatch
) -> None:
    """REGRESSION: the patch-compat filter must NOT iterate the dataset.

    We poison ``_load_item`` (the per-sample loader) so ANY materialisation
    raises. The fast path reads ``index``+``shape`` only, so the filter returns
    cleanly; the pre-fix slow path would touch ``__getitem__`` and explode —
    the exact 39 GB OOM that SIGKILLed queue-build on the cluster.
    """
    from mriforge.data.builders.torchio_queue_builder import TorchIOQueueBuilder

    ds = _build_single_contrast_ds(tmp_path, hw=(64, 64))

    def _boom(*_a, **_k):
        raise AssertionError("materialised a subject — slow path taken (OOM risk)")

    monkeypatch.setattr(M4RawRepetitionDataset, "_load_item", _boom)
    # patch fits (64x64 >= 32x32) → all kept, no drop, no iteration
    out = TorchIOQueueBuilder._filter_patch_compatible_subjects(ds, (32, 32))
    assert len(out) == 1


def test_queue_filter_drops_too_small_via_shape_metadata(tmp_path, monkeypatch) -> None:
    """A volume smaller than the patch is dropped from ``index`` with no voxel load."""
    from mriforge.data.builders.torchio_queue_builder import TorchIOQueueBuilder

    ds = _build_single_contrast_ds(tmp_path, hw=(16, 16))

    def _boom(*_a, **_k):
        raise AssertionError("slow path taken")

    monkeypatch.setattr(M4RawRepetitionDataset, "_load_item", _boom)
    out = TorchIOQueueBuilder._filter_patch_compatible_subjects(ds, (32, 32))
    assert len(out) == 0  # 16x16 < 32x32 patch → dropped via header shape


def test_all_retries_exhausted_raises_not_zero_fill(tmp_path, monkeypatch) -> None:
    """Regression (2026-07-01): when EVERY probed repetition group fails to load
    (systemic data loss — unmounted share, wrong ``data_root``, broken manifest),
    ``__getitem__`` must RAISE, not return a zero-filled synthetic ``tio.Subject``.

    The old zero-fill let training run to completion on meaningless all-zero
    input/target with no hard failure — the "loads zero/random" facade
    (pitfall #9/#16).
    """
    from mriforge.data.datasets.m4raw_dataset import _SkipSample

    ds = _build_single_contrast_ds(tmp_path, hw=(16, 16))

    def _always_skip(self, idx):  # noqa: ANN001
        raise _SkipSample("simulated unreadable k-space file")

    monkeypatch.setattr(M4RawRepetitionDataset, "_load_item", _always_skip)

    with pytest.raises(RuntimeError, match="retries exhausted"):
        _ = ds[0]


# ── Double k-space normalization: dataset AND transform both fired ────────────
# ``data.normalize_kspace`` is read by TWO independent normalizers that both run
# on the same tensor: ``M4RawRepetitionDataset.__getitem__`` (percentile divide +
# log1p) and ``KSpaceNormalizationTransform``, which the dataset then applies
# itself via ``self.transform(subject)``. The result is a double percentile
# divide + double log1p, and the transform OVERWRITES ``subject['kspace_scale']``
# so the dataset's scale is lost and the normalization is not invertible.
#
# Per CLAUDE.md the canonical home for a morphing op is ``data/transforms/`` —
# the dataset matches and serves, the transform morphs.


def _write_structured_kspace_h5(path, *, slices=1, coils=2, hw=(16, 16)) -> None:
    """M4Raw-style H5 whose k-space has a realistic DC-vs-periphery range."""
    h, w = hw
    rng = np.random.default_rng(0)
    img = np.zeros((slices, coils, h, w), dtype=np.complex64)
    img[:, :, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
    img += 0.01 * rng.standard_normal(img.shape).astype(np.float32)
    ksp = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(img, axes=(-2, -1)), norm="ortho"), axes=(-2, -1)
    ).astype(np.complex64)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("kspace", data=ksp)


def _stacked_to_complex(t: torch.Tensor) -> torch.Tensor:
    """``(2C, H, W, S)`` real-interleaved → ``(S, C, H, W)`` complex."""
    real, imag = t[0::2], t[1::2]  # (C, H, W, S)
    return torch.complex(real, imag).permute(3, 0, 1, 2)


def _build_normalizing_ds(tmp_path, *, transform=None):
    """Single-contrast M4Raw dataset with k-space normalization enabled."""
    files = []
    for rep in ("01", "02"):
        p = tmp_path / f"2022_T1{rep}.h5"
        _write_structured_kspace_h5(p)
        files.append(p)
    return M4RawRepetitionDataset(
        h5_files=files,
        single_contrast=True,
        coil_processing_mode="none",
        normalize_kspace=True,
        kspace_percentile=0.95,
        log_scaling=True,
        use_repetitions=True,
        transform=transform,
    )


@pytest.mark.parametrize("scale_domain", ["kspace", "image"])
def test_kspace_normalization_is_invertible_via_stored_scale(
    tmp_path, scale_domain
) -> None:
    """Round-trip: denormalizing the served k-space recovers the raw k-space.

    This holds only when the normalization ran EXACTLY ONCE and the scale it
    used is the one published as ``subject['kspace_scale']``. With the dataset
    and ``KSpaceNormalizationTransform`` both firing, two percentile divides and
    two ``log1p`` compressions were applied but only the transform's scale
    survived, so the inverse under-restored and the k-space was unrecoverable.

    Semantics-agnostic on purpose: it must hold for either scale domain.
    """
    from mriforge.data.transforms.normalization import (
        KSpaceNormalizationTransform,
        denormalize_kspace_robust,
    )

    # The transform the builder appends verbatim when data.normalize_kspace=True.
    tfm = KSpaceNormalizationTransform(
        percentile=0.95,
        log_scaling=True,
        center_fraction=0.25,
        scale_domain=scale_domain,
    )
    ds = _build_normalizing_ds(tmp_path, transform=tfm)
    subject = ds[0]

    raw = _load_raw_target(tmp_path)
    served = _stacked_to_complex(subject["target"].data)
    restored = denormalize_kspace_robust(
        served, subject["kspace_scale"], log_scaling=True
    )

    assert torch.allclose(restored, raw, rtol=1e-3, atol=1e-4 * raw.abs().max()), (
        "k-space is not recoverable from the stored scale — normalization was "
        "applied more than once (dataset AND transform)."
    )


def test_dataset_does_not_morph_kspace_when_a_transform_owns_normalization(
    tmp_path,
) -> None:
    """The dataset matches and serves; the transform morphs.

    With a no-op transform attached, the served k-space must still be the RAW
    k-space: percentile-divide + log1p compression belong to the transform
    layer (canonical home ``data/transforms/``), not to ``__getitem__``.
    """
    import torchio as tio

    class _NoOp(tio.Transform):
        def apply_transform(self, subject):
            return subject

    ds = _build_normalizing_ds(tmp_path, transform=_NoOp())
    served = _stacked_to_complex(ds[0]["target"].data)
    raw = _load_raw_target(tmp_path)

    assert torch.allclose(
        served, raw, rtol=1e-4, atol=1e-6 * raw.abs().max()
    ), "dataset morphed the k-space; normalization must live in the transform"


def test_normalize_kspace_without_a_transform_raises(tmp_path) -> None:
    """No silent skip: the dataset cannot honour normalize_kspace on its own.

    Since the transform owns normalization, asking for it with nothing attached
    to apply it would quietly serve raw k-space (pitfall #9).
    """
    with pytest.raises(ValueError, match="normalize_kspace=True but no transform"):
        _build_normalizing_ds(tmp_path, transform=None)


def _load_raw_target(tmp_path) -> torch.Tensor:
    """The unnormalized target the dataset should be serving: mean over reps."""
    reps = []
    for rep in ("01", "02"):
        with h5py.File(str(tmp_path / f"2022_T1{rep}.h5"), "r") as f:
            reps.append(torch.from_numpy(f["kspace"][()]).to(torch.complex64))
    return _average_reps(reps, "complex_mean")


# ── the leave-one-out NEX gate must not decline silently (issue #695) ────────
#
# `nex_target_exclude_input` drops the input rep from the averaged target so
# target and input noise are uncorrelated. It needs >=3 reps: with 2, LOO would
# leave a single noisy rep and defeat the purpose, so it falls back to the
# all-reps average. The fallback is correct; being silent about it is not.
#
# M4Raw ships FLAIR at 2 reps and T1/T2 at 3, and `single_contrast=True` puts
# all of them in one run as independent samples. So a run that asked for an
# unbiased target gets one for some samples and not others, and the reported
# PSNR/SSIM is an average over two different references. Nothing said so.


def _build_loo_ds(tmp_path, *, n_reps: int, exclude_input: bool = True):
    """Single-contrast dataset with exactly ``n_reps`` repetitions."""
    files = []
    for i in range(1, n_reps + 1):
        p = tmp_path / f"2022_T1{i:02d}.h5"
        _write_structured_kspace_h5(p)
        files.append(p)
    return M4RawRepetitionDataset(
        h5_files=files,
        single_contrast=True,
        coil_processing_mode="none",
        use_repetitions=True,
        target_mode="phase_aligned_mean",
        nex_target_exclude_input=exclude_input,
    )


def test_loo_declining_on_two_reps_is_reported(tmp_path, caplog) -> None:
    """The whole point: the silent fallback now leaves evidence."""
    ds = _build_loo_ds(tmp_path, n_reps=2)
    with caplog.at_level(logging.WARNING, logger=m4raw_mod.logger.name):
        ds[0]
    assert any(
        "nex_target_exclude_input=True" in r.getMessage()
        and "leave-one-out needs >=3" in r.getMessage()
        for r in caplog.records
    ), f"no decline warning; saw {[r.getMessage() for r in caplog.records]}"


def test_loo_firing_on_three_reps_is_silent(tmp_path, caplog) -> None:
    """Anti-vacuity: the warning must be about the DECLINE, not about the flag.

    Without this, a warning emitted unconditionally would satisfy the test
    above while telling the reader nothing.
    """
    ds = _build_loo_ds(tmp_path, n_reps=3)
    with caplog.at_level(logging.WARNING, logger=m4raw_mod.logger.name):
        ds[0]
    assert not [
        r for r in caplog.records if "nex_target_exclude_input" in r.getMessage()
    ]


def test_no_warning_when_loo_was_never_requested(tmp_path, caplog) -> None:
    """A 2-rep arm that never asked for LOO is not doing anything surprising."""
    ds = _build_loo_ds(tmp_path, n_reps=2, exclude_input=False)
    with caplog.at_level(logging.WARNING, logger=m4raw_mod.logger.name):
        ds[0]
    assert not [
        r for r in caplog.records if "nex_target_exclude_input" in r.getMessage()
    ]


def test_the_decline_is_reported_once_per_rep_count(tmp_path, caplog) -> None:
    """Per-sample logging would spam a real run; the evidence must still land.

    Pins the de-duplication key as the rep COUNT, so a run holding both a 2-rep
    and a (hypothetical) 1-rep group reports both rather than only the first.
    """
    ds = _build_loo_ds(tmp_path, n_reps=2)
    with caplog.at_level(logging.WARNING, logger=m4raw_mod.logger.name):
        for _ in range(4):
            ds[0]
    hits = [r for r in caplog.records if "nex_target_exclude_input" in r.getMessage()]
    assert len(hits) == 1, f"expected one report, got {len(hits)}"
    assert ds._loo_declined_reported == {2}


class TestRepetitionLoadFailuresAreLoud:
    """B8 / B14. A NEX target is the AVERAGE of repetitions, so losing them
    quietly lowers the SNR boost from sqrt(N) toward sqrt(1) while the config
    still says denoising."""

    @staticmethod
    def _paths(tmp_path, n_ok, n_missing):
        from mriforge.data.datasets import m4raw_dataset as m

        paths = []
        for i in range(n_ok):
            p = tmp_path / f"ok_{i}.h5"
            p.write_bytes(b"")
            paths.append(p)
        for i in range(n_missing):
            paths.append(tmp_path / f"missing_{i}.h5")
        return paths, m

    def test_all_unreadable_raises_skipsample_not_none(self, tmp_path) -> None:
        """The single-contrast path used to `return None` "for collate to
        filter". `m4raw` selects ImageCollateStrategy, which has NO
        `_filter_none` (only Robust/Physics do), so the None died deep in
        collate as `TypeError: 'NoneType' object is not subscriptable`."""
        paths, m = self._paths(tmp_path, n_ok=0, n_missing=3)
        with pytest.raises(m._SkipSample, match="all 3 repetition files"):
            m._load_reps_or_skip(paths, "idx=0")

    def test_partial_failure_is_a_counted_warning_not_debug(
        self, tmp_path, monkeypatch, caplog
    ) -> None:
        """Losing reps was logged at DEBUG, i.e. invisible at default level."""
        import logging

        import torch

        from mriforge.data.datasets import m4raw_dataset as m

        paths, _ = self._paths(tmp_path, n_ok=2, n_missing=2)
        monkeypatch.setattr(m, "_load_kspace", lambda _p: torch.zeros(1, 4, 4))
        with caplog.at_level(logging.WARNING):
            reps = m._load_reps_or_skip(paths, "idx=7")
        assert len(reps) == 2
        assert "2 of 4 repetitions unreadable" in caplog.text
        # The consequence, in the units the arm cares about.
        assert "sqrt(2)=1.41" in caplog.text
        assert "sqrt(4)=2.00" in caplog.text

    def test_a_clean_group_logs_nothing(self, tmp_path, monkeypatch, caplog) -> None:
        import logging

        import torch

        from mriforge.data.datasets import m4raw_dataset as m

        paths, _ = self._paths(tmp_path, n_ok=3, n_missing=0)
        monkeypatch.setattr(m, "_load_kspace", lambda _p: torch.zeros(1, 4, 4))
        with caplog.at_level(logging.WARNING):
            assert len(m._load_reps_or_skip(paths, "idx=1")) == 3
        assert caplog.text == ""

    def test_the_failure_message_names_every_bad_file(self, tmp_path) -> None:
        """An unreadable-file error that does not name the file makes the
        operator grep a manifest by hand."""
        paths, m = self._paths(tmp_path, n_ok=0, n_missing=2)
        with pytest.raises(m._SkipSample) as exc:
            m._load_reps_or_skip(paths, "idx=0")
        for p in paths:
            assert p.name in str(exc.value)

    def test_both_paths_use_the_one_loader(self) -> None:
        """The single-contrast and cross-contrast paths had drifted onto two
        different failure protocols — `return None` and `_SkipSample`. Deriving
        both from one helper is what keeps them from drifting again."""
        import inspect

        from mriforge.data.datasets import m4raw_dataset as m

        src = inspect.getsource(m.M4RawRepetitionDataset)
        assert src.count("_load_reps_or_skip(") == 2
        assert "return None  # DataLoader collate" not in src


class TestSingleSurvivingRepIsRefused:
    """B8's expensive half: with one rep, `target = kspace_reps[0]` — the
    INPUT — so the arm trains the identity while the config says denoising."""

    def test_source_refuses_the_degenerate_group(self) -> None:
        import inspect

        from mriforge.data.datasets.m4raw_dataset import M4RawRepetitionDataset

        src = inspect.getsource(M4RawRepetitionDataset)
        assert "self.use_repetitions and len(kspace_reps) < 2" in src, (
            "the single-surviving-rep guard is gone; a NEX arm can silently "
            "train input->input again"
        )

    def test_the_message_offers_the_deliberate_opt_out(self) -> None:
        """An arm that genuinely wants input==target sets use_repetitions:false.
        The error must say so, or the reader's only move is to delete data."""
        import inspect

        from mriforge.data.datasets.m4raw_dataset import M4RawRepetitionDataset

        src = inspect.getsource(M4RawRepetitionDataset)
        assert "data.use_repetitions: false" in src


class TestRepetitionCountsDecideWhetherLOOIsEvenPossible:
    """Issue #1172: M4Raw ships 3 reps for T1/T2 and 2 for FLAIR.

    The module docstring claimed 6/6/4, contradicting the ``#695`` comment in
    ``__init__`` (which was right). The counts are not trivia — they decide
    whether ``nex_target_exclude_input`` can engage at all, so a reader who
    trusted the wrong figures would conclude that FLAIR's input/target noise
    correlation is a configuration choice. It is not.

    Encoded as behaviour rather than prose so the correction cannot rot back.
    """

    #: Ground truth, confirmed 2026-08-17.
    M4RAW_REPS: ClassVar[dict[str, int]] = {"T1": 3, "T2": 3, "FLAIR": 2}

    def test_loo_is_available_for_t1_and_t2_but_not_flair(self) -> None:
        """The partition the threshold actually draws."""
        eligible = {
            contrast: n >= _MIN_REPS_FOR_LOO for contrast, n in self.M4RAW_REPS.items()
        }
        assert eligible == {"T1": True, "T2": True, "FLAIR": False}

    def test_the_stale_6_6_4_figures_would_have_hidden_the_flair_case(self) -> None:
        """Why the wrong docstring mattered, not just that it was wrong.

        Under 6/6/4 every contrast clears the threshold, so the FLAIR
        impossibility is invisible — the reader is told to "flip it cohort-wide"
        for a knob that cannot engage.
        """
        stale = {"T1": 6, "T2": 6, "FLAIR": 4}
        assert all(n >= _MIN_REPS_FOR_LOO for n in stale.values())
        assert not all(n >= _MIN_REPS_FOR_LOO for n in self.M4RAW_REPS.values())

    def test_loo_at_the_flair_count_would_leave_a_single_rep(self) -> None:
        """The reason the threshold is 3 and not 2.

        At 2 reps, excluding the input leaves rep1 alone: not an average, no
        sqrt(N) gain, and a *noisier* reference than the all-reps mean. The gate
        declining is therefore correct behaviour, not a limitation.
        """
        reps = [torch.randn(2, 4, 4, dtype=torch.complex64) for _ in range(2)]
        loo = _average_reps(reps, "complex_mean", exclude_index=0)
        assert torch.allclose(loo, reps[1]), (
            "at 2 reps a leave-one-out 'average' is just the other rep"
        )
        all_reps = _average_reps(reps, "complex_mean")
        assert not torch.allclose(loo, all_reps)

    def test_nex_snr_gain_follows_the_real_counts(self) -> None:
        """sqrt(3) / sqrt(2), not the sqrt(4) = 2 the architecture doc claimed."""
        import math

        assert math.sqrt(self.M4RAW_REPS["T1"]) == pytest.approx(1.732, abs=1e-3)
        assert math.sqrt(self.M4RAW_REPS["FLAIR"]) == pytest.approx(1.414, abs=1e-3)
        assert math.sqrt(4) == 2.0  # the figure that was wrong, for contrast


def test_getitem_publishes_the_contrast_name_beside_the_index(tmp_path) -> None:
    """The subject carries ``contrast`` ("T1"), not only ``contrast_idx`` (0).

    The per-case CSV writer is a generic reporting component in
    ``infrastructure/reporting/``; giving it only the integer would force it to
    carry a copy of M4Raw's 0=T1/1=T2/2=FLAIR vocabulary, coupling a generic
    component to one dataset. The name is read off the index record rather than
    re-parsed from the filename — ``stem.split("_")`` is already spelled in
    ``_build_index`` and ``_build_federated_pairs`` and a third copy would
    drift (non-negotiable 17).
    """
    ds = _build_single_contrast_ds(tmp_path)
    subject = ds[0]
    assert subject["contrast"] == "T1"
    assert int(subject["contrast_idx"]) == 0


def test_published_contrast_is_a_plain_string_so_collation_keeps_it_per_sample(
    tmp_path,
) -> None:
    """A tensor would be stacked; a string survives as a per-sample list.

    ``ImageCollateStrategy`` routes non-tensor values through its
    ``collated[key] = items`` branch, which is what lets a batch report the
    contrast of each sample it holds instead of one stacked number.
    """
    from mriforge.data.collation.strategies import ImageCollateStrategy

    ds = _build_single_contrast_ds(tmp_path)
    subject = ds[0]
    assert isinstance(subject["contrast"], str)

    batch = ImageCollateStrategy().collate(
        [
            {"input": torch.zeros(1, 2, 2), "contrast": "T1", "file_id": "a"},
            {"input": torch.zeros(1, 2, 2), "contrast": "FLAIR", "file_id": "b"},
        ]
    )
    assert batch["contrast"] == ["T1", "FLAIR"]


class TestM4RawRepetitionsByContrast:
    """``M4RAW_REPETITIONS_BY_CONTRAST`` is the producer-owned SSOT for the
    3/3/2 fact (#1172).

    Before it existed the counts lived only in prose -- this module's docstring
    and the ``_MIN_REPS_FOR_LOO`` comment -- so no checker could consume them
    and ``model_kwargs.num_repetitions`` was validated against nothing. Four
    arms shipped an unsatisfiable ``4`` as a result (#1173).
    """

    def test_counts_match_the_documented_corpus(self) -> None:
        from mriforge.data.datasets.m4raw_dataset import (
            M4RAW_REPETITIONS_BY_CONTRAST,
        )

        assert M4RAW_REPETITIONS_BY_CONTRAST == {"T1": 3, "T2": 3, "FLAIR": 2}

    def test_pd_is_absent_rather_than_guessed(self) -> None:
        """The module docstring records PD as "variable", so no literal is
        correct for it. Absent is a state to report; a consumer must skip, not
        read a missing key as zero (non-negotiable 3)."""
        from mriforge.data.datasets.m4raw_dataset import (
            M4RAW_REPETITIONS_BY_CONTRAST,
        )

        assert "PD" not in M4RAW_REPETITIONS_BY_CONTRAST

    def test_it_is_not_contrast_map(self) -> None:
        """PLANT: the coincidence that would have made a wrong fixture look
        confirmed.

        ``CONTRAST_MAP`` is a keyword -> integer CLASS INDEX map. Its
        ``"FLAIR": 2`` entry agrees with FLAIR's true repetition count by pure
        collision, so a checker that reached for ``CONTRAST_MAP`` would look
        right on the one contrast a reviewer is most likely to spot-check --
        and would then report T1 as having 0 repetitions and T2 as having 1.

        If someone ever collapses these two maps into one, this test is what
        catches it.
        """
        from mriforge.data.datasets.m4raw_dataset import (
            CONTRAST_MAP,
            M4RAW_REPETITIONS_BY_CONTRAST,
        )

        assert CONTRAST_MAP != M4RAW_REPETITIONS_BY_CONTRAST
        # The collision that makes the confusion plausible:
        assert CONTRAST_MAP["FLAIR"] == M4RAW_REPETITIONS_BY_CONTRAST["FLAIR"] == 2
        # ...and the readings that expose it as a collision:
        assert CONTRAST_MAP["T1"] == 0 and M4RAW_REPETITIONS_BY_CONTRAST["T1"] == 3
        assert CONTRAST_MAP["T2"] == 1 and M4RAW_REPETITIONS_BY_CONTRAST["T2"] == 3

    def test_the_loo_gate_agrees_with_the_map(self) -> None:
        """``_MIN_REPS_FOR_LOO`` and this map are two statements of one fact and
        must not drift: LOO is available exactly for the contrasts whose count
        reaches the threshold (T1/T2), and structurally impossible for FLAIR."""
        from mriforge.data.datasets.m4raw_dataset import (
            _MIN_REPS_FOR_LOO,
            M4RAW_REPETITIONS_BY_CONTRAST,
        )

        eligible = {
            c for c, n in M4RAW_REPETITIONS_BY_CONTRAST.items() if n >= _MIN_REPS_FOR_LOO
        }
        assert eligible == {"T1", "T2"}
