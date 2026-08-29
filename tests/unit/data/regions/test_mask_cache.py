"""The region cache (format v2), and the QC gate that can declare the tissue axis unsound.

v2 stores **raw FreeSurfer label ids**. v1 stored uint8 tissue ordinals, which reduced the
segmentation at WRITE time: thalamus (10/49) and hippocampus (17/53) were both already the
single ordinal ``SUBCORTICAL_GM`` before the file hit disk, so no reader could recover a
per-structure region from a v1 cache -- and ``--parc`` ids run to 2035, which does not fit
a uint8 at all.

The dangerous part of that migration is not the new format, it is the OLD one: v1 ordinal 3
is CSF, FreeSurfer id 3 is cerebral CORTEX. A v1 array read as raw labels is not garbage,
it is a *plausible* label map with tissues transposed. So the guard cannot be a value check
-- it has to make the misread structurally impossible.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mriforge.data.regions.mask_cache import (  # noqa: E402
    CACHE_FORMAT_VERSION,
    LABELS_KEY,
    REPORT_FILENAME,
    RegionCacheFormatError,
    RegionCacheReport,
    RegionMaskCache,
    TissueAxisUnsoundError,
    VolumeQC,
    load_region_cache,
    write_volume_cache,
)
from mriforge.data.regions.synthseg_labels import TissueClass  # noqa: E402

# Real FreeSurfer ids, not ordinals.
FS_CORTEX, FS_WM, FS_HIPPOCAMPUS, FS_THALAMUS = 3, 2, 17, 10

AFFINE = np.eye(4)


def _qc(file_id: str, *, dropped: bool = False, score: float = 0.9) -> VolumeQC:
    return VolumeQC(
        file_id=file_id,
        min_structure_score=score,
        mean_structure_score=score,
        n_slices=3,
        dropped=dropped,
        reason="qc_floor" if dropped else None,
        worst_structure="hippocampus" if dropped else None,
    )


def _label_volume(s: int = 3, h: int = 8, w: int = 8) -> torch.Tensor:
    """Background, cortex, WM -- and two distinct SUBCORTICAL structures, which is the
    whole point: in v1 these two would have been the same number on disk."""
    vol = torch.zeros((s, h, w), dtype=torch.int64)
    vol[:, 2:6, 2:4] = FS_CORTEX
    vol[:, 2:6, 4:6] = FS_WM
    vol[:, 6:7, 2:4] = FS_HIPPOCAMPUS
    vol[:, 6:7, 4:6] = FS_THALAMUS
    return vol


def _write(tmp_path, file_id="v.h5", vol=None, **kw):
    write_volume_cache(
        tmp_path,
        file_id,
        _label_volume() if vol is None else vol,
        qc_scores=kw.pop("qc_scores", {"hippocampus": 0.9}),
        affine=kw.pop("affine", AFFINE),
    )


def _cache(tmp_path, file_id="v") -> RegionMaskCache:
    return RegionMaskCache(
        tmp_path,
        RegionCacheReport(
            cache_dir=tmp_path,
            qc_floor=0.65,
            max_drop_rate=0.2,
            volumes=(_qc(file_id),),
        ),
    )


class TestRoundTrip:
    def test_raw_label_ids_survive_the_round_trip(self, tmp_path) -> None:
        vol = _label_volume()
        _write(tmp_path, "file_brain_001.h5", vol)
        got = _cache(tmp_path, "file_brain_001").labels("file_brain_001", 1)
        assert torch.equal(got.to(torch.int64), vol[1])

    def test_tissue_ordinals_are_derived_on_read_not_stored(self, tmp_path) -> None:
        """The reduction now happens on the way OUT, so the structure identities are
        still on disk when someone wants them."""
        _write(tmp_path)
        ords = _cache(tmp_path).tissue_ordinals("v", 0)
        assert ords.dtype is torch.uint8
        assert int((ords == TissueClass.CORTICAL_GM.ordinal).sum()) == 8
        assert int((ords == TissueClass.CEREBRAL_WM.ordinal).sum()) == 8

    def test_regions_derive_from_the_label_map(self, tmp_path) -> None:
        """A NEW region costs one registry entry and zero cache regeneration."""
        _write(tmp_path)
        cache = _cache(tmp_path)
        gm = cache.region("v", 0, [TissueClass.CORTICAL_GM])
        wm = cache.region("v", 0, [TissueClass.CEREBRAL_WM])
        assert int(gm.sum()) == 8
        assert int(wm.sum()) == 8
        assert int(cache.brain("v", 0).sum()) == 8 + 8 + 2 + 2
        assert not bool((gm & wm).any())

    def test_a_slice_out_of_range_raises(self, tmp_path) -> None:
        _write(tmp_path)
        with pytest.raises(IndexError, match="out of range"):
            _cache(tmp_path).labels("v", 99)


class TestStructuresSurviveTheWrite:
    """The reason v2 exists. Everything here is IMPOSSIBLE against a v1 cache -- not
    hard, impossible: the two structures were the same byte on disk."""

    def test_hippocampus_and_thalamus_are_distinguishable(self, tmp_path) -> None:
        _write(tmp_path)
        cache = _cache(tmp_path)

        hippo = cache.structure("v", 0, [FS_HIPPOCAMPUS])
        thal = cache.structure("v", 0, [FS_THALAMUS])

        assert int(hippo.sum()) == 2
        assert int(thal.sum()) == 2
        assert not bool((hippo & thal).any())

    def test_both_still_collapse_into_subcortical_gm_when_asked(self, tmp_path) -> None:
        """v2 does not lose the coarse view -- it just stops FORCING it."""
        _write(tmp_path)
        cache = _cache(tmp_path)
        coarse = cache.region("v", 0, [TissueClass.SUBCORTICAL_GM])
        assert int(coarse.sum()) == 4

    def test_a_structure_folds_left_and_right(self, tmp_path) -> None:
        vol = torch.zeros((1, 4, 4), dtype=torch.int64)
        vol[0, 0, 0] = 17  # left hippocampus
        vol[0, 0, 1] = 53  # right hippocampus
        _write(tmp_path, vol=vol)
        assert int(_cache(tmp_path).structure("v", 0, [17, 53]).sum()) == 2


class TestAV1CacheCannotBeMisread:
    def test_a_v1_npz_raises_instead_of_being_reinterpreted(self, tmp_path) -> None:
        """The core migration hazard. A v1 file forced open as v2 would yield a
        plausible label map with CSF read as cortex -- confidently, silently wrong."""
        np.savez_compressed(
            tmp_path / "v.npz",
            tissue=np.zeros((3, 8, 8), dtype=np.uint8),  # the v1 key
            qc_keys=np.array(["s0"], dtype=object),
            qc_values=np.array([0.9], dtype=np.float32),
        )
        with pytest.raises(RegionCacheFormatError, match="v1 region cache"):
            _cache(tmp_path).labels("v", 0)

    def test_the_error_says_there_is_no_migration(self, tmp_path) -> None:
        """There cannot be one: the structure ids were destroyed at v1 write time."""
        np.savez_compressed(tmp_path / "v.npz", tissue=np.zeros((1, 2, 2), dtype=np.uint8))
        with pytest.raises(RegionCacheFormatError, match="no migration"):
            _cache(tmp_path).labels("v", 0)

    def test_a_future_format_version_raises(self, tmp_path) -> None:
        np.savez_compressed(
            tmp_path / "v.npz",
            **{LABELS_KEY: np.zeros((1, 2, 2), dtype=np.uint16)},
            format_version=np.int32(99),
        )
        with pytest.raises(RegionCacheFormatError, match="format v99"):
            _cache(tmp_path).labels("v", 0)

    def test_a_stale_report_is_caught_even_when_the_npz_is_v2(self, tmp_path) -> None:
        """The rename cannot catch a MIXED directory -- someone rebuilding half the
        volumes. The report is what says which volumes exist, so it carries the version
        too."""
        _write(tmp_path)
        stale = tmp_path / "r.json"
        stale.write_text(
            json.dumps(
                {
                    "cache_format_version": 1,
                    "cache_dir": str(tmp_path),
                    "qc_floor": 0.65,
                    "max_drop_rate": 0.2,
                    "volumes": [],
                }
            )
        )
        with pytest.raises(RegionCacheFormatError, match="v1 region cache"):
            RegionCacheReport.from_json(stale)

    def test_a_report_with_no_version_at_all_is_treated_as_v1(self, tmp_path) -> None:
        """Every report ever written before this change. Absence of the key is not
        'unknown, proceed' -- it is v1, which is exactly the thing that must not open.
        """
        stale = tmp_path / "r.json"
        stale.write_text(
            json.dumps(
                {
                    "cache_dir": str(tmp_path),
                    "qc_floor": 0.65,
                    "max_drop_rate": 0.2,
                    "volumes": [],
                }
            )
        )
        with pytest.raises(RegionCacheFormatError, match="v1 region cache"):
            RegionCacheReport.from_json(stale)


class TestTheWriterRefusesWhatItCannotStore:
    def test_a_float_label_map_is_refused(self, tmp_path) -> None:
        """A fractional label id means something interpolated the segmentation, and an
        interpolated label is a tissue that does not exist."""
        with pytest.raises(TypeError, match="integer tensor"):
            write_volume_cache(
                tmp_path,
                "v",
                torch.zeros(2, 4, 4, dtype=torch.float32),
                qc_scores={"s": 0.9},
                affine=AFFINE,
            )

    def test_an_undeclared_label_is_refused(self, tmp_path) -> None:
        from mriforge.data.regions.synthseg_labels import UnmappedSynthSegLabelError

        vol = torch.zeros(2, 4, 4, dtype=torch.int64)
        vol[0, 0, 0] = 9999
        with pytest.raises(UnmappedSynthSegLabelError):
            write_volume_cache(tmp_path, "v", vol, qc_scores={"s": 0.9}, affine=AFFINE)

    def test_the_affine_is_required_and_checked(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="4x4 affine"):
            write_volume_cache(
                tmp_path,
                "v",
                _label_volume(),
                qc_scores={"s": 0.9},
                affine=np.eye(3),
            )

    def test_labels_are_stored_as_uint16(self, tmp_path) -> None:
        """uint8 cannot hold a --parc id (they run to 2035). This is not a size
        optimisation, it is a representability requirement."""
        vol = torch.zeros(1, 4, 4, dtype=torch.int64)
        vol[0, 0, 0] = 2035  # right transverse temporal, DK
        write_volume_cache(tmp_path, "v", vol, qc_scores={"s": 0.9}, affine=AFFINE)

        with np.load(tmp_path / "v.npz", allow_pickle=True) as z:
            assert z[LABELS_KEY].dtype == np.uint16
            assert int(z["format_version"]) == CACHE_FORMAT_VERSION
            assert int(z["slice_axis"]) == 0
            assert z["affine"].shape == (4, 4)
        assert int(_cache(tmp_path).structure("v", 0, [2035]).sum()) == 1


class TestQCScoresAreNamedNotPositional:
    def test_the_structure_name_survives_into_the_cache(self, tmp_path) -> None:
        """`{"structure_0": ...}` threw away WHICH structure scored badly, so a
        per-structure QC gate had nothing to gate on."""
        _write(tmp_path, qc_scores={"left_hippocampus": 0.66, "right_putamen": 0.98})
        with np.load(tmp_path / "v.npz", allow_pickle=True) as z:
            keys = [str(k) for k in z["qc_keys"]]
        assert keys == ["left_hippocampus", "right_putamen"]

    def test_the_report_records_which_structure_was_worst(self, tmp_path) -> None:
        report = RegionCacheReport(
            cache_dir=tmp_path,
            qc_floor=0.65,
            max_drop_rate=0.2,
            volumes=(_qc("bad", dropped=True, score=0.2),),
        )
        row = report.to_dict()["volumes"][0]
        assert row["worst_structure"] == "hippocampus"
        assert (
            RegionCacheReport.from_json(report.write_json(tmp_path / "r.json"))
            .volumes[0]
            .worst_structure
            == "hippocampus"
        )


class TestDroppedVolumesAreUnreadable:
    def test_a_qc_dropped_volume_cannot_be_scored(self, tmp_path) -> None:
        """A bad label map does not give a bad-but-usable region -- it gives a
        region that is not the anatomy it claims to be."""
        cache = RegionMaskCache(
            tmp_path,
            RegionCacheReport(
                cache_dir=tmp_path,
                qc_floor=0.65,
                max_drop_rate=0.5,
                volumes=(_qc("bad", dropped=True, score=0.1),),
            ),
        )
        assert not cache.has("bad")
        with pytest.raises(KeyError, match="dropped by SynthSeg QC"):
            cache.labels("bad", 0)


class TestTheTissueAxisGate:
    def test_a_low_drop_rate_leaves_the_axis_sound(self) -> None:
        report = RegionCacheReport(
            cache_dir="/tmp",
            qc_floor=0.65,
            max_drop_rate=0.2,
            volumes=(*(_qc(f"v{i}") for i in range(9)), _qc("bad", dropped=True)),
        )
        assert report.drop_rate == pytest.approx(0.1)
        assert report.tissue_axis_sound
        report.raise_if_tissue_axis_unsound()  # must not raise

    def test_too_many_drops_declares_the_whole_axis_unsound(self) -> None:
        """Proceeding on the survivors is WORSE than not running it: they are the
        volumes SynthSeg found easy -- the population with the cleanest boundaries
        and the most flattering region effect."""
        report = RegionCacheReport(
            cache_dir="/tmp",
            qc_floor=0.65,
            max_drop_rate=0.2,
            volumes=tuple(_qc(f"v{i}") for i in range(5))
            + tuple(_qc(f"b{i}", dropped=True) for i in range(5)),
        )
        assert report.drop_rate == pytest.approx(0.5)
        assert not report.tissue_axis_sound
        with pytest.raises(TissueAxisUnsoundError, match="not a random sample"):
            report.raise_if_tissue_axis_unsound()

    def test_the_error_says_the_pathology_axis_survives(self) -> None:
        """fastMRI+ boxes do not depend on SynthSeg, so a tissue-axis failure must
        not take the whole study down with it."""
        report = RegionCacheReport(
            cache_dir="/tmp",
            qc_floor=0.65,
            max_drop_rate=0.2,
            volumes=(_qc("a", dropped=True), _qc("b")),
        )
        with pytest.raises(TissueAxisUnsoundError, match="pathology axis"):
            report.raise_if_tissue_axis_unsound()


class TestReportProvenance:
    def test_the_report_round_trips_through_json(self, tmp_path) -> None:
        report = RegionCacheReport(
            cache_dir=tmp_path,
            qc_floor=0.65,
            max_drop_rate=0.2,
            volumes=(_qc("a"), _qc("b", dropped=True, score=0.2)),
        )
        path = report.write_json(tmp_path / "qc.json")
        back = RegionCacheReport.from_json(path)
        assert back.n_dropped == 1
        assert back.drop_rate == report.drop_rate
        assert back.tissue_axis_sound == report.tissue_axis_sound

    def test_the_report_stamps_the_cache_format(self, tmp_path) -> None:
        report = RegionCacheReport(
            cache_dir=tmp_path, qc_floor=0.65, max_drop_rate=0.2, volumes=(_qc("a"),)
        )
        assert report.to_dict()["cache_format_version"] == CACHE_FORMAT_VERSION

    def test_every_drop_is_itemised(self, tmp_path) -> None:
        report = RegionCacheReport(
            cache_dir=tmp_path,
            qc_floor=0.65,
            max_drop_rate=0.2,
            volumes=(_qc("bad", dropped=True),),
        )
        rows = report.to_dict()["volumes"]
        assert rows[0]["dropped"] is True
        assert rows[0]["reason"] == "qc_floor"


class TestLoadRegionCache:
    """The read side had no entry point at all: `RegionMaskCache` needs a
    `RegionCacheReport` that only the BUILDER held, so a sweep could not open a cache
    from a directory. `load_region_cache` is that entry point."""

    def test_it_opens_a_cache_the_builder_wrote(self, tmp_path) -> None:
        _write(tmp_path, "vol_a")
        RegionCacheReport(
            cache_dir=tmp_path,
            qc_floor=0.65,
            max_drop_rate=0.2,
            volumes=(VolumeQC("vol_a", 0.9, 0.9, 3, dropped=False),),
        ).write_json(tmp_path / REPORT_FILENAME)

        cache = load_region_cache(tmp_path)
        assert cache.has("vol_a")
        assert cache.region("vol_a", 0, [TissueClass.CEREBRAL_WM]).sum().item() == 8

    def test_a_cache_without_its_qc_report_will_not_open(self, tmp_path) -> None:
        """Not a convenience check. Without the report there is no record of which
        volumes SynthSeg REJECTED, so the sweep would serve exactly the masks whose
        segmentation is untrustworthy -- and they are not a random sample."""
        _write(tmp_path, "vol_a")
        with pytest.raises(FileNotFoundError, match=re.escape("report.json")):
            load_region_cache(tmp_path)
