"""Unit tests for the quantitative-map manifest loader.

Pairs with the 2026-06-21 ``load_quantitative_manifest`` addition that lets the
``quantitative`` dataset_type consume a committed-generator manifest (non-BIDS
corpora — the cluster NIST-MRF parameter maps) instead of the BIDS globber. The
loader must fail loud on an incomplete manifest (CLAUDE.md #9) rather than
silently yield half subjects.
"""

from __future__ import annotations

import json

import pytest

from spectramr.data.datasets.quantitative_dataset import load_quantitative_manifest


def _good_records():
    return [
        {
            "subject_id": "Subj_00#0",
            "file_id": "Subj_00#0",
            "input_paths": ["/d/T1.nii.gz", "/d/T2.nii.gz", "/d/M0.nii.gz"],
            "map_paths": {"t1": "/d/T1.nii.gz", "t2": "/d/T2.nii.gz", "m0": "/d/M0.nii.gz"},
        }
    ]


def _write(tmp_path, obj):
    p = tmp_path / "nist_mrf_qmaps.json"
    p.write_text(json.dumps(obj))
    return p


def test_loads_valid_manifest(tmp_path):
    p = _write(tmp_path, _good_records())
    recs = load_quantitative_manifest(p, ["t1", "t2", "m0"])
    assert len(recs) == 1
    assert recs[0]["map_paths"]["t1"].endswith("T1.nii.gz")


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        load_quantitative_manifest(tmp_path / "nope.json", ["t1"])


def test_non_list_manifest_raises(tmp_path):
    p = _write(tmp_path, {"not": "a list"})
    with pytest.raises(ValueError, match="must be a JSON list"):
        load_quantitative_manifest(p, ["t1"])


def test_empty_input_paths_fails_loud(tmp_path):
    bad = _good_records()
    bad[0]["input_paths"] = []
    p = _write(tmp_path, bad)
    with pytest.raises(ValueError, match="empty 'input_paths'"):
        load_quantitative_manifest(p, ["t1"])


def test_missing_requested_map_fails_loud(tmp_path):
    # manifest has t1/t2 but caller wants t1/t2/m0
    rec = _good_records()
    rec[0]["map_paths"] = {"t1": "/d/T1.nii.gz", "t2": "/d/T2.nii.gz"}
    rec[0]["input_paths"] = ["/d/T1.nii.gz"]
    p = _write(tmp_path, rec)
    with pytest.raises(ValueError, match="missing requested map"):
        load_quantitative_manifest(p, ["t1", "t2", "m0"])


def test_load_volume_reads_dicom_directory(tmp_path, monkeypatch):
    """A DICOM-stack *directory* (NIST-MRF .IMA maps) routes through
    DicomStrategy and yields a [1, X, Y, Z] tensor."""
    torch = pytest.importorskip("torch")
    from spectramr.data.datasets import quantitative_dataset as qd

    d = tmp_path / "MRF_7SLICE_5MM_T1_MAP_MASKED_0011"
    d.mkdir()
    (d / "s00.IMA").write_text("x")  # presence only; loader is mocked

    class _FakeDicom:
        def load(self, path, metadata=None):
            return {"data": torch.zeros(7, 64, 64)}  # 7-slice volume

    monkeypatch.setattr(qd, "_load_volume", qd._load_volume)  # ensure real fn
    monkeypatch.setattr(
        "spectramr.data.io_strategies.DicomStrategy", lambda: _FakeDicom()
    )
    vol = qd._load_volume(d)
    assert vol.dim() == 4 and vol.shape[0] == 1  # [1, X, Y, Z]


# ---------------------------------------------------------------------------
# Serving path: the `target` key, and the declared input_source (2026-08-05).
#
# Before this, QuantitativeMapDataset emitted `input` plus the named maps and
# no `target`, so BatchAdapter.from_dict rejected every batch and
# `dataset_type: quantitative` could not serve a single step. These pin the
# emitted key set and the pairing declaration that keeps a maps-style manifest
# from silently becoming an identity task.
# ---------------------------------------------------------------------------

MAPS = ["pd", "t1", "t2"]


def _corpus(tmp_path, *, self_paired: bool):
    """Write per-map .npy volumes; return an index of one record."""
    np = pytest.importorskip("numpy")
    map_paths = {}
    for m in MAPS:
        p = tmp_path / f"subj0_{m}.npy"
        np.save(p, np.ones((4, 4, 2), dtype="float32"))
        map_paths[m] = str(p)
    if self_paired:
        input_paths = [map_paths[m] for m in MAPS]
    else:
        input_paths = []
        for c in ("T1w", "T2w"):
            p = tmp_path / f"subj0_{c}.npy"
            np.save(p, np.zeros((4, 4, 2), dtype="float32"))
            input_paths.append(str(p))
    return [
        {
            "subject_id": "Subj_00#0",
            "file_id": "Subj_00#0",
            "input_paths": input_paths,
            "map_paths": map_paths,
        }
    ]


def _cfg(**over):
    from spectramr.config.schemas.data import QuantitativeConfigSchema

    base = {"enabled": True, "target_maps": MAPS, "input_source": "maps"}
    base.update(over)
    return QuantitativeConfigSchema(**base)


def _dataset(tmp_path, *, self_paired=True, **over):
    from spectramr.data.datasets.quantitative_dataset import QuantitativeMapDataset

    return QuantitativeMapDataset(
        index=_corpus(tmp_path, self_paired=self_paired),
        quantitative_config=_cfg(**over),
    )


class TestTargetIsEmitted:
    def test_subject_carries_the_canonical_target(self, tmp_path):
        pytest.importorskip("torchio")
        subject = _dataset(tmp_path)[0]
        assert "target" in subject, (
            "no 'target' key -> BatchAdapter.from_dict rejects the batch and "
            "dataset_type='quantitative' cannot serve a step"
        )

    def test_target_stacks_the_declared_maps_in_declaration_order(self, tmp_path):
        """Order is load-bearing: BlochRelaxationManifold reads channel 0 as M0."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("torchio")
        np = pytest.importorskip("numpy")
        index = _corpus(tmp_path, self_paired=True)
        # Give each map a distinct constant so the stacking order is visible.
        for value, m in enumerate(MAPS, start=1):
            np.save(index[0]["map_paths"][m], np.full((4, 4, 2), float(value), "float32"))
        from spectramr.data.datasets.quantitative_dataset import QuantitativeMapDataset

        subject = QuantitativeMapDataset(index=index, quantitative_config=_cfg())[0]
        target = subject["target"].data
        assert target.shape[0] == len(MAPS)
        for channel, _ in enumerate(MAPS):
            assert torch.allclose(
                target[channel], torch.full_like(target[channel], float(channel + 1))
            ), f"channel {channel} is not target_maps[{channel}]"

    def test_dry_iter_key_set_matches_getitem(self, tmp_path):
        """The Queue sizes itself against dry_iter; a key missing there is a key
        the patch sampler never learns to carry."""
        pytest.importorskip("torchio")
        ds = _dataset(tmp_path)
        assert set(ds.dry_iter()[0].keys()) == set(ds[0].keys())

    def test_named_map_keys_survive_alongside_target(self, tmp_path):
        """core/metrics/nr_cross.py reads qmaps by name; target is additive."""
        pytest.importorskip("torchio")
        subject = _dataset(tmp_path)[0]
        assert {"t1", "t2", "pd"}.issubset(set(subject.keys()))


class TestInputSourceIsDeclaredAndEnforced:
    def test_maps_declaration_accepts_a_maps_manifest(self, tmp_path):
        pytest.importorskip("torchio")
        assert len(_dataset(tmp_path, self_paired=True, input_source="maps")) == 1

    def test_contrasts_declaration_accepts_a_contrast_manifest(self, tmp_path):
        pytest.importorskip("torchio")
        assert len(_dataset(tmp_path, self_paired=False, input_source="contrasts")) == 1

    def test_contrasts_declaration_rejects_a_maps_manifest(self, tmp_path):
        """The identity trap: build_nist_mrf_manifest.py writes input_paths ==
        the maps, so a 'contrasts' arm fed that manifest would train input->input."""
        pytest.importorskip("torchio")
        with pytest.raises(ValueError, match="train the identity"):
            _dataset(tmp_path, self_paired=True, input_source="contrasts")

    def test_maps_declaration_rejects_a_contrast_manifest(self, tmp_path):
        pytest.importorskip("torchio")
        with pytest.raises(ValueError, match="input_paths are something else"):
            _dataset(tmp_path, self_paired=False, input_source="maps")

    def test_input_and_target_are_equal_only_under_the_maps_declaration(self, tmp_path):
        torch = pytest.importorskip("torch")
        pytest.importorskip("torchio")
        paired = _dataset(tmp_path, self_paired=True, input_source="maps")[0]
        assert torch.equal(paired["input"].data, paired["target"].data)
        split = _dataset(tmp_path, self_paired=False, input_source="contrasts")[0]
        assert not torch.equal(
            split["input"].data, split["target"].data[: split["input"].data.shape[0]]
        )


class TestSelfPairingIsOrderInsensitive:
    """``input_paths`` order and ``target_maps`` order are independent facts.

    ``build_nist_mrf_manifest.py`` orders ``input_paths`` by the script's
    ``--target-maps`` argument (default ``[t1, t2, pd]``), while an arm orders
    ``target_maps`` to match its consumer — ``[pd, t1, t2]`` for the Bloch
    manifold's (M0, T1, T2) coordinates. An ordered comparison made the one
    live arm unconstructible on the cluster, so the pairing check is a multiset
    compare: are the inputs the same *files* as the maps.
    """

    def _permuted_index(self, tmp_path):
        """Self-paired, but with input_paths in a different order to MAPS."""
        index = _corpus(tmp_path, self_paired=True)
        index[0]["input_paths"] = list(reversed(index[0]["input_paths"]))
        return index

    def test_permuted_self_pairing_is_still_recognised(self, tmp_path):
        from spectramr.data.datasets.quantitative_dataset import _input_is_the_maps

        assert _input_is_the_maps(self._permuted_index(tmp_path)[0], MAPS)

    def test_permuted_maps_manifest_constructs(self, tmp_path):
        pytest.importorskip("torchio")
        from spectramr.data.datasets.quantitative_dataset import QuantitativeMapDataset

        ds = QuantitativeMapDataset(
            index=self._permuted_index(tmp_path),
            quantitative_config=_cfg(input_source="maps"),
        )
        assert len(ds) == 1

    def test_permuted_maps_manifest_is_still_rejected_under_contrasts(self, tmp_path):
        pytest.importorskip("torchio")
        from spectramr.data.datasets.quantitative_dataset import QuantitativeMapDataset

        with pytest.raises(ValueError, match="train the identity"):
            QuantitativeMapDataset(
                index=self._permuted_index(tmp_path),
                quantitative_config=_cfg(input_source="contrasts"),
            )

    def test_target_channel_order_still_follows_target_maps_not_input_paths(
        self, tmp_path
    ):
        """Relaxing the pairing check must not relax the stacking order."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("torchio")
        np = pytest.importorskip("numpy")
        from spectramr.data.datasets.quantitative_dataset import QuantitativeMapDataset

        index = self._permuted_index(tmp_path)
        for value, m in enumerate(MAPS, start=1):
            np.save(index[0]["map_paths"][m], np.full((4, 4, 2), float(value), "float32"))
        subject = QuantitativeMapDataset(
            index=index, quantitative_config=_cfg(input_source="maps")
        )[0]
        target = subject["target"].data
        for channel, _ in enumerate(MAPS):
            assert torch.allclose(
                target[channel], torch.full_like(target[channel], float(channel + 1))
            ), f"target channel {channel} no longer follows target_maps"
