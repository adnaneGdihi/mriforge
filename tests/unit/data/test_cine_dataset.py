"""Phase 4c — CineMRIDataset.

Synthetic 4D cardiac-phantom fixtures avoid the need for ACDC. We
exercise NIfTI (when nibabel is available) and ``.pt`` as a portable
alternative.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchio as tio

from spectramr.config.schemas.data import TemporalConfigSchema
from spectramr.data.datasets.cine_dataset import (
    _VOLUME_SUFFIXES,
    CineMRIDataset,
    _frame_axis_to_channel,
    _volume_stem,
    build_cine_index,
)


def _make_synthetic_4d(
    tmp_path: Path,
    n_volumes: int = 2,
    shape: tuple[int, int, int, int] = (8, 8, 4, 12),
) -> Path:
    """Write n synthetic 4D .pt files under tmp_path matching the cine glob."""
    root = tmp_path / "cine"
    root.mkdir()
    for i in range(n_volumes):
        subj_dir = root / f"subj{i:03d}"
        subj_dir.mkdir()
        tensor = torch.randn(*shape)
        # Use .nii.gz extension carried by .pt content; the loader dispatches
        # on the actual extension so use .pt to skip nibabel.
        torch.save(tensor, str(subj_dir / f"subj{i:03d}_4d.pt"))
    return root


# ── Permute helper ──────────────────────────────────────────────────────────


def test_frame_axis_to_channel_moves_frame_to_dim_0() -> None:
    """frame_axis=3 → (H, W, D, F) → (F, H, W, D)."""
    x = torch.randn(8, 9, 4, 12)  # H, W, D, F
    out = _frame_axis_to_channel(x, frame_axis=3)
    assert out.shape == (12, 8, 9, 4)


def test_frame_axis_to_channel_handles_frame_axis_0() -> None:
    """frame_axis=0 → already in channel slot, no permute side effects."""
    x = torch.randn(12, 8, 9, 4)
    out = _frame_axis_to_channel(x, frame_axis=0)
    assert out.shape == (12, 8, 9, 4)


def test_frame_axis_to_channel_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="frame_axis"):
        _frame_axis_to_channel(torch.randn(2, 2, 2, 2), frame_axis=4)


# ── Dataset behavior ────────────────────────────────────────────────────────


def test_cine_dataset_rejects_disabled_config() -> None:
    cfg = TemporalConfigSchema()  # disabled
    with pytest.raises(ValueError, match="enabled=True"):
        CineMRIDataset(index=[], temporal_config=cfg)


def test_cine_dataset_yields_subject_with_frame_metadata(tmp_path: Path) -> None:
    """Per-Subject output carries n_frames, frame_order, file_id, etc."""
    root = _make_synthetic_4d(tmp_path, n_volumes=1)
    index = [
        {
            "path": str(next(root.rglob("*_4d.pt"))),
            "subject_id": "subj000",
            "file_id": "subj000_4d",
        }
    ]
    cfg = TemporalConfigSchema(
        enabled=True, frame_axis=3, frames_per_window=4, target_source="self"
    )
    ds = CineMRIDataset(index=index, temporal_config=cfg)
    subj = ds[0]
    assert subj["n_frames"] == 12
    assert subj["frame_order"] == list(range(12))
    assert isinstance(subj["input"], tio.ScalarImage)
    assert subj["input"].data.shape[0] == 12  # frames in channel slot
    assert subj["subject_id"] == "subj000"


def test_cine_dataset_validates_total_frames_when_set(tmp_path: Path) -> None:
    """temporal.total_frames mismatch ⇒ ValueError at load."""
    root = _make_synthetic_4d(tmp_path, n_volumes=1)
    index = [{"path": str(next(root.rglob("*_4d.pt")))}]
    cfg = TemporalConfigSchema(
        enabled=True,
        frame_axis=3,
        total_frames=99,
        target_source="self",  # actual is 12
    )
    ds = CineMRIDataset(index=index, temporal_config=cfg)
    with pytest.raises(ValueError, match="frames along"):
        _ = ds[0]


def test_cine_dataset_pairs_target_when_provided(tmp_path: Path) -> None:
    """target_path on record ⇒ Subject carries both 'input' and 'target'."""
    root = _make_synthetic_4d(tmp_path, n_volumes=1)
    input_path = next(root.rglob("*_4d.pt"))
    target_path = input_path.parent / "subj000_target.pt"
    torch.save(torch.randn(8, 8, 4, 12), str(target_path))

    cfg = TemporalConfigSchema(
        enabled=True, frame_axis=3, target_source="sibling", target_suffix="_target.pt"
    )
    ds = CineMRIDataset(
        index=[{"path": str(input_path), "target_path": str(target_path)}],
        temporal_config=cfg,
    )
    subj = ds[0]
    assert "target" in subj
    assert isinstance(subj["target"], tio.ScalarImage)


def test_cine_dataset_dry_iter_returns_subject_list(tmp_path: Path) -> None:
    root = _make_synthetic_4d(tmp_path, n_volumes=3)
    index = [{"path": str(p)} for p in root.rglob("*_4d.pt")]
    cfg = TemporalConfigSchema(enabled=True, frame_axis=3, target_source="self")
    ds = CineMRIDataset(index=index, temporal_config=cfg)
    stubs = ds.dry_iter()
    assert isinstance(stubs, list)
    assert len(stubs) == len(index)
    assert all(isinstance(s, tio.Subject) for s in stubs)


# ── Index building ──────────────────────────────────────────────────────────


def test_build_cine_index_globs_correctly(tmp_path: Path) -> None:
    """The default glob matches '*_4d.nii.gz'."""
    root = tmp_path / "cine"
    root.mkdir()
    for sub in ("subj-001", "subj-002"):
        d = root / sub
        d.mkdir()
        (d / f"{sub}_4d.nii.gz").write_bytes(b"placeholder")
    records = build_cine_index(root)
    assert len(records) == 2
    assert all("path" in r and r["path"].endswith("_4d.nii.gz") for r in records)


def test_build_cine_index_returns_empty_on_no_match(tmp_path: Path) -> None:
    """No matches ⇒ empty list, no crash. Caller decides what to do."""
    records = build_cine_index(tmp_path)
    assert records == []


# ── Target pairing (the reason cine could not reach a strategy) ──────────────


def test_self_pairing_emits_a_target_that_is_a_clone_not_an_alias(
    tmp_path: Path,
) -> None:
    """``target_source='self'`` yields an equal-but-independent target.

    Equality alone would also hold for ``target = input`` (the same object),
    and a transform that corrupts ``input`` in place would then corrupt the
    target too -- silently making the pair trivial again.
    """
    root = _make_synthetic_4d(tmp_path, n_volumes=1)
    cfg = TemporalConfigSchema(enabled=True, frame_axis=3, target_source="self")
    ds = CineMRIDataset(
        index=[{"path": str(next(root.rglob("*_4d.pt")))}], temporal_config=cfg
    )
    subj = ds[0]
    inp, tgt = subj["input"].data, subj["target"].data
    assert torch.equal(inp, tgt)
    assert tgt.data_ptr() != inp.data_ptr()
    tgt[0, 0, 0, 0] += 1.0
    assert not torch.equal(inp, tgt)


def test_sibling_pairing_without_a_target_path_raises(tmp_path: Path) -> None:
    """A hand-built index that declares sibling pairing but carries no path.

    ``build_cine_index`` raises first, so this is the defence for any other
    index producer -- the alternative is a missing ``target`` key surfacing
    much later as ``BatchAdapter requires canonical keys``.
    """
    root = _make_synthetic_4d(tmp_path, n_volumes=1)
    cfg = TemporalConfigSchema(
        enabled=True, frame_axis=3, target_source="sibling", target_suffix="_gt.pt"
    )
    ds = CineMRIDataset(
        index=[{"path": str(next(root.rglob("*_4d.pt")))}], temporal_config=cfg
    )
    with pytest.raises(ValueError, match="no 'target_path'"):
        _ = ds[0]


@pytest.mark.parametrize("target_source", ["self", "sibling"])
def test_dry_iter_key_set_matches_getitem(tmp_path: Path, target_source: str) -> None:
    """The Queue sizes itself against ``dry_iter``; a differing key set lies.

    Asserted as an invariant over both pairing modes rather than as one
    example, because the previous ``dry_iter`` attached ``target`` only when
    the record happened to carry ``target_path``.
    """
    root = _make_synthetic_4d(tmp_path, n_volumes=1)
    input_path = next(root.rglob("*_4d.pt"))
    record: dict = {"path": str(input_path)}
    kwargs: dict = {"target_source": target_source}
    if target_source == "sibling":
        target_path = input_path.parent / "sibling_gt.pt"
        torch.save(torch.randn(8, 8, 4, 12), str(target_path))
        record["target_path"] = str(target_path)
        kwargs["target_suffix"] = "_gt.pt"

    cfg = TemporalConfigSchema(enabled=True, frame_axis=3, **kwargs)
    ds = CineMRIDataset(index=[record], temporal_config=cfg)
    assert set(ds.dry_iter()[0].keys()) == set(ds[0].keys())


def test_build_cine_index_reports_every_missing_sibling_at_once(
    tmp_path: Path,
) -> None:
    """One raise naming all gaps, not one run per missing file."""
    root = tmp_path / "cine"
    root.mkdir()
    for i in range(3):
        d = root / f"subj{i:03d}"
        d.mkdir()
        (d / f"subj{i:03d}_4d.nii.gz").write_bytes(b"placeholder")
    (root / "subj000" / "subj000_4d_gt.nii.gz").write_bytes(b"placeholder")

    with pytest.raises(FileNotFoundError) as exc:
        build_cine_index(root, target_suffix="_gt.nii.gz")
    message = str(exc.value)
    assert "2 of 3" in message
    assert "subj001_4d.nii.gz" in message and "subj002_4d.nii.gz" in message
    assert "subj000" not in message  # the one that IS paired is not reported


@pytest.mark.parametrize("suffix", _VOLUME_SUFFIXES)
def test_sibling_name_never_resolves_to_the_input_itself(
    tmp_path: Path, suffix: str
) -> None:
    """Parametrised over the whole loadable vocabulary, not just NIfTI.

    The rule was ``name.replace(".nii.gz", target_suffix)``, a no-op on every
    other extension -- and a no-op makes ``target_path == path``, pairing a
    volume with itself. Unreachable only while ``glob_pattern`` was hardcoded.
    """
    root = tmp_path / "cine"
    (root / "s0").mkdir(parents=True)
    stem = "s0_4d"
    (root / "s0" / f"{stem}{suffix}").write_bytes(b"placeholder")
    (root / "s0" / f"{stem}_gt{suffix}").write_bytes(b"placeholder")

    records = build_cine_index(
        root, glob_pattern=f"**/{stem}{suffix}", target_suffix=f"_gt{suffix}"
    )
    assert len(records) == 1
    assert records[0]["target_path"] != records[0]["path"]
    assert records[0]["target_path"].endswith(f"_gt{suffix}")


def test_volume_stem_and_loader_share_one_extension_vocabulary() -> None:
    """Every suffix the stem rule strips is one ``_load_4d_volume`` accepts.

    Two lists would let the index pair a file the loader then refuses.
    """
    for suffix in _VOLUME_SUFFIXES:
        assert _volume_stem(Path(f"vol{suffix}")) == "vol"
    with pytest.raises(ValueError, match="ends with none of"):
        _volume_stem(Path("vol.mha"))
