"""Phase 4b-extra — LoadDWIMetadata transform."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
import torchio as tio

from spectramr.data.transforms import dwi_metadata
from spectramr.data.transforms.dwi_metadata import (
    LoadDWIMetadata,
    parse_bval_file,
    parse_bvec_file,
)


def test_docstring_names_the_real_b_value_consumer() -> None:
    """F3/I2: the transform must not claim a non-existent consumer. The
    b_values are consumed by the (now real, registered) monoexp ADC loss."""
    doc = dwi_metadata.__doc__ or ""
    assert "dwi_adc_monoexp" in doc
    # And it must be honest that no live arm exercises the path yet.
    assert "roadmap" in doc.lower()


# ── File parsers ────────────────────────────────────────────────────────────


def test_parse_bval_single_line(tmp_path: Path) -> None:
    p = tmp_path / "dwi.bval"
    p.write_text("0 1000 1000 2000 2000\n")
    assert parse_bval_file(p) == [0.0, 1000.0, 1000.0, 2000.0, 2000.0]


def test_parse_bval_multi_line(tmp_path: Path) -> None:
    p = tmp_path / "dwi.bval"
    p.write_text("0 1000\n1000 2000 2000\n")
    assert parse_bval_file(p) == [0.0, 1000.0, 1000.0, 2000.0, 2000.0]


def test_parse_bvec_3_rows(tmp_path: Path) -> None:
    p = tmp_path / "dwi.bvec"
    p.write_text("0 1 0\n0 0 1\n0 0 0\n")
    vecs = parse_bvec_file(p)
    assert vecs == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_parse_bvec_rejects_wrong_row_count(tmp_path: Path) -> None:
    p = tmp_path / "bad.bvec"
    p.write_text("0 1\n0 0\n")  # only 2 rows
    with pytest.raises(ValueError, match="3 rows"):
        parse_bvec_file(p)


def test_parse_bvec_rejects_ragged_columns(tmp_path: Path) -> None:
    p = tmp_path / "ragged.bvec"
    p.write_text("0 1 0\n0 0\n0 0 0\n")  # row 2 short
    with pytest.raises(ValueError, match="inconsistent"):
        parse_bvec_file(p)


# ── Transform integration ──────────────────────────────────────────────────


def _make_dwi_subject_with_sidecars(tmp_path: Path) -> tio.Subject:
    """Write a NIfTI + .bval/.bvec sidecars and return the Subject."""
    try:
        import nibabel as nib
        import numpy as np
    except ImportError:
        pytest.skip("nibabel not installed")

    img_path = tmp_path / "sub-001_dwi.nii.gz"
    nib.save(  # type: ignore[no-untyped-call]
        nib.Nifti1Image(np.zeros((4, 4, 2, 5), dtype=np.float32), affine=np.eye(4)),
        str(img_path),
    )
    (tmp_path / "sub-001_dwi.bval").write_text("0 1000 1000 2000 2000\n")
    (tmp_path / "sub-001_dwi.bvec").write_text("0 1 0 1 0\n0 0 1 0 1\n0 0 0 0 0\n")
    return tio.Subject(input=tio.ScalarImage(str(img_path)))


def test_dwi_loader_attaches_b_values_and_vectors_as_tensors(tmp_path: Path) -> None:
    subj = _make_dwi_subject_with_sidecars(tmp_path)
    out = LoadDWIMetadata()(subj)
    assert out["n_directions"] == 5
    # Attached as tensors (not Python lists) so the default collate stacks them and
    # they reach QSpaceDiffusionStrategy / the ADC loss as tensors (#350).
    bvals, bvecs = out["b_values"], out["b_vectors"]
    assert isinstance(bvals, torch.Tensor) and bvals.shape == (5,)
    assert torch.equal(bvals, torch.tensor([0.0, 1000.0, 1000.0, 2000.0, 2000.0]))
    assert isinstance(bvecs, torch.Tensor) and bvecs.shape == (5, 3)
    assert torch.equal(bvecs[1], torch.tensor([1.0, 0.0, 0.0]))


def test_dwi_loader_strict_raises_on_missing_sidecars(tmp_path: Path) -> None:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError:
        pytest.skip("nibabel not installed")
    img_path = tmp_path / "sub-002_dwi.nii.gz"
    nib.save(  # type: ignore[no-untyped-call]
        nib.Nifti1Image(np.zeros((4, 4, 2, 1), dtype=np.float32), affine=np.eye(4)),
        str(img_path),
    )
    # No .bval or .bvec written.
    subj = tio.Subject(input=tio.ScalarImage(str(img_path)))
    with pytest.raises(FileNotFoundError, match="DWI sidecars"):
        LoadDWIMetadata(strict=True)(subj)


def test_dwi_loader_lenient_skips_when_sidecars_missing(tmp_path: Path) -> None:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError:
        pytest.skip("nibabel not installed")
    img_path = tmp_path / "sub-003_dwi.nii.gz"
    nib.save(  # type: ignore[no-untyped-call]
        nib.Nifti1Image(np.zeros((4, 4, 2, 1), dtype=np.float32), affine=np.eye(4)),
        str(img_path),
    )
    subj = tio.Subject(input=tio.ScalarImage(str(img_path)))
    out = LoadDWIMetadata(strict=False)(subj)
    assert "b_values" not in out


def test_dwi_loader_no_path_returns_unchanged() -> None:
    """No resolvable source ⇒ lenient mode attaches nothing and does not crash.

    It now also WARNS; see
    ``test_lenient_mode_warns_rather_than_returning_in_silence``.
    """
    subj = tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 2, 2, 1)))
    out = LoadDWIMetadata()(subj)
    assert "b_values" not in out


def test_dwi_loader_strict_raises_on_inconsistent_lengths(tmp_path: Path) -> None:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError:
        pytest.skip("nibabel not installed")
    img_path = tmp_path / "sub-004_dwi.nii.gz"
    nib.save(  # type: ignore[no-untyped-call]
        nib.Nifti1Image(np.zeros((4, 4, 2, 5), dtype=np.float32), affine=np.eye(4)),
        str(img_path),
    )
    (tmp_path / "sub-004_dwi.bval").write_text("0 1000 1000\n")  # 3 b-values
    (tmp_path / "sub-004_dwi.bvec").write_text(
        "0 1 0 1 0\n0 0 1 0 1\n0 0 0 0 0\n"  # 5 directions
    )
    subj = tio.Subject(input=tio.ScalarImage(str(img_path)))
    with pytest.raises(ValueError, match="inconsistent"):
        LoadDWIMetadata(strict=True)(subj)


# ── Reachability: the transform could not see any tensor-backed Subject ─────


def _write_dwi_files(tmp_path: Path, stem: str = "sub-001_dwi") -> Path:
    """Write a 4-D NIfTI plus its .bval/.bvec siblings; return the image path."""
    try:
        import nibabel as nib
        import numpy as np
    except ImportError:
        pytest.skip("nibabel not installed")

    img_path = tmp_path / f"{stem}.nii.gz"
    nib.save(  # type: ignore[no-untyped-call]
        nib.Nifti1Image(np.zeros((4, 4, 2, 5), dtype=np.float32), affine=np.eye(4)),
        str(img_path),
    )
    (tmp_path / f"{stem}.bval").write_text("0 1000 1000 2000 2000\n")
    (tmp_path / f"{stem}.bvec").write_text("0 1 0 1 0\n0 0 1 0 1\n0 0 0 0 0\n")
    return img_path


@pytest.mark.parametrize("path_key", dwi_metadata._SUBJECT_PATH_KEYS)
def test_every_recorded_source_path_key_resolves(tmp_path: Path, path_key: str) -> None:
    """Parametrised over the whole lookup, not one example of it.

    ``TorchIOSubjectBuilder`` constructs every image with ``tensor=``, so
    ``tio.Image.path`` is None on every route through it. Before the Subject-level
    lookup existed, this transform was invisible to all of them.
    """
    img_path = _write_dwi_files(tmp_path)
    subj = tio.Subject(
        input=tio.ScalarImage(tensor=torch.zeros(1, 2, 2, 1)),
        **{path_key: str(img_path)},
    )
    assert dwi_metadata._find_image_path(subj) == img_path

    out = LoadDWIMetadata(strict=True)(subj)
    assert out["n_directions"] == 5
    assert isinstance(out["b_values"], torch.Tensor)


def test_image_path_still_wins_over_a_recorded_key(tmp_path: Path) -> None:
    """The torchio-read form keeps priority — no regression for manifest_roles."""
    real = _write_dwi_files(tmp_path, stem="real_dwi")
    subj = tio.Subject(
        input=tio.ScalarImage(str(real)),
        source_path=str(tmp_path / "does_not_exist.nii.gz"),
    )
    assert dwi_metadata._find_image_path(subj) == real


def test_strict_raises_when_no_source_can_be_resolved() -> None:
    """The silent-skip hole: strict=True used to return the Subject untouched.

    This transform is only appended when the arm declared
    ``acquisition_metadata.fields: [b_value, bvec]``, so reaching this branch
    means the run asked for DWI metadata and will not get it (pitfall #9).
    """
    subj = tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 2, 2, 1)))
    with pytest.raises(ValueError, match="no source file recorded"):
        LoadDWIMetadata(strict=True)(subj)


@pytest.mark.parametrize(
    ("make_subject", "expected"),
    [
        (lambda p: tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 2, 2, 1))),
         "no source file recorded"),
        (lambda p: tio.Subject(
            input=tio.ScalarImage(tensor=torch.zeros(1, 2, 2, 1)),
            source_path=str(p / "unpaired.nii.gz"),
         ),
         "sidecars not found"),
    ],
    ids=["no-source", "missing-sidecars"],
)
def test_lenient_mode_warns_rather_than_returning_in_silence(
    tmp_path: Path, caplog, make_subject, expected: str
) -> None:
    """Both lenient branches used to `return subject` with no log line at all."""
    (tmp_path / "unpaired.nii.gz").write_bytes(b"placeholder")
    with caplog.at_level(logging.WARNING, logger=dwi_metadata.__name__):
        out = LoadDWIMetadata(strict=False)(make_subject(tmp_path))
    assert "b_values" not in out
    assert any(expected in r.message for r in caplog.records), (
        f"expected a WARNING containing {expected!r}; got "
        f"{[r.message[:60] for r in caplog.records]}"
    )


def test_missing_sidecar_message_names_which_one(tmp_path: Path) -> None:
    """'.bval and .bvec are missing' and 'the .bvec is missing' are different bugs."""
    try:
        import nibabel as nib
        import numpy as np
    except ImportError:
        pytest.skip("nibabel not installed")

    img_path = tmp_path / "half_dwi.nii.gz"
    nib.save(  # type: ignore[no-untyped-call]
        nib.Nifti1Image(np.zeros((4, 4, 2, 5), dtype=np.float32), affine=np.eye(4)),
        str(img_path),
    )
    (tmp_path / "half_dwi.bval").write_text("0 1000\n")  # .bvec deliberately absent

    with pytest.raises(FileNotFoundError) as exc:
        LoadDWIMetadata(strict=True)(tio.Subject(input=tio.ScalarImage(str(img_path))))
    assert "'bvec'" in str(exc.value)
    assert "'bval'" not in str(exc.value)
