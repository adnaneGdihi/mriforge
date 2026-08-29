"""DWI b-value / bvec metadata loader (Phase 4b-extra).

BIDS DWI files come with two sidecar files:

- ``<name>.bval`` — text file with N b-values, one per direction.
- ``<name>.bvec`` — text file with a 3×N matrix of unit gradient vectors.

This transform reads both and attaches them to the Subject as **tensors** for
downstream Bloch / DWI loss and regulariser code::

    subject["b_values"] = tensor([0., 1000., 1000., ...])       # (N,)
    subject["b_vectors"] = tensor([[0,0,0], [1,0,0], ...])      # (N, 3)
    subject["n_directions"] = N

Tensors (not Python lists) so the default collate stacks them to ``(B, N)`` /
``(B, N, 3)`` and they reach the strategy as tensors, not object lists.

Use alongside :class:`LoadAcquisitionMetadata` to populate the rest of
the scan params (TE/TR/FA/B0).

Consumer status (audit 2026-07 F3/I2; #350 fixed 2026-07-18): the ``b_values``
are consumed by ``mriforge.models.losses.dwi_adc_monoexp_loss.DWIADCMonoexpLoss``
(a monoexponential ``S(b)=S0·e^{-b·ADC}`` fit + consistency loss) and graded by
the ``adc_mae`` metric. The ``b_vectors`` (gradient directions) are consumed by
``QSpaceDiffusionStrategy`` to build the real spherical-harmonic angular basis
(Descoteaux 2007) for its Laplace-Beltrami regulariser — it reads this exact key
and raises if it is absent. All registered and unit-tested on synthetic diffusion
signals. NOTE: no dataset on the current corpus emits multi-b DWI images, so no
live training arm exercises this path yet (roadmap).

Reachability
============

The sidecar rule needs the **file on disk**, so this transform can only work on
a Subject whose source path is knowable. Two lookups provide it (see
``_find_image_path``):

1. ``tio.Image.path`` -- set only by the ``ScalarImage(path)`` form, used by
   ``index_builder.load_from_manifest_roles`` with ``format: nifti`` and by the
   image-folder loader.
2. A Subject-level ``source_path`` string, recorded by the subject builders.

Before the second lookup existed, every route through ``TorchIOSubjectBuilder``
was invisible to this transform: it builds every image with ``tensor=``, so
``Image.path`` is None, ``_find_image_path`` returned None, and
``apply_transform`` returned the Subject untouched -- **even at
``strict=True``**. The knob was real (``acquisition_metadata.fields`` admits
``b_value``/``bvec`` as Literals) and the config looked satisfied.

DWI volumes themselves are servable on the ordinary NIfTI route:
``NiftiStrategy`` maps a 4-D ``(W, H, D, N)`` file to ``(N, W, H, D)``, putting
the gradient directions in the channel slot, which is the layout both consumers
expect.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torchio as tio

__all__ = ["LoadDWIMetadata", "parse_bval_file", "parse_bvec_file"]

logger = logging.getLogger(__name__)


def parse_bval_file(path: Path) -> list[float]:
    """Read a BIDS .bval file → list of b-values (s/mm²).

    The file is whitespace-separated; some vendors put all values on one
    line, others split across lines.
    """
    text = path.read_text()
    # Replace newlines with spaces, then split on any whitespace.
    tokens = text.replace("\n", " ").split()
    return [float(t) for t in tokens if t.strip()]


def parse_bvec_file(path: Path) -> list[list[float]]:
    """Read a BIDS .bvec file → list of [bx, by, bz] gradient vectors.

    BIDS spec: 3 rows × N columns, whitespace-separated. We transpose to
    a list of length N where each entry is [x, y, z].
    """
    text = path.read_text()
    rows = [
        [float(t) for t in line.split() if t.strip()] for line in text.split("\n") if line.strip()
    ]
    if len(rows) != 3:
        raise ValueError(
            f"Expected 3 rows in .bvec file {path}; got {len(rows)}. "
            "BIDS spec: rows = (x, y, z), columns = N gradient directions."
        )
    n_directions = len(rows[0])
    if any(len(r) != n_directions for r in rows):
        raise ValueError(
            f".bvec file {path} has inconsistent column counts across rows: "
            f"{[len(r) for r in rows]}."
        )
    return [[rows[0][i], rows[1][i], rows[2][i]] for i in range(n_directions)]


def _find_dwi_sidecars(image_path: Path) -> tuple[Path | None, Path | None]:
    """Return the (.bval, .bvec) sibling paths for a DWI NIfTI input."""
    stem = image_path.name
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    bval = image_path.with_name(f"{stem}.bval")
    bvec = image_path.with_name(f"{stem}.bvec")
    return (bval if bval.exists() else None, bvec if bvec.exists() else None)


#: Subject-level string keys that name the file a Subject was built from,
#: consulted in order after the images' own ``path``. Needed because almost
#: every producer in this repo constructs ``tio.ScalarImage(tensor=...)``, and
#: torchio only populates ``Image.path`` for the ``ScalarImage(path)`` form --
#: which is used by exactly two routes (``index_builder.load_from_manifest_roles``
#: with ``format: nifti``, and the image-folder loader). On every other route
#: the sidecar rule had nothing to resolve against.
_SUBJECT_PATH_KEYS: tuple[str, ...] = (
    "source_path",
    "input_path",
    "image_path",
    "primary_path",
    "target_path",
)


def _find_image_path(subject: tio.Subject) -> Path | None:
    """Resolve the file a Subject was loaded from, or None.

    Two lookups, in order: the images' own ``path`` (set only when torchio
    itself read the file), then the Subject's recorded source-path keys.
    """
    for key in ("input", "mri", "image", "hr", "target"):
        img = subject.get(key)
        if img is None:
            continue
        path = getattr(img, "path", None)
        if path is not None:
            return Path(path)
    for key in _SUBJECT_PATH_KEYS:
        recorded = subject.get(key)
        if isinstance(recorded, str | Path) and str(recorded):
            return Path(recorded)
    return None


class LoadDWIMetadata(tio.Transform):
    """Attach DWI b-values + bvecs to each Subject.

    Args:
        strict: When True, raise if either sidecar is missing for a
            DWI scan; when False, log a warning and skip.

    After this transform, ``subject["b_values"]`` and
    ``subject["b_vectors"]`` carry per-direction lists. The transform
    also validates that ``len(b_values) == len(b_vectors)`` — a common
    BIDS export bug that silently corrupts downstream loss code.
    """

    def __init__(self, strict: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.strict = strict

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        image_path = _find_image_path(subject)
        if image_path is None:
            # This transform is only appended when the arm declared
            # ``acquisition_metadata.fields: [b_value, bvec]``, so reaching here
            # means the run asked for DWI metadata and will not get it. It used
            # to return silently -- not even at ``strict=True`` -- so the ADC
            # loss and QSpaceDiffusionStrategy's angular basis went without
            # their inputs on a config that looked satisfied (pitfall #9).
            message = (
                "LoadDWIMetadata: no source file recorded on this Subject, so "
                ".bval/.bvec siblings cannot be located. The images were built "
                "with tensor= (torchio only sets Image.path when it reads the "
                f"file itself) and none of {list(_SUBJECT_PATH_KEYS)} is set. "
                "b_values/b_vectors will be absent, and any consumer of them "
                "will fail or silently degrade."
            )
            if self.strict:
                raise ValueError(message)
            logger.warning(message)
            return subject

        bval_path, bvec_path = _find_dwi_sidecars(image_path)
        if bval_path is None or bvec_path is None:
            missing = [
                name for name, found in (("bval", bval_path), ("bvec", bvec_path)) if found is None
            ]
            message = (
                f"DWI sidecars not found for {image_path}: missing "
                f"{missing}. Expected sibling .bval and .bvec files."
            )
            if self.strict:
                raise FileNotFoundError(message)
            logger.warning(message)
            return subject

        b_values = parse_bval_file(bval_path)
        b_vectors = parse_bvec_file(bvec_path)
        if len(b_values) != len(b_vectors):
            msg = (
                f"DWI sidecars inconsistent for {image_path}: "
                f"{len(b_values)} b-values but {len(b_vectors)} bvecs."
            )
            if self.strict:
                raise ValueError(msg)
            logger.warning(msg)
            return subject

        # Attach as tensors (not Python lists): the default collate then stacks
        # them to (B, N) / (B, N, 3), so QSpaceDiffusionStrategy and the ADC loss
        # receive tensors, not object lists (#350).
        subject["b_values"] = torch.as_tensor(b_values, dtype=torch.float32)
        subject["b_vectors"] = torch.as_tensor(b_vectors, dtype=torch.float32)
        subject["n_directions"] = len(b_values)
        return subject
