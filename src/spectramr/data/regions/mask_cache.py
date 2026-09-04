"""Per-volume SynthSeg region cache: writer, reader, and the QC gate.

What is cached (format v2)
--------------------------
One ``.npz`` per volume holding the **raw FreeSurfer label ids** (uint16, ``[S, H, W]``) --
not pre-baked per-region bool masks, and no longer the coarse tissue ordinals of v1.

Deriving on read is deliberate. Every region (``brain``, ``tissue:wm``, ``structure:
hippocampus``, and any composition someone adds next month) *derives* from the label map,
so adding a region costs one entry in ``region_registry`` and **zero** cache regeneration.
Pre-baked masks would freeze today's region list into the cache, and a cache you must
rebuild whenever you add a region is not really a cache. A label field is also mostly long
uniform runs, so DEFLATE compresses it well.

v1 broke that promise in the one place it mattered. It stored *tissue ordinals*, which
means the reduction happened **at write time**: thalamus (10/49) and hippocampus (17/53)
were both already the single ordinal ``SUBCORTICAL_GM`` by the time the file hit disk, so
no reader, however clever, could recover a per-structure region from it. ``--parc`` labels
run to 2035 and do not fit a uint8 at all. v2 stores what SynthSeg actually said and
reduces on the way out.

The QC gate (the honest part)
-----------------------------
SynthSeg on ~5 mm anisotropic fastMRI brain is **out of distribution** even in
``--robust`` mode. Some volumes will segment badly. Two rules:

1. A volume below the QC floor is **dropped**, not repaired. A bad label map does
   not produce a bad-but-usable region -- it produces a region that is not the
   anatomy it claims to be, and every metric scored inside it is then measuring
   something nobody can name.
2. If the drop rate exceeds ``max_drop_rate`` (default 20%), **the tissue-region
   axis is unsound and the run must say so** rather than quietly proceeding on
   the survivors. Surviving volumes are not a random sample of the cohort: they
   are the ones SynthSeg found easy, which is exactly the population where the
   region boundaries are cleanest and the region effect is most likely to look
   real.

The pathology axis (fastMRI+ boxes) does not depend on SynthSeg, so it survives a
tissue-axis failure. That is why the gate is a separate, explicit call rather than
a hard raise at load time.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from spectramr.data.regions.synthseg_labels import (
    ORDINAL_TO_TISSUE,
    TissueClass,
    brain_mask,
    label_map_to_tissue,
    structure_mask,
    tissue_mask,
)

__all__ = [
    "CACHE_FORMAT_VERSION",
    "DEFAULT_MAX_DROP_RATE",
    "DEFAULT_QC_FLOOR",
    "LABELS_KEY",
    "REPORT_FILENAME",
    "RegionCacheFormatError",
    "RegionCacheReport",
    "RegionMaskCache",
    "TissueAxisUnsoundError",
    "VolumeQC",
    "load_region_cache",
    "write_volume_cache",
]

CACHE_FORMAT_VERSION = 2
"""v2 stores **raw FreeSurfer label ids**; v1 stored uint8 tissue ordinals.

The bump is not cosmetic. v1 threw the structure identities away *at write time*: thalamus
(10/49) and hippocampus (17/53) both became the single ordinal ``SUBCORTICAL_GM``, so no
per-structure region was recoverable from a v1 cache no matter what the reader did. And a
``--parc`` label runs to 2035, which does not fit in a uint8 at all."""

LABELS_KEY = "labels"
"""The npz key v2 writes. v1 wrote ``tissue``.

**The rename is the primary v1 guard, and it has to be.** v1 ordinal ``2`` is
``CEREBRAL_WM``; FreeSurfer id ``2`` is *also* cerebral white matter -- but v1 ordinal
``3`` is ``CSF`` while FreeSurfer ``3`` is cerebral CORTEX. A v1 array read as raw labels
is therefore not garbage, it is a *plausible* label map with grey and white matter partly
transposed, and every downstream mask would be confidently wrong. A version int alone would
not save us if a stale file were read before the version was checked; a renamed key makes
the misread structurally impossible -- a v2 reader on a v1 file finds no ``labels`` array
and raises."""

REPORT_FILENAME = "report.json"
"""The QC report always lives beside the masks it describes.

A cache directory with no report is not openable: the report is what says which volumes
SynthSeg DROPPED, and reading the masks without it would silently score the volumes QC
had rejected -- the exact population whose region boundaries cannot be trusted."""

# SynthSeg's own per-structure QC score is calibrated so that < 0.65 flags a
# structure the model is not confident in. We gate the VOLUME on its minimum
# per-structure score: one badly-segmented structure poisons the region it defines.
DEFAULT_QC_FLOOR = 0.65

DEFAULT_MAX_DROP_RATE = 0.20


class TissueAxisUnsoundError(RuntimeError):
    """Too many volumes failed SynthSeg QC for the tissue axis to mean anything."""


class RegionCacheFormatError(RuntimeError):
    """The cache on disk is not the format this reader understands."""

    @classmethod
    def stale_v1(cls, path: Path) -> RegionCacheFormatError:
        return cls(
            f"{path} is a v1 region cache (it stores the 'tissue' array: uint8 tissue "
            "ordinals). v2 stores raw FreeSurfer label ids under 'labels'. The two are "
            "NOT interchangeable and the difference is invisible if you force it: v1 "
            "ordinal 3 means CSF, FreeSurfer id 3 means cerebral cortex. Reading one as "
            "the other yields a plausible label map with tissues transposed. There is no "
            "migration -- the structure identities were destroyed at v1 write time. "
            "Rebuild: scripts/data/build_synthseg_region_cache.py"
        )


@dataclass(frozen=True, slots=True)
class VolumeQC:
    """SynthSeg's verdict on one volume."""

    file_id: str
    min_structure_score: float
    mean_structure_score: float
    n_slices: int
    dropped: bool
    reason: str | None = None
    worst_structure: str | None = None
    """WHICH structure scored lowest -- not just how low.

    The volume gate takes the minimum across structures, which is right for 8 tissue
    classes and wrong for 62 structures: a volume passing at min=0.66 has a hippocampus at
    0.66 and a putamen at 0.98, and reporting both as equally trustworthy is how a
    structure axis quietly launders a bad segmentation. Recording the name is what lets
    PR5 gate per structure instead of per volume."""

    def as_row(self) -> dict[str, object]:
        return {
            "file_id": self.file_id,
            "min_structure_score": round(self.min_structure_score, 4),
            "mean_structure_score": round(self.mean_structure_score, 4),
            "worst_structure": self.worst_structure,
            "n_slices": self.n_slices,
            "dropped": self.dropped,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RegionCacheReport:
    """What the cache build kept and threw away."""

    cache_dir: Path
    qc_floor: float
    max_drop_rate: float
    volumes: tuple[VolumeQC, ...] = field(default_factory=tuple)

    @property
    def n_total(self) -> int:
        return len(self.volumes)

    @property
    def n_dropped(self) -> int:
        return sum(v.dropped for v in self.volumes)

    @property
    def n_kept(self) -> int:
        return self.n_total - self.n_dropped

    @property
    def drop_rate(self) -> float:
        return self.n_dropped / self.n_total if self.n_total else 0.0

    @property
    def tissue_axis_sound(self) -> bool:
        return self.drop_rate <= self.max_drop_rate

    def raise_if_tissue_axis_unsound(self) -> None:
        """Called before any ``tissue:*`` region is scored.

        Not called for pathology regions -- those do not depend on SynthSeg, so a
        tissue-axis failure must not take the whole study down with it.
        """
        if self.tissue_axis_sound:
            return
        raise TissueAxisUnsoundError(
            f"SynthSeg dropped {self.n_dropped}/{self.n_total} volumes "
            f"({self.drop_rate:.1%}) below the QC floor {self.qc_floor}, above the "
            f"{self.max_drop_rate:.0%} ceiling. The tissue-region axis is UNSOUND on "
            "this cohort and must not be reported. Proceeding on the survivors would "
            "be worse than not running it: they are not a random sample, they are the "
            "volumes SynthSeg found easy -- precisely the population with the "
            "cleanest region boundaries and the most flattering region effect. "
            "SynthSeg is out of distribution on ~5mm anisotropic fastMRI brain; that "
            "is the finding, and the report must say so. The pathology axis "
            "(fastMRI+ boxes) does not depend on SynthSeg and is unaffected."
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "cache_dir": str(self.cache_dir),
            "qc_floor": self.qc_floor,
            "max_drop_rate": self.max_drop_rate,
            "volumes_total": self.n_total,
            "volumes_kept": self.n_kept,
            "volumes_dropped": self.n_dropped,
            "drop_rate": round(self.drop_rate, 4),
            "tissue_axis_sound": self.tissue_axis_sound,
            "volumes": [v.as_row() for v in self.volumes],
        }

    def write_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @classmethod
    def from_json(cls, path: str | Path) -> RegionCacheReport:
        d = json.loads(Path(path).read_text())
        # The per-file key rename catches a v1 npz. This catches the case the rename
        # cannot: a directory where someone rebuilt SOME volumes as v2 and left the rest,
        # so the report -- the thing that says which volumes exist at all -- is stale.
        version = int(d.get("cache_format_version", 1))
        if version != CACHE_FORMAT_VERSION:
            raise RegionCacheFormatError(
                f"{path} describes a v{version} region cache; this reader is "
                f"v{CACHE_FORMAT_VERSION}. Rebuild the whole cache directory -- a "
                "partial rebuild leaves a mix, and the report is what decides which "
                "volumes are readable. scripts/data/build_synthseg_region_cache.py"
            )
        return cls(
            cache_dir=Path(d["cache_dir"]),
            qc_floor=float(d["qc_floor"]),
            max_drop_rate=float(d["max_drop_rate"]),
            volumes=tuple(
                VolumeQC(
                    file_id=v["file_id"],
                    min_structure_score=v["min_structure_score"],
                    mean_structure_score=v["mean_structure_score"],
                    n_slices=v["n_slices"],
                    dropped=v["dropped"],
                    reason=v["reason"],
                    worst_structure=v.get("worst_structure"),
                )
                for v in d["volumes"]
            ),
        )


_MAX_UINT16 = 65535


def write_volume_cache(
    cache_dir: str | Path,
    file_id: str,
    labels: torch.Tensor,
    *,
    qc_scores: Mapping[str, float],
    affine: np.ndarray,
) -> Path:
    """Write one volume's ``[S, H, W]`` raw FreeSurfer label map + its QC scores.

    Stored as **uint16**: the ids run to 2035 under ``--parc``, so uint8 is not merely
    lossy, it cannot represent them. Kept as an int tensor in memory (torch's uint16
    support is partial), which costs nothing -- a slice is a few hundred KB.

    ``qc_scores`` is keyed by **structure name**, not by position. The old
    ``{"structure_0": ...}` form threw away which structure each score belonged to, so a
    per-structure QC gate (PR5) had nothing to gate on.
    """
    if labels.ndim != 3:
        raise ValueError(f"expected [S, H, W] label map, got {tuple(labels.shape)}")
    if labels.is_floating_point():
        raise TypeError(
            f"label map must be an integer tensor, got {labels.dtype}. A float label id "
            "means something upstream interpolated the segmentation -- and an "
            "interpolated label is a tissue that does not exist."
        )
    if labels.numel():
        lo, hi = int(labels.min()), int(labels.max())
        if lo < 0 or hi > _MAX_UINT16:
            raise ValueError(
                f"label ids span [{lo}, {hi}], outside uint16. These are not SynthSeg "
                "FreeSurfer ids (max 2035 with --parc)."
            )
    if affine.shape != (4, 4):
        raise ValueError(f"expected a 4x4 affine, got {affine.shape}")

    # Every id must be a declared FreeSurfer label. This does NOT catch a v1 ordinal array
    # being passed here by mistake -- ordinals 0/2/3/4/5/7 are all real FreeSurfer ids, so
    # such an array is indistinguishable from a (wrong) label map at write time. That is
    # precisely why the READ-side guard is a renamed npz key rather than a value check:
    # the misread has to be made structurally impossible, because it cannot be detected.
    label_map_to_tissue(labels)

    out = Path(cache_dir) / f"{Path(file_id).stem}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        **{LABELS_KEY: labels.cpu().numpy().astype(np.uint16)},
        format_version=np.int32(CACHE_FORMAT_VERSION),
        slice_axis=np.int32(0),
        affine=np.asarray(affine, dtype=np.float64),
        qc_keys=np.array(list(qc_scores), dtype=object),
        qc_values=np.array(list(qc_scores.values()), dtype=np.float32),
    )
    return out


class RegionMaskCache:
    """Read-side of the cache. The sweep touches this and never a segmenter."""

    def __init__(self, cache_dir: str | Path, report: RegionCacheReport) -> None:
        self.cache_dir = Path(cache_dir)
        self.report = report
        self._dropped = {v.file_id for v in report.volumes if v.dropped}

    def has(self, file_id: str) -> bool:
        return (
            file_id not in self._dropped and (self.cache_dir / f"{Path(file_id).stem}.npz").exists()
        )

    def labels(self, file_id: str, slice_index: int) -> torch.Tensor:
        """Raw FreeSurfer label ids ``[H, W]`` for one slice of the CLEAN reference.

        This is the cache's ground truth. Everything else on this class -- ``brain``,
        ``region``, ``structure`` -- is derived from it *on read*, so adding a region
        never costs a cache rebuild.
        """
        if file_id in self._dropped:
            raise KeyError(
                f"{file_id} was dropped by SynthSeg QC and has no trustworthy "
                "segmentation. Scoring a tissue region on it would measure a region "
                "that is not the anatomy it claims to be."
            )
        path = self.cache_dir / f"{Path(file_id).stem}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"no region cache for {file_id} at {path}. Regenerate with "
                "scripts/data/build_synthseg_region_cache.py -- the cache is a "
                "derived artifact and is never shipped."
            )
        with np.load(path, allow_pickle=True) as z:
            if LABELS_KEY not in z:
                raise RegionCacheFormatError.stale_v1(path)
            version = int(z["format_version"]) if "format_version" in z else 0
            if version != CACHE_FORMAT_VERSION:
                raise RegionCacheFormatError(
                    f"{path} is cache format v{version}; this reader is "
                    f"v{CACHE_FORMAT_VERSION}. Rebuild the cache."
                )
            vol = z[LABELS_KEY]

        if not 0 <= slice_index < vol.shape[0]:
            raise IndexError(
                f"slice {slice_index} out of range for {file_id} ({vol.shape[0]} slices)"
            )
        # uint16 -> int32: torch's uint16 support is partial (isin/unique are unhappy),
        # and the widening is free at slice scale.
        return torch.from_numpy(np.ascontiguousarray(vol[slice_index]).astype(np.int32))

    def tissue_ordinals(self, file_id: str, slice_index: int) -> torch.Tensor:
        """The ``[H, W]`` tissue-class ordinals, derived from the raw labels on read."""
        return label_map_to_tissue(self.labels(file_id, slice_index))

    def brain(self, file_id: str, slice_index: int) -> torch.Tensor:
        return brain_mask(self.tissue_ordinals(file_id, slice_index))

    def region(
        self, file_id: str, slice_index: int, tissues: Sequence[TissueClass]
    ) -> torch.Tensor:
        return tissue_mask(self.tissue_ordinals(file_id, slice_index), *tissues)

    def structure(self, file_id: str, slice_index: int, label_ids: Sequence[int]) -> torch.Tensor:
        """Bool mask for a named anatomical structure, by raw FreeSurfer id.

        The thing v1 could not do at any price: its ordinals had already merged thalamus
        and hippocampus into ``SUBCORTICAL_GM`` before the file was written.
        """
        return structure_mask(self.labels(file_id, slice_index), label_ids)

    def tissue_histogram(self, file_id: str, slice_index: int) -> dict[TissueClass, int]:
        ords = self.tissue_ordinals(file_id, slice_index)
        vals, counts = torch.unique(ords, return_counts=True)
        return {
            ORDINAL_TO_TISSUE[int(v)]: int(c)
            for v, c in zip(vals.tolist(), counts.tolist(), strict=True)
        }


def load_region_cache(cache_dir: str | Path) -> RegionMaskCache:
    """Open a region cache written by ``scripts/data/build_synthseg_region_cache.py``.

    The read side had no entry point: ``RegionMaskCache`` wants a
    :class:`RegionCacheReport` that only the *builder* held, so a sweep had no way to
    open a cache from a directory at all. This is that entry point, and it insists on the
    report rather than reconstructing a permissive one -- a cache opened without its QC
    report would happily serve the masks of volumes SynthSeg had **rejected**, which are
    not a random sample but exactly the volumes whose segmentation is untrustworthy.
    """
    d = Path(cache_dir)
    report_path = d / REPORT_FILENAME
    if not report_path.exists():
        raise FileNotFoundError(
            f"no {REPORT_FILENAME} in {d}. A region cache is only openable together with "
            "the QC report that says which volumes SynthSeg dropped. Rebuild with "
            "scripts/data/build_synthseg_region_cache.py --cache-dir "
            f"{d} (it writes the report there by default)."
        )
    return RegionMaskCache(d, RegionCacheReport.from_json(report_path))
