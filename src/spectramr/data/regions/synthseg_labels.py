"""FreeSurfer label id -> tissue class, for the SynthSeg region tiers.

Why SynthSeg runs per-volume and never per-slice
------------------------------------------------
SynthSeg is a **5-level 3-D U-Net** with four ``MaxPool3d(2, 2)`` stages
(``scripts/eval/convert_synthseg_h5_to_pt.py``). A single slice has ``D = 1``, so
the first pool computes ``floor((1 - 2) / 2) + 1 = 0`` and the forward pass dies.
Zero-padding ``D`` to 16 does not rescue it: the network then sees a slab that is
15/16 empty, and its BatchNorm running statistics were fit on real 3-D context.
It would emit *a* label map, and that map would be meaningless -- a textbook
facade (pitfall #16).

So the segmenter runs **once per volume, offline**, and the sweep consumes a
cached 2-D slice of the resulting 3-D label map. It never calls a segmenter.

Masks come from the CLEAN reference only
----------------------------------------
Re-segmenting each degraded frame would let the region boundary move with the
artifact, and you would be measuring "how much did this artifact confuse
SynthSeg" instead of "how well does metric M track degradation inside region R".
A region is a property of the ground truth: segment once, reuse across every
artifact and severity.

An unmapped label raises, for the same reason there is no ``other`` lesion group:
a silently-bucketed label is a region whose anatomical meaning nobody can state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum

import torch

__all__ = [
    "DK_PARCELLATION_LABELS",
    "LABEL_TO_TISSUE",
    "TissueClass",
    "UnmappedSynthSegLabelError",
    "brain_mask",
    "label_map_to_tissue",
    "structure_mask",
    "tissue_mask",
]


class TissueClass(StrEnum):
    """Coarse tissue classes composed from the SynthSeg label set."""

    BACKGROUND = "background"
    CORTICAL_GM = "cortical_gm"
    CEREBRAL_WM = "cerebral_wm"
    CSF = "csf"
    SUBCORTICAL_GM = "subcortical_gm"
    CEREBELLAR_GM = "cerebellar_gm"
    CEREBELLAR_WM = "cerebellar_wm"
    BRAINSTEM = "brainstem"

    @property
    def ordinal(self) -> int:
        """Stable uint8 code for the cached label map."""
        return _ORDINALS[self]


_ORDINALS: Mapping[TissueClass, int] = {
    TissueClass.BACKGROUND: 0,
    TissueClass.CORTICAL_GM: 1,
    TissueClass.CEREBRAL_WM: 2,
    TissueClass.CSF: 3,
    TissueClass.SUBCORTICAL_GM: 4,
    TissueClass.CEREBELLAR_GM: 5,
    TissueClass.CEREBELLAR_WM: 6,
    TissueClass.BRAINSTEM: 7,
}

ORDINAL_TO_TISSUE: Mapping[int, TissueClass] = {v: k for k, v in _ORDINALS.items()}

# The standard SynthSeg output label set (FreeSurfer ids). Left/right are folded
# together: the region axis asks "does the metric behave differently in WM than in
# GM", not "differently in LEFT WM". Laterality is still recoverable from the
# column centroid, which is what the contralateral-mirror control uses.
LABEL_TO_TISSUE: Mapping[int, TissueClass] = {
    0: TissueClass.BACKGROUND,
    # Cerebral white matter
    2: TissueClass.CEREBRAL_WM,
    41: TissueClass.CEREBRAL_WM,
    # Cerebral cortex
    3: TissueClass.CORTICAL_GM,
    42: TissueClass.CORTICAL_GM,
    # Ventricles + CSF
    4: TissueClass.CSF,
    43: TissueClass.CSF,
    5: TissueClass.CSF,
    44: TissueClass.CSF,
    14: TissueClass.CSF,
    15: TissueClass.CSF,
    24: TissueClass.CSF,
    # Deep grey
    10: TissueClass.SUBCORTICAL_GM,
    49: TissueClass.SUBCORTICAL_GM,
    11: TissueClass.SUBCORTICAL_GM,
    50: TissueClass.SUBCORTICAL_GM,
    12: TissueClass.SUBCORTICAL_GM,
    51: TissueClass.SUBCORTICAL_GM,
    13: TissueClass.SUBCORTICAL_GM,
    52: TissueClass.SUBCORTICAL_GM,
    17: TissueClass.SUBCORTICAL_GM,
    53: TissueClass.SUBCORTICAL_GM,
    18: TissueClass.SUBCORTICAL_GM,
    54: TissueClass.SUBCORTICAL_GM,
    26: TissueClass.SUBCORTICAL_GM,
    58: TissueClass.SUBCORTICAL_GM,
    28: TissueClass.SUBCORTICAL_GM,
    60: TissueClass.SUBCORTICAL_GM,
    # Cerebellum
    8: TissueClass.CEREBELLAR_GM,
    47: TissueClass.CEREBELLAR_GM,
    7: TissueClass.CEREBELLAR_WM,
    46: TissueClass.CEREBELLAR_WM,
    # Brainstem
    16: TissueClass.BRAINSTEM,
}

# Desikan-Killiany cortical parcels, emitted only under `mri_synthseg --parc`:
# 1001-1035 (left) and 2001-2035 (right).
#
# These MUST resolve to CORTICAL_GM. Under --parc SynthSeg stops emitting the whole-cortex
# labels 3/42 and emits the parcels instead, so a table that knows 3/42 but not 1001+ has
# two failure modes and both are silent-ish: every parcel is an unmapped label (loud, at
# least), or -- if anyone "fixes" that by bucketing them to BACKGROUND -- `brain()` loses
# the ENTIRE CORTEX while the run goes on producing confident numbers for `tissue:*` and
# `brain`. The whole cortical ribbon would simply not be in the brain mask.
#
# Folding all parcels into CORTICAL_GM keeps `brain()` and `tissue:cortical_gm` identical
# whether or not --parc was used, which is the invariant the builder test pins. The parcel
# identities are not lost: the cache now stores the RAW label ids, so PR5's
# `structure:*` / `parcellation:*` regions read them straight out of the label map.
DK_PARCELLATION_LABELS: tuple[int, ...] = (*range(1001, 1036), *range(2001, 2036))

LABEL_TO_TISSUE = {
    **LABEL_TO_TISSUE,
    **dict.fromkeys(DK_PARCELLATION_LABELS, TissueClass.CORTICAL_GM),
}


class UnmappedSynthSegLabelError(KeyError):
    """A FreeSurfer label id with no declared :class:`TissueClass`."""

    def __init__(self, label: int) -> None:
        self.label = label
        super().__init__(
            f"SynthSeg emitted FreeSurfer label {label}, which has no declared "
            "TissueClass. Silently bucketing it would create a region whose "
            "anatomical meaning nobody can state. Declare it in LABEL_TO_TISSUE "
            "(spectramr/data/regions/synthseg_labels.py). If SynthSeg is emitting "
            "labels outside its documented set, that is itself the finding -- the "
            "segmentation is probably out of distribution on this data."
        )


_UNMAPPED_ORDINAL = 255
"""Sentinel in the lookup table. Not a TissueClass -- it is the absence of one."""

_MAX_LABEL = max(LABEL_TO_TISSUE)

_LUT = torch.full((_MAX_LABEL + 1,), _UNMAPPED_ORDINAL, dtype=torch.uint8)
for _fs_id, _tissue in LABEL_TO_TISSUE.items():
    _LUT[_fs_id] = _tissue.ordinal
# Built from a dict LITERAL, not from a decorator-populated registry, so it cannot be
# order-dependent: every entry exists before this module finishes importing.


def label_map_to_tissue(label_map: torch.Tensor) -> torch.Tensor:
    """FreeSurfer label ids ``[..., H, W]`` -> :class:`TissueClass` ordinals (uint8).

    A single gather, not one full-tensor compare per label. The old form scanned the whole
    volume once per declared id -- 32 passes for the standard label set, ~100 with
    ``--parc`` -- on every read, and the cache is read once per (sample, region, metric).

    Raises :class:`UnmappedSynthSegLabelError` on any undeclared id.
    """
    ids = label_map.to(torch.long)
    if ids.numel():
        out_of_range = (ids < 0) | (ids > _MAX_LABEL)
        if bool(out_of_range.any()):
            raise UnmappedSynthSegLabelError(int(ids[out_of_range][0]))

    out = _LUT.to(ids.device)[ids]
    unmapped = out == _UNMAPPED_ORDINAL
    if bool(unmapped.any()):
        raise UnmappedSynthSegLabelError(int(ids[unmapped][0]))
    return out


def structure_mask(label_map: torch.Tensor, label_ids: Sequence[int]) -> torch.Tensor:
    """Bool mask for the union of raw FreeSurfer ``label_ids``.

    This is what the v2 cache buys: thalamus (10/49) and hippocampus (17/53) are distinct
    here, where the v1 tissue-ordinal cache had already collapsed both into
    ``SUBCORTICAL_GM`` before anything was written to disk.
    """
    if not label_ids:
        raise ValueError("structure_mask() needs at least one FreeSurfer label id")
    wanted = torch.as_tensor(
        sorted({int(i) for i in label_ids}),
        dtype=label_map.dtype,
        device=label_map.device,
    )
    return torch.isin(label_map, wanted)


def tissue_mask(tissue_ordinals: torch.Tensor, *tissues: TissueClass) -> torch.Tensor:
    """Bool mask for the union of ``tissues`` in an ordinal map."""
    if not tissues:
        raise ValueError("tissue_mask() needs at least one TissueClass")
    mask = torch.zeros_like(tissue_ordinals, dtype=torch.bool)
    for t in tissues:
        mask |= tissue_ordinals == t.ordinal
    return mask


def brain_mask(tissue_ordinals: torch.Tensor) -> torch.Tensor:
    """Everything SynthSeg did not call background."""
    return tissue_ordinals != TissueClass.BACKGROUND.ordinal
