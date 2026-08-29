"""CLI-based SynthSeg segmentation for the sim2rank volumetric segmentation stage.

FreeSurfer's ``mri_synthseg`` is the SynthSeg available in the target environment
(no inline torch/Keras model). This wraps the NIfTI round-trip: write a
slice-first ``[S, H, W]`` volume with real geometry, run ``mri_synthseg
--robust``, read the FreeSurfer label volume back. It is the honest counterpart
to the Otsu proxy the whole-image leaderboard falls back to — real anatomical
Dice between a degraded/predicted reconstruction and the clean reference.

Contract (see :mod:`mriforge.data.nifti_export`): the volume is exported as
``.nii.gz``, slice-first, float32-finite, with a real affine, and SynthSeg is run
with ``--robust`` (fastMRI brain is ~5 mm anisotropic, out of SynthSeg's nominal
range). There is NO silent fallback — a missing ``mri_synthseg`` or a non-zero
exit RAISES. This runs offline in a dedicated FreeSurfer-gated stage, never
inside the FreeSurfer-free STEP-3 leaderboard core.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch

from mriforge.core.metrics.quantitative.challenge_metrics import DGM_LABELS_14
from mriforge.data.nifti_export import VoxelGeometry, write_slice_first_nifti

#: Default FreeSurfer entry point. A tuple so callers can inject a wrapper
#: (``("srun", "mri_synthseg")`` etc.) without shell string-splitting.
DEFAULT_SYNTHSEG_CMD: tuple[str, ...] = ("mri_synthseg",)


def segment_volume_via_synthseg(
    volume: torch.Tensor,
    geometry: VoxelGeometry,
    *,
    workdir: str | Path,
    cmd: tuple[str, ...] = DEFAULT_SYNTHSEG_CMD,
    robust: bool = True,
    timeout: float = 1800.0,
) -> torch.Tensor:
    """Segment a ``[S, H, W]`` volume with ``mri_synthseg``.

    Returns a ``[S, H, W]`` int64 volume of raw FreeSurfer label ids. Raises
    ``RuntimeError`` if the CLI fails or emits no output (no silent fallback).
    """
    import nibabel as nib

    if volume.dim() != 3:
        raise ValueError(f"volume must be [S, H, W]; got shape {tuple(volume.shape)}")

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    in_path = work / "volume.nii.gz"
    out_path = work / "volume_seg.nii.gz"

    write_slice_first_nifti(in_path, volume.float().cpu(), geometry)

    argv = [*cmd, "--i", str(in_path), "--o", str(out_path)]
    if robust:
        argv.append("--robust")
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"mri_synthseg failed (rc={proc.returncode}) for {in_path}:\n"
            f"{(proc.stderr or '')[-2000:]}"
        )
    if not out_path.exists():
        raise RuntimeError(f"mri_synthseg returned 0 but produced no label volume at {out_path}.")

    labels = nib.load(str(out_path)).get_fdata()
    return torch.from_numpy(labels).round().to(torch.int64)


def synthseg_dice(
    pred_volume: torch.Tensor,
    ref_volume: torch.Tensor,
    geometry: VoxelGeometry,
    *,
    workdir: str | Path,
    labels: tuple[int, ...] = DGM_LABELS_14,
    cmd: tuple[str, ...] = DEFAULT_SYNTHSEG_CMD,
) -> float:
    """Volumetric SynthSeg Dice over ``labels`` between two ``[S, H, W]`` volumes.

    Both volumes are segmented with the SAME CLI, then scored with the honest-
    denominator multi-class Dice (a class absent from both maps contributes
    nothing — no free Dice=1). ``labels`` defaults to the 14 deep-grey-matter
    FreeSurfer ids (:data:`DGM_LABELS_14`).
    """
    from mriforge.core.metrics.quantitative.segmentation import dice_score

    work = Path(workdir)
    pred_seg = segment_volume_via_synthseg(pred_volume, geometry, workdir=work / "pred", cmd=cmd)
    ref_seg = segment_volume_via_synthseg(ref_volume, geometry, workdir=work / "ref", cmd=cmd)
    # Volume-level Dice: flatten the whole [S, H, W] label map as one "image".
    d = dice_score(pred_seg[None], ref_seg[None], n_classes=0, labels=labels)
    return float(d.mean())


def synthseg_dice_trajectories(
    clean_volume: torch.Tensor,
    geometry: VoxelGeometry,
    *,
    degrade_fn,
    families: tuple[str, ...] | list[str],
    severities: tuple[float, ...] | list[float],
    workdir: str | Path,
    labels: tuple[int, ...] = DGM_LABELS_14,
    cmd: tuple[str, ...] = DEFAULT_SYNTHSEG_CMD,
) -> dict[str, list[float]]:
    """Per-family SynthSeg-Dice severity trajectories for one clean 3-D volume.

    Segments the clean reference ONCE (segment-once), then for each
    ``(family, theta)`` applies ``degrade_fn(clean_volume, family, theta) ->
    volume`` (the digital-twin degradation at the volume level), segments the
    degraded volume, and scores volumetric Dice against the reference.

    Returns ``{family: [dice at each severity]}`` — an ADR-style trajectory that
    folds into the leaderboard alongside the swept per-image metrics: Dice falls
    monotonically with severity for a metric that tracks anatomical fidelity, so
    ADR/SFA can rank ``synthseg_dice`` exactly like any other swept metric.

    This is the FreeSurfer-gated stage's compute core; it runs one
    ``mri_synthseg`` subprocess per ``(clean + family x severity)`` volume, so it
    belongs in an offline pass, never the FreeSurfer-free STEP-3 leaderboard core.
    """
    from mriforge.core.metrics.quantitative.segmentation import dice_score

    work = Path(workdir)
    ref_seg = segment_volume_via_synthseg(clean_volume, geometry, workdir=work / "ref", cmd=cmd)
    out: dict[str, list[float]] = {}
    for family in families:
        traj: list[float] = []
        for i, theta in enumerate(severities):
            deg = degrade_fn(clean_volume, family, float(theta))
            seg = segment_volume_via_synthseg(
                deg, geometry, workdir=work / f"{family}_{i}", cmd=cmd
            )
            d = dice_score(seg[None], ref_seg[None], n_classes=0, labels=labels)
            traj.append(float(d.mean()))
        out[family] = traj
    return out


__all__ = [
    "DEFAULT_SYNTHSEG_CMD",
    "segment_volume_via_synthseg",
    "synthseg_dice",
    "synthseg_dice_trajectories",
]
