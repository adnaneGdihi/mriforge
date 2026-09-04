"""#350 — the q-space SH regulariser fires end-to-end (transform → collate → strategy).

The pre-fix facade was invisible to the strategy's own unit tests: they passed a
directions tensor **directly**, so they proved the SH maths works while the real
pipeline fed the strategy ``None`` forever (the transform wrote ``b_vectors`` but the
strategy read ``bvecs``). These tests assert the **seam**: the directions the
``LoadDWIMetadata`` transform attaches, collated the way the dataloader collates them,
reach the strategy under the key it actually reads and make the Laplace-Beltrami term
appear — and that a diffusion batch missing directions RAISES rather than silently
degrading to vanilla reconstruction.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import default_collate

from spectramr.infrastructure.training.strategies.qspace_diffusion_strategy import (
    QSpaceDiffusionStrategy,
)
from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)

N_DIR = 15  # enough directions for max_order=4 (K=15 SH coefficients)


def _write_dwi_subject(tmp_path: Path, sub: str, directions: torch.Tensor):
    """Write a NIfTI + .bval/.bvec for ``directions`` (N,3); return the Subject."""
    try:
        import nibabel as nib
        import numpy as np
        import torchio as tio
    except ImportError:  # pragma: no cover
        pytest.skip("nibabel/torchio not installed")

    img_path = tmp_path / f"{sub}_dwi.nii.gz"
    nib.save(  # type: ignore[no-untyped-call]
        nib.Nifti1Image(
            np.zeros((4, 4, 2, directions.shape[0]), dtype=np.float32), affine=np.eye(4)
        ),
        str(img_path),
    )
    (tmp_path / f"{sub}_dwi.bval").write_text(
        " ".join("1000" for _ in directions) + "\n"
    )
    rows = [
        " ".join(f"{directions[i, ax].item():.6f}" for i in range(directions.shape[0]))
        for ax in range(3)
    ]
    (tmp_path / f"{sub}_dwi.bvec").write_text("\n".join(rows) + "\n")
    return tio.Subject(input=tio.ScalarImage(str(img_path)))


def _unit_directions(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(N_DIR, 3, generator=g)
    return v / v.norm(dim=-1, keepdim=True)


def _make_strategy() -> QSpaceDiffusionStrategy:
    return object.__new__(QSpaceDiffusionStrategy)


def _base_dict() -> dict[str, torch.Tensor]:
    total = (torch.zeros(1, requires_grad=True) + 1.0).squeeze(0)
    return {"loss_total": total, "g_total_loss": total}


def _collated_bvectors_from_transform(
    tmp_path: Path, dirs_per_sample: list[torch.Tensor]
):
    """Transform each sample's sidecars, then collate the way the dataloader does."""
    from spectramr.data.transforms.dwi_metadata import LoadDWIMetadata

    load = LoadDWIMetadata()
    per_sample = []
    for i, dirs in enumerate(dirs_per_sample):
        subj = _write_dwi_subject(tmp_path, f"sub-{i:03d}", dirs)
        out = load(subj)
        bvec = out["b_vectors"]
        assert isinstance(bvec, torch.Tensor) and bvec.shape == (N_DIR, 3)
        per_sample.append({"b_vectors": bvec})
    # The default collate stacks per-subject (N,3) tensors -> (B,N,3).
    return default_collate(per_sample)["b_vectors"]


def test_transform_output_collates_and_fires_the_regulariser(tmp_path: Path) -> None:
    """The end-to-end seam: transform-written directions reach the strategy and the
    Laplace-Beltrami term appears (the witness the SH fit ran on real directions)."""
    dirs = _unit_directions(0)
    b_vectors = _collated_bvectors_from_transform(tmp_path, [dirs, dirs])  # (2,N,3)
    assert b_vectors.shape == (2, N_DIR, 3)

    signal = torch.randn(2, N_DIR, 4, 4, requires_grad=True)
    batch = {"prediction": signal, "b_vectors": b_vectors}

    from unittest.mock import patch

    with patch.object(
        ReconstructionTrainingStrategy,
        "_compute_losses_impl",
        return_value=_base_dict(),
    ):
        out = _make_strategy()._compute_losses_impl(input_batch=batch, epoch=0)

    assert "loss_sh_laplace_beltrami" in out, (
        "the q-space regulariser did not fire on directions produced by "
        "LoadDWIMetadata — the transform→collate→strategy seam is broken (#350)"
    )
    assert torch.isfinite(out["loss_sh_laplace_beltrami"])


def test_missing_b_vectors_raises_not_silent_base() -> None:
    """A diffusion batch (prediction present) with no directions must RAISE — the
    old facade returned the base recon losses silently (pitfall #9/#16)."""
    signal = torch.randn(2, N_DIR, 4, 4)
    batch = {"prediction": signal}  # no b_vectors

    from unittest.mock import patch

    with patch.object(
        ReconstructionTrainingStrategy,
        "_compute_losses_impl",
        return_value=_base_dict(),
    ), pytest.raises(ValueError, match="b_vectors"):
        _make_strategy()._compute_losses_impl(input_batch=batch, epoch=0)


def test_heterogeneous_directions_across_batch_raise() -> None:
    """Samples with different diffusion directions cannot share one SH basis —
    reduce-to-shared must raise rather than silently pick sample 0's directions."""
    b_vectors = torch.stack([_unit_directions(1), _unit_directions(2)])  # differ
    signal = torch.randn(2, N_DIR, 4, 4)
    batch = {"prediction": signal, "b_vectors": b_vectors}

    from unittest.mock import patch

    with patch.object(
        ReconstructionTrainingStrategy,
        "_compute_losses_impl",
        return_value=_base_dict(),
    ), pytest.raises(ValueError, match="heterogeneous"):
        _make_strategy()._compute_losses_impl(input_batch=batch, epoch=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
