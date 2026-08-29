"""ISMRMRD measured-trajectory k-space dataset (spec E1 follow-up — kasper spiral).

Reconstructs from raw ISMRMRD acquisitions using the file's OWN measured
trajectory (the spiral/non-Cartesian k-space coordinates stored per acquisition),
closing the ``trajectory_source='ismrmrd'`` branch ``BartKspaceDataset`` raises
on. The recon composes the existing ``NUFFTForwardModel`` / ``IterativeDCF`` —
never a new gridding implementation. ``IsmrmrdStrategy`` (the SSOT reader) decodes
the container; this dataset is the semantic/recon layer.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torchio as tio
from torch.utils.data import Dataset

from mriforge.config.schemas.data import IsmrmrdConfigSchema
from mriforge.data.io_strategies import IsmrmrdStrategy

__all__ = ["IsmrmrdKspaceDataset", "build_ismrmrd_index", "resolve_nominal_file"]

_SUFFIXES = (".mrd", ".h5")


def build_ismrmrd_index(data_root: str | Path, exclude_glob: str | None = None) -> list[str]:
    """List ISMRMRD files (``.mrd`` / ``.h5``) under ``data_root``, sorted.

    ``exclude_glob`` (e.g. ``"*nominal*"``) drops auxiliary files from the
    primary index so they are not reconstructed as standalone samples — they are
    instead resolved per-measured-file by :func:`resolve_nominal_file`.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"ISMRMRD data_root does not exist: {root}")
    return [
        str(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in _SUFFIXES
        and not (exclude_glob and fnmatch.fnmatch(p.name, exclude_glob))
    ]


def resolve_nominal_file(measured_path: str | Path, nominal_glob: str) -> str:
    """Resolve the single nominal-trajectory sibling of ``measured_path``.

    Globs the measured file's directory for ``nominal_glob`` (excluding the
    measured file itself) and requires exactly one match — no silent fallback
    on 0 or >1 (the pairing must be unambiguous for ``Δk_true = measured -
    nominal`` to be well-defined).
    """
    p = Path(measured_path)
    matches = sorted(
        str(c)
        for c in p.parent.glob("*")
        if c.is_file()
        and c.suffix.lower() in _SUFFIXES
        and fnmatch.fnmatch(c.name, nominal_glob)
        and str(c) != str(p)
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"resolve_nominal_file: expected exactly one file matching "
            f"{nominal_glob!r} beside {p.name} (excluding it), found "
            f"{len(matches)}: {matches}"
        )
    return matches[0]


class IsmrmrdKspaceDataset(Dataset):
    """ISMRMRD raw acquisitions → ``tio.Subject`` (measured-trajectory NUFFT recon)."""

    def __init__(
        self,
        index: list[str],
        ismrmrd_config: IsmrmrdConfigSchema,
        transform: Callable[[tio.Subject], tio.Subject] | None = None,
    ) -> None:
        if not ismrmrd_config.enabled:
            raise ValueError("IsmrmrdKspaceDataset requires data.ismrmrd.enabled=true.")
        self.index = list(index)
        self.cfg = ismrmrd_config
        self.transform = transform
        self._reader = IsmrmrdStrategy()

    def __len__(self) -> int:
        return len(self.index)

    def dry_iter(self) -> list[tio.Subject]:
        """Cheap Subject shells for ``tio.Queue`` length probing (no voxel read).

        ``tio.Queue.iterations_per_epoch`` calls ``subjects_dataset.dry_iter()``
        and only needs an indexable ``Sequence[Subject]`` of the right length.
        Without it a queued arm crashes at step 0 with ``'IsmrmrdKspaceDataset'
        object has no attribute 'dry_iter'`` (the 2026-06 oracle_bssfp /
        exp_vf_29 incident class). Mirrors :meth:`OracleBssfpDataset.dry_iter`.
        """
        stub = torch.zeros(1, 1, 1, 1)
        return [tio.Subject(input=tio.ScalarImage(tensor=stub)) for _ in range(len(self))]

    def __getitem__(self, idx: int) -> tio.Subject:
        loaded = self._reader.load(self.index[idx])
        kspace = loaded["kspace"]  # [n_acq, n_coil, n_samp] complex
        traj = loaded.get("trajectory")
        if traj is None:
            raise ValueError(
                f"ismrmrd_kspace requires a measured trajectory but "
                f"{self.index[idx]} has trajectory_dimensions=0 (Cartesian?)."
            )
        recon = self._nufft_recon(kspace, traj, loaded.get("header", {}))  # [1, H, W]
        ksp_chan = self._kspace_to_channels(kspace)

        subject = tio.Subject(
            input=tio.ScalarImage(tensor=recon.unsqueeze(-1)),
            target=tio.ScalarImage(tensor=recon.clone().unsqueeze(-1)),
            kspace=tio.ScalarImage(tensor=ksp_chan.unsqueeze(-1)),
        )
        subject["trajectory"] = traj.float()  # measured traj for vf_35
        if self.cfg.emit_paired_trajectory:
            nominal_glob = self.cfg.nominal_file_glob
            assert nominal_glob  # guaranteed by IsmrmrdConfigSchema._validate
            nominal_path = resolve_nominal_file(self.index[idx], nominal_glob)
            nom = self._reader.load(nominal_path).get("trajectory")
            if nom is None:
                raise ValueError(
                    f"paired nominal file {nominal_path} carries no measured "
                    "trajectory (trajectory_dimensions=0) — cannot form Δk_true."
                )
            # Δk_true = measured - nominal is computed downstream by the scoring
            # seam; the dataset just emits the two trajectories it has.
            subject["trajectory_measured"] = traj.float()
            subject["trajectory_nominal"] = nom.float()
        if self.transform is not None:
            subject = self.transform(subject)
        return subject

    def _im_size(self, header: Any) -> tuple[int, int]:
        if self.cfg.im_size:
            return int(self.cfg.im_size[0]), int(self.cfg.im_size[1])
        enc = header.get("encoded_matrix") if isinstance(header, dict) else None
        if enc and len(enc) >= 2 and enc[0] > 0 and enc[1] > 0:
            return int(enc[0]), int(enc[1])
        raise ValueError(
            "ismrmrd_kspace: no im_size set and the XML header carries no "
            "encodedSpace matrixSize — declare data.ismrmrd.im_size=[H, W]."
        )

    def _nufft_recon(self, kspace: torch.Tensor, traj: torch.Tensor, header: Any) -> torch.Tensor:
        """Density-compensated NUFFT adjoint over the MEASURED trajectory + RSS."""
        from mriforge.infrastructure.physics.nufft_ops import NUFFTForwardModel

        n_acq, n_coil, n_samp = (int(s) for s in kspace.shape)
        im_size = self._im_size(header)
        # measured (kx, ky) flattened to [2, N], scaled into the [-pi, pi] convention
        t = traj[..., :2].reshape(-1, 2).float() * float(self.cfg.trajectory_scale)
        t = t.transpose(0, 1).contiguous()  # [2, N]
        kdata = kspace.permute(1, 0, 2).reshape(n_coil, n_acq * n_samp)  # [coil, N]

        if self.cfg.density_compensation == "iterative":
            from mriforge.infrastructure.physics.gridding import IterativeDCF

            kdata = kdata * IterativeDCF().forward(t.transpose(0, 1)).to(kdata.dtype)
        elif self.cfg.density_compensation == "radial":
            from mriforge.infrastructure.physics.gridding import RadialDCF

            dcf = RadialDCF(n_acq, n_samp).forward(t.transpose(0, 1))
            kdata = kdata * dcf.to(kdata.dtype)

        img = (
            NUFFTForwardModel(im_size=im_size).adjoint_project(kdata.unsqueeze(0), t).squeeze(0)
        )  # [coil, H, W]
        recon = (img.abs() ** 2).sum(0).sqrt()  # RSS [H, W]
        return recon.unsqueeze(0).float()  # [1, H, W]

    @staticmethod
    def _kspace_to_channels(kspace: torch.Tensor) -> torch.Tensor:
        """Carry the non-Cartesian k-space as interleaved real/imag channels over
        the (acq, sample) grid: ``[2*coil, n_acq, n_samp]`` real."""
        n_acq, n_coil, n_samp = (int(s) for s in kspace.shape)
        k = kspace.permute(1, 0, 2)  # [coil, acq, samp]
        ri = torch.stack([k.real, k.imag], dim=1)  # [coil, 2, acq, samp]
        return ri.reshape(2 * n_coil, n_acq, n_samp).float()
