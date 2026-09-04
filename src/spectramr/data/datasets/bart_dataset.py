"""BART ``.cfl``/``.hdr`` non-Cartesian / multi-coil k-space dataset (spec E2).

Turns a directory of BART arrays into ``tio.Subject`` instances carrying a
fully-sampled RSS magnitude recon (``input`` / ``target``) plus the complex
k-space (``kspace``), matching the fastMRI / m4raw contract so the existing
physics transform pipeline can undersample to form the model input.

Separation of concerns:

- ``BartCflStrategy`` (``data/io_strategies.py``) is the *pure format decoder* —
  it returns the raw complex array + BART dim vector and assumes nothing.
- This dataset is the *semantic layer*: it interprets the array via the
  validated ``data.bart.bart_dim_map`` and reconstructs through **SSOT physics**
  (``ifft2c`` for Cartesian, composed ``NUFFTForwardModel`` / ``GoldenAngleTrajectory``
  / ``RadialDCF`` for non-Cartesian) — never a new FFT / gridding implementation.

The dim map is required and validated upstream (:class:`BartConfigSchema`); the
canonicalizer refuses to silently drop a non-singleton, unmapped dim (pitfall #9).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset

from spectramr.config.schemas.data import BartConfigSchema
from spectramr.data.io_strategies import BartCflStrategy
from spectramr.infrastructure.physics.fft_ops import ifft2c

__all__ = ["BartKspaceDataset", "build_bart_index", "canonicalize_bart_array"]

# Leading→trailing role order for the canonical k-space tensor. Spatial roles
# (readout/phase/spoke) come last so the recon can address them as the final axes.
_CANONICAL_ROLE_ORDER = (
    "coil",
    "echo",
    "flip",
    "frame",
    "repetition",
    "slice",
    "map",
    "readout",
    "phase",
    "phase2",
    "spoke",
)


def build_bart_index(data_root: str | Path, file_pattern: str | None = None) -> list[str]:
    """List BART array basenames under ``data_root`` (excludes ``*_traj.cfl``).

    ``file_pattern`` (e.g. ``'b1map'``) restricts to basenames containing it, so a
    mixed directory loads only the matching arrays. Returns the suffix-stripped
    basenames (BART addresses arrays by basename), sorted for determinism. Raises
    if ``data_root`` does not exist.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"BART data_root does not exist: {root}")
    cfls = sorted(
        p
        for p in root.rglob("*.cfl")
        if not p.stem.endswith("_traj") and (file_pattern is None or file_pattern in p.stem)
    )
    return [str(p.with_suffix("")) for p in cfls]


def canonicalize_bart_array(
    array: torch.Tensor, bart_dim_map: dict[str, int]
) -> tuple[torch.Tensor, list[str]]:
    """Permute a raw BART array into canonical role order, squeezing undeclared
    singleton dims.

    Args:
        array: complex tensor whose shape equals the BART dim vector.
        bart_dim_map: role → BART dim index (validated by :class:`BartConfigSchema`).

    Returns:
        ``(canonical_tensor, present_roles)`` — ``present_roles`` lists the role
        of each output axis in order.

    Raises:
        ValueError: if a dim index exceeds the array rank, or an *undeclared* dim
            is non-singleton (refusing to silently drop a meaningful axis —
            CLAUDE.md pitfall #9).
    """
    ndim = array.dim()
    for role, idx in bart_dim_map.items():
        if idx >= ndim:
            raise ValueError(
                f"bart_dim_map['{role}']={idx} exceeds BART array rank {ndim} "
                f"(shape {tuple(array.shape)})."
            )
    mapped = set(bart_dim_map.values())
    for d in range(ndim):
        if d not in mapped and array.shape[d] != 1:
            raise ValueError(
                f"BART dim {d} has size {array.shape[d]} but no role in "
                f"bart_dim_map={bart_dim_map}; refusing to silently drop a "
                "non-singleton dimension (pitfall #9)."
            )
    present = [r for r in _CANONICAL_ROLE_ORDER if r in bart_dim_map]
    perm = [bart_dim_map[r] for r in present]
    remaining = [d for d in range(ndim) if d not in perm]  # undeclared singletons
    out_shape = [array.shape[bart_dim_map[r]] for r in present]
    canonical = array.permute(*perm, *remaining).contiguous().view(*out_shape)
    return canonical, present


class BartKspaceDataset(Dataset):
    """BART k-space → ``tio.Subject`` with fully-sampled recon + complex k-space."""

    def __init__(
        self,
        index: list[str],
        bart_config: BartConfigSchema,
        transform: Callable[[tio.Subject], tio.Subject] | None = None,
    ) -> None:
        if not bart_config.enabled:
            raise ValueError(
                "BartKspaceDataset requires data.bart.enabled=true (and a non-empty bart_dim_map)."
            )
        # cfg/reader must precede index construction: ``_build_index_record``
        # reads ``self.cfg.bart_dim_map`` / ``self.cfg.sampling`` to derive the
        # per-record recon shape from the .hdr header.
        self.cfg = bart_config
        self.transform = transform
        self._reader = BartCflStrategy()
        # F5: dict records ``{"path", "shape": [1, H, W]}`` (shape from the .hdr
        # header alone) so the queue-build patch-compat filter runs its no-voxel
        # fast path instead of NUFFT-reconstructing every volume → host OOM.
        self.index = [self._build_index_record(item) for item in index]

    def __len__(self) -> int:
        return len(self.index)

    def dry_iter(self) -> list[tio.Subject]:
        """Lightweight ``Subject`` shells for TorchIO ``Queue`` compatibility.

        ``tio.Queue.iterations_per_epoch`` calls ``self.subjects_dataset.dry_iter()``
        (torchio/data/queue.py: ``all_subjects_list = self.subjects_dataset.dry_iter()``)
        and only needs an *indexable* ``Sequence[Subject]`` of the right length — it
        counts subjects and reads each subject's optional ``num_samples`` attribute,
        never the voxel data. So we return one 1x1x1x1-stub ``Subject`` per index
        record (no ``.cfl`` payload read, no recon). Mirrors
        :meth:`UniversalMRIDataset.dry_iter`.

        Without this method a BART-backed arm wrapped in a ``tio.Queue`` crashes at
        the first training step with ``'BartKspaceDataset' object has no attribute
        'dry_iter'`` (cluster VF smoke 2026-06-16: exp_vf_21..30, the b0/b1 sim arms).
        """
        stub = torch.zeros(1, 1, 1, 1)
        subjects: list[tio.Subject] = []
        for record in self.index:
            path = str(record.get("path", "unknown"))
            subjects.append(
                tio.Subject(
                    input=tio.ScalarImage(tensor=stub),
                    path=path,
                    file_id=record.get("file_id", Path(path).stem),
                )
            )
        return subjects

    @staticmethod
    def _recon_hw_header_only(
        base: str, bart_dim_map: dict[str, int], sampling: str
    ) -> tuple[int, int] | None:
        """Recon image ``(H, W)`` from the BART ``.hdr`` header ALONE.

        No ``.cfl`` payload read, no NUFFT. Mirrors the recon geometry used by
        :meth:`_cartesian_recon` / :meth:`_noncartesian_recon`: a Cartesian
        recon is ``(readout, phase)``; a radial NUFFT recon is the
        ~2x-oversampled square ``(readout // 2)²``. This lets the queue-build
        filter drop volumes smaller than ``patch_size`` without reconstructing
        them (the F5 fix for the radial queue-build OOM). Returns ``None`` if the
        header is missing/malformed or lacks the ``readout`` role.
        """
        try:
            p = Path(base)
            hdr = (p.with_suffix("") if p.suffix == ".cfl" else p).with_suffix(".hdr")
            dims = BartCflStrategy._parse_dims(hdr.read_text())
            ro = int(dims[bart_dim_map["readout"]])
            if sampling == "cartesian":
                return (ro, int(dims[bart_dim_map["phase"]]))
            return (max(ro // 2, 1), max(ro // 2, 1))
        except (OSError, KeyError, IndexError, ValueError):
            return None

    def _build_index_record(self, item: str | dict[str, Any]) -> dict[str, Any]:
        """Normalise an index entry to ``{"path", ["shape"]}``.

        Accepts the flat string indices both :func:`build_bart_index` and
        :func:`build_index_from_manifest` return (BART addresses arrays by
        basename), plus a pre-built dict for forward-compat. Attaches the recon
        ``shape`` ``[1, H, W]`` when derivable from the header alone (F5).
        """
        rec: dict[str, Any] = dict(item) if isinstance(item, dict) else {"path": item}
        base = rec.get("path")
        if base is not None and "shape" not in rec:
            hw = self._recon_hw_header_only(str(base), self.cfg.bart_dim_map, self.cfg.sampling)
            if hw is not None:
                rec["shape"] = [1, hw[0], hw[1]]
        return rec

    def __getitem__(self, idx: int) -> tio.Subject:
        base = self.index[idx]["path"]
        array: Any = self._reader.load(base)["data"]
        if not torch.is_complex(array):
            array = array.to(torch.complex64)
        kspace, roles = canonicalize_bart_array(array, self.cfg.bart_dim_map)

        if self.cfg.sampling == "cartesian":
            recon = self._cartesian_recon(kspace, roles)
        else:
            recon = self._noncartesian_recon(kspace, roles)
        recon = recon.to(torch.float32)
        ksp_chan = self._kspace_to_channels(kspace)

        subject = tio.Subject(
            input=tio.ScalarImage(tensor=recon.unsqueeze(-1)),
            target=tio.ScalarImage(tensor=recon.clone().unsqueeze(-1)),
            kspace=tio.ScalarImage(tensor=ksp_chan.unsqueeze(-1)),
        )
        if self.cfg.b0_from_echo:
            # Derive a REAL B0 field (Hz) from the multi-echo phase evolution and
            # carry it as `b0_map` for the VF real-reference field-scoring seam.
            b0 = self._b0_map_from_echoes(kspace, roles).to(torch.float32)  # [H, W]
            subject["b0_map"] = tio.ScalarImage(
                tensor=b0.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
            )
        if self.cfg.b1_from_dam:
            # Derive a REAL B1+ efficiency map (double-angle) and carry it as
            # `b1_map` for the multi-acquisition B1 real-reference seam.
            b1 = self._b1_map_from_dam(kspace, roles).to(torch.float32)  # [H, W]
            subject["b1_map"] = tio.ScalarImage(
                tensor=b1.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
            )
        if self.transform is not None:
            subject = self.transform(subject)
        return subject

    # ── reconstruction (SSOT physics; never new FFT/gridding) ────────────────
    def _cartesian_recon(self, kspace: torch.Tensor, roles: list[str]) -> torch.Tensor:
        """Centered IFFT over (readout, phase) + RSS over coil → magnitude image."""
        h, w = roles.index("readout"), roles.index("phase")
        others = [a for a in range(len(roles)) if a not in (h, w)]
        k = kspace.permute(*others, h, w).contiguous()  # [*others, H, W]
        img = ifft2c(k)
        if "coil" in roles:
            coil_pos = others.index(roles.index("coil"))
            img = (img.abs() ** 2).sum(dim=coil_pos).sqrt()
        else:
            img = img.abs()
        return self._fold_leading_to_channel(img)

    def _noncartesian_recon(self, kspace: torch.Tensor, roles: list[str]) -> torch.Tensor:
        """Density-compensated NUFFT adjoint over (readout, spoke) + RSS over coil.

        Composes the already-tested ``GoldenAngleTrajectory`` / ``NUFFTForwardModel``
        / ``RadialDCF`` physics modules. The golden-angle trajectory is the assumed
        acquisition model (valid for the radial datasets); a measured-trajectory
        source (``sibling_cfl`` / ``ismrmrd``) is a tracked follow-up and raises
        loudly rather than fabricating a wrong recon.
        """
        from spectramr.infrastructure.physics.nufft_ops import (
            GoldenAngleTrajectory,
            NUFFTForwardModel,
        )

        if self.cfg.trajectory_source != "golden_angle":
            raise NotImplementedError(
                "BART non-Cartesian recon with trajectory_source="
                f"'{self.cfg.trajectory_source}' is not yet wired; only "
                "'golden_angle' is implemented (spec E2 follow-up)."
            )
        ro, sp = roles.index("readout"), roles.index("spoke")
        others = [a for a in range(len(roles)) if a not in (ro, sp)]
        # Order trailing dims (spoke, readout) to match the trajectory's
        # spoke-outer / sample-inner flattening.
        k = kspace.permute(*others, sp, ro).contiguous()
        lead = k.shape[:-2]
        n_spokes, n_ro = int(k.shape[-2]), int(k.shape[-1])
        im_side = max(n_ro // 2, 1)  # radial readout is ~2x oversampled
        im_size = (im_side, im_side)

        traj = GoldenAngleTrajectory(im_size, n_spokes, n_ro).generate(num_frames=1)
        traj = traj.squeeze(0)  # [2, N], coords in [-pi, pi]
        c_flat = int(np.prod(lead)) if len(lead) else 1
        kdata = k.reshape(c_flat, n_spokes * n_ro)  # [C, N]

        if self.cfg.density_compensation == "radial":
            from spectramr.infrastructure.physics.gridding import RadialDCF

            # RadialDCF wants [..., 2]; the NUFFT trajectory is [2, N].
            dcf = RadialDCF(n_spokes, n_ro).forward(traj.transpose(0, 1))  # [N]
            kdata = kdata * dcf.to(kdata.dtype)
        elif self.cfg.density_compensation == "iterative":
            from spectramr.infrastructure.physics.gridding import IterativeDCF

            dcf = IterativeDCF().forward(traj.transpose(0, 1))
            kdata = kdata * dcf.to(kdata.dtype)

        nufft = NUFFTForwardModel(im_size=im_size)
        img = nufft.adjoint_project(kdata.unsqueeze(0), traj).squeeze(0)  # [C, H, W]
        if "coil" in roles:
            coil_pos = others.index(roles.index("coil"))
            img = img.reshape(*lead, *img.shape[-2:])
            img = (img.abs() ** 2).sum(dim=coil_pos).sqrt()
        else:
            img = img.abs()
        return self._fold_leading_to_channel(img)

    def _radial_coil_images(self, k_crs: torch.Tensor) -> torch.Tensor:
        """NUFFT-adjoint a ``[coil, readout, spoke]`` radial block → ``[coil, H, W]``
        complex coil images (phase preserved — no RSS)."""
        from spectramr.infrastructure.physics.nufft_ops import (
            GoldenAngleTrajectory,
            NUFFTForwardModel,
        )

        coil, n_ro, n_sp = int(k_crs.shape[0]), int(k_crs.shape[1]), int(k_crs.shape[2])
        im_side = max(n_ro // 2, 1)
        im_size = (im_side, im_side)
        # spoke-outer / sample-inner ordering to match the trajectory flattening
        k2 = k_crs.permute(0, 2, 1).reshape(coil, n_sp * n_ro)  # [coil, N]
        traj = GoldenAngleTrajectory(im_size, n_sp, n_ro).generate(num_frames=1)
        traj = traj.squeeze(0)  # [2, N]
        if self.cfg.density_compensation == "radial":
            from spectramr.infrastructure.physics.gridding import RadialDCF

            dcf = RadialDCF(n_sp, n_ro).forward(traj.transpose(0, 1))
            k2 = k2 * dcf.to(k2.dtype)
        img = NUFFTForwardModel(im_size=im_size).adjoint_project(k2.unsqueeze(0), traj)
        return img.squeeze(0)  # [coil, H, W]

    def _b0_map_from_echoes(self, kspace: torch.Tensor, roles: list[str]) -> torch.Tensor:
        """Derive a REAL B0 field map (Hz) from the first two echoes.

        Uses the multi-coil generalisation of ``physics.b0_mapping.phase_difference``:
        the per-coil conjugate product cancels each coil's static phase, and its
        sum over coils gives the coil-combined inter-echo phase difference. Then
        ``B0[Hz] = angle(Σ_c echo1·conj(echo0)) / (2π·ΔTE)``.
        """
        if self.cfg.delta_te is None:  # guaranteed by BartConfigSchema; defensive
            raise ValueError("bart.b0_from_echo requires bart.delta_te (seconds).")
        delta_te_s = float(self.cfg.delta_te)  # plain float (closure-safe narrowing)
        e_ax = roles.index("echo")
        if int(kspace.shape[e_ax]) < 2:
            raise ValueError(
                "bart.b0_from_echo requires >= 2 echoes; got "
                f"{int(kspace.shape[e_ax])} along the 'echo' axis."
            )
        roles_e = [r for r in roles if r != "echo"]
        c, ro, sp = (
            roles_e.index("coil"),
            roles_e.index("readout"),
            roles_e.index("spoke"),
        )

        def _echo_img(e: int) -> torch.Tensor:
            ke = kspace.select(e_ax, e).permute(c, ro, sp).contiguous()
            return self._radial_coil_images(ke)  # [coil, H, W]

        img0, img1 = _echo_img(0), _echo_img(1)
        cross = (img1 * img0.conj()).sum(dim=0)  # [H, W] coil-combined
        return torch.angle(cross) / (2.0 * math.pi * delta_te_s)  # Hz

    def _b1_map_from_dam(self, kspace: torch.Tensor, roles: list[str]) -> torch.Tensor:
        """Derive a REAL B1+ transmit efficiency map from a double-angle (DAM)
        acquisition (Cartesian, two flip angles).

        ``B1_efficiency = arccos(|S_2alpha| / (2·|S_alpha|)) / alpha_nominal`` (Insko & Bolinger
        1993) — the same textbook inversion ``vf_field_extraction.MarkerB1Transmit
        DoubleAngle`` implements, applied to the coil-combined magnitude of the two
        flip measurements.
        """
        if self.cfg.b1_nominal_flip_deg is None:  # guaranteed by schema; defensive
            raise ValueError("bart.b1_from_dam requires bart.b1_nominal_flip_deg.")
        alpha_nominal = math.radians(float(self.cfg.b1_nominal_flip_deg))
        flip_ax = roles.index("flip")
        if int(kspace.shape[flip_ax]) < 2:
            raise ValueError(
                "bart.b1_from_dam requires >= 2 flip measurements; got "
                f"{int(kspace.shape[flip_ax])} along the 'flip' axis."
            )
        roles_f = [r for r in roles if r != "flip"]
        h, w = roles_f.index("readout"), roles_f.index("phase")

        def _flip_mag(fi: int) -> torch.Tensor:
            kf = kspace.select(flip_ax, fi)  # roles_f
            others = [a for a in range(len(roles_f)) if a not in (h, w)]
            k = kf.permute(*others, h, w).contiguous()
            img = ifft2c(k)  # Cartesian readout/phase
            if "coil" in roles_f:
                coil_pos = others.index(roles_f.index("coil"))
                mag = (img.abs() ** 2).sum(dim=coil_pos).sqrt()
            else:
                mag = img.abs()
            return self._fold_leading_to_channel(mag)[0]  # [H, W]

        s1, s2 = _flip_mag(0), _flip_mag(1)
        ratio = (s2 / (2.0 * s1 + 1e-8)).clamp(-1.0, 1.0)
        return torch.arccos(ratio) / alpha_nominal  # [H, W] efficiency (1.0 = nominal)

    @staticmethod
    def _fold_leading_to_channel(img: torch.Tensor) -> torch.Tensor:
        """Collapse any leading (echo/frame/...) axes into the channel slot → (C, H, W)."""
        h, w = int(img.shape[-2]), int(img.shape[-1])
        lead = img.shape[:-2]
        c = int(np.prod(lead)) if len(lead) else 1
        return img.reshape(c, h, w)

    @staticmethod
    def _kspace_to_channels(kspace: torch.Tensor) -> torch.Tensor:
        """Carry complex k-space as interleaved real/imag channels (TorchIO-safe).

        ``[*lead, H, W]`` complex → ``[2 * prod(lead), H, W]`` real, with the
        channel order ``[Re(c0), Im(c0), Re(c1), Im(c1), ...]`` matching
        ``KSpaceCoilTransforms``' interleave convention.
        """
        h, w = int(kspace.shape[-2]), int(kspace.shape[-1])
        flat = kspace.reshape(-1, h, w)  # [Cflat, H, W] complex
        ri = torch.stack([flat.real, flat.imag], dim=1)  # [Cflat, 2, H, W]
        return ri.reshape(-1, h, w).to(torch.float32)  # [2*Cflat, H, W]
