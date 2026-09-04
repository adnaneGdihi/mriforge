"""Sample a few records/slices per dataset for EDA.

Reads through the data-layer SSOT (``io_strategies``) plus a directory-scan fallback,
because the repo manifests are frequently out of sync with what is physically mirrored
on a given machine (a manifest ``data_root`` may be nested below where the files actually
sit, or absent entirely). The EDA's job is to reflect *what is on disk*, so the loader
resolves a real scan directory (manifest ``data_root`` → nearest existing ancestor) and
globs the data files it finds there. Everything degrades gracefully: a missing directory,
an unreadable file, or an oversized file yields a note and a smaller sample, never a crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from spectramr.data.eda.catalog import _SHARED_CONTAINERS, resolve_scan_dir
from spectramr.data.io_strategies import IOStrategyFactory, load_tensor_from_file
from spectramr.data.profiler import profile_dataset
from spectramr.infrastructure.physics.fft_ops import coil_combine_rss, ifft2c

# Extensions we know how to read for EDA, in rough priority order.
_DATA_EXTS = (
    ".h5",
    ".nii.gz",
    ".nii",
    ".png",
    ".pt",
    ".npy",
    ".mat",
    ".mrd",
    ".cfl",
    ".dcm",
    ".ima",
)
# Reconstructed-image formats — surfaced as a fallback for datasets whose primary read yields only
# raw k-space (osi2one_47mt_recon ships 176 .dcm next to its ISMRMRD .h5).
_RECON_IMAGE_EXTS = (".dcm", ".ima", ".nii.gz", ".nii", ".png")
# Files larger than this are profiled (header/card) but not voxel-loaded — guards against
# the multi-GB kasper SPIFI spiral files (~2.2 GB each).
_MAX_VOXEL_BYTES = 1_200_000_000


@dataclass
class DatasetSample:
    images: list[torch.Tensor] = field(default_factory=list)  # magnitude [H,W]
    kspace: list[torch.Tensor] = field(default_factory=list)  # complex [C,H,W] or [H,W]
    complex_images: list[torch.Tensor] = field(default_factory=list)  # complex recon [H,W]
    frames: list[torch.Tensor] = field(default_factory=list)  # temporal frames [H,W] (cine/4D)
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    scan_dir: Path | None = None


def _suffix(path: Path) -> str:
    s = path.suffix.lower()
    if s == ".gz" and path.stem.lower().endswith(".nii"):
        return ".nii.gz"
    return s


def _scan_dir(entry) -> Path | None:
    """Resolve the on-disk directory to scan for this dataset's real files.

    Delegates to :func:`spectramr.data.eda.catalog.resolve_scan_dir` (the single SSOT used by the
    catalog's tier decision too): the manifest ``data_root`` when it holds files, else a sibling
    ``extracted/`` (external sets unpacked away from ``…/raw``), else the nearest dataset-specific
    ancestor (a ``data_root`` nested *below* the files, m4raw). The walk stops at shared-container
    names so an absent dataset never anchors on a sibling's data.
    """
    return resolve_scan_dir(entry.data_root)


def _find_files(scan_dir: Path, file_type: str, budget: int) -> list[Path]:
    """Recursively collect up to ``budget`` data files, preferring ``file_type``'s ext.

    Collects a bounded pool *lazily* (never materialising every path under a huge cluster
    directory), then sorts the pool so the result is deterministic.
    """
    pref = {
        "h5": ".h5",
        "nifti": ".nii.gz",
        "png": ".png",
        "mat": ".mat",
        "ismrmrd": ".h5",
        "mrd": ".mrd",
        "bart": ".cfl",
    }.get((file_type or "").lower())
    pool: list[Path] = []
    for p in scan_dir.rglob("*"):
        if p.is_file() and _suffix(p) in _DATA_EXTS:
            pool.append(p)
            if len(pool) >= budget * 12:  # bounded pool, then prioritise
                break
    pool.sort(key=lambda p: (pref is not None and _suffix(p) != pref, str(p)))
    return pool[: max(budget, 1)]


def _find_recon_images(scan_dir: Path, budget: int) -> list[Path]:
    """A few reconstructed-image files (:data:`_RECON_IMAGE_EXTS`) under ``scan_dir`` — the
    fallback when a dataset's raw k-space filled the sample budget but it also ships recon images
    (DICOM/NIfTI/PNG). Bounded, deterministic."""
    pool: list[Path] = []
    for p in scan_dir.rglob("*"):
        if p.is_file() and _suffix(p) in _RECON_IMAGE_EXTS:
            pool.append(p)
            if len(pool) >= budget * 4:
                break
    pool.sort(key=str)
    return pool[: max(budget, 1)]


_ARCHIVE_EXTS = (".zip", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar")


def _find_archives(scan_dir: Path, budget: int) -> list[Path]:
    """Up to ``budget`` archives under ``scan_dir`` — a downloaded-but-not-unpacked dataset keeps
    its images inside ``.tar.gz`` / ``.zip`` (kirby_kki_multimodal: MMRR-*.tar.gz)."""
    pool: list[Path] = []
    for p in scan_dir.rglob("*"):
        if p.is_file() and any(p.name.lower().endswith(e) for e in _ARCHIVE_EXTS):
            pool.append(p)
            if len(pool) >= budget * 8:
                break
    pool.sort(key=str)
    return pool[: max(budget, 1)]


def _decode_member_image(name: str, data: bytes) -> torch.Tensor | None:
    """Decode an archive member's bytes into a 2-D image — NIfTI / PNG / DICOM (extension-less
    members are tried as DICOM). Returns the reduced 2-D plane, or ``None``."""
    import io

    import numpy as np

    low = name.lower()
    leaf = low.rsplit("/", 1)[-1]
    try:
        if low.endswith((".nii.gz", ".nii")):
            import gzip

            import nibabel as nib

            raw = gzip.decompress(data) if low.endswith(".gz") else data
            img = nib.Nifti1Image.from_bytes(raw)
            return _to_2d(torch.from_numpy(np.asarray(img.dataobj, dtype=np.float32)))
        if low.endswith(".png"):
            from PIL import Image

            arr = np.asarray(Image.open(io.BytesIO(data)).convert("F")).copy()
            return _to_2d(torch.from_numpy(arr).float())
        if low.endswith((".dcm", ".ima")) or "." not in leaf:
            import pydicom

            ds = pydicom.dcmread(io.BytesIO(data), force=True)
            return _to_2d(torch.from_numpy(np.asarray(ds.pixel_array).astype(np.float32)))
    except Exception:
        return None
    return None


def _archive_image(path: Path, max_try: int = 80) -> torch.Tensor | None:
    """Read one 2-D image slice from inside an archive (``.zip`` / ``.tar.*``) without unpacking to
    disk — reads only the first member that decodes as a NIfTI / PNG / DICOM image. ``None`` when
    the archive holds no readable image (e.g. a nested-archive or TWIX-only zip)."""
    import tarfile
    import zipfile

    low = path.name.lower()
    img_exts = (".nii.gz", ".nii", ".png", ".dcm", ".ima")
    try:
        if low.endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                names = [n for n in z.namelist() if not n.endswith("/")]
                names.sort(key=lambda n: (not n.lower().endswith(img_exts), n))
                for n in names[:max_try]:
                    img = _decode_member_image(n, z.read(n))
                    if img is not None:
                        return img
        elif any(low.endswith(e) for e in (".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar")):
            with tarfile.open(path, "r:*") as t:
                members = [m for m in t.getmembers() if m.isfile()]
                members.sort(key=lambda m: (not m.name.lower().endswith(img_exts), m.name))
                for m in members[:max_try]:
                    fh = t.extractfile(m)
                    if fh is None:
                        continue
                    img = _decode_member_image(m.name, fh.read())
                    if img is not None:
                        return img
    except Exception:
        return None
    return None


def _h5_mid_slice(path: Path) -> int | None:
    """Middle slice index of an h5 k-space / recon dataset — a header read, no voxel load.

    Returned so the io_strategy reads a single ``f[key][mid]`` slice instead of the whole
    multi-slice multi-coil volume (which OOMs on full-size cluster data at any real budget).
    """
    try:
        import h5py

        with h5py.File(path, "r", swmr=True) as f:
            for key in ("kspace", "reconstruction_rss", "reconstruction_esc"):
                shp = getattr(f.get(key), "shape", None)
                if shp is not None:
                    # Only 3-D+ (slices, [coils,] H, W) has a slice axis; a 2-D (H, W) image
                    # must be loaded whole — slicing it would read a single row.
                    return (shp[0] // 2) if len(shp) >= 3 else None
    except Exception:  # fall back to a full read only if the peek fails
        return None
    return None


def _cfl_central_slice(path: Path) -> torch.Tensor | None:
    """Central 2-D plane of a BART ``.cfl`` via ``memmap`` — O(plane) memory, never materializes
    the (possibly multi-GB) payload.

    BART stores a complex64 array column-major with its dim vector in the sibling ``.hdr``. We
    memmap the payload, view it at the header shape, and reduce to the two largest axes (the
    spatial / readout-by-spokes plane), taking the centre of every other axis (coils, slices,
    echoes). Returns complex ``[H, W]``, or ``None`` if the pair is missing / inconsistent.
    """
    base = path.with_suffix("")
    hdr, cfl = base.with_suffix(".hdr"), base.with_suffix(".cfl")
    if not (hdr.is_file() and cfl.is_file()):
        return None
    try:
        import numpy as np

        from spectramr.data.io_strategies import BartCflStrategy

        dims = [d for d in BartCflStrategy._parse_dims(hdr.read_text()) if d > 0]
        if len(dims) < 2:
            return None
        mm = np.memmap(cfl, dtype=np.complex64, mode="r")
        if mm.size != int(np.prod(dims)):
            return None
        view = mm.reshape(dims, order="F").squeeze()
        while view.ndim > 2:  # collapse the smallest (least informative) axis to its centre
            ax = int(np.argmin(view.shape))
            view = np.take(view, view.shape[ax] // 2, axis=ax)
        if view.ndim != 2:
            return None
        return torch.from_numpy(np.ascontiguousarray(view))  # materializes only the 2-D plane
    except Exception:
        return None


def _load_traj_cfl(data_path: Path) -> torch.Tensor | None:
    """A sibling BART trajectory (``*traj*.cfl``) for a radial data ``.cfl``, as ``(2, N)`` in
    radians — the FAITHFUL coordinates (ssa_fary_radial ships ``traj.cfl``). BART stores the
    trajectory in k-space-sample units; we scale by ``π/max`` to torchkbnufft's ``[-π, π]``.
    Returns ``None`` when no trajectory sibling exists or it can't be parsed.
    """
    try:
        import numpy as np

        from spectramr.data.io_strategies import BartCflStrategy

        cands = sorted(data_path.parent.glob("*traj*.cfl"))
        if not cands:
            return None
        tp = cands[0]
        hdr = tp.with_suffix(".hdr")
        if not hdr.is_file():
            return None
        dims = [d for d in BartCflStrategy._parse_dims(hdr.read_text()) if d > 0]
        mm = np.memmap(tp, dtype=np.complex64, mode="r")
        if not dims or mm.size != int(np.prod(dims)):
            return None
        view = np.real(mm.reshape(dims, order="F").squeeze())  # coords are real-valued
        if view.ndim < 2 or view.shape[0] < 2:
            return None
        coords = np.ascontiguousarray(view[:2].reshape(2, -1))  # (2, N) — kx, ky
        t = torch.from_numpy(coords).float()
        m = float(t.abs().max())
        return t / m * torch.pi if m > 0 else None
    except Exception:
        return None


def _radial_nufft_recon(ksp: torch.Tensor, traj: torch.Tensor | None = None) -> torch.Tensor | None:
    """Adjoint-NUFFT image from a 2-D radial k-space plane ``(spokes, samples)`` (single coil).

    Uses the measured ``traj`` ``(2, N)`` when supplied (faithful — ssa_fary_radial), else
    synthesizes a golden-angle radial trajectory (**best-effort**: assumes the standard radial
    golden-angle scheme and a ``(spokes, samples)`` layout, so the image may be rotated/incorrect
    if the acquisition differs — the raw k-space log-magnitude remains the ground truth). Returns
    magnitude ``[H, W]``, or ``None`` on any failure.
    """
    try:
        from spectramr.infrastructure.physics.nufft_utils import apply_nufft_adjoint
        from spectramr.infrastructure.physics.trajectories import TrajectoryFactory

        if ksp.ndim != 2:
            return None
        spokes, samples = int(ksp.shape[0]), int(ksp.shape[1])
        if spokes < 2 or samples < 4:
            return None
        n = min(max(samples, 32), 512)  # image grid ~ readout length, bounded
        im_size = (n, n)
        dcf: torch.Tensor | None = None
        if traj is None or traj.ndim != 2 or traj.shape[1] != spokes * samples:
            traj, dcf = TrajectoryFactory.get_radial_trajectory(
                im_size,
                num_spokes=spokes,
                samples_per_spoke=samples,
                golden_angle=True,
            )
        data = ksp.reshape(1, 1, -1).to(torch.complex64)  # (B, C, N) spoke-major
        img = apply_nufft_adjoint(
            data,
            traj.unsqueeze(0),
            None if dcf is None else dcf.unsqueeze(0),
            im_size=im_size,
            device=torch.device("cpu"),
        )
        return img.abs().squeeze().float()
    except Exception:
        return None


def _recon_from_kspace(ksp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (magnitude[H,W], complex_recon[H,W]). Multicoil → RSS."""
    img = ifft2c(ksp)  # [C,H,W]
    if ksp.shape[0] > 1:
        mag = coil_combine_rss(img.unsqueeze(0)).squeeze()
        cplx = img[0]
    else:
        cplx = img[0]
        mag = cplx.abs()
    return mag.float(), cplx


def _load_one(
    path: Path, sample: DatasetSample, cartesian: bool = True, temporal: bool = False
) -> None:
    """Read one file and append its image / k-space / temporal frames to the sample (best-effort).

    ``cartesian=False`` (radial / spiral / non-Cartesian k-space): the raw k-space is kept for
    the log-magnitude view, but **no** ``ifft2c`` image is produced — Cartesian-reconstructing
    non-Cartesian samples yields a meaningless 'strip' image. Proper recon needs NUFFT gridding,
    which the EDA does not attempt.

    ``temporal=True`` (cine / 4D-flow / 5D): the array's time axis is sampled into ``frames`` (the
    temporal figure suite) instead of being collapsed to a single still.
    """
    suf = _suffix(path)
    # Some formats are read one slice at a time, so their on-disk size is irrelevant — only
    # formats we must read wholesale are size-gated:
    #   * h5 with a slice axis → lazy h5py indexing (kasper SPIFI ~2.2 GB, mridata_org);
    #   * BART .cfl → memmap of its column-major central plane (radial .cfl volumes);
    #   * v7.3 .mat (HDF5) → lazy h5py mid-slice / per-frame indexing (bssfp_qsm vol*.mat, osu cine).
    # Without these, the blanket size guard reduced large-but-sliceable datasets to a card.
    h5_mid = _h5_mid_slice(path) if suf == ".h5" else None
    cfl_sliceable = suf == ".cfl" and path.with_suffix(".hdr").is_file()
    mat_lazy = suf == ".mat" and _is_hdf5(path)
    if (
        h5_mid is None
        and not cfl_sliceable
        and not mat_lazy
        and path.stat().st_size > _MAX_VOXEL_BYTES
    ):
        sample.notes.append(f"skipped voxel load (>{_MAX_VOXEL_BYTES // 10**9} GB): {path.name}")
        return
    try:
        if suf == ".png":
            import numpy as np
            from PIL import Image

            arr = np.asarray(Image.open(path).convert("F")).copy()
            _append_image_or_note(sample, _to_2d(torch.from_numpy(arr).float()), path)
            return
        if suf in (".pt", ".npy"):
            t = load_tensor_from_file(str(path)).float()
            _append_array(sample, t, path, temporal)
            return
        if suf == ".cfl":  # BART k-space — memmap a central plane (radial/spiral, non-Cartesian)
            ksp = _cfl_central_slice(path)
            if ksp is not None and _append_kspace_or_note(sample, ksp, path):
                if cartesian:
                    mag, cplx = _recon_from_kspace(ksp if ksp.ndim == 3 else ksp.unsqueeze(0))
                    sample.images.append(mag)
                    sample.complex_images.append(cplx)
                else:
                    # radial/spiral: adjoint-NUFFT recon — measured trajectory if a sibling
                    # traj.cfl exists (ssa_fary), else best-effort golden-angle (labeled).
                    traj = _load_traj_cfl(path)
                    measured = traj is not None and traj.shape[1] == ksp.numel()
                    recon = _radial_nufft_recon(ksp, traj if measured else None)
                    if recon is not None and _is_plane(recon):
                        sample.images.append(recon)
                        label = (
                            "measured trajectory"
                            if measured
                            else "golden-angle assumed — approximate"
                        )
                        sample.notes.append(f"radial NUFFT recon ({label}): {path.name}")
            return
        if suf == ".mat":
            _load_mat(path, sample, temporal, lazy=mat_lazy)
            return
        # lazy single-slice read for sliceable h5 — avoids full-volume OOM
        meta = {"slice_index": h5_mid} if h5_mid is not None else None
        loaded = IOStrategyFactory.get("auto").load(str(path), meta)
        ksp = loaded.get("kspace")
        data = loaded.get("data")
        if ksp is not None and torch.is_tensor(ksp):
            ksp = _drop_singletons(_middle_slice(ksp))  # (N,1,M) raw acquisition → (N,M) plane
            if _append_kspace_or_note(sample, ksp, path) and cartesian:
                # non-Cartesian: skip ifft2c — it would fabricate a 'strip' image
                mag, cplx = _recon_from_kspace(ksp if ksp.ndim == 3 else ksp.unsqueeze(0))
                sample.images.append(mag)
                sample.complex_images.append(cplx)
        elif data is not None and torch.is_tensor(data):
            _append_array(sample, data, path, temporal)
        else:
            arr = _largest_array(loaded.get("variables") if isinstance(loaded, dict) else None)
            if arr is not None:
                _append_array(sample, arr, path, temporal)
    except Exception as exc:
        sample.notes.append(f"error reading {path.name}: {type(exc).__name__}")


def _is_hdf5(path: Path) -> bool:
    """True if ``path`` is an HDF5 file (v7.3 ``.mat`` is HDF5 → lazily h5py-sliceable). A cheap
    8-byte magic-number read; no full open."""
    try:
        with open(path, "rb") as fh:
            return fh.read(8) == b"\x89HDF\r\n\x1a\n"
    except Exception:
        return False


def _load_mat(path: Path, sample: DatasetSample, temporal: bool, lazy: bool) -> None:
    """Read a MATLAB ``.mat`` for EDA. A ``.mat`` is a *container of named variables*, so the main
    array is found by size. v7.3 (HDF5) is read lazily one plane at a time (``lazy=True``); legacy
    v5 is read whole via ``MatStrategy`` only when under the size cap.
    """
    if temporal:
        frames = _mat_time_frames(path) if lazy else []
        if not frames and not lazy and path.stat().st_size <= _MAX_VOXEL_BYTES:
            arr = _largest_array(IOStrategyFactory.get("auto").load(str(path)).get("variables"))
            frames = _to_time_frames(arr.float()) if arr is not None else []
        if frames:
            sample.frames.extend(frames)
            sample.images.append(frames[len(frames) // 2])  # representative still for the montage
            return
    arr = _mat_mid_slice(path) if lazy else None
    if arr is None and not lazy and path.stat().st_size <= _MAX_VOXEL_BYTES:
        arr = _largest_array(IOStrategyFactory.get("auto").load(str(path)).get("variables"))
    if arr is not None:
        _append_array(sample, arr, path, temporal=False)  # already reduced to a plane / volume
    elif lazy or path.stat().st_size > _MAX_VOXEL_BYTES:
        sample.notes.append(f"no readable array in {path.name}")


def _mat_largest_dataset(f, min_ndim: int) -> str | None:
    """Name of the largest (by element count) ``>= min_ndim``-D numeric dataset in an open HDF5
    ``.mat`` — the container's main array (skips scalar / metadata variables)."""
    import numpy as np

    best: tuple[int, str] | None = None

    def visit(name: str, obj: object) -> None:
        nonlocal best
        shp = getattr(obj, "shape", None)
        dt = getattr(obj, "dtype", None)
        if shp is None or dt is None or len(shp) < min_ndim:
            return
        names = getattr(dt, "names", None)
        if not ((names and {"real", "imag"} <= set(names)) or dt.kind in "fci"):
            return
        n = int(np.prod(shp))
        if best is None or n > best[0]:
            best = (n, name)

    f.visititems(visit)
    return best[1] if best else None


def _decode_mat_array(arr: object) -> torch.Tensor:
    """Decode a raw h5py-read MATLAB array (incl. the complex ``[('real',...),('imag',...)]``
    compound dtype) into a tensor."""
    import numpy as np

    names = getattr(getattr(arr, "dtype", None), "names", None)
    if names and {"real", "imag"} <= set(names):
        arr = arr["real"].astype(np.float32) + 1j * arr["imag"].astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr))


def _mat_mid_slice(path: Path) -> torch.Tensor | None:
    """Lazy middle plane of the largest array in a v7.3 (HDF5) ``.mat`` — reads ONE slice via
    h5py instead of the whole multi-GB volume. Returns a 2-D+ tensor, or None."""
    try:
        import h5py

        with h5py.File(path, "r", swmr=True) as f:
            name = _mat_largest_dataset(f, min_ndim=2)
            if name is None:
                return None
            ds = f[name]
            assert isinstance(ds, h5py.Dataset)
            raw = ds[ds.shape[0] // 2] if ds.ndim >= 3 else ds[()]
            return _decode_mat_array(raw)
    except Exception:
        return None


def _mat_time_frames(path: Path, k: int = 6) -> list[torch.Tensor]:
    """Lazy ``k`` evenly-spaced frames along the leading (time) axis of the largest array in a
    v7.3 (HDF5) ``.mat`` — reads ``k`` planes, never the whole volume (cine / 4D-flow)."""
    try:
        import h5py
        import numpy as np

        with h5py.File(path, "r", swmr=True) as f:
            name = _mat_largest_dataset(f, min_ndim=3)
            if name is None:
                return []
            ds = f[name]
            assert isinstance(ds, h5py.Dataset)
            tt = ds.shape[0]
            idxs = sorted(set(np.linspace(0, tt - 1, min(k, tt)).round().astype(int).tolist()))
            return [
                _to_2d(
                    (lambda t: (t.abs() if t.is_complex() else t).float())(_decode_mat_array(ds[i]))
                )
                for i in idxs
            ]
    except Exception:
        return []


def _to_time_frames(t: torch.Tensor, k: int = 6) -> list[torch.Tensor]:
    """Up to ``k`` 2-D frames sampled along the largest non-spatial (time) axis of an in-memory
    N-D volume — the two largest axes are taken as spatial. Returns ``[]`` when there is no time
    axis (< 3-D after squeeze). Used for whole-read v5 ``.mat`` / 4-D ``.npy``."""
    import numpy as np

    t = t.squeeze()
    if t.ndim < 3:
        return []
    order = sorted(range(t.ndim), key=lambda i: t.shape[i], reverse=True)
    spatial = set(order[:2])  # the two largest axes are H, W
    t_ax = max((i for i in range(t.ndim) if i not in spatial), key=lambda i: t.shape[i])
    tt = t.shape[t_ax]
    idxs = sorted(set(np.linspace(0, tt - 1, min(k, tt)).round().astype(int).tolist()))
    return [_to_2d(t.select(t_ax, i)) for i in idxs]


def _drop_singletons(t: torch.Tensor) -> torch.Tensor:
    """Collapse size-1 dims — a singleton coil/echo in a raw acquisition read ((N,1,M)→(N,M),
    (1,H,W)→(H,W)) — so a squeezable plane is not mistaken for a malformed read. A genuine 1-D
    vector ((1,W)→(W,)) stays 1-D and is still rejected by :func:`_is_plane`.
    """
    return t.squeeze() if 1 in tuple(t.shape) else t


def _append_kspace_or_note(sample: DatasetSample, ksp: torch.Tensor, path: Path) -> bool:
    """Append a k-space plane, or record a ``skipped malformed read`` note for a degenerate
    (1-D / non-plane) read (the osi2one wrong-key bug). Returns whether it was appended."""
    if _is_plane(ksp):
        sample.kspace.append(ksp)
        return True
    sample.notes.append(f"skipped malformed read (shape {tuple(ksp.shape)}): {path.name}")
    return False


def _largest_array(obj: object) -> torch.Tensor | None:
    """The largest ≥2-D tensor inside a (possibly nested) MAT variables dict — the main data
    array a ``.mat`` container holds alongside scalars / metadata structs. Returns the tensor
    with the most elements, or ``None``. Bounded recursion depth guards against cyclic structs.
    """
    best: torch.Tensor | None = None
    stack: list[tuple[object, int]] = [(obj, 0)]
    while stack:
        x, d = stack.pop()
        if isinstance(x, torch.Tensor):
            if x.ndim >= 2 and (best is None or x.numel() > best.numel()):
                best = x
        elif d < 6 and isinstance(x, dict):
            stack.extend((v, d + 1) for v in x.values())
        elif d < 6 and isinstance(x, (list, tuple)):
            stack.extend((v, d + 1) for v in x)
    return best


def _append_image_or_note(sample: DatasetSample, img: torch.Tensor, path: Path) -> None:
    """Append a 2-D image, or record a ``skipped malformed read`` note. Guards every image path
    (png / pt / npy / strategy ``data``) against a 1-D read reaching ``render_montage`` (imshow) —
    the mrf_100mt_optimum fingerprint-dictionary ``(1423170,)`` crash."""
    img = _drop_singletons(img)
    if _is_plane(img):
        sample.images.append(img)
    else:
        sample.notes.append(f"skipped malformed read (shape {tuple(img.shape)}): {path.name}")


def _append_array(sample: DatasetSample, arr: torch.Tensor, path: Path, temporal: bool) -> None:
    """Append a raw N-D array as temporal frames (cine/4D — the temporal suite) or a single 2-D
    still. For temporal data it also keeps a representative mid-frame in ``images`` so the standard
    montage / intensity figures still render."""
    arr = (arr.abs() if arr.is_complex() else arr).float()
    if temporal:
        frames = _to_time_frames(arr)
        if frames:
            sample.frames.extend(frames)
            sample.images.append(frames[len(frames) // 2])
            return
    _append_image_or_note(sample, _to_2d(arr), path)


def _middle_slice(t: torch.Tensor) -> torch.Tensor:
    """Reduce an N-D array to a single [C,H,W] or [H,W] slice (middle along leading dims)."""
    while t.ndim > 3:
        t = t[t.shape[0] // 2]
    return t


def _is_plane(t: torch.Tensor) -> bool:
    """True when ``t`` is usable as a 2-D figure: ≥2 dims with both trailing dims ≥2.

    Guards against a reader that returns a 1-D vector or a degenerate ``(1, W)`` row — the
    osi2one 'wrong key / flattened read' bug, where ``IOStrategyFactory.get("auto")`` returns a
    ``(150,)`` / ``(48,)`` array. Feeding that to ``render_montage`` (``imshow``) or
    ``radial_energy`` (``np.gradient``) crashes the whole figure suite; degrade to a note instead.
    """
    return t.ndim >= 2 and min(t.shape[-2:]) >= 2


def _to_2d(t: torch.Tensor) -> torch.Tensor:
    """Reduce an N-D real image to a 2-D [H,W] slice, taking the middle of the smallest axis.

    NIfTI volumes arrive as (H, W, slices) or (slices, H, W); the through-plane axis is the
    smallest, so slicing its midpoint yields a representative in-plane image.
    """
    t = t.squeeze()
    while t.ndim > 2:
        ax = min(range(t.ndim), key=lambda i: t.shape[i])
        t = t.movedim(ax, 0)[t.shape[ax] // 2]
    return t


def _maybe_pairs(scan_dir: Path, budget: int, sample: DatasetSample) -> None:
    """For paired datasets with source/ + target/ subdirs, pair by matching filename."""
    src_dir, tgt_dir = scan_dir / "source", scan_dir / "target"
    if not (src_dir.is_dir() and tgt_dir.is_dir()):
        return
    for sp in sorted(src_dir.rglob("*"))[: budget * 3]:
        if not (sp.is_file() and _suffix(sp) in _DATA_EXTS):
            continue
        tp = tgt_dir / sp.name
        if not tp.exists():
            continue
        s_tmp, t_tmp = DatasetSample(), DatasetSample()
        _load_one(sp, s_tmp)
        _load_one(tp, t_tmp)
        if s_tmp.images and t_tmp.images:
            sample.pairs.append((s_tmp.images[0], t_tmp.images[0]))
        if len(sample.pairs) >= budget:
            break


def sample(entry, budget: int = 8) -> DatasetSample:
    out = DatasetSample()
    if budget <= 0:
        out.notes.append("dry-run: voxel load skipped")
        return out

    scan_dir = _scan_dir(entry)
    out.scan_dir = scan_dir
    if entry.tier == "absent" or scan_dir is None:
        out.notes.append(
            f"absent or no scan dir: status={entry.status}, data_root={entry.data_root}"
        )
        return out

    cartesian = entry.modality != "noncartesian"
    temporal = entry.modality == "temporal"
    if entry.modality == "paired":
        _maybe_pairs(scan_dir, budget, out)
    for path in _find_files(scan_dir, entry.file_type, budget):
        _load_one(path, out, cartesian=cartesian, temporal=temporal)
    # When the primary read yields no displayable image, surface reconstructed images. This covers
    # two cases: a non-Cartesian dataset that *also* ships recon images beside the raw k-space
    # (osi2one_47mt_recon: .dcm next to the ISMRMRD .h5), and a dataset whose scan dir holds only
    # unreadable/oversized files (kasper raw/ has huge field-monitoring .h5) while the real images
    # live in a sibling — so we also retry the parent dataset dir.
    if not out.images and not out.frames and not out.pairs:
        search_dirs = [scan_dir]
        parent = scan_dir.parent
        # Retry the parent dataset dir (kasper's images live in a sibling extracted/), but NEVER a
        # shared container (external/, databases/) — that would pull a *sibling dataset's* images
        # (the whole_heart_5d→brain contamination).
        if (
            parent != scan_dir
            and "databases" in str(parent)
            and parent.name not in _SHARED_CONTAINERS
        ):
            search_dirs.append(parent)
        for sdir in search_dirs:
            for path in _find_recon_images(sdir, budget):
                _load_one(path, out, cartesian=True, temporal=temporal)
            if out.images:
                break
        # Still nothing loose? Reconstructed/raw images are often shipped inside archives —
        # oracle_bssfp's *_DICOM.zip beside the TWIX .dat, kirby_kki_multimodal's NIfTI inside
        # MMRR-*.tar.gz — so read a slice straight out of the archive (no unpack to disk).
        if not out.images:
            for sdir in search_dirs:
                for arc in _find_archives(sdir, budget):
                    img = _archive_image(arc)
                    if img is not None:
                        _append_image_or_note(out, img, arc)
                    if out.images:
                        break
                if out.images:
                    break
    if not out.images and not out.pairs and not out.kspace and not out.frames:
        out.notes.append(f"no readable files under {scan_dir}")

    try:
        if entry.records:
            out.profile = profile_dataset(
                data_root=str(entry.data_root),
                file_type=entry.file_type,
                records=list(entry.records),
                sample=min(budget, 8),
            )
    except Exception as exc:
        out.notes.append(f"profile failed: {type(exc).__name__}")
    return out
