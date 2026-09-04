r"""Synthetic-pathology augmentation transform.

Implements §7 of the multi-contrast brainstorm — inserts physically
plausible lesions into clean MRI volumes by *modulating local tissue
parameters* and re-synthesising the contrast through the Bloch layer.
This is the principled alternative to image-space blob insertion: a
lesion has a local :math:`(\Delta\rho, \Delta T_1, \Delta T_2)` shift
relative to surrounding tissue, and the *spectral* signature of that
shift in k-space differs systematically from a smooth image-domain blob.
Models pre-trained with image-domain-only blob augmentation tend to
produce false signatures in k-space at inference; this transform
removes that failure mode.

The default lesion type is "white matter hyperintensity" (WMH) — a
common MS / vascular finding with $\Delta T_1 \approx 0$ ms and
$\Delta T_2 \approx +200$ ms. Other presets cover acute stroke
($\Delta T_1 \approx +400$, $\Delta T_2 \approx +50$ — slight) and
tumour core ($\Delta\rho \approx -0.2$, $\Delta T_1 \approx +800$,
$\Delta T_2 \approx +400$ — strong).

Lesion geometry is a smooth Gaussian blob with random radius and
position. The blob is multiplied into a *delta map* added to the
underlying tissue parameters; the resulting parameters are run through
:class:`~spectramr.infrastructure.physics.multi_physics_bloch.MultiPhysicsBlochLayer`
to produce the lesioned image.

Limitations
-----------
- The transform requires base tissue-parameter estimates :math:`(\rho_0,
  T_1^0, T_2^0)` for the input image. If they're not supplied via
  ``subject["tissue_params"]``, the transform falls back to a *crude
  tissue-class lookup* using a brain-extraction mask: white matter, grey
  matter, and CSF get canonical 3T values. This is sufficient for
  augmentation purposes but not quantitatively accurate.
- Single-contrast acquisition metadata is required to re-synthesise the
  lesioned image; supplied via ``subject["acquisition_params"]``.

The transform is registered with the standard ``tio.Transform`` API and
composes with every other transform in ``src/data/transforms/``.

References
----------
[1]  E. M. Haacke *et al.*, *Magnetic Resonance Imaging: Physical
     Principles and Sequence Design*, Wiley, 2014.
[3]  B. Billot *et al.*, "SynthSeg: Segmentation of brain MRI scans of
     any contrast and resolution without retraining," *Med. Image
     Anal.*, vol. 86, p. 102789, 2023.

§7 of ``docs/multi_contrast.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torchio as tio

from spectramr.infrastructure.physics.multi_physics_bloch import MultiPhysicsBlochLayer

__all__ = ["SyntheticLesionTransform"]


# Canonical lesion parameter shifts (ms / unitless for ρ).
_LESION_PRESETS: dict[str, dict[str, float]] = {
    "wmh": {"d_rho": 0.0, "d_t1": 0.0, "d_t2": 200.0},
    "stroke_acute": {"d_rho": 0.0, "d_t1": 400.0, "d_t2": 50.0},
    "tumour_core": {"d_rho": -0.2, "d_t1": 800.0, "d_t2": 400.0},
    "edema": {"d_rho": 0.05, "d_t1": 200.0, "d_t2": 300.0},
}

# Canonical 3T tissue parameters (ms / unitless).
_TISSUE_TABLE_3T: dict[str, dict[str, float]] = {
    "wm": {"rho": 0.65, "t1": 832.0, "t2": 79.6},
    "gm": {"rho": 0.78, "t1": 1331.0, "t2": 110.0},
    "csf": {"rho": 1.00, "t1": 4000.0, "t2": 2000.0},
    "background": {"rho": 0.0, "t1": 1.0, "t2": 1.0},
}


def _gaussian_blob_3d(
    shape: tuple[int, int, int],
    centre: tuple[float, float, float],
    radius: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a (X, Y, Z) Gaussian blob centred at ``centre``.

    The amplitude at distance r is ``exp(-r**2 / (2 * (radius/2)**2))``
    so ``radius`` is roughly the FWHM of the blob.
    """
    X, Y, Z = shape
    xs = torch.arange(X, device=device, dtype=dtype) - centre[0]
    ys = torch.arange(Y, device=device, dtype=dtype) - centre[1]
    zs = torch.arange(Z, device=device, dtype=dtype) - centre[2]
    xx = xs.view(-1, 1, 1)
    yy = ys.view(1, -1, 1)
    zz = zs.view(1, 1, -1)
    r2 = xx**2 + yy**2 + zz**2
    sigma = max(radius / 2.0, 1e-3)
    return torch.exp(-r2 / (2.0 * sigma**2))


class SyntheticLesionTransform(tio.Transform):
    r"""Insert one or more synthetic lesions into clean MRI volumes.

    Args:
        lesion_type: Preset name in ``{"wmh", "stroke_acute",
            "tumour_core", "edema"}`` selecting the
            :math:`(\Delta\rho, \Delta T_1, \Delta T_2)` shift, or a
            custom dict with those keys.
        n_lesions: Number of lesions to insert per call.
        radius_voxels: Half-FWHM range (min, max) for the lesion blob.
        intensity_jitter: Multiplicative jitter on the lesion amplitude
            (mean 1, std controlled here). 0.0 disables jitter.
        keys: Subject keys to apply the transform to.
        require_tissue_params: If True, raise when the subject doesn't
            carry ``tissue_params``; if False, fall back to the
            crude WM lookup (acceptable for augmentation but not
            quantitative).
        seed: Optional integer seed for the per-call RNG.
    """

    def __init__(
        self,
        lesion_type: str | dict[str, float] = "wmh",
        n_lesions: int = 1,
        radius_voxels: tuple[float, float] = (3.0, 8.0),
        intensity_jitter: float = 0.1,
        keys: Sequence[str] = ("input", "target"),
        require_tissue_params: bool = False,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(lesion_type, str):
            if lesion_type not in _LESION_PRESETS:
                raise ValueError(
                    f"lesion_type must be one of {list(_LESION_PRESETS)} "
                    f"or a custom dict, got {lesion_type!r}."
                )
            self.shifts = dict(_LESION_PRESETS[lesion_type])
            self.lesion_label = lesion_type
        else:
            for k in ("d_rho", "d_t1", "d_t2"):
                if k not in lesion_type:
                    raise ValueError(
                        f"custom lesion_type must contain {k!r}; got {sorted(lesion_type)}."
                    )
            self.shifts = dict(lesion_type)
            self.lesion_label = "custom"
        if n_lesions < 1:
            raise ValueError(f"n_lesions must be >= 1, got {n_lesions}")
        if radius_voxels[0] <= 0 or radius_voxels[1] <= 0:
            raise ValueError(f"radius_voxels must be positive; got {radius_voxels}")
        if intensity_jitter < 0:
            raise ValueError(f"intensity_jitter must be >= 0, got {intensity_jitter}")
        self.n_lesions = n_lesions
        self.radius_voxels = (float(radius_voxels[0]), float(radius_voxels[1]))
        self.intensity_jitter = intensity_jitter
        self.keys = tuple(keys)
        self.require_tissue_params = require_tissue_params
        self.seed = seed
        self._bloch = MultiPhysicsBlochLayer(learnable_flip_angle=False).eval()
        for p in self._bloch.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _crude_tissue_params(
        image: torch.Tensor, brain_mask: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        """Fallback (ρ, T1, T2) from a magnitude image + brain mask.

        Bins voxels by intensity quartile and assigns the canonical
        WM / GM / CSF entry from ``_TISSUE_TABLE_3T``. Crude but enough
        to make the Bloch synthesis well-conditioned for augmentation.
        """
        rho = torch.full_like(image, _TISSUE_TABLE_3T["background"]["rho"])
        t1 = torch.full_like(image, _TISSUE_TABLE_3T["background"]["t1"])
        t2 = torch.full_like(image, _TISSUE_TABLE_3T["background"]["t2"])
        if brain_mask is None:
            inside = image > image.mean()
        else:
            inside = brain_mask > 0.5
        # Within the brain mask, use intensity tertiles → WM/GM/CSF.
        if inside.any():
            inside_vals = image[inside]
            q1 = torch.quantile(inside_vals, 0.33)
            q2 = torch.quantile(inside_vals, 0.66)
            wm = inside & (image <= q1)
            gm = inside & (image > q1) & (image <= q2)
            csf = inside & (image > q2)
            rho[wm] = _TISSUE_TABLE_3T["wm"]["rho"]
            t1[wm] = _TISSUE_TABLE_3T["wm"]["t1"]
            t2[wm] = _TISSUE_TABLE_3T["wm"]["t2"]
            rho[gm] = _TISSUE_TABLE_3T["gm"]["rho"]
            t1[gm] = _TISSUE_TABLE_3T["gm"]["t1"]
            t2[gm] = _TISSUE_TABLE_3T["gm"]["t2"]
            rho[csf] = _TISSUE_TABLE_3T["csf"]["rho"]
            t1[csf] = _TISSUE_TABLE_3T["csf"]["t1"]
            t2[csf] = _TISSUE_TABLE_3T["csf"]["t2"]
        return {"rho": rho, "t1": t1, "t2": t2}

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        rng = torch.Generator(device="cpu")
        if self.seed is not None:
            rng.manual_seed(self.seed)
        else:
            rng.seed()

        for key in self.keys:
            if key not in subject:
                continue
            image = subject[key]
            data = image.data  # (C, X, Y, Z)
            if data.shape[0] != 1:
                # Multi-channel inputs are typically real/imag pairs; the
                # synthesis here operates on magnitude — caller should pass
                # magnitude images.
                continue
            magnitude = data[0]  # (X, Y, Z)
            shape = tuple(magnitude.shape)

            # Resolve tissue parameters
            tissue = subject.get("tissue_params") if hasattr(subject, "get") else None
            if tissue is None:
                if self.require_tissue_params:
                    raise ValueError(
                        f"SyntheticLesionTransform requires subject['tissue_params'] "
                        f"but none was found for key={key!r}."
                    )
                brain_mask = subject["brain_mask"].data[0] if "brain_mask" in subject else None
                tissue = self._crude_tissue_params(magnitude, brain_mask)

            rho = tissue["rho"].to(magnitude.dtype)
            t1 = tissue["t1"].to(magnitude.dtype)
            t2 = tissue["t2"].to(magnitude.dtype)

            # Sample lesion positions inside the brain mask if available.
            inside = None
            if "brain_mask" in subject:
                inside = subject["brain_mask"].data[0] > 0.5
            for _ in range(self.n_lesions):
                # Sample a centre voxel inside the brain (or anywhere).
                if inside is not None and inside.any():
                    flat_idx = torch.randint(
                        0, int(inside.sum().item()), (1,), generator=rng
                    ).item()
                    voxels = inside.nonzero(as_tuple=False)
                    cx, cy, cz = voxels[flat_idx].tolist()
                else:
                    cx = torch.randint(0, shape[0], (1,), generator=rng).item()
                    cy = torch.randint(0, shape[1], (1,), generator=rng).item()
                    cz = torch.randint(0, shape[2], (1,), generator=rng).item()
                radius = (
                    self.radius_voxels[0]
                    + (self.radius_voxels[1] - self.radius_voxels[0])
                    * torch.rand(1, generator=rng).item()
                )
                blob = _gaussian_blob_3d(
                    shape, (cx, cy, cz), radius, magnitude.device, magnitude.dtype
                )
                if self.intensity_jitter > 0:
                    jitter = 1.0 + self.intensity_jitter * (torch.randn(1, generator=rng).item())
                    blob = blob * jitter
                rho = rho + blob * self.shifts["d_rho"]
                t1 = t1 + blob * self.shifts["d_t1"]
                t2 = t2 + blob * self.shifts["d_t2"]

            # Resolve acquisition metadata
            acq = subject["acquisition_params"] if "acquisition_params" in subject else None
            if acq is None:
                # Default to spin-echo PD-like contrast — magnitude image
                # roughly preserved, just modulates with the new tissue.
                acq = {"TE": 15.0, "TR": 2000.0, "TI": 0.0, "contrast_type": "spin_echo"}

            meta = dict(acq) if isinstance(acq, dict) else dict(acq.items())
            ct = meta.get("contrast_type")
            if isinstance(ct, str):
                ct_int = {
                    "spin_echo": MultiPhysicsBlochLayer.SEQ_SPIN_ECHO,
                    "inversion_recovery": MultiPhysicsBlochLayer.SEQ_INVERSION_RECOVERY,
                    "gradient_echo": MultiPhysicsBlochLayer.SEQ_GRADIENT_ECHO,
                    "diffusion_weighted": MultiPhysicsBlochLayer.SEQ_DIFFUSION_WEIGHTED,
                }.get(ct, MultiPhysicsBlochLayer.SEQ_SPIN_ECHO)
                meta["contrast_type"] = ct_int

            # Run Bloch synthesis on the lesioned tissue parameters.
            params = {
                "rho": rho.unsqueeze(0).unsqueeze(0),
                "t1": t1.unsqueeze(0).unsqueeze(0),
                "t2": t2.unsqueeze(0).unsqueeze(0),
            }
            with torch.no_grad():
                signal = self._bloch(tissue_params=params, metadata=meta)
            new_data = signal.squeeze(0)  # (1, X, Y, Z) → (1, X, Y, Z)
            image.set_data(new_data.to(data.dtype))

        return subject
