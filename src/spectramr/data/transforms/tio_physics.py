import logging

import torch
import torchio as tio

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

logger = logging.getLogger(__name__)


class PhysicsInformedMasking(tio.transforms.Transform):
    """
    Physics-Informed Masking: Simulates Cartesian k-space undersampling.

    CRITICAL: Undersampling applies the SAME mask to ALL coils.
    CRITICAL: Undersampling applies the SAME mask to ALL coils.
    - Input: (C=coils*2, W, H, D)
    - Output: 'input' with IDENTICAL shape, but with zeros applied by mask
    -         'target' (copy of original full kspace)

    Per-coil behavior:
    - Generates 1D mask in frequency encoding direction
    - Expands to 2D (W, H)
    - Broadcasts to ALL coils
    - Result: all coils have same sampling pattern (Cartesian property)
    """

    def __init__(self, acceleration=4, center_fraction=0.08, validate_shapes=True, **kwargs):
        """__init__.

        Args:
            acceleration (Any): Description.
            center_fraction (Any): Description.
            validate_shapes (Any): Description.
        """
        super().__init__(**kwargs)
        self.acc = acceleration
        self.cf = center_fraction
        self.validate_shapes = validate_shapes
        # Deterministic per-shape mask cache. The old ``torch.randperm`` was
        # unseeded → a genuinely different mask per item, unlike every other
        # generator in the physics SSOT (all seeded). We seed a scoped
        # ``torch.Generator`` per ``width`` so masks are reproducible (and thus
        # cacheable) without touching global RNG. Cache keyed by
        # ``(width, height)``; forkserver-pickle-safe (plain dict + a CPU
        # Generator built lazily / non-stored key).
        self._mask_cache: dict[tuple[int, int], torch.Tensor] = {}

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # [PI-FIX] Prefer 'kspace_raw' as source of truth
        """apply_transform.

        Args:
            subject (tio.Subject): Description.
        Returns:
            tio.Subject: Description.
        """
        key = "kspace_raw" if "kspace_raw" in subject else "kspace"
        if key not in subject:
            logger.debug(f"No k-space found ({key} not in subject keys: {subject.keys()})")
            return subject

        # 1. Extract Tensor: (C, W, H, D)
        # C should be even (Re, Im pairs)
        raw_tensor = subject[key].data
        c, w, h, d = raw_tensor.shape
        device = raw_tensor.device

        logger.debug(
            f"PhysicsInformedMasking input: shape={raw_tensor.shape}, "
            f"device={device}, channels (coils*2)={c}"
        )

        # 2. Complex Conversion: (C, W, H, D) -> (C/2, W, H, D) Complex
        if c % 2 != 0:
            raise ValueError(
                f"K-Space channels must be even for Re/Im pairs. Got {c}. "
                f"This suggests coil count mismatch or wrong data format."
            )

        # Reshape to (C//2, 2, W, H, D) then view as complex
        # Assuming layout is [Re1, Im1, Re2, Im2...] or [Re1..N, Im1..N]?
        # FastMRISubjectWrapper outputs: kspace_t.permute(...) -> (C*2, ...)
        # It flattens (C, 2) -> C*2.
        # So we can reshape back.
        num_coils = c // 2
        logger.debug(f"  Reshaping to complex: {c} channels → {num_coils} coils + real/imag")

        try:
            reshaped = raw_tensor.contiguous().view(num_coils, 2, w, h, d)
            logger.debug(f"  After view: {reshaped.shape}")

            permuted = reshaped.permute(0, 2, 3, 4, 1).contiguous()
            logger.debug(f"  After permute: {permuted.shape} (coils, W, H, D, complex)")

            kspace_complex = torch.view_as_complex(permuted)
            logger.debug(f"  After complex view: {kspace_complex.shape}")
        except RuntimeError as e:
            raise ValueError(
                f"Failed to convert to complex format. Input shape {raw_tensor.shape}, "
                f"expected (C={c}, W={w}, H={h}, D={d}) with even C. "
                f"Error: {e}"
            )

        # 3. Generate & Apply Mask
        # CRITICAL: Generate ONE mask, apply to ALL coils (Cartesian property)
        logger.debug(f"  Generating {self.acc}x undersampling mask (w={w}, h={h})")
        mask = self._generate_mask(w, h).to(device)

        # Reshape for broadcasting: (1, W, H, 1) - One mask per volume (Cartesian)
        mask_complex = mask.unsqueeze(0).unsqueeze(-1).expand(1, w, h, d)
        logger.debug(f"  Mask shape for broadcasting: {mask_complex.shape}")

        # Apply same mask to all coils
        undersampled_complex = kspace_complex * mask_complex
        logger.debug(f"  After masking: {undersampled_complex.shape}")

        # 4. Convert Back to IO Format: (C*2, W, H, D)
        try:
            real_view = torch.view_as_real(undersampled_complex)
            logger.debug(f"  After view_as_real: {real_view.shape}")

            permuted = real_view.permute(0, 4, 1, 2, 3)
            logger.debug(f"  After permute back: {permuted.shape}")

            undersampled_ri = permuted.flatten(0, 1)
            logger.debug(f"  After flatten to IO format: {undersampled_ri.shape}")
        except RuntimeError as e:
            raise ValueError(
                f"Failed to convert back from complex. Complex shape {undersampled_complex.shape}. "
                f"Error: {e}"
            )

        # 5. Store Results & Validate
        # 'input': Undersampled K-Space (Model Input)
        logger.debug(f"PhysicsInformedMasking output: input shape={undersampled_ri.shape}")

        # VALIDATION: Check shapes match (critical for avoiding downstream errors)
        if self.validate_shapes:
            if undersampled_ri.shape != raw_tensor.shape:
                raise ValueError(
                    f"Shape mismatch after undersampling! "
                    f"Input: {raw_tensor.shape}, Output: {undersampled_ri.shape}. "
                    f"Undersampling should preserve tensor shape."
                )

            # Check sparsity increased (mask was applied)
            input_zeros = (raw_tensor == 0).sum().float()
            output_zeros = (undersampled_ri == 0).sum().float()
            input_sparsity = input_zeros / raw_tensor.numel() * 100
            output_sparsity = output_zeros / undersampled_ri.numel() * 100

            logger.debug(f"  Sparsity: input={input_sparsity:.1f}%, output={output_sparsity:.1f}%")

            if output_sparsity <= input_sparsity:
                logger.warning(
                    f"Masking may not have been applied! "
                    f"Output sparsity {output_sparsity:.1f}% not greater than input {input_sparsity:.1f}%"
                )

        # Store as 'input' (Canonical)
        subject.add_image(
            tio.ScalarImage(tensor=undersampled_ri, affine=subject[key].affine),
            "input",
        )

        # Store original as 'target' (Canonical) if not present
        if "target" not in subject:
            target_data = subject[key].data

            # [FIX] Ensure target dtype matches input (real-stack if input was converted)
            if not torch.is_complex(undersampled_ri) and torch.is_complex(target_data):
                # Convert target to real stack matching input's Re/Im layout
                # (C, W, H, D) complex -> (C, W, H, D, 2) real -> (C, 2, W, H, D) real -> (2C, W, H, D) real
                target_real = torch.view_as_real(target_data)
                target_permuted = target_real.permute(0, 4, 1, 2, 3)
                target_data = target_permuted.flatten(0, 1)

            subject.add_image(
                tio.ScalarImage(tensor=target_data, affine=subject[key].affine),
                "target",
            )
        else:
            # [FIX] If target exists but is complex while input is real, convert it
            target_img = subject["target"]
            target_data = target_img.data
            if not torch.is_complex(undersampled_ri) and torch.is_complex(target_data):
                target_real = torch.view_as_real(target_data)
                target_permuted = target_real.permute(0, 4, 1, 2, 3)
                target_data = target_permuted.flatten(0, 1)
                target_img.set_data(target_data)

        # 'mask': The Sampling Mask
        # Expand mask to proper shape for viewing/loss: (1, W, H, D)
        mask_ri = mask.unsqueeze(0).unsqueeze(-1).expand(1, w, h, d)
        subject.add_image(tio.LabelMap(tensor=mask_ri, affine=subject[key].affine), "mask")

        logger.debug(
            f"  Added to subject: input {undersampled_ri.shape}, target, mask {mask_ri.shape}"
        )

        # [PI-FIX] Update 'mri_sub' (Zero-filled Recon) for consistency check?
        # Usually handled by the model or on-the-fly, but having a quick recon is good for debug.
        # Simple IFFT for QC
        # ifft_img = torch.fft.ifft2(undersampled_complex)
        # ifft_mag = ifft_img.abs().sum(dim=0, keepdim=True) # RSS-like
        # subject.add_image(tio.ScalarImage(tensor=ifft_mag, affine=subject[key].affine), 'mri_sub')

        return subject

    def _generate_mask(self, width, height) -> torch.Tensor:
        """Generate a Cartesian undersampling mask.

        CRITICAL DESIGN:
        - Undersamples along ``width``, which this transform treats as the
          PHASE-ENCODE axis (the 1D line pattern varies along ``width`` and is
          held constant across ``height``). The old docstring called ``width``
          "frequency encoding" while the code undersampled it and a sibling
          comment called it phase encoding — a self-contradiction; the code's
          actual assumption is width == phase-encode, stated plainly here.
        - Expands the 1D line pattern to 2D and applies it to ALL coils.

        NOTE (review 2026-07-01, canonical-homes / pitfall #12): the physics SSOT
        ``infrastructure/physics/sampling.py::KSpaceMaskGenerator`` is the single
        home for k-space masks. This transform is NOT yet delegated to it because
        the two differ in BOTH the undersampled axis (this one → ``width``;
        ``KSpaceMaskGenerator`` → ``height``) AND the sampling distribution (this
        one → uniform-random outside the ACS; SSOT → variable-density power-law).
        Consolidating is therefore a physics change, not a refactor, and needs the
        M4Raw phase-encode axis confirmed first. See docs/review_fixes_2026_07_01.rst.

        Args:
            width: Phase-encode dimension (the axis actually undersampled here).
            height: The fully-sampled (readout) dimension; the mask is constant
                along it.

        Returns:
            Binary mask (W, H) with 1=acquired, 0=missing.
        """
        # Return a cached mask when this shape was already built — the mask is
        # deterministic (seeded below), so recomputing it per item is waste.
        cached = self._mask_cache.get((width, height))
        if cached is not None:
            return cached

        # Simple Cartesian Mask Logic
        mask = torch.zeros(width)
        center = width // 2
        cf_width = int(width * self.cf)

        # Safety check for center fraction
        start = max(0, center - cf_width // 2)
        end = min(width, center + cf_width // 2)
        mask[start:end] = 1

        # Random sampling outside center — deterministic via a scoped generator
        # seeded per width (reproducible across items/runs, no global RNG).
        outer_indices = torch.where(mask == 0)[0]
        if len(outer_indices) > 0 and self.acc > 0:
            num_samples = int(len(outer_indices) / self.acc)
            gen = torch.Generator(device="cpu").manual_seed(1234 + int(width))
            perm = torch.randperm(len(outer_indices), generator=gen)[:num_samples]
            mask[outer_indices[perm]] = 1

        # Expand 1D mask to 2D (assuming Phase Encoding is Width)
        # Returns (W, H). ``contiguous`` so the cached tensor is not an
        # expand-view aliasing ``mask`` (kept minimal-memory but safe to reuse).
        mask_2d = mask.unsqueeze(1).expand(width, height).contiguous()
        self._mask_cache[(width, height)] = mask_2d

        # Calculate and log undersampling ratio
        acquired = mask_2d.sum()
        total = mask_2d.numel()
        actual_acceleration = total / max(acquired, 1)
        logger.debug(
            f"  Mask statistics: {acquired}/{total} acquired ({100 * acquired / total:.1f}%), "
            f"acceleration={actual_acceleration:.2f}x"
        )

        return mask_2d


class DigitalTwinDegradation(tio.transforms.Transform):
    """Transversal Digital-Twin degradation as a data-pipeline transform.

    Mirrors :class:`PhysicsInformedMasking` (the accelerator) but applies the
    ``DigitalTwinSimulator`` corruption — composite motion + B0/B1 fields +
    noise — so **any** experiment can opt into synthetic degradation without
    a VF strategy. The twin operates in the image domain, so a k-space source
    is round-tripped ``ifft2c → corrupt → fft2c``.

    Produces the corrupted acquisition as ``input`` and preserves the clean
    ``target``, giving supervised (corrupted → clean) pairs.

    This is a K-SPACE transform. The former ``source_domain`` knob advertised an
    ``"image"`` value that could never work: the apply path resolved its source
    from the k-space key either way, so ``"image"`` meant "read k-space and
    pretend it is an image". No caller ever passed it either, since the transform
    builder hardcodes the k-space route. Corrupting an image-domain subject is a
    real capability but a separate design decision (which key is the clean
    source, and what happens on a dataset whose ``target`` is an independently
    acquired volume), so the knob is gone rather than accepted-but-broken.

    Args:
        simulator: A pre-built ``DigitalTwinSimulator`` (constructed from
            ``config.physics.digital_twin`` by the transform builder).
        degradation_only: skip VF marker embedding (default True — markers are
            a VF concept and have no place in a general degradation).
    """

    #: Where a k-space-serving dataset puts the raw acquisition, in lookup order.
    SOURCE_KEYS: tuple[str, ...] = ("kspace_raw", "kspace")

    def __init__(
        self,
        simulator,
        degradation_only: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.simulator = simulator
        self.degradation_only = degradation_only

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Corrupt the subject's acquisition with the digital twin."""
        key = next((k for k in self.SOURCE_KEYS if k in subject), None)
        if key is None:
            # Was a DEBUG log and a pass-through. The builder logs "[PHYSICS]
            # Injecting transversal DigitalTwinDegradation" at INFO one step
            # earlier, so silence here meant the run REPORTED a degradation it
            # never applied and trained on clean data (pitfall #16), invisible
            # at default verbosity.
            raise ValueError(
                "DigitalTwinDegradation found no acquisition to corrupt: the "
                f"subject carries {sorted(subject.keys())} but none of "
                f"{list(self.SOURCE_KEYS)}. It is a k-space transform. Serve a "
                "dataset_type that emits k-space (fastmri_kspace, ismrmrd, "
                "m4raw, bart), or turn physics.digital_twin.apply_as_transform "
                "off."
            )

        raw = subject[key].data  # (C, W, H, D)
        c, w, h, d = raw.shape
        if c % 2 != 0:
            raise ValueError(
                f"DigitalTwinDegradation expects even channels (Re/Im pairs); got {c}."
            )
        num_coils = c // 2

        # (C, W, H, D) -> (num_coils, W, H, D) complex
        kspace_complex = torch.view_as_complex(
            raw.contiguous().view(num_coils, 2, w, h, d).permute(0, 2, 3, 4, 1).contiguous()
        )
        # Treat depth as batch: (D, num_coils, W, H)
        kc = kspace_complex.permute(3, 0, 1, 2).contiguous()

        # Twin works in image space.
        image = ifft2c(kc)
        corrupted_img, _marker_prior, _clean = self.simulator(
            image, degradation_only=self.degradation_only
        )
        corrupted_k = fft2c(corrupted_img)

        # (D, num_coils, W, H) -> (num_coils, W, H, D) -> (C, W, H, D) real
        corrupted_k = corrupted_k.permute(1, 2, 3, 0).contiguous()
        corrupted_ri = (
            torch.view_as_real(corrupted_k).permute(0, 4, 1, 2, 3).flatten(0, 1).contiguous()
        )
        if corrupted_ri.shape != raw.shape:
            raise ValueError(
                f"DigitalTwinDegradation must preserve shape: in {raw.shape}, "
                f"out {corrupted_ri.shape}."
            )

        subject.add_image(tio.ScalarImage(tensor=corrupted_ri, affine=subject[key].affine), "input")
        if "target" not in subject:
            subject.add_image(
                tio.ScalarImage(tensor=raw.clone(), affine=subject[key].affine), "target"
            )
        return subject
