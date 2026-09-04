from spectramr.shared.utils.mask_generator import MaskGenerator


def verify_cartesian_mask():
    print("Verifying Cartesian Line Mask...")
    gen = MaskGenerator(seed=42)
    shape = (256, 256)
    accel = 4.0

    # Generate mask
    mask = gen.generate_mask("cartesian_lines", shape, acceleration_factor=accel)

    # 1. Check Shape
    assert mask.shape == shape, f"Shape mismatch: {mask.shape} vs {shape}"

    # 2. Check Line Consistency
    # In Cartesian mapping:
    # Readout (dim 1, W) should be fully sampled if a line is selected.
    # So if mask[y, 0] is 1, mask[y, :] should be all 1s.
    # If mask[y, 0] is 0, mask[y, :] should be all 0s.

    line_means = mask.mean(dim=1)
    is_line = (line_means == 0.0) | (line_means == 1.0)
    assert is_line.all(), "Mask is not line-based! Partial lines found."

    # 3. Check Center Sampling (ACS)
    center_y = shape[0] // 2
    acs_width = int(shape[0] * 0.08)
    # Check a few lines around center
    center_start = (shape[0] - acs_width) // 2
    center_end = center_start + acs_width

    center_region = mask[center_start:center_end, :]
    assert center_region.all(), "ACS region not fully sampled!"

    print("Cartesian Line Mask Verified: Line-based integrity and ACS checks passed.")


if __name__ == "__main__":
    verify_cartesian_mask()
