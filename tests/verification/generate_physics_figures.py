from pathlib import Path

import matplotlib.pyplot as plt
from spectramr.shared.utils.mask_generator import MaskGenerator, MaskType


def save_mask_figure(mask, name, acceleration, output_dir):
    """Save mask visualization."""
    plt.figure(figsize=(10, 10))
    plt.imshow(mask.cpu().numpy(), cmap="gray", interpolation="nearest")
    plt.title(f"{name} (R={acceleration}x)")
    plt.axis("off")

    # Calculate actual acceleration
    mask_sum = mask.sum().item()
    total = mask.numel()
    actual_R = total / mask_sum if mask_sum > 0 else float("inf")

    plt.text(
        10,
        20,
        f"Actual R={actual_R:.2f}",
        color="white",
        fontsize=12,
        backgroundcolor="black",
    )

    filename = output_dir / f"{name}_x{acceleration}.png"
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {filename}")


def main():
    # Setup
    output_dir = Path("tests/verification/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = MaskGenerator(seed=42)
    shape = (256, 256)
    accelerations = [2, 4, 8, 16, 32, 64]

    types = [
        (MaskType.SPIRAL, "spiral"),
        (MaskType.VARIABLE_DENSITY, "variable_density"),
        (MaskType.RADIAL, "radial"),
        (MaskType.UNIFORM_CARTESIAN, "uniform"),
    ]

    print(f"Generating figures in {output_dir}...")

    for accel in accelerations:
        print(f"\n--- Acceleration x{accel} ---")
        for mask_type, name in types:
            try:
                # Generate mask
                # Check for specific kwargs if needed
                kwargs = {}
                if mask_type == MaskType.SPIRAL:
                    # Spiral uses geometric arms now.
                    # Base arms defaults to 48, which at R=64 might be < 1 arm if not handled?
                    # The code says max(1, int(base_arms / acceleration_factor))
                    pass

                mask = generator.generate_mask(
                    mask_type, shape, acceleration_factor=float(accel), **kwargs
                )

                # Save figure
                save_mask_figure(mask, name, accel, output_dir)

            except Exception as e:
                print(f"Failed to generate {name} x{accel}: {e}")

    print("\nDone. Please inspect the figures in tests/verification/figures/")


if __name__ == "__main__":
    main()
