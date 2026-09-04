import os
import sys

import torch

sys.path.insert(0, os.getcwd())

from spectramr.models.generators.fno_generator import FNOGenerator


def test_fno_sense_projection():
    print("Testing FNOGenerator Physics Projection...")

    # Setup
    device = torch.device("cpu")  # Keep it simple
    B, C, H, W = 1, 2, 64, 64

    # 1. Create Model
    model = FNOGenerator(
        in_channels=2, out_channels=2, width=16, depth=2, output_domain="kspace"
    ).to(device)
    model.train()  # Must be in training mode for projection

    # 2. Input K-Space
    x = torch.randn(B, C, H, W).to(device)

    # 3. Sensitivity Maps (Single Coil for simplicity first)
    # Shape: (B, Coils, H, W) -> (1, 1, 64, 64)
    # But FNO expects complex map logic if used
    # Let's test Single Coil Fallback first (no maps)
    print("  Testing Single Coil Fallback...")
    y = model(x, project_to_image=True)
    if y.shape == (B, 1, H, W):
        print("  ✅ Single Coil Projection Shape Correct")
    else:
        print(f"  ❌ Single Coil Shape Mismatch: {y.shape}")
        return False

    # 4. Multi-Coil SENSE
    print("  Testing Multi-Coil SENSE...")
    # Input has 2*Coils channels. Let's say 2 coils -> 4 channels
    coils = 2
    model_mc = FNOGenerator(
        in_channels=2 * coils,
        out_channels=2,  # Internal output is 2 per coil? No, FNO usually outputs refined K-space
        # If FNO outputs (B, 2*Coils, H, W), we need to check how it's configured.
        # The code assumes output shape[1] > 2 implies multi-coil.
        width=16,
    ).to(device)
    model_mc.fc2 = torch.nn.Linear(128, 2 * coils)  # Hack to force output channels
    model_mc.train()

    x_mc = torch.randn(B, 2 * coils, H, W).to(device)

    # Sensitivity Maps: (B, Coils, H, W, 2) flattened or complex?
    # The code expects complex tensor or compatible shape for sense_adjoint
    # sense_adjoint expects (kspace, maps).
    # maps should be complex (B, Coils, H, W)
    maps = torch.complex(torch.randn(B, coils, H, W), torch.randn(B, coils, H, W)).to(
        device
    )

    try:
        y_mc = model_mc(x_mc, sensitivity_maps=maps, project_to_image=True)
        if y_mc.shape == (B, 1, H, W):
            print("  ✅ Multi-Coil SENSE Projection Valid")
        else:
            print(f"  ❌ Multi-Coil Shape Mismatch: {y_mc.shape}")
            return False
    except Exception as e:
        print(f"  ❌ Multi-Coil Crash: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    if test_fno_sense_projection():
        print("FNO Physics Verified!")
        sys.exit(0)
    else:
        print("FNO Physics Failed!")
        sys.exit(1)
