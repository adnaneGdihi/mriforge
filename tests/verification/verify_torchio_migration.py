import sys
from pathlib import Path

import torch

# Ensure src is in path
sys.path.append(str(Path(__file__).parents[2]))

from spectramr.config.data_config import DataConfig
from spectramr.infrastructure.builders.context import BuilderContext
from spectramr.infrastructure.builders.directors.data_pipeline_director import (
    DataPipelineDirector,
)


def verify_migration():
    print(">>> Verifying TorchIO Migration...")

    # 1. Setup Config
    # Ensure index exists or fallback to dummy path if not verifying real data index yet
    index_path = "data/manifests/fastmri_knee.pkl"
    if not Path(index_path).exists():
        print(
            f"!!! Index file not found at {index_path}. Cannot verify real data loading."
        )
        # Create a dummy index for verification if needed?
        # Or just bail out as per script logic.
        print("Please run Step 9 first or ensure index exists.")
        return

    config = DataConfig(
        dataset_type="fastmri_kspace",
        data_path="./data/processed/dummy",  # Won't be used if index exists
        index_path=index_path,  # Must exist
        batch_size=2,
        patch_size=(320, 320, 1),  # 2D Slices
        samples_per_volume=4,
        queue_length=20,
        num_workers=0,  # Debug on main thread
    )

    # 2. Create Loader
    try:
        loader, _ = DataPipelineDirector(BuilderContext(config=config)).build_dataloaders()
    except FileNotFoundError:
        print("!!! Index file not found. Please run Step 9 first.")
        return
    except Exception as e:
        print(f"!!! Error creating loader: {e}")
        return

    # 3. Fetch Batch
    print(">>> Fetching batch from Queue...")
    try:
        batch = next(iter(loader))
    except Exception as e:
        print(f"!!! Error fetching batch: {e}")
        return

    # 4. Analyze Structure (TorchIO outputs a dictionary-like object)
    if "kspace" in batch and "data" in batch["kspace"]:
        kspace = batch["kspace"]["data"]  # [B, C, W, H, D]
    else:
        print(f"!!! Unexpected batch structure. Keys: {batch.keys()}")
        return

    mask = batch["mask"]["data"]

    print(f"Batch Keys: {batch.keys()}")
    print(f"K-Space Shape: {kspace.shape}")
    print(f"Mask Shape:    {mask.shape}")

    # 5. Assertions
    # TorchIO adds a Depth dimension (D=1) for 2D slices
    assert (
        kspace.dim() == 5
    ), f"TorchIO should output 5D tensors (B, C, W, H, D), got {kspace.shape}"
    assert (
        kspace.shape[-1] == 1
    ), f"Depth should be 1 for 2D training, got {kspace.shape[-1]}"

    # Mask C=1 is valid for K-Space C>1 (Broadcasting)
    assert (
        mask.shape[1] == 1 or mask.shape[1] == kspace.shape[1]
    ), f"Mask channels {mask.shape[1]} must be 1 or match K-Space {kspace.shape[1]}"
    assert (
        mask.shape[2:] == kspace.shape[2:]
    ), f"Spatial dimensions must match. Mask: {mask.shape}, KSpace: {kspace.shape}"

    print(">>> SUCCESS: Pipeline is producing valid tensors.")

    # 6. [PI-REQ] L2 Fidelity Check (Identity Transform)
    print(">>> Running L2 Fidelity Check (Identity)...")

    # Create a config with NO acceleration (Identity)
    fidelity_config = DataConfig(
        dataset_type="fastmri_kspace",
        index_path=index_path,
        batch_size=1,
        patch_size=(320, 320, 1),
        acceleration=1,  # No undersampling
        center_fraction=1.0,  # Keep all frequencies
        num_workers=0,
    )

    try:
        fid_loader, _ = DataPipelineDirector(BuilderContext(config=fidelity_config)).build_dataloaders()
        fid_batch = next(iter(fid_loader))

        # Input (Original) vs Output (Transform applied)
        # With acc=1, kspace_sub should equal kspace_raw

        # Since our Transform writes to 'kspace_sub', and 'kspace_raw' is the source
        orig = fid_batch["kspace_raw"]["data"]
        processed = fid_batch["kspace_sub"]["data"]

        # Assert L2 Error is negligible
        error = torch.norm(orig.float() - processed.float())
        print(f"    L2 Error: {error.item():.2e}")

        assert (
            error < 1e-6
        ), f"Fidelity Check Failed! Error {error.item()} > 1e-6. Data corrupted by transforms."
        print(">>> SUCCESS: L2 Fidelity Check Passed.")

    except Exception as e:
        print(f"!!! Fidelity Check Failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    verify_migration()
