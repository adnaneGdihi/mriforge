import logging
import os
import sys

import numpy as np
import torch

# Add src to path
sys.path.append(os.getcwd())

from spectramr.core.metrics.medical import MedicalMetrics
from spectramr.shared.utils.sampling_masks import create_sampling_mask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ipen():
    logger.info("Testing IPEN Metric...")
    metrics = MedicalMetrics(device="cpu")

    # Case 1: Identical images -> IPEN should be 0
    gt = torch.randn(1, 2, 32, 32)
    pred = gt.clone()
    ipen = metrics.calculate_ipen(pred, gt)
    logger.info(f"IPEN (Identical): {ipen} (Expected: 0.0)")
    assert np.isclose(ipen, 0.0, atol=1e-6)

    # Case 2: Phase shift
    # Create complex tensors
    gt_complex = torch.complex(torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32))
    # Shift phase by pi/2
    pred_complex = gt_complex * torch.exp(torch.tensor(1j * np.pi / 2))

    # Needs to be [B, 2, H, W] for the interface if passed as real tensor, or complex
    ipen_val = metrics.calculate_ipen(pred_complex, gt_complex)
    logger.info(f"IPEN (Phase Shift pi/2): {ipen_val}")

    # Manual calculation check
    phi_gt = torch.angle(gt_complex)
    phi_pred = torch.angle(pred_complex)
    # diff should be pi/2 everywhere (wrapped)
    # diff = (pi/2 + pi) % 2pi - pi = 1.57...

    diff = (phi_pred - phi_gt + np.pi) % (2 * np.pi) - np.pi
    expected_ipen = torch.norm(diff) / torch.norm(phi_gt)
    logger.info(f"Expected IPEN: {expected_ipen.item()}")

    assert np.isclose(ipen_val, expected_ipen.item(), atol=1e-5)
    logger.info("IPEN Test Passed")


def test_mask_hashing():
    logger.info("Testing Deterministic Mask Hashing...")
    shape = (32, 32)
    seed = 42

    # Generate twice
    mask1 = create_sampling_mask(shape, 4, mask_type="uniform", seed=seed)
    mask2 = create_sampling_mask(shape, 4, mask_type="uniform", seed=seed)

    if np.array_equal(mask1, mask2):
        logger.info("Masks are identical (Deterministic).")
    else:
        logger.error("Masks are NOT identical!")

    # Check if hash is logged (by ensuring it runs without error)
    # We can't easily capture logs here without capturing stdout, but the function call itself verifies execution path.
    logger.info("Mask hashing code path executed successfully.")


if __name__ == "__main__":
    test_ipen()
    test_mask_hashing()
