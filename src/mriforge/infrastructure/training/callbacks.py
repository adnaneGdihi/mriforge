import logging

import torch
import torch.nn as nn


class GradientExplosionError(RuntimeError):
    """GradientExplosionError class."""

    pass


def check_gradients(
    model: nn.Module, logger: logging.Logger = None, raise_error: bool = True
) -> bool:
    """
    Check for NaN or Inf in gradients.

    Args:
        model: Model to check.
        logger: Optional logger to report issues.
        raise_error: Whether to raise an exception immediately.

    Returns:
        True if gradients are healthy, False if NaN/Inf detected (if not raising).
    """
    has_issue = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                msg = f"NaN gradient detected in {name}"
                if logger:
                    logger.error(msg)
                has_issue = True
                if raise_error:
                    raise GradientExplosionError(msg)

            if torch.isinf(param.grad).any():
                msg = f"Inf gradient detected in {name}"
                if logger:
                    logger.error(msg)
                has_issue = True
                if raise_error:
                    raise GradientExplosionError(msg)

    return not has_issue
