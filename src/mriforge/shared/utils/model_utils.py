"""
Consolidated and generic model utility functions.

This module serves as the canonical location for shared, model-agnostic
helper functions as per Task 1.4 of the refactoring roadmap.
"""

from torch import nn


def maybe_dropout(dropout: float) -> nn.Module:
    """Return a dropout layer when `dropout` is positive, otherwise identity."""
    return nn.Dropout(dropout) if dropout > 0 else nn.Identity()
