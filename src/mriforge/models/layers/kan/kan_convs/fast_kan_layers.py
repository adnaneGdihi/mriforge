"""Shim exposing FastKANConv2DLayer for tests.

The implementation lives in `fast_kan_conv.py`.
This module provides the expected symbol path
`models.kan_convs.fast_kan_layers.FastKANConv2DLayer` for patching.
"""

from .fast_kan_conv import FastKANConv2DLayer  # re-export

__all__ = ["FastKANConv2DLayer"]
