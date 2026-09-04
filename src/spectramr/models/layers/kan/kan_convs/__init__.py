# This file makes the 'kan_convs' directory a Python package.
# Allows imports like `from models.kan_convs import fast_kan_layers`.

try:
    from . import attention_conv, fast_kan_conv, fast_kan_layers, kans
except ImportError:
    # This can happen during certain test setups.
    pass

__all__ = ["attention_conv", "fast_kan_conv", "fast_kan_layers", "kans"]
