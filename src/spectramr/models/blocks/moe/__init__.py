"""Mixture-of-experts routing blocks.

Single source of truth for the top-1 router used by ``moe_vae``.
"""

from .top1_router import Top1Router

__all__ = ["Top1Router"]
