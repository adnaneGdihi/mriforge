"""Reusable VAE building blocks.

Houses the ladder-VAE precision-merge block and the von Mises-Fisher
distribution used by the hyperspherical VAE. These blocks are the
single source of truth for those primitives — they must never be
re-inlined inside a generator (CLAUDE.md canonical-homes rule).
"""

from .ladder_block import LadderBlock
from .von_mises_fisher import VonMisesFisher

__all__ = ["LadderBlock", "VonMisesFisher"]
