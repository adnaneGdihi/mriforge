"""Space-filling-curve blocks (Beltrami-adaptive variants).

The legacy fixed-SFC builders live in
``mriforge.models.blocks.topology_linearizer`` and
``mriforge.models.blocks.hilbert_order``. This subpackage hosts the
adaptive, Beltrami-parameterised SFCs introduced by
``TODO/audit_plan_novel.md`` §§194-344.
"""

from .beltrami_sfc import BeltramiSFCBlock
from .mamba_4d import BiMamba4DBlock, MRFMambaBlock4D

__all__ = ["BeltramiSFCBlock", "BiMamba4DBlock", "MRFMambaBlock4D"]
