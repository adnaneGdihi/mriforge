"""KIDOT: Knowledge-Informed Dynamic Optimal Transport
=====================================================

This module implements unpaired VLF→HF translation via physics-informed
optimal transport. Instead of learning a direct mapping, we learn a
continuous transport path that respects the forward imaging physics.

Key Innovation:
- Standard OT: min distance between distributions
- KIDOT: min distance + physics consistency penalty

Core Equation:
    C(x, y) = ||x - y||² + λ||y - A(x)||²

Where A is the forward operator (Bloch + k-space).
"""

from .kidot_cost import KIDOTCost, PhysicsOperator
from .kidot_transport import KIDOTTransport, create_kidot_transport
from .velocity_network import VelocityNetwork

__all__ = [
    "KIDOTCost",
    "KIDOTTransport",
    "PhysicsOperator",
    "VelocityNetwork",
    "create_kidot_transport",
]
