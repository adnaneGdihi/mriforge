"""Operator-identification models (Proposal 1).

Importing this package registers the BCH operator conditioner with the
model registry (via the ``@register_model`` decorator).
"""

from .bch_conditioner import BCHOperatorConditioner

__all__ = ["BCHOperatorConditioner"]
