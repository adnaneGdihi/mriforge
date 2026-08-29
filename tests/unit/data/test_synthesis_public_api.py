"""Pin the audit-16-F6 re-export.

``src/data/synthesis/`` was an implicit namespace package with no
``__init__.py``, which made ``VLFSimulator`` invisible as a public
API even though it was meant to be reused. The post-fix package
re-exports the public class so callers can import it from
``mriforge.data.synthesis``.
"""

from __future__ import annotations


def test_vlf_simulator_reachable_via_package_root() -> None:
    """The whole point of the re-export."""
    from mriforge.data import synthesis

    assert hasattr(synthesis, "VLFSimulator"), (
        "mriforge.data.synthesis no longer re-exports VLFSimulator. "
        "If the class moved, update __init__.py; if it was deleted, "
        "remove the import here too."
    )
    assert "VLFSimulator" in synthesis.__all__
