"""Registry wiring for ``GraphEncodingTransform`` (finding D9).

``graph_encoding`` was the ONE name the old ``data.processing.transforms``
consumer recognised -- it matched the literal string and ``break``ed, which is
why every other declaration was silently discarded. It now resolves through the
registry like everything else, so the special case is gone.
"""

from __future__ import annotations

from mriforge.data.transforms.graph_transform import GraphEncodingTransform
from mriforge.data.transforms.registry import build_transform, get_transform


def test_graph_encoding_is_registered() -> None:
    entry = get_transform("graph_encoding")
    assert entry.cls is GraphEncodingTransform


def test_graph_encoding_declares_the_keys_it_adds() -> None:
    """``produces`` is what lets an audit ask 'did the mechanism fire?'."""
    produces = get_transform("graph_encoding").produces
    assert "graph_nodes" in produces
    assert "graph_h" in produces and "graph_w" in produces


def test_graph_encoding_reads_kspace() -> None:
    assert get_transform("graph_encoding").requires == ("kspace",)


def test_declared_kwargs_reach_the_constructor() -> None:
    t = build_transform("graph_encoding", k_neighbors=12, max_nodes=1024)
    assert t.k_neighbors == 12
    assert t.max_nodes == 1024
