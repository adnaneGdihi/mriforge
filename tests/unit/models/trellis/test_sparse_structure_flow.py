"""``SparseStructureFlowModel`` — the block import its constructor depends on.

``ModulatedTransformerCrossBlock`` is defined in ``blocks/trellis_attention.py``
and named in its ``__all__``; the constructor imported it from
``blocks/trellis_transformer``, which has never defined it. Because the import is
FUNCTION-LOCAL, the model registered cleanly and the failure only appeared at
construction — no import-time check, and no registry check, could see it
(non-negotiable 15: a gate only catches the shape it has watched fail).
"""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")


def test_modulated_transformer_cross_block_resolves_from_trellis_attention():
    """The class the constructor needs, from the module that actually defines it."""
    from spectramr.models.blocks.trellis_attention import ModulatedTransformerCrossBlock

    assert ModulatedTransformerCrossBlock is not None


def test_the_class_is_not_in_trellis_transformer():
    """Pin the premise. If it is ever re-exported there, this test says so
    rather than letting the two spellings silently both work."""
    from spectramr.models.blocks import trellis_transformer

    assert not hasattr(trellis_transformer, "ModulatedTransformerCrossBlock"), (
        "trellis_transformer now exports the class too — elect one owner "
        "(non-negotiable 17) instead of leaving both spellings valid"
    )


def test_constructor_imports_it_from_the_right_module():
    """Source-level, because the import is function-local.

    An import inside a method body is invisible to import-time linting and to
    any registry walk, so the only cheap check is the source itself.
    """
    from spectramr.models.trellis import sparse_structure_flow

    source = inspect.getsource(sparse_structure_flow)
    assert "blocks.trellis_attention import ModulatedTransformerCrossBlock" in source
    assert "blocks.trellis_transformer import ModulatedTransformerCrossBlock" not in source
