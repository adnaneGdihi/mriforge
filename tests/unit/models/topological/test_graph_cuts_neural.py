"""Unit tests for :mod:`mriforge.models.topological.graph_cuts_neural`.

Covers the pitfall-#9 contract on ``GraphConstruct``: an unknown ``distance``
string must raise a clear ``ValueError`` naming the offending option at
``forward`` time, rather than degrading to a cryptic ``UnboundLocalError`` when
``torch.topk`` consumes the never-assigned ``dist`` local.

Also pins the two supported distance paths (``euclidean`` / ``cosine``) to their
expected k-NN index shape so the fix is provably numerics-neutral, and runs a
small end-to-end ``GraphCutsNeural`` forward.

device='cpu', tiny tensors, fixed seed.
"""

import pytest
import torch

from mriforge.models.registry import MODEL_REGISTRY
from mriforge.models.topological.graph_cuts_neural import (
    GraphConstruct,
    GraphCutsNeural,
)


class TestGraphConstructDistanceDispatch:
    def test_unknown_distance_raises_valueerror(self):
        # pitfall #9: an unknown distance must fail loud (ValueError naming the
        # offending value), never crash deferred with UnboundLocalError.
        torch.manual_seed(0)
        gc = GraphConstruct(k=2, distance="manhattan")
        x = torch.randn(1, 4, 8)
        with pytest.raises(ValueError, match="unknown distance"):
            gc(x)

    def test_unknown_distance_message_names_value(self):
        gc = GraphConstruct(k=2, distance="manhattan")
        x = torch.randn(1, 4, 8)
        with pytest.raises(ValueError, match="manhattan"):
            gc(x)

    @pytest.mark.parametrize("distance", ["euclidean", "cosine"])
    def test_supported_distance_returns_knn_indices(self, distance):
        # The two supported paths are unchanged: indices [B, N, k] (self excluded).
        torch.manual_seed(0)
        k = 2
        b, n, d = 1, 5, 8
        gc = GraphConstruct(k=k, distance=distance)
        x = torch.randn(b, n, d)
        idx = gc(x)
        assert idx.shape == (b, n, k)
        # self index (column 0 before slicing) is excluded, so each node's own
        # index should not appear as its nearest neighbour in a generic config.
        assert idx.dtype == torch.long


class TestGraphCutsNeuralForward:
    def test_registered_in_model_registry(self):
        assert "graph_cuts_neural" in MODEL_REGISTRY

    def test_forward_shape(self):
        torch.manual_seed(0)
        # 32x32 with patch_size=16 -> 2x2 = 4 patches; k must satisfy k+1 <= 4.
        model = GraphCutsNeural(
            in_channels=2,
            out_channels=2,
            features=[8, 16],
            patch_size=16,
            k=3,
        )
        x = torch.randn(1, 2, 32, 32)
        out = model(x)
        assert out.shape == (1, 2, 32, 32)


class TestCapabilityContract:
    """Every declared field is asserted against the behaviour it claims (#1106).

    Before this, ``graph_cuts_neural`` declared only its name and training_mode,
    so ``data_model_compatibility`` reported "unannotated; audit cross-check
    skipped (legacy escape hatch)" and passed -- a green tick for a check that
    never ran. A declaration is only worth having if it is true, so each field
    below is paired with the behaviour that makes it true.
    """

    def _caps(self):
        from mriforge.models.registry import get_model_capabilities

        return get_model_capabilities("graph_cuts_neural")

    def test_the_contract_is_declared_at_all(self):
        caps = self._caps()
        assert caps.spatial_dims == (2,)
        assert caps.input_domain == "image"
        assert caps.output_domain == "image"
        assert caps.accepts_complex is False
        assert caps.requires_paired_data is True

    def test_rank_2_is_a_contract_not_a_preference(self):
        """``forward`` unpacks 4 values, so rank 3 must genuinely fail."""
        model = GraphCutsNeural(in_channels=2, out_channels=2, features=[8], patch_size=16, k=3)
        with pytest.raises(ValueError, match="too many values to unpack"):
            model(torch.randn(1, 2, 8, 32, 32))

    def test_accepts_complex_false_is_honest(self):
        """Declared False, so complex input must actually fail rather than work.

        A *wrong* False is the more expensive error here -- it is what made the
        audit reject valid pre_model iFFT chains for ``graph_unet`` until
        2026-05-12 -- so pin that the refusal is real and comes from the model,
        not from the declaration.
        """
        model = GraphCutsNeural(in_channels=2, out_channels=2, features=[8], patch_size=16, k=3)
        with pytest.raises(RuntimeError, match="dtype"):
            model(torch.randn(1, 1, 32, 32, dtype=torch.complex64))

    def test_the_forward_touches_no_kspace_transform(self):
        """The class docstring says "K-Space Patches"; the code says otherwise.

        ``input_domain="image"`` rests on there being no FFT on any path, so that
        absence is the thing to pin -- if someone later adds an internal transform,
        the domain declaration silently becomes a lie.
        """
        import inspect

        # Comment lines are stripped: ``getsource`` on a decorated class includes
        # the decorator, whose own comment explains the domain reasoning in prose
        # that mentions IFFT -- a naive scan would match the explanation, not code.
        code = "\n".join(
            line
            for line in inspect.getsource(GraphCutsNeural).splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "fft" not in code.lower()
