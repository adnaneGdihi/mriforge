"""Unit tests for the gated-graph block and GatedGNNReconstructor.

Covers:
* :func:`mriforge.models.blocks.graph.gated_gnn.scatter_add` aggregation.
* :class:`mriforge.models.blocks.graph.gated_gnn.GatedGraphConv` forward
  shape and message propagation across the graph (a perturbation at one
  vertex influences a distant vertex after T >= diameter steps).
* :class:`mriforge.models.generators.gated_gnn_reconstructor.GatedGNNReconstructor`
  forward shape and registration as ``gated_gnn``.

Written but not executed here; they run on the cluster.
"""

import pytest
import torch

from mriforge.models.blocks.graph.gated_gnn import GatedGraphConv, scatter_add
from mriforge.models.generators.gated_gnn_reconstructor import GatedGNNReconstructor
from mriforge.models.registry import MODEL_REGISTRY, get_model_class


class TestScatterAdd:
    def test_aggregates_into_buckets(self):
        src = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        index = torch.tensor([0, 0, 1])
        out = scatter_add(src, index, dim_size=3)
        assert out.shape == (3, 2)
        assert torch.allclose(out[0], torch.tensor([3.0, 3.0]))
        assert torch.allclose(out[1], torch.tensor([3.0, 3.0]))
        assert torch.allclose(out[2], torch.tensor([0.0, 0.0]))


class TestGatedGraphConv:
    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            GatedGraphConv(hidden_dim=0)
        with pytest.raises(ValueError):
            GatedGraphConv(num_edge_types=0)
        with pytest.raises(ValueError):
            GatedGraphConv(num_steps=0)

    def test_forward_shape(self):
        gnn = GatedGraphConv(hidden_dim=8, num_edge_types=2, num_steps=3)
        h = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_type = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        out = gnn(h, edge_index, edge_type)
        assert out.shape == (5, 8)
        assert torch.isfinite(out).all()

    def test_message_propagates_across_chain(self):
        # Path graph 0-1-2-3-4 (bidirectional). With T >= diameter (4),
        # perturbing vertex 0's input should change vertex 4's output.
        gnn = GatedGraphConv(hidden_dim=4, num_edge_types=1, num_steps=5)
        src = [0, 1, 1, 2, 2, 3, 3, 4]
        dst = [1, 0, 2, 1, 3, 2, 4, 3]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type = torch.zeros(len(src), dtype=torch.long)

        h0 = torch.randn(5, 4, requires_grad=True)
        out = gnn(h0, edge_index, edge_type)
        # Gradient of distant vertex 4 wrt source vertex 0 must be non-zero.
        out[4].sum().backward()
        assert h0.grad is not None
        assert h0.grad[0].abs().sum() > 0, "message did not propagate to distant vertex"


class TestGatedGNNReconstructor:
    def test_registered_name(self):
        assert "gated_gnn" in MODEL_REGISTRY
        assert get_model_class("gated_gnn") is GatedGNNReconstructor
        assert MODEL_REGISTRY["gated_gnn"]["mode"] == "reconstruction"
        caps = MODEL_REGISTRY["gated_gnn"]["capabilities"]
        assert caps.input_domain == "kspace"
        assert caps.output_domain == "image"

    def test_constructible_with_no_args(self):
        model = GatedGNNReconstructor()
        assert isinstance(model, GatedGNNReconstructor)

    def test_invalid_connectivity_raises(self):
        with pytest.raises(ValueError):
            GatedGNNReconstructor(connectivity=6)

    def test_forward_shape(self):
        model = GatedGNNReconstructor(
            in_channels=1, out_channels=1, hidden_dim=8, num_edge_types=4, num_steps=3
        )
        x = torch.randn(2, 1, 8, 8)
        out = model(x)
        assert out.shape == (2, 1, 8, 8)
        assert torch.isfinite(out).all()

    def test_grid_graph_cached(self):
        model = GatedGNNReconstructor(hidden_dim=8, num_steps=2)
        x = torch.randn(1, 1, 6, 6)
        model(x)
        assert (6, 6) in model._edge_cache

    def test_external_edges_accepted(self):
        model = GatedGNNReconstructor(hidden_dim=8, num_edge_types=2, num_steps=2)
        x = torch.randn(1, 1, 4, 4)
        n = 16
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_type = torch.tensor([0, 1, 0], dtype=torch.long)
        out = model(x, edge_index=edge_index, edge_type=edge_type)
        assert out.shape == (1, 1, 4, 4)

    def test_mismatched_edge_args_raise(self):
        model = GatedGNNReconstructor(hidden_dim=8, num_steps=2)
        x = torch.randn(1, 1, 4, 4)
        with pytest.raises(ValueError):
            model(x, edge_index=torch.tensor([[0], [1]], dtype=torch.long))
