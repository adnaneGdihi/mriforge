"""GNN Coil Fusion
===============

Graph Neural Network for Coil Fusion in MRI.
This model treats coils as nodes in a graph to learn optimal fusion strategies.
"""

from spectramr.models.registry import register_model

from .graph_unet_generator import GraphUNetGenerator


@register_model(
    name="gnn_coil_fusion",
    training_mode="reconstruction",
    # Mirrors ``graph_unet``, the base this subclasses. These were absent, which is not
    # the same as inherited: an undeclared field reads as None and the audit checks that
    # consume it -- check_workflow_spatial_rank, the signal-domain checks, the spec card
    # -- do not run at all (#1084).
    #
    # This entry was deliberately held back when the other four in #1084 were settled,
    # on the theory that a COIL-FUSION model's input domain should differ from its
    # base's. It does not, and the reason is that "coil" is not a domain here at all:
    # ``Domain`` is {image, kspace, complex_image, latent, pde_grid, mesh, spectrum}.
    # Multi-coil-ness lives in the CHANNEL count, not the domain -- this model's own
    # docstring says in_channels is "2 * num_coils" and out_channels is "2 for combined
    # image". So fusing coil maps or image-space data IS image-domain work, with
    # ``accepts_complex=True`` covering the real/imag-interleaved case.
    #
    # Corroborated independently by its one and only arm,
    # experiments/training/architecture/experiment_78_gnn_coil.yaml, which declares
    # input_domain: image / output_domain: image over multicoil data (num_coils: 32,
    # in_channels: 2) -- i.e. the arm already asserts what this decorator now says.
    #
    # NB the base's "graph" name is misleading in the same direction: its decorator
    # records that the forward consumes standard [B, C, H, W] image tensors and uses
    # graph adjacency only as an internal connectivity prior. Arms that begin in k-space
    # (e.g. experiment_42 NUFFT reconstruction) convert BEFORE the model -- which is why
    # accepts_complex was corrected to True in 2026-05, so the audit would stop
    # rejecting pre_model iFFT chains.
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=True,
    requires_paired_data=True,
)
class GNNCoilFusion(GraphUNetGenerator):
    """GNN Coil Fusion Network.

    This model extends GraphUNetGenerator to specifically handle coil fusion tasks.
    It treats the input channels (representing coils) as nodes in a graph
    to exploit inter-coil correlations.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        hidden_dim: int = 64,
        depth: int = 3,
        **kwargs,
    ):
        """Initialize GNN Coil Fusion.

        Args:
            in_channels: Number of input channels (e.g. 2 * num_coils)
            out_channels: Number of output channels (e.g. 2 for combined image)
            hidden_dim: Hidden dimension size
            depth: Network depth
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            depth=depth,
        )

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "gnn_coil_fusion"
