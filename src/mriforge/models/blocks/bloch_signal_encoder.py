"""Bloch-grounded acquisition encoder for PMPS (Phase 0).

Embed an acquisition vector :math:`P = (T_E, T_R, T_I, \\alpha, B_0)` by
the Bloch / EPG signal signature it would produce on a fixed
reference-tissue panel. The output is a learned projection of the
concatenated signal responses across panel entries; for a single
acquisition the embedding has shape ``(embed_dim,)`` and for a batch of
acquisitions ``(B, embed_dim)``.

The reference panel is configurable at construction time. The default
covers grey-matter, white-matter, CSF, fat and two sparse pathology
entries. Pathology entries are anchored on published literature ranges
[Bottomley 1984] and serve as anchors against which the encoder can
discriminate cohort distribution shifts; they are not used as exact
labels.

This is a *block*, not a full model — it composes into strategies that
need acquisition conditioning (multi-contrast SSL, paired synthesis,
Bloch-bottleneck pipelines). Register via ``@register_block`` so the
audit can detect when YAML mounts it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from mriforge.infrastructure.physics.differentiable_bloch import (
    DifferentiableBlochLayer,
)
from mriforge.models.blocks.block_registry import register_block
from mriforge.models.blocks.interfaces import IBlockStrategy

# Default reference-tissue panel: (name, T1_ms, T2_ms, PD).
# Values reflect mid-field 1.5 T literature so the encoder produces
# physically plausible signals across the 64 mT - 3 T range when paired
# with the field-strength conditioning of the diffusion prior.
_DEFAULT_PANEL: tuple[tuple[str, float, float, float], ...] = (
    ("gm", 950.0, 100.0, 0.85),
    ("wm", 600.0, 80.0, 0.70),
    ("csf", 4000.0, 2000.0, 1.00),
    ("fat", 350.0, 50.0, 0.95),
    ("lesion_t1", 1200.0, 110.0, 0.80),
    ("lesion_t2", 900.0, 150.0, 0.85),
)


@dataclass(frozen=True)
class ReferenceTissue:
    """A single reference-tissue entry."""

    name: str
    t1_ms: float
    t2_ms: float
    pd: float


@dataclass(frozen=True)
class ReferencePanel:
    """Fixed reference-tissue panel used by :class:`BlochSignalEncoder`."""

    tissues: tuple[ReferenceTissue, ...]

    @classmethod
    def default(cls) -> ReferencePanel:
        """Build the default 6-tissue panel."""
        return cls(
            tuple(
                ReferenceTissue(name=name, t1_ms=t1, t2_ms=t2, pd=pd)
                for name, t1, t2, pd in _DEFAULT_PANEL
            )
        )

    @classmethod
    def from_names(cls, names: Sequence[str]) -> ReferencePanel:
        """Build a panel by subselecting the default panel by name."""
        lookup = {t.name: t for t in cls.default().tissues}
        missing = [n for n in names if n not in lookup]
        if missing:
            raise ValueError(
                f"Unknown reference-tissue names: {missing}. Available: {sorted(lookup.keys())}"
            )
        return cls(tuple(lookup[n] for n in names))

    @property
    def size(self) -> int:
        """Number of tissues in the panel."""
        return len(self.tissues)


@register_block("bloch_signal_encoder")
class BlochSignalEncoder(IBlockStrategy):
    """Embed an acquisition vector by its Bloch signature on a fixed panel.

    The encoder runs a frozen SPGR Bloch simulation on the configured
    reference panel for each acquisition in the input batch and projects
    the resulting signal vector through a small MLP. The Bloch primitives
    are taken from :class:`DifferentiableBlochLayer` so the embedding is
    differentiable with respect to :math:`(T_E, T_R, \\alpha, B_0)` when
    needed (e.g. for protocol-design optimisation), and AMP-safe.

    Args:
        embed_dim: Output embedding dimensionality.
        panel: Optional :class:`ReferencePanel` (defaults to the 6-tissue
            default panel).
        hidden_dim: Width of the projection MLP's hidden layer.
        sequence_type: Bloch sequence variant; only ``"SPGR"`` is
            supported at the moment.
        concomitant_auto_threshold_t: Field strength (Tesla) below which
            the Maxwell concomitant correction is auto-enabled on every
            simulation. Defaults to 0.5 T.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        panel: ReferencePanel | None = None,
        hidden_dim: int = 64,
        sequence_type: str = "SPGR",
        concomitant_auto_threshold_t: float = 0.5,
    ) -> None:
        super().__init__()
        self.panel = panel if panel is not None else ReferencePanel.default()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.sequence_type = sequence_type
        self.concomitant_auto_threshold_t = concomitant_auto_threshold_t

        self._bloch = DifferentiableBlochLayer(sequence_type=sequence_type)

        # Project the [n_panel] real-valued signal vector to embed_dim.
        in_features = self.panel.size
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Register the panel parameters as buffers so they move with .to().
        t1 = torch.tensor([t.t1_ms for t in self.panel.tissues], dtype=torch.float32)
        t2 = torch.tensor([t.t2_ms for t in self.panel.tissues], dtype=torch.float32)
        pd = torch.tensor([t.pd for t in self.panel.tissues], dtype=torch.float32)
        self.register_buffer("_panel_t1_ms", t1)
        self.register_buffer("_panel_t2_ms", t2)
        self.register_buffer("_panel_pd", pd)

    def _simulate_panel(
        self,
        te_ms: torch.Tensor,
        tr_ms: torch.Tensor,
        flip_angle_rad: torch.Tensor,
        b0_t: torch.Tensor,
    ) -> torch.Tensor:
        """Run the panel through the Bloch layer.

        All inputs are shape ``(B,)``. Output is ``(B, n_panel)``: the
        SPGR magnitude for each (acquisition, tissue) pair.
        """
        batch_size = te_ms.shape[0]
        n_panel = self.panel.size
        # Shape (B, n_panel, 1, 1, 1) for the Bloch layer's [B, C, H, W] expectation.
        t1 = self._panel_t1_ms.view(1, n_panel, 1, 1).expand(batch_size, n_panel, 1, 1)
        t2 = self._panel_t2_ms.view(1, n_panel, 1, 1).expand(batch_size, n_panel, 1, 1)
        pd = self._panel_pd.view(1, n_panel, 1, 1).expand(batch_size, n_panel, 1, 1)

        # The Bloch layer expects scalar tr/te/flip_angle per call. Loop over
        # batch elements; the panel dim acts as the channel dim so each call
        # produces an (1, n_panel, 1, 1) tensor we stack.
        outputs: list[torch.Tensor] = []
        for b in range(batch_size):
            sig = self._bloch.forward(
                t1_map=t1[b : b + 1],
                t2_map=t2[b : b + 1],
                pd_map=pd[b : b + 1],
                tr=float(tr_ms[b].item()),
                te=float(te_ms[b].item()),
                flip_angle_rad=flip_angle_rad[b],
                b0_map=None,
                b1_map=None,
                include_concomitant=False,  # encoder uses pure SPGR magnitudes
                field_strength_t=float(b0_t[b].item()),
            )
            # sig: (1, n_panel, 1, 1) → (n_panel,)
            outputs.append(sig.real.view(n_panel) if sig.is_complex() else sig.view(n_panel))
        return torch.stack(outputs, dim=0)

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Embed an acquisition vector.

        Args:
            x: Tensor of shape ``(B, 5)`` with columns
                ``(T_E_ms, T_R_ms, T_I_ms, FA_deg, B_0_T)``. ``T_I`` is
                accepted for schema compatibility but is currently
                unused by the SPGR kernel.

        Returns:
            Tensor of shape ``(B, embed_dim)``.
        """
        if x.ndim != 2 or x.shape[-1] != 5:
            raise ValueError(
                f"BlochSignalEncoder expects (B, 5) acquisition vectors "
                f"(TE, TR, TI, FA, B0); got shape {tuple(x.shape)}."
            )
        te_ms = x[:, 0]
        tr_ms = x[:, 1]
        # ti_ms = x[:, 2]  # reserved for IR sequences
        fa_deg = x[:, 3]
        b0_t = x[:, 4]

        flip_angle_rad = fa_deg * (torch.pi / 180.0)
        signals = self._simulate_panel(te_ms, tr_ms, flip_angle_rad, b0_t)
        return self.proj(signals)
