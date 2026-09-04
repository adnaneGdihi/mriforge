"""Forward-pass + guard contract for the pure-k-space backbone.

2026-07-14 (issue #278): ``test_kspace_generator_forward`` built
``backbone_type='unet' + force_pure_kspace=True`` while leaving ``attention_type``
at its constructor default of ``'self'`` — a combination the generator
**correctly rejects**, because ``PureKSpaceUNet`` has no attention seam and would
silently drop the request (pitfall #16). The guard was right; the test predated it
and had been red on ``dev`` ever since, unnoticed because Actions is disabled here.

The forward test now opts out of attention explicitly, and the guard itself is
covered positively below rather than merely being tripped over.
"""

import pytest
import torch

from spectramr.models.generators.kspace_cold_diffusion_generator import (
    KSpaceColdDiffusionGenerator,
)

# ``condition_with_smaps=False`` because the default conditioning path doubles the
# backbone's expected input channels (noisy k-space + S-maps), and these unit tests
# call the model directly without the strategy-layer concat.
_PURE_KSPACE_KWARGS = dict(
    in_channels=2,
    out_channels=2,
    base_channels=16,
    num_layers=2,
    time_embedding_dim=32,
    use_complex_conv=True,
    force_pure_kspace=True,
    condition_with_smaps=False,
)


def test_kspace_generator_forward() -> None:
    """The pure-k-space backbone round-trips a (B, C, H, W) batch."""
    model = KSpaceColdDiffusionGenerator(
        **_PURE_KSPACE_KWARGS,
        attention_type="none",  # PureKSpaceUNet has no attention seam
    )

    x = torch.randn(2, 2, 64, 64)
    timesteps = torch.randint(0, 1000, (2,))

    # `dc_method` defaults to 'hard', so the training forward builds a DC layer
    # and refuses to run without the measurement it is gated on. Supplying it is
    # what production does (`_build_generator_kwargs`); a bare call used to pass
    # here only because the skip was silent, which a shape assertion cannot see.
    mask = torch.zeros(2, 1, 64, 64)
    mask[..., ::2] = 1.0

    output = model(x, timestep=timesteps, kspace_measured=x * mask, mask=mask)
    if isinstance(output, tuple):  # (output, scale) when training=True
        output = output[0]

    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_pure_kspace_unet_rejects_explicit_attention() -> None:
    """An attention request that cannot be honoured must RAISE, not be dropped.

    Silently ignoring it would ship an arm advertising ``attention_type=self``
    that actually ran vanilla — a mislabeled entry in the sim2rank pool.
    """
    with pytest.raises(ValueError, match="no attention seam"):
        KSpaceColdDiffusionGenerator(**_PURE_KSPACE_KWARGS, attention_type="self")


def test_pure_kspace_unet_rejects_defaulted_attention() -> None:
    """Omitting the key is not an escape hatch: the default is ``'self'``.

    This is the case the old forward test tripped over. An arm that simply never
    declares ``attention_type`` must fail at build time too, otherwise the facade
    returns through the back door.
    """
    with pytest.raises(ValueError, match="no attention seam"):
        KSpaceColdDiffusionGenerator(**_PURE_KSPACE_KWARGS)
