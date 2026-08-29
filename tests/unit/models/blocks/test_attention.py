"""Energy-stability contracts for the exp_11 shootout attention blocks.

The exp_11 energy probe (PR #398; cluster run 2026-07-20) measured
``KernelizedAttention`` at ``worst_rho ~ 3.6e3``: the block computed the
linear-attention numerator ``phi(Q) (phi(K)^T V)`` with no ``D^{-1}``
normalizer, so its gain scaled with sequence length (issue #405). These tests
pin the FAVOR+ contract (positive features, softmax-kernel estimate,
row-stochastic attention, seq-len-invariant gain) and the identity-at-init
contract for the residual family (zero-init output projections).
"""

import pytest
import torch
from torch import nn

from mriforge.models.blocks.attention import (
    KernelizedAttention,
    LinearAttention,
    WindowAttention,
)


def _gain(module: torch.nn.Module, x: torch.Tensor) -> float:
    with torch.no_grad():
        y = module(x)
    return (y.norm() / x.norm()).item()


class TestKernelizedFavorPlus:
    """FAVOR+ numerics (Choromanski et al., 2021) on the internal seam."""

    def test_features_strictly_positive(self):
        torch.manual_seed(0)
        attn = KernelizedAttention(32, num_heads=4, num_features=64)
        x = torch.randn(2, 4, 50, 8)  # [B, heads, L, head_dim]
        feat = attn._favor_plus_features(x)
        assert (feat > 0).all(), "FAVOR+ features must be strictly positive"

    def test_feature_dot_estimates_softmax_kernel(self):
        # E[phi(q) . phi(k)] = exp(q . k / sqrt(d)) -- the softmax kernel.
        torch.manual_seed(0)
        d, m = 8, 4096
        attn = KernelizedAttention(d, num_heads=1, num_features=m)
        q = 0.5 * torch.randn(1, 1, 16, d)
        k = 0.5 * torch.randn(1, 1, 16, d)
        est = torch.einsum(
            "bhlf,bhmf->bhlm",
            attn._favor_plus_features(q),
            attn._favor_plus_features(k),
        )
        truth = torch.exp(torch.einsum("bhld,bhmd->bhlm", q, k) * d**-0.5)
        torch.testing.assert_close(est, truth, rtol=0.25, atol=0.05)

    def test_row_stochastic_maps_constant_v_to_itself(self):
        # With D^{-1} normalization the implied attention matrix is
        # row-stochastic, so a constant value field is a fixed point --
        # regardless of Q, K, and sequence length.
        torch.manual_seed(0)
        attn = KernelizedAttention(32, num_heads=4, num_features=64)
        shape = (2, 4, 50, 8)  # [batch, heads, seq_len, head_dim]
        q, k = torch.randn(shape), torch.randn(shape)
        v = torch.full(shape, 3.0)
        out = attn._favor_attention(q, k, v)
        torch.testing.assert_close(out, v, rtol=1e-4, atol=1e-4)

    def test_gain_does_not_scale_with_sequence_length(self):
        # The #405 defect: without the normalizer the un-averaged sum over
        # L = H*W positions makes the gain grow ~linearly with L (measured
        # worst_rho ~3.6e3 on full-res k-space). 16x the tokens must not
        # mean ~16x the gain.
        torch.manual_seed(0)
        attn = KernelizedAttention(32, num_heads=4, num_features=64).eval()
        with torch.no_grad():
            for p in attn.parameters():
                torch.nn.init.normal_(p, std=0.05)
        g_small = _gain(attn, torch.randn(1, 32, 8, 8))
        g_large = _gain(attn, torch.randn(1, 32, 32, 32))
        assert g_large < 4.0 * g_small, (
            f"gain scales with seq_len: {g_small:.3f} @ 64 tokens vs {g_large:.3f} @ 1024 tokens"
        )

    def test_identity_at_init(self):
        # Residual + zero-init out_proj: the block starts as the identity,
        # rho = 1.0, and learns its gain during training.
        torch.manual_seed(0)
        attn = KernelizedAttention(32, num_heads=4, num_features=64)
        x = torch.randn(2, 32, 16, 16)
        torch.testing.assert_close(attn(x), x)


class TestResidualFamilyInitIdentity:
    """The residual blocks must start as the identity (zero-init proj)."""

    def test_linear_attention_identity_at_init(self):
        torch.manual_seed(0)
        attn = LinearAttention(32)
        x = torch.randn(2, 32, 16, 16)
        torch.testing.assert_close(attn(x), x)

    def test_window_attention_identity_at_init(self):
        torch.manual_seed(0)
        attn = WindowAttention(32)
        x = torch.randn(2, 32, 16, 16)
        torch.testing.assert_close(attn(x), x)

    @pytest.mark.parametrize("cls", [LinearAttention, WindowAttention])
    def test_gradients_reach_attention_path(self, cls):
        # Zero-init must not dead-end the branch: the attention parameters
        # still receive gradient through the residual sum.
        torch.manual_seed(0)
        attn = cls(32)
        x = torch.randn(2, 32, 16, 16, requires_grad=True)
        attn(x).square().mean().backward()
        proj = attn.proj.weight.grad
        assert proj is not None and proj.abs().sum() > 0


# ─────────────────────────────────────────────────────────────────────────────
# IdentityAtInitAttention (issue #471)
#
# LinearAttention states the family contract in its own initialiser: "zero output
# projection => forward(x) == x, so the residual family starts at energy gain
# rho = 1.0". Only 3 of the 8 blocks the complex_unet dispatch can build honoured
# it. Measured rho at init over 5 seeds on a 1/f k-space feature map:
# self/kernelized/sparse 1.000, dual_domain ~1.05, channel ~0.65,
# wavelet_freq ~0.54, spatial ~0.96 (as low as 0.23), kan_dual_domain 7.4-9.2.
# The call site is REPLACE, so a non-identity block discards its input and emits
# that -- the shootout was comparing eight different starting points.
# ─────────────────────────────────────────────────────────────────────────────


def _kspace_feature(seed: int = 0, C: int = 32, H: int = 32, W: int = 32) -> torch.Tensor:
    """1/f magnitude, random phase: the dynamic range a k-space feature map has."""
    torch.manual_seed(seed)
    ky = torch.fft.fftshift(torch.fft.fftfreq(H))[:, None]
    kx = torch.fft.fftshift(torch.fft.fftfreq(W))[None, :]
    amp = 1.0 / (ky**2 + kx**2).sqrt().clamp_min(1e-3)
    return (amp * torch.randn(2, C, H, W)).contiguous()


def test_identity_at_init_is_bit_exact() -> None:
    """Not "close to" identity: the output must BE the input at step 0."""
    from mriforge.models.blocks.attention import ChannelAttention, IdentityAtInitAttention

    x = _kspace_feature()
    wrapped = IdentityAtInitAttention(ChannelAttention(x.shape[1])).eval()

    with torch.no_grad():
        out = wrapped(x)

    assert torch.equal(out, x), "gamma=0 must give an exact identity"
    assert (out.norm() / x.norm()).item() == pytest.approx(1.0, abs=0.0)


def test_gamma_one_reproduces_the_raw_block() -> None:
    """No expressivity is given up: gamma interpolates identity -> native block."""
    from mriforge.models.blocks.attention import ChannelAttention, IdentityAtInitAttention

    torch.manual_seed(3)
    inner = ChannelAttention(32)
    wrapped = IdentityAtInitAttention(inner).eval()
    with torch.no_grad():
        wrapped.gamma.fill_(1.0)

    x = _kspace_feature()
    with torch.no_grad():
        torch.testing.assert_close(wrapped(x), inner(x))


def test_gamma_receives_gradient_so_it_leaves_zero() -> None:
    """A zero-init knob that cannot be trained is a dead parameter (#15).

    The loss must not be purely quadratic in the block's own output -- for a
    multiplicative zero scale that gives exactly 0 and says nothing.
    """
    from mriforge.models.blocks.attention import ChannelAttention, IdentityAtInitAttention

    torch.manual_seed(0)
    wrapped = IdentityAtInitAttention(ChannelAttention(16))
    x = torch.randn(1, 16, 8, 8)
    target = torch.randn(1, 16, 8, 8)

    ((wrapped(x) - target) ** 2).sum().backward()

    assert wrapped.gamma.grad is not None
    assert wrapped.gamma.grad.abs().item() > 0.0


def test_t_emb_is_forwarded_by_signature_not_by_class_tuple() -> None:
    """The old dispatch was an isinstance check against a hardcoded class tuple.

    Wrapping made that check false, which would have silently blinded the KAN
    block's timestep conditioning -- and the same trap waits for any future
    time-conditioned block. Detection is by signature instead.
    """
    from mriforge.models.blocks.attention import ChannelAttention, IdentityAtInitAttention
    from mriforge.models.blocks.dual_domain_attention_kan import (
        KANGatedDualDomainAttention,
    )

    assert IdentityAtInitAttention(ChannelAttention(16)).takes_t_emb is False
    kan = IdentityAtInitAttention(
        KANGatedDualDomainAttention(
            in_channels=16, time_embedding_dim=32, feature_domain="kspace", num_heads=2
        )
    )
    assert kan.takes_t_emb is True

    x = _kspace_feature(C=16, H=16, W=16)
    with torch.no_grad():
        out = kan.eval()(x, torch.randn(x.shape[0], 32))
    assert torch.equal(out, x)


def test_shape_changing_inner_raises_rather_than_broadcasting() -> None:
    """A block that alters the feature shape is a wiring error, not something to
    silently broadcast against the residual (pitfall #9)."""
    from mriforge.models.blocks.attention import IdentityAtInitAttention

    class _Shrinks(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, : x.shape[1] // 2]

    wrapped = IdentityAtInitAttention(_Shrinks())
    with pytest.raises(ValueError, match="changed the feature shape"):
        wrapped(torch.randn(1, 8, 4, 4))


def test_cross_contrast_olmpa_is_identity_at_init() -> None:
    """The bottleneck block ComplexUNet adds residually (``attended + target_feat``).

    Without the zero output scale it emitted 150-190x the target's norm across
    seeds, so the bottleneck carried almost pure attention output at step 0.
    """
    from mriforge.models.blocks.attention import CrossContrastOLMPA

    for seed in range(3):
        torch.manual_seed(seed)
        block = CrossContrastOLMPA(in_channels=32, num_contrasts=1, phase_safe_dim=64).eval()
        target = torch.randn(2, 64, 32)
        with torch.no_grad():
            out = block(target, torch.randn(2, 64, 32))
        assert out.abs().max().item() == 0.0, f"seed {seed}: not identity at init"


def test_cross_contrast_olmpa_out_scale_trains_off_zero() -> None:
    from mriforge.models.blocks.attention import CrossContrastOLMPA

    torch.manual_seed(0)
    block = CrossContrastOLMPA(in_channels=8, num_contrasts=1, phase_safe_dim=8)
    target, ref = torch.randn(1, 16, 8), torch.randn(1, 16, 8)
    # Mirror the ComplexUNet call site: the block's output is added residually.
    ((block(target, ref) + target - torch.randn(1, 16, 8)) ** 2).sum().backward()

    assert block.out_scale.grad is not None
    assert block.out_scale.grad.abs().item() > 0.0


class TestPhaseSafeDualAttentionReductionIsLive:
    """``reduction`` must actually reduce (pitfall #15).

    The block computed ``hidden_dim = max(stacked_channels // reduction,
    stacked_channels)``. Since ``reduction >= 1`` makes the left operand no
    larger than the right, that ``max`` always returned ``stacked_channels`` --
    an advertised knob that provably could not change the module. It reads as
    consumed (it is stored on ``self.reduction`` and appears in the signature),
    which is why it survived: only evaluating the expression exposes it.
    """

    @staticmethod
    def _hidden_dim(block) -> int:
        # The Conv2d in query_proj maps stacked_channels -> hidden_dim.
        return block.query_proj[0].out_channels

    def test_reduction_changes_the_projection_width(self):
        from mriforge.models.blocks.attention import PhaseSafeDualAttention

        wide = PhaseSafeDualAttention(in_channels=8, num_heads=1, reduction=1)
        narrow = PhaseSafeDualAttention(in_channels=8, num_heads=1, reduction=4)
        assert self._hidden_dim(wide) == 16  # in_channels * 2
        assert self._hidden_dim(narrow) == 4
        assert self._hidden_dim(narrow) < self._hidden_dim(wide)

    def test_default_reduction_is_unchanged_so_checkpoints_still_load(self):
        """reduction=1 must give the pre-fix width, or every state_dict breaks."""
        from mriforge.models.blocks.attention import PhaseSafeDualAttention

        block = PhaseSafeDualAttention(in_channels=4, num_heads=1, reduction=1)
        assert self._hidden_dim(block) == 8  # in_channels * 2, as before

    def test_reduction_never_collapses_the_projection_to_zero(self):
        """The floor is 1 channel, not 0 -- Conv2d(out_channels=0) is invalid."""
        from mriforge.models.blocks.attention import PhaseSafeDualAttention

        block = PhaseSafeDualAttention(in_channels=1, num_heads=1, reduction=64)
        assert self._hidden_dim(block) == 1

    def test_a_reduction_below_one_raises_rather_than_degrading(self):
        """Non-negotiable #3: an illegal value raises, it does not fall back."""
        from mriforge.models.blocks.attention import PhaseSafeDualAttention

        with pytest.raises(ValueError, match="reduction must be >= 1"):
            PhaseSafeDualAttention(in_channels=8, num_heads=1, reduction=0)
