"""Tests for the normalization SSOT.

Targets ``mriforge.data.transforms.normalization``. The module is the single
source of truth for normalization across datasets, training strategies,
and TorchIO transforms — every consumer must delegate here, so the
tests verify the documented contracts:

- ``compute_magnitude`` handles every documented input layout
- ``normalize_percentile`` round-trips via ``denormalize_percentile``
- ``normalize_zscore`` produces zero-mean, unit-std output
- ``normalize_minmax`` maps the input range to ``out_range`` exactly
- ``normalize_tensor`` dispatcher routes to the right strategy
- ``NormalizationStrategy.from_string`` accepts case-insensitive aliases
"""

from __future__ import annotations

import math
from typing import ClassVar

import pytest
import torch
import torchio as tio

from mriforge.data.transforms.normalization import (
    DECOMPRESS_MAGNITUDE_CEILING,
    NormalizationConfig,
    NormalizationStrategy,
    compress_kspace_log,
    compute_magnitude,
    decompress_kspace_log,
    denormalize_percentile,
    normalize_minmax,
    normalize_percentile,
    normalize_tensor,
    normalize_zscore,
)


class TestLogKspaceCompression:
    """Phase-preserving log1p magnitude compression for k-space dynamic range.

    Regression for the experiment_11 DC-blob: raw k-space has a ~200x dynamic
    range the CNN can't represent, so data-consistency injection of the
    measured DC dominates the model's squashed output -> centre blob. Log
    compression reduces the range; this verifies it round-trips, preserves
    phase, and actually compresses.
    """

    def _make_complex_kspace(self):
        torch.manual_seed(0)
        # DC-dominated complex k-space: huge centre, tiny periphery.
        z = torch.randn(2, 4, 16, 16, dtype=torch.complex64) * 0.2
        z[:, :, 8, 8] = 50.0 + 10.0j  # DC spike
        return z

    def test_round_trip_complex(self):
        z = self._make_complex_kspace()
        recovered = decompress_kspace_log(compress_kspace_log(z))
        assert torch.allclose(recovered, z, atol=1e-3, rtol=1e-3)

    def test_round_trip_real_stacked_interleaved(self):
        z = self._make_complex_kspace()  # [B, C, H, W] complex
        # interleave R/I along channel dim 1 -> [B, 2C, H, W]
        stacked = torch.stack([z.real, z.imag], dim=2).reshape(z.shape[0], -1, *z.shape[2:])
        recovered = decompress_kspace_log(
            compress_kspace_log(stacked, channel_dim=1), channel_dim=1
        )
        assert torch.allclose(recovered, stacked, atol=1e-3, rtol=1e-3)

    def test_compression_reduces_dynamic_range(self):
        z = self._make_complex_kspace()
        comp = compress_kspace_log(z)
        raw_range = z.abs().max() / z.abs().median().clamp_min(1e-8)
        comp_range = comp.abs().max() / comp.abs().median().clamp_min(1e-8)
        assert comp_range < raw_range / 5, (
            f"log compression should shrink the dynamic range markedly: "
            f"raw={raw_range:.1f} comp={comp_range:.1f}"
        )

    def test_phase_is_preserved(self):
        z = self._make_complex_kspace()
        comp = compress_kspace_log(z)
        # Where magnitude is non-trivial, the phase angle must be unchanged.
        mask = z.abs() > 1e-3
        assert torch.allclose(
            torch.angle(comp)[mask], torch.angle(z)[mask], atol=1e-4
        )

    def test_odd_channel_count_raises(self):
        with pytest.raises(ValueError, match="even channel count"):
            compress_kspace_log(torch.randn(1, 3, 8, 8), channel_dim=1)

    def test_decompress_clamps_expm1_overflow(self):
        # Regression (experiment_11 kernelized-attention arm, iter-1000 smoke):
        # an under-trained model emitted a *compressed* |k| ~1750. Un-clamped
        # expm1(1750) overflows float32 -> inf -> inf*scale -> NaN that poisoned
        # every validation metric and made EarlyStopping pick "best" on NaN.
        # The decompress must stay finite for both memory layouts.
        huge_stacked = torch.full((1, 2, 4, 4), 1750.0, dtype=torch.float32)
        out = decompress_kspace_log(huge_stacked, channel_dim=1)
        assert torch.isfinite(out).all(), "decompress emitted inf/NaN on large |k|"

        huge_complex = torch.full((1, 1, 4, 4), 1750.0 + 0.0j, dtype=torch.complex64)
        out_c = decompress_kspace_log(huge_complex)
        assert torch.isfinite(out_c.abs()).all()

    def test_decompress_clamp_boundary(self):
        # Below the ceiling decompresses exactly (expm1(m)); far above is capped
        # at expm1(ceiling) rather than inf -- the ceiling sits well above any
        # physical compressed magnitude (<= ~6) so legitimate data is untouched.
        ceiling = DECOMPRESS_MAGNITUDE_CEILING
        below = torch.full((1, 1, 2, 2), ceiling - 5.0, dtype=torch.complex64)
        above = torch.full((1, 1, 2, 2), 500.0, dtype=torch.complex64)
        out_below = decompress_kspace_log(below).abs()
        out_above = decompress_kspace_log(above).abs()
        assert torch.allclose(
            out_below, torch.tensor(math.expm1(ceiling - 5.0)), rtol=1e-3
        )
        assert torch.allclose(out_above, torch.tensor(math.expm1(ceiling)), rtol=1e-3)
        assert torch.isfinite(out_above).all()


class TestSSOTRobustKspaceNormalization:
    """The single normalize/denormalize pair every k-space normalizer must use.

    Round-trips input AND output the same way (the experiment_11 fix requires
    input, prediction, and target to all live in the same compressed domain).
    """

    def _ksp(self):
        torch.manual_seed(1)
        z = torch.randn(2, 4, 16, 16, dtype=torch.complex64) * 0.2
        z[:, :, 8, 8] = 60.0 + 5.0j
        return z

    @pytest.mark.parametrize("log_scaling", [False, True])
    def test_round_trip_complex(self, log_scaling):
        from mriforge.data.transforms.normalization import (
            denormalize_kspace_robust,
            normalize_kspace_robust,
        )

        z = self._ksp()
        norm, scale = normalize_kspace_robust(z, percentile=0.95, log_scaling=log_scaling)
        back = denormalize_kspace_robust(norm, scale, log_scaling=log_scaling)
        assert torch.allclose(back, z, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("log_scaling", [False, True])
    def test_round_trip_real_stacked(self, log_scaling):
        from mriforge.data.transforms.normalization import (
            denormalize_kspace_robust,
            normalize_kspace_robust,
        )

        z = self._ksp()
        stk = torch.stack([z.real, z.imag], dim=2).reshape(z.shape[0], -1, *z.shape[2:])
        norm, scale = normalize_kspace_robust(
            stk, percentile=0.95, log_scaling=log_scaling, channel_dim=1
        )
        back = denormalize_kspace_robust(
            norm, scale, log_scaling=log_scaling, channel_dim=1
        )
        assert torch.allclose(back, stk, atol=1e-2, rtol=1e-2)

    def test_log_scaling_compresses_dynamic_range(self):
        from mriforge.data.transforms.normalization import normalize_kspace_robust

        z = self._ksp()
        lin, _ = normalize_kspace_robust(z, percentile=0.95, log_scaling=False)
        log, _ = normalize_kspace_robust(z, percentile=0.95, log_scaling=True)
        assert log.abs().max() < lin.abs().max() / 3

    def test_precomputed_scale_is_used(self):
        from mriforge.data.transforms.normalization import normalize_kspace_robust

        z = self._ksp()
        fixed = torch.tensor(4.0)
        norm, scale = normalize_kspace_robust(z, scale=fixed, log_scaling=False)
        assert torch.allclose(scale, fixed)
        assert torch.allclose(norm, z / fixed)


# ---------------------------------------------------------------------------
# NormalizationStrategy enum
# ---------------------------------------------------------------------------


def test_strategy_from_string_case_insensitive() -> None:
    """``from_string`` accepts mixed case."""
    assert NormalizationStrategy.from_string("PERCENTILE") == NormalizationStrategy.PERCENTILE
    assert NormalizationStrategy.from_string("ZScore") == NormalizationStrategy.ZSCORE


def test_strategy_robust_percentile_alias() -> None:
    """The legacy ``robust_percentile`` alias maps to ``PERCENTILE``."""
    assert (
        NormalizationStrategy.from_string("robust_percentile")
        == NormalizationStrategy.PERCENTILE
    )


def test_strategy_unknown_raises_with_valid_options() -> None:
    """Unknown strategy fails loud, listing the valid options."""
    with pytest.raises(ValueError, match="Unknown normalization strategy"):
        NormalizationStrategy.from_string("magic")


# ---------------------------------------------------------------------------
# compute_magnitude — multi-layout support
# ---------------------------------------------------------------------------


def test_compute_magnitude_complex_input() -> None:
    """Complex tensor → ``torch.abs(data)``."""
    cplx = torch.complex(torch.ones(2, 1, 4, 4), torch.zeros(2, 1, 4, 4))
    out = compute_magnitude(cplx)
    assert torch.allclose(out, torch.ones_like(out))


def test_compute_magnitude_b2hw_layout() -> None:
    """``[B, 2, H, W]`` real-stacked → sqrt(re² + im²)."""
    re = torch.full((1, 1, 2, 2), 3.0)
    im = torch.full((1, 1, 2, 2), 4.0)
    stacked = torch.cat([re, im], dim=1)  # [1, 2, 2, 2]
    out = compute_magnitude(stacked)
    assert torch.allclose(out, torch.full_like(out, 5.0), atol=1e-3)


def test_compute_magnitude_real_input_is_abs() -> None:
    """Real-only fallback returns ``abs(data)``."""
    data = torch.tensor([-2.0, 3.0, -5.0])
    out = compute_magnitude(data)
    assert torch.allclose(out, torch.tensor([2.0, 3.0, 5.0]))


# ---------------------------------------------------------------------------
# normalize_percentile + denormalize_percentile (round-trip)
# ---------------------------------------------------------------------------


def test_percentile_normalize_round_trip() -> None:
    """``denormalize(normalize(x))`` recovers x within float32 tolerance.

    Out-of-default range disables clamping in this test by setting
    ``clamp=False``.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 1, 16, 16) * 10.0
    norm, scale = normalize_percentile(
        x, percentile=0.99, clamp=False, out_range=(0.0, 1.0)
    )
    recon = denormalize_percentile(norm, scale)
    assert torch.allclose(recon, x, atol=1e-4)


def test_percentile_normalize_scale_is_positive() -> None:
    """The returned scale is always positive (clamped to ``min_scale``)."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 8, 8)
    _, scale = normalize_percentile(x, percentile=0.99)
    assert scale.item() > 0


def test_percentile_normalize_handles_all_zero_input() -> None:
    """All-zero input → finite output (no NaN), scale clamped to eps."""
    x = torch.zeros(1, 1, 8, 8)
    norm, _ = normalize_percentile(x)
    assert torch.isfinite(norm).all()


def test_percentile_normalize_clamps_to_out_range() -> None:
    """``clamp=True`` enforces output range."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 16, 16) * 100.0
    norm, _ = normalize_percentile(
        x, percentile=0.5, clamp=True, out_range=(0.0, 1.0)
    )
    assert norm.min().item() >= 0.0
    assert norm.max().item() <= 1.0


# ---------------------------------------------------------------------------
# normalize_zscore
# ---------------------------------------------------------------------------


def test_zscore_produces_zero_mean_unit_std() -> None:
    """Output has mean ≈ 0, std ≈ 1."""
    torch.manual_seed(0)
    x = torch.randn(1000) * 5.0 + 3.0  # large mean and std
    norm, (mean, std) = normalize_zscore(x)
    assert abs(norm.mean().item()) < 1e-5
    assert abs(norm.std().item() - 1.0) < 1e-3
    assert torch.allclose(mean, x.mean())
    assert torch.allclose(std, x.std() + 1e-8, atol=1e-9)


def test_zscore_constant_input_does_not_explode() -> None:
    """Constant input → std = 0 + eps; output is finite."""
    x = torch.full((100,), 7.0)
    norm, _ = normalize_zscore(x)
    assert torch.isfinite(norm).all()


# ---------------------------------------------------------------------------
# normalize_minmax
# ---------------------------------------------------------------------------


def test_minmax_maps_to_out_range_exactly() -> None:
    """min(out) = out_min, max(out) = out_max."""
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
    norm, _ = normalize_minmax(x, out_range=(-2.0, 2.0))
    assert torch.allclose(norm.min(), torch.tensor(-2.0))
    assert torch.allclose(norm.max(), torch.tensor(2.0))


def test_minmax_constant_input_returned_unchanged() -> None:
    """Constant input (range < eps) → returned unchanged with dummy stats."""
    x = torch.full((10,), 5.0)
    norm, (min_val, max_val) = normalize_minmax(x, eps=1e-3)
    assert torch.equal(norm, x)
    assert min_val == max_val


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_none_returns_input_unchanged() -> None:
    """Strategy NONE is identity."""
    x = torch.randn(2, 1, 4, 4)
    out = normalize_tensor(x, NormalizationConfig(strategy=NormalizationStrategy.NONE))
    assert torch.equal(out, x)


def test_dispatcher_percentile_routes_correctly() -> None:
    """Strategy PERCENTILE produces a normalised output."""
    x = torch.randn(1, 1, 8, 8) * 10.0
    cfg = NormalizationConfig(strategy=NormalizationStrategy.PERCENTILE, percentile=0.99)
    out = normalize_tensor(x, cfg)
    assert out.shape == x.shape
    assert not torch.equal(out, x)


def test_dispatcher_zscore_routes_correctly() -> None:
    """Strategy ZSCORE produces zero-mean, unit-std."""
    x = torch.randn(2000)
    cfg = NormalizationConfig(strategy=NormalizationStrategy.ZSCORE)
    out = normalize_tensor(x, cfg)
    assert abs(out.mean().item()) < 1e-3
    assert abs(out.std().item() - 1.0) < 1e-2


def test_dispatcher_minmax_routes_correctly() -> None:
    """Strategy MINMAX maps to default ``[0, 1]`` range."""
    x = torch.tensor([-2.0, 0.0, 2.0, 4.0])
    cfg = NormalizationConfig(strategy=NormalizationStrategy.MINMAX)
    out = normalize_tensor(x, cfg)
    assert torch.allclose(out.min(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(out.max(), torch.tensor(1.0), atol=1e-6)


# ---------------------------------------------------------------------------
# Sanity-shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(1, 1, 8, 8), (2, 2, 16, 16), (1, 4, 32, 32)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_percentile_shape_matrix(shape: tuple[int, ...]) -> None:
    """Percentile normalize preserves shape across the matrix."""
    x = torch.randn(*shape)
    norm, scale = normalize_percentile(x)
    assert norm.shape == shape
    assert torch.isfinite(norm).all()


# ---------------------------------------------------------------------------
# kspace_image_domain_scale — Parseval-compliant scale
# ---------------------------------------------------------------------------
# Ported out of ``M4RawRepetitionDataset.__getitem__``, which computed it inline
# while the transform ALSO normalized, double-scaling every arm. The dataset now
# matches and serves; this is the transform-layer home for the same physics.


class TestKSpaceImageDomainScale:
    """The scale is measured on the image-domain coil RSS, not on |k|."""

    @staticmethod
    def _kspace_of(img: "torch.Tensor") -> "torch.Tensor":
        from mriforge.infrastructure.physics.fft_ops import fft2c

        return fft2c(img)

    def test_matches_the_image_rss_quantile(self) -> None:
        """It equals the quantile of |RSS(ifft2c(k))| — the definition."""
        from mriforge.data.transforms.normalization import kspace_image_domain_scale

        torch.manual_seed(0)
        img = torch.randn(3, 16, 16, dtype=torch.complex64)  # (coils, H, W)
        k = self._kspace_of(img)

        expected_rss = torch.sqrt((img.abs() ** 2).sum(dim=0) + 1e-8)
        expected = torch.quantile(expected_rss.flatten().float(), 0.95)

        got = kspace_image_domain_scale(k, percentile=0.95)
        assert torch.allclose(got, expected, rtol=1e-4)

    def test_differs_from_the_kspace_magnitude_quantile(self) -> None:
        """The two domains are NOT interchangeable — k-space is DC-heavy.

        Guards the migration: swapping the domains silently would change what
        the network is trained on.
        """
        from mriforge.data.transforms.normalization import (
            compute_magnitude,
            kspace_image_domain_scale,
        )

        img = torch.zeros(1, 32, 32, dtype=torch.complex64)
        img[:, 8:24, 8:24] = 1.0
        k = self._kspace_of(img)

        image_scale = kspace_image_domain_scale(k, percentile=0.95)
        kspace_scale = torch.quantile(
            compute_magnitude(k).flatten().float(), 0.95
        )
        assert not torch.isclose(image_scale, kspace_scale, rtol=0.05)

    def test_accepts_real_interleaved_layout(self) -> None:
        """Real/imag interleaved along the channel axis gives the same scale."""
        from mriforge.data.transforms.normalization import kspace_image_domain_scale

        torch.manual_seed(1)
        img = torch.randn(2, 16, 16, dtype=torch.complex64)
        k = self._kspace_of(img)

        stacked = torch.empty(4, 16, 16)
        stacked[0::2] = k.real
        stacked[1::2] = k.imag

        assert torch.allclose(
            kspace_image_domain_scale(stacked, percentile=0.9),
            kspace_image_domain_scale(k, percentile=0.9),
            rtol=1e-4,
        )

    def test_odd_channel_count_falls_back_to_kspace_magnitude(self) -> None:
        """An odd channel count is not a real/imag stack — do not invent phase."""
        from mriforge.data.transforms.normalization import (
            compute_magnitude,
            kspace_image_domain_scale,
        )

        x = torch.rand(3, 8, 8)  # cannot be real/imag interleaved
        expected = torch.quantile(compute_magnitude(x).flatten().float(), 0.9)
        assert torch.allclose(
            kspace_image_domain_scale(x, percentile=0.9), expected, rtol=1e-4
        )


class TestKSpaceNormalizationScaleDomain:
    """``scale_domain`` selects the measurement domain and rejects typos."""

    def test_unknown_scale_domain_raises(self) -> None:
        """No silent fallback to a default domain (pitfall #9)."""
        from mriforge.data.transforms.normalization import KSpaceNormalizationTransform

        with pytest.raises(ValueError, match="Unknown scale_domain"):
            KSpaceNormalizationTransform(scale_domain="parseval")

    def test_image_domain_publishes_the_image_scale(self) -> None:
        """The published kspace_scale is the one actually divided by."""
        import torchio as tio

        from mriforge.data.transforms.normalization import (
            KSpaceNormalizationTransform,
            kspace_image_domain_scale,
        )
        from mriforge.infrastructure.physics.fft_ops import fft2c

        torch.manual_seed(2)
        img = torch.randn(1, 16, 16, dtype=torch.complex64)
        k = fft2c(img)
        stacked = torch.empty(2, 16, 16, 1)
        stacked[0::2, ..., 0] = k.real
        stacked[1::2, ..., 0] = k.imag

        subject = tio.Subject(kspace=tio.ScalarImage(tensor=stacked))
        out = KSpaceNormalizationTransform(
            percentile=0.95, log_scaling=False, scale_domain="image"
        )(subject)

        expected = kspace_image_domain_scale(stacked, percentile=0.95)
        assert torch.allclose(out["kspace_scale"], expected, rtol=1e-4)
        assert torch.allclose(
            out["kspace"].data, stacked / expected, rtol=1e-4, atol=1e-6
        )


class TestImageDomainScaleAxes:
    """The image-domain scale must FFT over (H, W), whatever the layout.

    Regression: the first implementation took the trailing two axes, so a
    multi-slice TorchIO subject ``(2C, H, W, D)`` was transformed over ``(W, D)``
    — a ~2.6x wrong scale on real anatomy.

    Fixtures here are STRUCTURED on purpose. White-noise k-space is invariant
    under any unitary transform, so a noise fixture cannot detect an axis bug
    (and the round-trip test cannot either — a scalar scale cancels regardless
    of how wrongly it was computed).
    """

    @staticmethod
    def _phantom(slices, coils, h, w):
        from mriforge.infrastructure.physics.fft_ops import fft2c

        torch.manual_seed(0)
        img = torch.zeros(slices, coils, h, w, dtype=torch.complex64)
        img[:, :, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
        img = img + 0.02 * torch.randn(slices, coils, h, w, dtype=torch.complex64)
        k = fft2c(img.reshape(-1, h, w)).reshape(slices, coils, h, w)
        return img, k

    @staticmethod
    def _expected(img, percentile=0.95):
        rss = torch.sqrt((img.abs() ** 2).sum(dim=1) + 1e-8)
        return torch.quantile(rss.flatten().float(), percentile)

    @pytest.mark.parametrize(
        "slices,coils,h,w",
        [(4, 2, 32, 32), (1, 2, 32, 24), (3, 3, 40, 24)],
        ids=["multislice", "single-slice", "non-square"],
    )
    def test_torchio_layout_c_h_w_d(self, slices, coils, h, w):
        """TorchIO ``(2C, H, W, D)`` with channel_dim=0."""
        from mriforge.data.transforms.normalization import kspace_image_domain_scale

        img, k = self._phantom(slices, coils, h, w)
        stacked = torch.empty(coils * 2, h, w, slices)
        stacked[0::2] = k.real.permute(1, 2, 3, 0)
        stacked[1::2] = k.imag.permute(1, 2, 3, 0)

        got = kspace_image_domain_scale(stacked, percentile=0.95, channel_dim=0)
        assert torch.allclose(got, self._expected(img), rtol=1e-2)

    def test_batched_layout_b_c_h_w(self):
        """Inference ``(B, 2C, H, W)`` with channel_dim=1."""
        from mriforge.data.transforms.normalization import kspace_image_domain_scale

        img, k = self._phantom(2, 3, 32, 32)  # slices axis reused as batch
        stacked = torch.empty(2, 6, 32, 32)
        stacked[:, 0::2] = k.real
        stacked[:, 1::2] = k.imag

        got = kspace_image_domain_scale(stacked, percentile=0.95, channel_dim=1)
        assert torch.allclose(got, self._expected(img), rtol=1e-2)

    def test_too_few_spatial_axes_raises(self):
        """A tensor with no (H, W) after the channel axis cannot form an image."""
        from mriforge.data.transforms.normalization import kspace_image_domain_scale

        with pytest.raises(ValueError, match="two spatial axes"):
            kspace_image_domain_scale(torch.randn(4, 8), channel_dim=0)


class TestKSpaceNormalizationSpec:
    """One resolver for the k-space scale, shared by training and inference."""

    CFG = {
        "normalize_kspace": True,
        "kspace_percentile": 0.95,
        "log_scaling": True,
        "kspace_scale_domain": "image",
        "normalization_kwargs": {"percentile": 0.99, "clamp": False},
    }

    def test_from_dict_and_from_object_agree(self):
        """Inference passes a dict, training a pydantic model."""
        from mriforge.data.transforms.normalization import KSpaceNormalizationSpec

        class _Obj:
            normalize_kspace = True
            kspace_percentile = 0.95
            log_scaling = True
            kspace_scale_domain = "image"

        assert KSpaceNormalizationSpec.from_data_config(
            self.CFG
        ) == KSpaceNormalizationSpec.from_data_config(_Obj())

    def test_disabled_spec_is_a_no_op_with_unit_scale(self):
        """Callers never branch on ``enabled`` themselves."""
        from mriforge.data.transforms.normalization import KSpaceNormalizationSpec

        spec = KSpaceNormalizationSpec.from_data_config(
            dict(self.CFG, normalize_kspace=False)
        )
        x = torch.randn(1, 4, 8, 8)
        out, scale = spec.normalize(x)
        assert out is x and float(scale) == 1.0
        assert spec.denormalize(x, scale) is x

    @pytest.mark.parametrize("domain", ["kspace", "image"])
    @pytest.mark.parametrize("log_scaling", [False, True])
    def test_round_trip(self, domain, log_scaling):
        from mriforge.data.transforms.normalization import KSpaceNormalizationSpec
        from mriforge.infrastructure.physics.fft_ops import fft2c

        spec = KSpaceNormalizationSpec.from_data_config(
            dict(self.CFG, kspace_scale_domain=domain, log_scaling=log_scaling)
        )
        torch.manual_seed(0)
        img = torch.zeros(1, 2, 16, 16, dtype=torch.complex64)
        img[:, :, 4:12, 4:12] = 1.0
        k = fft2c(img.reshape(-1, 16, 16)).reshape(1, 2, 16, 16)
        x = torch.empty(1, 4, 16, 16)
        x[:, 0::2], x[:, 1::2] = k.real, k.imag

        normed, scale = spec.normalize(x, channel_dim=1)
        restored = spec.denormalize(normed, scale, channel_dim=1)
        assert torch.allclose(restored, x, rtol=1e-3, atol=1e-4 * x.abs().max())

    def test_unknown_scale_domain_raises(self):
        from mriforge.data.transforms.normalization import KSpaceNormalizationSpec

        with pytest.raises(ValueError, match="scale_domain"):
            KSpaceNormalizationSpec(scale_domain="parseval")


# ---------------------------------------------------------------------------
# Image-domain normalization SSOT (#760) — the image twin of the #571 k-space fix
# ---------------------------------------------------------------------------


class TestKSpaceNormalizationSpecReadsTheNestedBlock:
    """The resolver reads ``data.processing``, not the flat legacy names.

    The phase-9 block decomposition re-parented all five knobs under
    ``data.processing`` and renamed two of them, but ``from_data_config`` kept
    reading the flat pre-decomposition spellings off ``data`` itself with
    ``getattr(data, name, default)``. Post-decomposition those names exist on no
    schema object and there is no forwarding shim, so every read missed and
    every declared value was replaced by its default.

    ``normalize`` is a *silent* no-op when disabled, so the
    ``apply_kspace_normalization`` fallback — which exists precisely to normalize
    a batch the dataloader served raw — returned the raw batch and a unit scale
    while reporting success. ``experiment_11_attention_none`` trained on raw
    k-space at ``|k|max ~ 2400`` and its k-space renders were ``expm1``-ed on
    the strength of the declaration, clamping the whole contrast-carrying band
    to ``DECOMPRESS_MAGNITUDE_CEILING`` and drawing a phase-only edge map.

    Every test here feeds the REAL ``DataProcessingConfigSchema``: a fixture
    that hand-writes the knobs would keep passing through the next rename, which
    is exactly how the flat reader survived a whole schema migration. The
    pre-existing suites above feed flat dicts, which is why they stayed green.
    """

    #: The exp_11 kspace_filling attention-shootout declaration.
    ARM: ClassVar[dict[str, object]] = {
        "enable_kspace_normalization": True,
        "enable_log_scaling": True,
        "kspace_percentile": 0.95,
        "kspace_scale_domain": "image",
    }

    @staticmethod
    def _data(**overrides):
        """A ``data``-shaped holder carrying the real processing schema block."""
        from types import SimpleNamespace

        from mriforge.config.schemas.data import DataProcessingConfigSchema

        return SimpleNamespace(
            processing=DataProcessingConfigSchema(
                **{**TestKSpaceNormalizationSpecReadsTheNestedBlock.ARM, **overrides}
            )
        )

    def test_reads_the_declared_nested_values(self):
        """The regression: this resolved to every default before the fix."""
        from mriforge.data.transforms.normalization import KSpaceNormalizationSpec

        spec = KSpaceNormalizationSpec.from_data_config(self._data())

        assert spec.enabled is True  # was False -> normalize() was a no-op
        assert spec.log_scaling is True  # was False -> no log1p compression
        assert spec.percentile == 0.95  # was the 0.99 default
        assert spec.scale_domain == "image"  # was the 'kspace' default

    def test_center_fraction_comes_from_the_block(self):
        """Parity with the transform, which is handed the same 0.25 default."""
        from mriforge.data.transforms.normalization import KSpaceNormalizationSpec

        assert KSpaceNormalizationSpec.from_data_config(self._data()).center_fraction == 0.25
        spec = KSpaceNormalizationSpec.from_data_config(
            self._data(log_scaling_center_fraction=0.1)
        )
        assert spec.center_fraction == 0.1

    def test_nested_block_wins_over_stale_flat_names(self):
        """A leftover flat key must not override the block that supersedes it."""
        from types import SimpleNamespace

        from mriforge.config.schemas.data import DataProcessingConfigSchema
        from mriforge.data.transforms.normalization import KSpaceNormalizationSpec

        data = SimpleNamespace(
            processing=DataProcessingConfigSchema(**self.ARM),
            normalize_kspace=False,
            log_scaling=False,
            kspace_percentile=0.5,
        )
        spec = KSpaceNormalizationSpec.from_data_config(data)

        assert (spec.enabled, spec.log_scaling, spec.percentile) == (True, True, 0.95)

    def test_absent_block_and_absent_knobs_resolve_to_defaults(self):
        """Nothing declared anywhere: a standalone strategy, a dataless paradigm."""
        from types import SimpleNamespace

        from mriforge.data.transforms.normalization import KSpaceNormalizationSpec

        for empty in ({}, SimpleNamespace()):
            spec = KSpaceNormalizationSpec.from_data_config(empty)
            assert spec == KSpaceNormalizationSpec()
            assert spec.enabled is False

    def test_block_that_omits_a_knob_raises(self):
        """Absent-block and absent-field are different facts (CLAUDE.md #3b).

        Defaulting here is how a rename disables the mechanism in silence: the
        reader keeps the old spelling, every read misses, and nothing goes red
        because "absent" and "off" are the same boolean.
        """
        from types import SimpleNamespace

        from mriforge.data.transforms.normalization import KSpaceNormalizationSpec

        partial = SimpleNamespace(
            enable_kspace_normalization=True,
            kspace_percentile=0.95,
            kspace_scale_domain="image",
            log_scaling_center_fraction=0.25,
        )  # `enable_log_scaling` renamed away
        with pytest.raises(AttributeError, match="enable_log_scaling"):
            KSpaceNormalizationSpec.from_data_config(SimpleNamespace(processing=partial))

        with pytest.raises(AttributeError, match="enable_log_scaling"):
            KSpaceNormalizationSpec.from_data_config(
                {"processing": {k: v for k, v in self.ARM.items() if k != "enable_log_scaling"}
                 | {"log_scaling_center_fraction": 0.25}}
            )

    def test_explicit_none_raises_for_a_non_nullable_knob(self):
        """An explicit None must not read as the schema default."""
        from mriforge.data.transforms.normalization import KSpaceNormalizationSpec

        with pytest.raises(ValueError, match="kspace_percentile"):
            KSpaceNormalizationSpec.from_data_config(
                {"processing": dict(self.ARM, kspace_percentile=None,
                                    log_scaling_center_fraction=0.25)}
            )

    def test_nullable_center_fraction_may_be_none(self):
        """The one knob with a real null meaning: 'use the full magnitude'."""
        from mriforge.data.transforms.normalization import KSpaceNormalizationSpec

        spec = KSpaceNormalizationSpec.from_data_config(
            {"processing": dict(self.ARM, log_scaling_center_fraction=None)}
        )
        assert spec.center_fraction is None

    def test_legacy_flat_dict_still_resolves_and_says_so(self, caplog):
        """A checkpoint's stored `data` dict predates the block; honour it, loudly."""
        import logging

        from mriforge.data.transforms.normalization import KSpaceNormalizationSpec

        flat = {
            "normalize_kspace": True,
            "log_scaling": True,
            "kspace_percentile": 0.95,
            "kspace_scale_domain": "image",
        }
        with caplog.at_level(logging.WARNING):
            spec = KSpaceNormalizationSpec.from_data_config(flat)

        assert (spec.enabled, spec.log_scaling, spec.percentile) == (True, True, 0.95)
        assert spec.scale_domain == "image"
        assert "pre-decomposition FLAT names" in caplog.text

    def test_nested_resolution_matches_what_the_transform_is_handed(self):
        """The fallback's stated job: reproduce the transform's normalization.

        ``TorchIOTransformBuilder`` reads the same block and hands those values
        to ``KSpaceNormalizationTransform``. If the two resolvers disagree, the
        fallback silently trains on a different distribution than the chain.
        """
        from mriforge.data.transforms.normalization import (
            KSpaceNormalizationSpec,
            KSpaceNormalizationTransform,
        )

        block = self._data().processing
        transform = KSpaceNormalizationTransform(
            percentile=block.kspace_percentile,
            log_scaling=block.enable_log_scaling,
            center_fraction=block.log_scaling_center_fraction,
            scale_domain=block.kspace_scale_domain,
        )
        resolved = KSpaceNormalizationSpec.from_data_config(self._data())

        for field in ("percentile", "log_scaling", "center_fraction", "scale_domain"):
            assert getattr(resolved, field) == getattr(transform.spec, field), field


class TestImageNormalizationSpecResolution:
    """One resolver decides WHAT normalizes image intensity, and records WHY."""

    @staticmethod
    def _spec(nt, dt=None, **kw):
        from mriforge.data.transforms.normalization import ImageNormalizationSpec

        return ImageNormalizationSpec.from_declared(nt, dt, kw or None)

    def test_declared_type_wins(self) -> None:
        from mriforge.data.transforms.normalization import NormalizationStrategy

        spec = self._spec("percentile", "nifti_paired")
        assert spec.source == "declared"
        assert spec.config.strategy is NormalizationStrategy.PERCENTILE

    def test_contrast_aware_with_no_declaration_inherits_the_dataset_default(
        self,
    ) -> None:
        """66 corpus arms declare NO ``normalization_type`` and relied entirely
        on the dataset's internal pass. Resolving them to "none" would have
        trained those arms on unnormalized intensities — the single most
        dangerous way this port could have gone wrong."""
        from mriforge.data.transforms.normalization import NormalizationStrategy

        for dtype in ("contrast_aware_paired", "nifti_paired"):
            spec = self._spec("none", dtype)
            assert spec.source == "contrast_aware_legacy_default"
            assert spec.config.strategy is NormalizationStrategy.PERCENTILE
            assert spec.enabled

    def test_other_dataset_types_with_none_really_get_none(self) -> None:
        """The inheritance is scoped. A ``nifti`` arm never had a dataset pass,
        so "none" must keep meaning none rather than silently gaining one."""
        spec = self._spec("none", "nifti")
        assert spec.source == "disabled"
        assert not spec.enabled

    def test_config_vocabulary_aliases_resolve(self) -> None:
        """The schema Literal and the strategy enum were named independently:
        the config says ``robust_percentile`` / ``standard`` where the enum says
        PERCENTILE / ZSCORE. 270 corpus arms use ``robust_percentile``."""
        from mriforge.data.transforms.normalization import NormalizationStrategy

        assert (
            self._spec("robust_percentile", "nifti").config.strategy
            is NormalizationStrategy.PERCENTILE
        )
        assert (
            self._spec("standard", "nifti").config.strategy
            is NormalizationStrategy.ZSCORE
        )

    def test_unknown_type_raises_rather_than_degrading(self) -> None:
        # The resolver reports in CONFIG vocabulary ("normalization_type"), not
        # enum vocabulary -- `NormalizationStrategy.from_string` is the one that
        # says "Unknown normalization strategy" (tested separately above). This
        # regex tracked the enum wording and had been red on dev since the two
        # messages diverged.
        with pytest.raises(ValueError, match="Unknown normalization_type"):
            self._spec("robust_percentil", "nifti")  # a typo, not a member

    def test_percentile_accepts_the_0_100_spelling(self) -> None:
        spec = self._spec("percentile", "nifti", percentile=99.5)
        assert spec.config.percentile == pytest.approx(0.995)


class TestTheLegacyDefaultIsTheDATASETsDEFAULT:
    """Port fidelity, asserted against the thing it was ported FROM.

    The k-space port (#571) established the trap: the dataset scale and the
    transform scale were NOT equivalent, so deleting either side silently
    changed the physics. Pinning the ported numbers as literals would only
    prove they match the literals. This derives the expected config from
    ``ContrastConfig``'s own defaults — the values the dataset actually fed the
    SSOT — so a drift in either place fails here.
    """

    def test_it_matches_what_the_subject_builder_used_to_build(self) -> None:
        from mriforge.data.datasets.contrast_aware import ContrastConfig
        from mriforge.data.transforms.normalization import (
            _CONTRAST_AWARE_LEGACY_DEFAULT,
            NormalizationConfig,
            NormalizationStrategy,
        )

        cc = ContrastConfig(name="T1")  # the defaults every arm silently got
        percentile = cc.percentile / 100.0 if cc.percentile > 1.0 else cc.percentile
        expected = NormalizationConfig(
            strategy=NormalizationStrategy.from_string(cc.normalization),
            percentile=percentile,
            out_range=cc.out_range,
            clamp=cc.clamp,
            min_scale=0.05,  # the noise floor the builder hardcoded
        )
        assert expected == _CONTRAST_AWARE_LEGACY_DEFAULT

    def test_the_noise_floor_survived_the_port(self) -> None:
        """``min_scale`` is the one parameter torchio's RescaleIntensity had no
        equivalent for. Losing it means amplifying background noise on a
        low-signal ULF volume."""
        from mriforge.data.transforms.normalization import (
            _CONTRAST_AWARE_LEGACY_DEFAULT,
        )

        assert _CONTRAST_AWARE_LEGACY_DEFAULT.min_scale == 0.05


class TestImageNormalizationTransformAppliesOnce:
    """The served tensor must equal ONE application of the resolved config."""

    @staticmethod
    def _subject(data):
        import torchio as tio

        return tio.Subject(
            input=tio.ScalarImage(tensor=data.clone()),
            target=tio.ScalarImage(tensor=data.clone()),
        )

    def test_output_equals_a_single_ssot_application(self) -> None:
        import torch

        from mriforge.data.transforms.normalization import (
            ImageNormalizationSpec,
            ImageNormalizationTransform,
            normalize_tensor,
        )

        torch.manual_seed(0)
        data = torch.rand(1, 8, 8, 1) * 100.0
        spec = ImageNormalizationSpec.from_declared("percentile", "nifti_paired")
        out = ImageNormalizationTransform(spec)(self._subject(data))

        expected = normalize_tensor(data, spec.config)
        assert torch.allclose(out["input"].data, expected, atol=1e-6)
        assert torch.allclose(out["target"].data, expected, atol=1e-6)

    def test_a_second_pass_is_detectable(self) -> None:
        """The guard behind the guard: if applying twice were indistinguishable
        from once, the assertion above could not catch a re-introduced second
        normalizer. Percentile+clamp is NOT idempotent here, so it can."""
        import torch

        from mriforge.data.transforms.normalization import (
            ImageNormalizationSpec,
            ImageNormalizationTransform,
        )

        torch.manual_seed(0)
        data = torch.rand(1, 8, 8, 1) * 100.0
        spec = ImageNormalizationSpec.from_declared(
            "percentile", "nifti", {"percentile": 0.5}
        )
        tf = ImageNormalizationTransform(spec)
        once = tf(self._subject(data))["input"].data
        twice = tf(self._subject(once))["input"].data
        assert not torch.allclose(once, twice, atol=1e-6)

    def test_kspace_keys_are_left_alone(self) -> None:
        """Image normalizers clamp and shift, which destroys complex k-space."""
        import torch
        import torchio as tio

        from mriforge.data.transforms.normalization import (
            ImageNormalizationSpec,
            ImageNormalizationTransform,
        )

        data = torch.randn(2, 8, 8, 1) * 50.0
        subject = tio.Subject(kspace=tio.ScalarImage(tensor=data.clone()))
        spec = ImageNormalizationSpec.from_declared("percentile", "nifti")
        out = ImageNormalizationTransform(spec)(subject)
        assert torch.equal(out["kspace"].data, data)

    def test_the_source_is_stamped_onto_the_subject(self) -> None:
        """Which rule chose the numbers is what a reader needs when intensities
        move between two arms that look identically configured."""
        import torch

        from mriforge.data.transforms.normalization import (
            ImageNormalizationSpec,
            ImageNormalizationTransform,
        )

        spec = ImageNormalizationSpec.from_declared("none", "nifti_paired")
        out = ImageNormalizationTransform(spec)(self._subject(torch.rand(1, 4, 4, 1)))
        assert out["image_normalization_source"] == "contrast_aware_legacy_default"


class TestPercentileDividesRatherThanClips:
    """Plan item B5, fixed as a side effect of routing through the SSOT.

    ``tio.RescaleIntensity(percentiles=(0, p))`` CLIPS at the percentile before
    rescaling, while the builder's own comment described the operation as
    "divide by the 99th". With ``clamp=False`` the SSOT divides, so values above
    the percentile stay above 1 instead of being flattened onto it.
    """

    def test_values_above_the_percentile_are_not_flattened(self) -> None:
        import torch

        from mriforge.data.transforms.normalization import (
            ImageNormalizationSpec,
            ImageNormalizationTransform,
        )

        data = torch.cat([torch.ones(1, 1, 99, 1), torch.full((1, 1, 1, 1), 50.0)], 2)
        spec = ImageNormalizationSpec.from_declared(
            "percentile", "nifti", {"percentile": 0.9, "clamp": False}
        )
        out = ImageNormalizationTransform(spec)(
            TestImageNormalizationTransformAppliesOnce._subject(data)
        )
        assert out["input"].data.max() > 1.0, (
            "the bright voxel was clipped to the percentile — that is the "
            "RescaleIntensity behaviour this port replaced"
        )


class TestUnknownTypeRaisesInTheConfigVocabulary:
    """The unknown-``normalization_type`` raise moved here, and had to be re-worded.

    Two PRs collided on this. One made an unimplemented ``normalization_type``
    raise instead of warning-and-appending-nothing (#9), with a message quoting
    the CONFIG spellings. The other replaced both hand-written dispatch chains
    in the builder with this one resolver, which routed the raise through
    ``NormalizationStrategy.from_string`` -- whose message quotes the ENUM
    members. That regression is invisible to either PR alone: ``standard`` is
    the config spelling and ``zscore`` is the enum spelling, so the error told
    the author to write a value the schema Literal rejects.
    """

    @staticmethod
    def _resolve(value):
        from mriforge.data.transforms.normalization import ImageNormalizationSpec

        return ImageNormalizationSpec.from_declared(value, None, None)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown normalization_type"):
            self._resolve("scalar")

    def test_message_names_the_offending_value(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            self._resolve("scalar")
        assert "'scalar'" in str(excinfo.value)

    def test_message_quotes_the_config_spellings_not_the_enum_members(self) -> None:
        """``standard`` must appear; ``zscore`` is not writable in YAML."""
        from mriforge.data.transforms.normalization import (
            IMPLEMENTED_NORMALIZATION_TYPES,
        )

        with pytest.raises(ValueError) as excinfo:
            self._resolve("scalar")
        message = str(excinfo.value)
        for valid in IMPLEMENTED_NORMALIZATION_TYPES:
            assert repr(valid) in message, f"{valid!r} missing from the error text"
        assert "standard" in message

    def test_the_original_enum_error_is_chained(self) -> None:
        """Re-worded, not swallowed -- the root cause stays reachable."""
        with pytest.raises(ValueError) as excinfo:
            self._resolve("scalar")
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert "normalization strategy" in str(excinfo.value.__cause__)

    def test_the_builder_re_exports_the_same_object(self) -> None:
        """One definition. A copy in the builder is how the two sets drift."""
        from mriforge.data.builders import torchio_transform_builder
        from mriforge.data.transforms import normalization

        assert (
            torchio_transform_builder.IMPLEMENTED_NORMALIZATION_TYPES
            is normalization.IMPLEMENTED_NORMALIZATION_TYPES
        )

    def test_robust_percentile_is_not_in_the_set_but_still_resolves(self) -> None:
        """The one licensed gap: folded upstream, accepted here, not advertised."""
        from mriforge.data.transforms.normalization import (
            IMPLEMENTED_NORMALIZATION_TYPES,
            NormalizationStrategy,
        )

        assert "robust_percentile" not in IMPLEMENTED_NORMALIZATION_TYPES
        spec = self._resolve("robust_percentile")
        assert spec.config.strategy is NormalizationStrategy.PERCENTILE


class TestKSpaceNormalizationSurvivesPatchExtraction:
    """The transform's output must reach the batch, not just the Subject (#1213).

    ``tio.Subject`` mirrors its entries into ``self.__dict__`` and
    ``Subject.__setitem__`` is not defined, so the ``kspace_scale`` /
    ``kspace_log_scaled`` / ``kspace_normalized`` keys this transform publishes
    never reached ``__dict__``. ``tio.Crop`` — the engine behind every
    ``PatchSampler``, hence every ``tio.Queue`` — builds its output *solely* from
    ``__dict__``, so those keys were **dropped** at patch extraction.

    The consequence was not cosmetic. With no ``kspace_scale`` on the batch,
    ``DiffusionTrainingStrategy._batch_is_already_normalized`` took its
    ``kspace_scale is None`` route (the silent one, #1211), concluded the batch
    had never been normalized, and re-normalized with a scale of its own —
    discarding the declared percentile and domain while reporting success.
    """

    @staticmethod
    def _subject() -> tio.Subject:
        """A DC-heavy k-space Subject shaped like the M4Raw loader's output."""
        torch.manual_seed(0)
        k = torch.randn(2, 16, 16, 4) * 0.1
        k[:, 8, 8, :] = 2500.0  # DC blob: the ~200x range the transform tames
        return tio.Subject(input=tio.ScalarImage(tensor=k))

    def _normalized(self) -> tio.Subject:
        from mriforge.data.transforms.normalization import KSpaceNormalizationTransform

        return KSpaceNormalizationTransform(percentile=0.95, log_scaling=True)(self._subject())

    def test_the_markers_reach_the_patch(self) -> None:
        """A cropped subject still declares that it was normalized, and by how much."""
        normalized = self._normalized()
        assert normalized["kspace_normalized"] is True  # sanity: it ran

        patch = tio.Crop((0, 0, 0, 0, 0, 1))(normalized)

        for key in ("kspace_scale", "kspace_log_scaled", "kspace_normalized"):
            assert key in patch, (
                f"{key!r} dropped at patch extraction — the batch cannot tell the "
                "strategy it was already normalized (#1211/#1213)"
            )
        assert bool(patch["kspace_normalized"]) is True
        assert float(patch["kspace_scale"]) == pytest.approx(float(normalized["kspace_scale"]))

    def test_the_patch_carries_compressed_magnitudes_after_a_replacing_transform(
        self,
    ) -> None:
        """The patch's |k| is the transform's output, not the raw DC blob.

        Composed in the PRODUCTION order — ``EnsureSpatialConsistency`` first.
        That ordering is what made the image half of the defect reachable: this
        transform edits in place via ``Image.set_data``, which is safe on its own,
        but the chain's first member *replaces* every image object, so the tensor
        being edited was no longer the one ``__dict__`` (and therefore the crop)
        would read. A single-transform test cannot see this.

        A ``log1p``-compressed float32 magnitude cannot exceed ~44, so a patch
        anywhere near the input's 2500 proves the compression was discarded.
        """
        from mriforge.data.transforms.geometric import EnsureSpatialConsistency
        from mriforge.data.transforms.normalization import KSpaceNormalizationTransform

        chain = tio.Compose(
            [
                EnsureSpatialConsistency(),
                KSpaceNormalizationTransform(percentile=0.95, log_scaling=True),
            ]
        )
        normalized = chain(self._subject())
        patch = tio.Crop((0, 0, 0, 0, 0, 1))(normalized)

        assert float(patch["input"].data.abs().max()) < 44.0
        assert float(patch["input"].data.abs().max()) == pytest.approx(
            float(normalized["input"].data.abs().max())
        )
