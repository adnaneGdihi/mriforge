"""Per-block forward-latency benchmark for the slow model families.

Companion to ``docs/perf_hotspot_fixes_attention_kan_mamba.rst`` (2026-07-01
review): times the blocks whose hot loops were fixed, so before/after wins
can be recorded and future regressions eyeballed. Run manually:

    source .venv/bin/activate && python tests/benchmarks/benchmark_attention_kan_mamba.py

Deliberately NOT a pytest module with timing asserts — wall-clock gates in CI
are flaky by construction; this is a measurement harness (same convention as
``hotspots_benchmark.py``).
"""

import os
import time

# The SE3 block imports MambaBlock, which hard-requires mamba_ssm unless the
# CPU wiring fallback is allowed; a timing harness is exactly that use case.
os.environ.setdefault("SPECTRAMR_ALLOW_MAMBA_FALLBACK", "1")
os.environ.setdefault("SPECTRAMR_SUPPRESS_CLINICAL_WARNING", "1")

import torch


def _time(fn, warmup: int = 3, iters: int = 10) -> float:
    """Median-ish wall time per call, CUDA-synchronized when available."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def bench_radial_band_attention(device: torch.device) -> None:
    from spectramr.models.blocks.dual_domain_attention_kan import RadialBandAttention

    block = RadialBandAttention(channels=8, num_heads=2, num_bands=8).to(device)
    x = torch.randn(1, 8, 64, 64, dtype=torch.complex64, device=device)
    with torch.no_grad():
        t = _time(lambda: block(x))
    print(f"  RadialBandAttention  B=1 C=8 64x64      : {t * 1e3:8.2f} ms/fwd")


def bench_kan_bases(device: torch.device) -> None:
    from spectramr.models.layers.kan.fourier_basis import FourierBasis
    from spectramr.models.layers.kan.wavelet_basis import WaveletBasis

    wb = WaveletBasis(num_bases=8, wavelet_family="haar").to(device)
    fb = FourierBasis(num_bases=8).to(device)
    x4 = torch.randn(2, 16, 64, 64, device=device)
    with torch.no_grad():
        tw = _time(lambda: wb(x4))
        tf = _time(lambda: fb(x4))
    print(f"  WaveletBasis         B=2 C=16 64x64 K=8 : {tw * 1e3:8.2f} ms/fwd")
    print(f"  FourierBasis         B=2 C=16 64x64 K=8 : {tf * 1e3:8.2f} ms/fwd")


def bench_diff_ssm(device: torch.device) -> None:
    from spectramr.models.mamba.diff_mamba import DiffSSM

    ssm = DiffSSM(d_model=32, d_state=16, dt_step=0.5).to(device)
    x = torch.randn(1, 256, 32, device=device)
    with torch.no_grad():
        t = _time(lambda: ssm(x), warmup=2, iters=5)
    print(f"  DiffSSM              L=256 D=32 dt=0.5  : {t * 1e3:8.2f} ms/fwd")


def bench_d2_mamba_indices(device: torch.device) -> None:
    from spectramr.models.mamba.d2_mamba import D2MambaBlock

    block = D2MambaBlock(d_model=8, d_state=4, d_conv=2).to(device)

    def fresh_build() -> None:
        block._scan_index_cache.clear()
        block._get_scan_indices("zorder", 128, 128, device)

    t_cold = _time(fresh_build, warmup=1, iters=3)
    block._scan_index_cache.clear()
    block._get_scan_indices("zorder", 128, 128, device)
    t_hot = _time(
        lambda: block._get_scan_indices("zorder", 128, 128, device), iters=100
    )
    print(f"  D2Mamba zorder build 128x128 cold        : {t_cold * 1e3:8.2f} ms")
    print(f"  D2Mamba zorder build 128x128 cached      : {t_hot * 1e6:8.2f} us")


def bench_bloch_mamba(device: torch.device) -> None:
    from spectramr.models.mamba.bloch_mamba import BlochMambaBlock

    block = BlochMambaBlock(d_model=32, d_state=16).to(device)
    x = torch.randn(1, 256, 32, device=device)
    with torch.no_grad():
        t = _time(lambda: block(x), warmup=2, iters=5)
    print(f"  BlochMambaBlock      L=256 D=32 bidir    : {t * 1e3:8.2f} ms/fwd")


def bench_se3_oscillator(device: torch.device) -> None:
    from spectramr.models.blocks.se3_lie_algebra_mamba import SE3LieAlgebraMambaBlock

    block = SE3LieAlgebraMambaBlock(d_state=8, group="SE3", sample_rate_hz=100.0)
    block = block.to(device).eval()
    norms = torch.rand(2, 512, 2, device=device) + 0.1
    with torch.no_grad():
        t = _time(lambda: block._oscillator_energy(norms), iters=20)
    print(f"  SE3 oscillator energy L=512              : {t * 1e3:8.2f} ms/call")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"benchmark_attention_kan_mamba on {device} (torch {torch.__version__})")
    bench_radial_band_attention(device)
    bench_kan_bases(device)
    bench_diff_ssm(device)
    bench_d2_mamba_indices(device)
    bench_bloch_mamba(device)
    bench_se3_oscillator(device)


if __name__ == "__main__":
    main()
