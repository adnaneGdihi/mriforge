"""Performance benchmarks for FFT/IFFT operations.

Measures:
- Forward FFT performance
- Inverse FFT performance
- Centered vs uncentered FFT
- Comparison with raw torch.fft
"""

import time

import pytest
import torch

from mriforge.infrastructure.physics.fft_ops import FFTTransformer


@pytest.fixture
def fft_transformer():
    """Create FFT transformer instance."""
    return FFTTransformer()


class TestFFTPerformance:
    """Benchmark FFT operation performance."""

    @pytest.mark.parametrize("image_size", [128, 256, 512])
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_fft2_performance(self, fft_transformer, image_size, batch_size):
        """Benchmark 2D FFT performance."""
        # Create test image
        image = torch.randn(batch_size, 1, image_size, image_size)

        # Warm up
        for _ in range(5):
            _ = fft_transformer.fft2_uncentered(image)

        # Benchmark
        num_iterations = 100

        start_time = time.perf_counter()
        for _ in range(num_iterations):
            kspace = fft_transformer.fft2_uncentered(image)
        elapsed = time.perf_counter() - start_time

        avg_time = (elapsed / num_iterations) * 1000  # ms

        print("\nFFT2 Performance:")
        print(f"  Image size: {image_size}x{image_size}, Batch: {batch_size}")
        print(f"  Avg time:   {avg_time:.3f}ms")
        print(f"  Throughput: {num_iterations / elapsed:.1f} FFTs/sec")

        # Performance threshold (should be fast)
        assert avg_time < 100, f"FFT2 too slow: {avg_time:.3f}ms"

    @pytest.mark.parametrize("image_size", [128, 256, 512])
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_ifft2_performance(self, fft_transformer, image_size, batch_size):
        """Benchmark 2D IFFT performance."""
        # Create test k-space
        kspace = torch.randn(
            batch_size, 1, image_size, image_size, dtype=torch.complex64
        )

        # Warm up
        for _ in range(5):
            _ = fft_transformer.ifft2_uncentered(kspace)

        # Benchmark
        num_iterations = 100

        start_time = time.perf_counter()
        for _ in range(num_iterations):
            image = fft_transformer.ifft2_uncentered(kspace)
        elapsed = time.perf_counter() - start_time

        avg_time = (elapsed / num_iterations) * 1000  # ms

        print("\nIFFT2 Performance:")
        print(f"  Image size: {image_size}x{image_size}, Batch: {batch_size}")
        print(f"  Avg time:   {avg_time:.3f}ms")
        print(f"  Throughput: {num_iterations / elapsed:.1f} IFFTs/sec")

        assert avg_time < 100, f"IFFT2 too slow: {avg_time:.3f}ms"

    def test_fft_roundtrip_performance(self, fft_transformer):
        """Benchmark FFT-IFFT roundtrip."""
        image = torch.randn(4, 1, 256, 256)

        # Warm up
        for _ in range(5):
            kspace = fft_transformer.fft2_uncentered(image)
            _ = fft_transformer.ifft2_uncentered(kspace)

        # Benchmark
        num_iterations = 50

        start_time = time.perf_counter()
        for _ in range(num_iterations):
            kspace = fft_transformer.fft2_uncentered(image)
            recon = fft_transformer.ifft2_uncentered(kspace)
        elapsed = time.perf_counter() - start_time

        avg_time = (elapsed / num_iterations) * 1000  # ms

        print("\nFFT-IFFT Roundtrip Performance:")
        print(f"  Avg time:   {avg_time:.3f}ms")
        print(f"  Throughput: {num_iterations / elapsed:.1f} roundtrips/sec")

        assert avg_time < 200, f"Roundtrip too slow: {avg_time:.3f}ms"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fft_gpu_vs_cpu(self, fft_transformer):
        """Compare FFT performance on GPU vs CPU."""
        image_cpu = torch.randn(4, 1, 256, 256)
        image_gpu = image_cpu.to("cuda")

        # CPU benchmark
        start_time = time.perf_counter()
        for _ in range(50):
            _ = fft_transformer.fft2_uncentered(image_cpu)
        cpu_time = time.perf_counter() - start_time

        # GPU benchmark (with warmup)
        for _ in range(5):
            _ = fft_transformer.fft2_uncentered(image_gpu)
        torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(50):
            _ = fft_transformer.fft2_uncentered(image_gpu)
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        speedup = (cpu_time / gpu_time - 1) * 100

        print("\nGPU vs CPU FFT Performance:")
        print(f"  CPU time: {cpu_time * 1000:.1f}ms")
        print(f"  GPU time: {gpu_time * 1000:.1f}ms")
        print(f"  Speedup:  {speedup:+.1f}%")

        # GPU should be faster or similar
        assert gpu_time <= cpu_time, "GPU FFT should not be slower than CPU"


class TestFFTComparison:
    """Compare custom FFT with raw torch.fft."""

    def test_centered_vs_raw_fft(self):
        """Compare centered FFT with raw torch.fft."""
        fft_transformer = FFTTransformer()
        image = torch.randn(4, 1, 256, 256)

        # Warm up
        for _ in range(5):
            _ = fft_transformer.fft2_uncentered(image)
            _ = torch.fft.fft2(image, norm="ortho")

        # Benchmark custom centered FFT
        num_iterations = 100

        start_time = time.perf_counter()
        for _ in range(num_iterations):
            _ = fft_transformer.fft2_uncentered(image)
        custom_time = time.perf_counter() - start_time

        # Benchmark raw torch.fft
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            _ = torch.fft.fft2(image, norm="ortho")
        raw_time = time.perf_counter() - start_time

        overhead = (custom_time / raw_time - 1) * 100

        print("\nCentered FFT vs Raw FFT:")
        print(f"  Custom FFT: {custom_time * 1000:.1f}ms")
        print(f"  Raw FFT:    {raw_time * 1000:.1f}ms")
        print(f"  Overhead:   {overhead:+.1f}%")

        # Overhead should be reasonable (centering + normalization)
        assert overhead < 50, f"Custom FFT overhead too high: {overhead:.1f}%"


class Test3DFFTPerformance:
    """Benchmark 3D FFT operations."""

    def test_fft3_performance(self):
        """Benchmark 3D FFT performance."""
        fft_transformer = FFTTransformer()

        # 3D volume
        volume = torch.randn(2, 1, 64, 64, 64)

        # Warm up
        for _ in range(5):
            _ = fft_transformer.fft3(volume)

        # Benchmark
        num_iterations = 20

        start_time = time.perf_counter()
        for _ in range(num_iterations):
            kspace = fft_transformer.fft3(volume)
        elapsed = time.perf_counter() - start_time

        avg_time = (elapsed / num_iterations) * 1000  # ms

        print("\nFFT3 Performance (64x64x64 volume):")
        print(f"  Avg time:   {avg_time:.1f}ms")
        print(f"  Throughput: {num_iterations / elapsed:.1f} FFTs/sec")

        assert avg_time < 500, f"3D FFT too slow: {avg_time:.1f}ms"
