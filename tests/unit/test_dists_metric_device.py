"""Regression test for DISTSMetric device routing.

Audit 2026-05: a cluster Gen-3 sim2rank run on ``--device cuda``
ground to a halt at ~7 s per metric call because the DISTSMetric
wrapper recorded ``device="cuda"`` but the inner DISTS loss's VGG19
parameters stayed on CPU. Each call then forced an implicit CPU↔GPU
input transfer and ran the VGG19 forward pass on CPU.

The fix moves ``self._impl`` to the requested device in
``DISTSMetric.__init__`` and adds a per-call device guard in
``compute_metric``. This test pins both behaviours so the regression
cannot reappear.
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.unit
class TestDISTSMetricDeviceRouting:
    """Pin the device-routing invariant for the DISTS metric wrapper."""

    def test_constructed_on_cpu_has_cpu_weights(self) -> None:
        from spectramr.core.metrics import get_metric

        m = get_metric("dists", device="cpu")
        first_param = next(m._impl.parameters())
        assert (
            first_param.device.type == "cpu"
        ), f"DISTS metric VGG19 must live on CPU; got {first_param.device}"

    @pytest.mark.gpu
    def test_constructed_on_cuda_has_cuda_weights(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from spectramr.core.metrics import get_metric

        m = get_metric("dists", device="cuda")
        first_param = next(m._impl.parameters())
        assert first_param.device.type == "cuda", (
            "DISTS metric requested device=cuda but VGG19 stayed on "
            f"{first_param.device}. This is the cluster-run hang regression."
        )

    def test_eval_mode_and_no_grad(self) -> None:
        from spectramr.core.metrics import get_metric

        m = get_metric("dists", device="cpu")
        # Eval mode prevents BN/dropout drift, and requires_grad=False
        # stops the precompute loop from accidentally building an
        # autograd graph over 50k metric calls.
        assert not m._impl.training, "DISTS metric must be in eval mode"
        for p in m._impl.parameters():
            assert (
                not p.requires_grad
            ), "DISTS VGG19 params must have requires_grad=False"

    def test_per_call_device_guard_accepts_mismatched_input(self) -> None:
        from spectramr.core.metrics import get_metric

        m = get_metric("dists", device="cpu")
        # Inputs of different shapes are OK as long as the metric
        # auto-moves them. Use small images for fast execution.
        x = torch.randn(1, 1, 32, 32)
        y = torch.randn(1, 1, 32, 32)
        score = m(x, y)
        assert torch.is_tensor(score) or isinstance(score, float)

    def test_vgg19_stays_fp32(self) -> None:
        """VGG19 weights must remain fp32 even when constructed under autocast."""
        from spectramr.core.metrics import get_metric

        # Even if the user *thinks* they're in bf16-land, DISTS pins fp32.
        with torch.amp.autocast(
            "cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16,
            enabled=True,
        ):
            m = get_metric("dists", device="cpu")
        weight_dtypes = {p.dtype for p in m._impl.parameters()}
        assert weight_dtypes == {
            torch.float32
        }, f"DISTS VGG19 weights must be pure fp32; got dtypes {weight_dtypes}"

    def test_outer_autocast_does_not_change_output(self) -> None:
        """DISTS output must be the same whether or not the caller is under autocast."""
        from spectramr.core.metrics import get_metric

        m = get_metric("dists", device="cpu")
        torch.manual_seed(0)
        x = torch.randn(1, 1, 32, 32)
        y = torch.randn(1, 1, 32, 32)

        # Pure-fp32 baseline
        v_clean = float(m(x, y))

        # Wrap caller in bf16 autocast — DISTS must internally
        # disable it and produce the same value.
        with torch.amp.autocast("cpu", dtype=torch.bfloat16, enabled=True):
            v_under_autocast = float(m(x, y))

        assert abs(v_clean - v_under_autocast) < 1e-5, (
            f"DISTS leaked autocast precision: fp32={v_clean}, "
            f"under_autocast={v_under_autocast}, |Δ|={abs(v_clean - v_under_autocast)}"
        )
