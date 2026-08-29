"""Unit tests for the compute-device SSOT (:mod:`mriforge.core.compute_device`).

Pins the accelerated-run contract: a *heavy* pipeline (train / infer /
validate / hpo / ablation / probe / ...) must never silently degrade to CPU.
Either it gets an accelerator, or it raises — unless the user explicitly
dictated CPU.

The policy is a pure function of ``(requested, pipeline, cuda_available)``,
so every branch is exercised here without a GPU.
"""

from __future__ import annotations

import dataclasses

import pytest

from mriforge.core.compute_device import (
    HEAVY_PIPELINES,
    AcceleratorRequiredError,
    DeviceDecision,
    cpu_opt_in_from_env,
    resolve_compute_device,
)

# ── auto: the default path ────────────────────────────────────────────


class TestAutoResolution:
    def test_auto_picks_cuda_when_available(self) -> None:
        d = resolve_compute_device(
            "auto", pipeline="train", cuda_available=True, environ={}
        )
        assert d.device == "cuda"
        assert d.accelerated is True
        assert d.cpu_opt_in is False

    def test_auto_without_gpu_raises_for_heavy_pipeline(self) -> None:
        """The load-bearing case: an sbatch that lands on a GPU-less node.

        Pre-fix this returned ``cpu`` and burned the whole wall-clock
        allocation at ~100x slowdown while reporting success.
        """
        with pytest.raises(AcceleratorRequiredError, match="no accelerator"):
            resolve_compute_device(
                "auto", pipeline="train", cuda_available=False, environ={}
            )

    def test_auto_without_gpu_allows_light_pipeline(self) -> None:
        """A non-heavy caller (e.g. a config lint) may run on CPU."""
        d = resolve_compute_device(
            "auto", pipeline="lint", cuda_available=False, environ={}
        )
        assert d.device == "cpu"
        assert d.accelerated is False

    def test_none_and_empty_are_treated_as_auto(self) -> None:
        for spec in (None, "", "  "):
            with pytest.raises(AcceleratorRequiredError):
                resolve_compute_device(
                    spec, pipeline="train", cuda_available=False, environ={}
                )

    def test_auto_prefers_cuda_over_mps(self) -> None:
        d = resolve_compute_device(
            "auto",
            pipeline="train",
            cuda_available=True,
            mps_available=True,
            environ={},
        )
        assert d.device == "cuda"

    def test_auto_falls_to_mps_when_no_cuda(self) -> None:
        d = resolve_compute_device(
            "auto",
            pipeline="train",
            cuda_available=False,
            mps_available=True,
            environ={},
        )
        assert d.device == "mps"
        assert d.accelerated is True


# ── explicit cuda: never silently downgraded ──────────────────────────


class TestExplicitCuda:
    def test_cuda_without_gpu_raises(self) -> None:
        """An explicit ``cuda`` request on a CPU-only host is a broken
        environment, never a preference — it raises even with FORCE_CPU."""
        with pytest.raises(AcceleratorRequiredError, match="CUDA is not available"):
            resolve_compute_device(
                "cuda", pipeline="train", cuda_available=False, environ={}
            )

    def test_cuda_request_not_relaxed_by_force_cpu_env(self) -> None:
        with pytest.raises(AcceleratorRequiredError):
            resolve_compute_device(
                "cuda",
                pipeline="train",
                cuda_available=False,
                environ={"FORCE_CPU": "true"},
            )

    def test_the_message_does_not_recommend_a_remedy_that_cannot_work(self) -> None:
        """The advice must match the branch it is raised from.

        The message used to end *"run on CPU deliberately with --device cpu /
        FORCE_CPU=true"* — six lines below a comment reading *"Deliberate: NOT
        relaxed by FORCE_CPU"*, and directly contradicted by
        ``test_cuda_request_not_relaxed_by_force_cpu_env`` above. Three of
        cluster job 8004252's failures landed on this branch, so the first thing
        a reader would have tried is the one thing that cannot help.
        """
        with pytest.raises(AcceleratorRequiredError) as exc:
            resolve_compute_device(
                "cuda", pipeline="train", cuda_available=False, environ={}
            )
        msg = str(exc.value)
        assert "--device cpu" in msg, "the remedy that DOES work must be named"
        assert "run.device: cpu" in msg
        assert "FORCE_CPU does NOT apply" in msg

    def test_indexed_cuda_is_preserved(self) -> None:
        d = resolve_compute_device(
            "cuda:3", pipeline="train", cuda_available=True, environ={}
        )
        assert d.device == "cuda:3"
        assert d.accelerated is True

    def test_malformed_cuda_index_raises(self) -> None:
        with pytest.raises(ValueError, match="Malformed"):
            resolve_compute_device(
                "cuda:x", pipeline="train", cuda_available=True, environ={}
            )


# ── explicit cpu: the user-dictated escape hatch ──────────────────────


class TestExplicitCpuOptIn:
    def test_explicit_cpu_is_allowed_on_heavy_pipeline(self) -> None:
        d = resolve_compute_device(
            "cpu", pipeline="train", cuda_available=True, environ={}
        )
        assert d.device == "cpu"
        assert d.accelerated is False
        assert d.cpu_opt_in is True, "explicit cpu == the user dictated it"

    def test_force_cpu_env_relaxes_auto(self) -> None:
        d = resolve_compute_device(
            "auto",
            pipeline="train",
            cuda_available=False,
            environ={"FORCE_CPU": "1"},
        )
        assert d.device == "cpu"
        assert d.cpu_opt_in is True
        assert d.source == "env:FORCE_CPU"

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_force_cpu_truthy_spellings(self, value: str) -> None:
        assert cpu_opt_in_from_env({"FORCE_CPU": value}) is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_force_cpu_falsy_spellings(self, value: str) -> None:
        assert cpu_opt_in_from_env({"FORCE_CPU": value}) is False

    def test_force_cpu_garbage_raises(self) -> None:
        """Pitfall #15: an advertised knob validates and RAISES on an
        unknown value — it never degrades to a default."""
        with pytest.raises(ValueError, match="FORCE_CPU"):
            cpu_opt_in_from_env({"FORCE_CPU": "maybe"})


# ── unknown specs raise (pitfall #9: no silent fallback) ──────────────


class TestUnknownSpec:
    def test_unknown_device_raises(self) -> None:
        with pytest.raises(ValueError, match="unrecognized"):
            resolve_compute_device(
                "tpu", pipeline="train", cuda_available=True, environ={}
            )

    def test_mps_without_mps_raises(self) -> None:
        with pytest.raises(AcceleratorRequiredError, match="MPS"):
            resolve_compute_device(
                "mps", pipeline="train", cuda_available=False, environ={}
            )


# ── the heavy-pipeline registry ───────────────────────────────────────


class TestHeavyPipelineRegistry:
    @pytest.mark.parametrize(
        "pipeline",
        ["train", "infer", "validate", "hpo", "ablation", "probe", "predict"],
    )
    def test_user_named_pipelines_are_heavy(self, pipeline: str) -> None:
        """The pipelines the user called out must all be gated."""
        assert pipeline in HEAVY_PIPELINES

    def test_decision_is_frozen(self) -> None:
        d = resolve_compute_device(
            "cpu", pipeline="train", cuda_available=False, environ={}
        )
        assert isinstance(d, DeviceDecision)
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.device = "cuda"  # type: ignore[misc]


# ── provenance (pitfall #15c: stamp the resolved value) ───────────────


class TestProvenance:
    def test_source_is_recorded(self) -> None:
        d = resolve_compute_device(
            "cuda", pipeline="train", cuda_available=True, environ={}, source="cli"
        )
        assert d.source == "cli"

    def test_to_dict_is_json_serialisable(self) -> None:
        import json

        d = resolve_compute_device(
            "cuda", pipeline="train", cuda_available=True, environ={}
        )
        payload = json.loads(json.dumps(d.to_dict()))
        assert payload["device"] == "cuda"
        assert payload["accelerated"] is True
