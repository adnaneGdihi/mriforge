"""Unit tests for the Tier-2 synthetic forward probe.

Targets :mod:`mriforge.infrastructure.validation.forward_probe` —
:class:`ProbeResult` plus :func:`synthetic_forward_probe`. The probe is
the only check that catches *runtime* model pathologies (shape, NaN,
gradient explosion, AMP unscale traps) at config-load time, so its
behavioural contract is load-bearing for the smoke / audit pipeline.

The probe is tolerant by design: missing torch, missing registry
entries, and structurally-broken configs all surface as structured
:class:`ProbeResult` records, never as raised exceptions that would
crash the audit CLI. These tests pin that contract.

When the source module is not yet present on the branch (e.g. a tree
branched off ``main`` before the Tier-2 work landed), the top-level
``pytest.importorskip`` skips the whole module rather than failing
collection.
"""

from __future__ import annotations

import json
import types
from typing import Any

import pytest

from tests.utils.data_config_stub import DataConfigStub

torch = pytest.importorskip("torch")
forward_probe = pytest.importorskip("mriforge.infrastructure.validation.forward_probe")

ProbeResult = forward_probe.ProbeResult
synthetic_forward_probe = forward_probe.synthetic_forward_probe


# ── ProbeResult JSON-serialisation contract ────────────────────────────


class TestProbeResultSerialisation:
    """The audit aggregator stores probe results as JSON — the dataclass
    must round-trip cleanly and expose every documented field."""

    def test_to_dict_contains_documented_keys(self) -> None:
        r = ProbeResult(passed=True, category="forward_pass_shape", message="ok")
        d = r.to_dict()
        for key in (
            "passed",
            "category",
            "message",
            "severity",
            "yaml_keys",
            "fix_hint",
            "device",
            "elapsed_seconds",
            "traceback",
        ):
            assert key in d, f"ProbeResult.to_dict missing key {key!r}"

    def test_to_json_round_trips(self) -> None:
        r = ProbeResult(
            passed=False,
            category="oom",
            message="cuda oom",
            yaml_keys=["data.batch_size"],
            fix_hint="lower batch size",
        )
        parsed = json.loads(r.to_json())
        assert parsed["passed"] is False
        assert parsed["category"] == "oom"
        assert parsed["yaml_keys"] == ["data.batch_size"]
        assert parsed["fix_hint"] == "lower batch size"

    def test_defaults_are_safe(self) -> None:
        r = ProbeResult(passed=True, category="forward_pass_shape", message="m")
        assert r.severity == "error"  # documented default
        assert r.yaml_keys == []
        assert r.fix_hint is None
        assert r.device == "cpu"
        assert r.elapsed_seconds == 0.0
        assert r.traceback is None


# ── Synthetic configs + fake model classes ─────────────────────────────


def _cfg(
    model_type: str = "test_probe_model",
    in_channels: int = 1,
    out_channels: int = 1,
    patch_size: list[int] | None = None,
    batch_size: int = 1,
    model_kwargs: dict | None = None,
    spatial_dims: int | None = None,
) -> Any:
    """Build a minimal duck-typed config that matches the probe's contract.

    The probe only reads ``config.model.*`` and ``config.data.*`` — a
    real :class:`TrainingSettings` is not required.
    """
    model_ns = types.SimpleNamespace(
        model_type=model_type,
        in_channels=in_channels,
        out_channels=out_channels,
        model_kwargs=model_kwargs or {},
    )
    if spatial_dims is not None:
        model_ns.spatial_dims = spatial_dims
    return types.SimpleNamespace(
        model=model_ns,
        # `data.sampling.patch_size` / `data.loader.batch_size` since the block
        # decomposition -- forward_probe walks the canonical paths, so a flat
        # stand-in raises `no attribute 'sampling'` before the probe ever runs.
        data=DataConfigStub(
            patch_size=patch_size or [16, 16],
            batch_size=batch_size,
        ),
    )


class _GoodModel(torch.nn.Module):
    """Shape-preserving 2D conv. Forward + backward should always succeed."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _ShapeBreakingModel(torch.nn.Module):
    """Strided conv → output spatial size doesn't match target."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _RaisingForwardModel(torch.nn.Module):
    """Forward unconditionally raises — exercises the runtime-error path."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "mat1 and mat2 shapes cannot be multiplied (65536x2 and 256x8)"
        )


class _IdentityModel(torch.nn.Module):
    """Returns input verbatim — identity-collapse pathology."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FieldConditionedModel(torch.nn.Module):
    """Mimics the MICCAI cross-field score nets (field_flow / cross_field /
    field_guided_diffusion): forward REQUIRES a continuous ``field_strength`` AND a
    source ``cond_image`` (``cond_image=None`` raises). The probe must synthesise
    both — otherwise it TypeErrors on field_strength or passes cond_image=None."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x, *, field_strength, cond_image=None, **_):
        if cond_image is None:
            raise ValueError("FieldConditionedModel requires cond_image (source).")
        scale = field_strength.reshape(-1, 1, 1, 1).float()
        return self.conv(x) * scale


class _IdentityModelOptedOut(torch.nn.Module):
    """Returns input verbatim BUT opts out of the identity-collapse check.

    Mirrors the real-world residual / consistency / normalising-flow case:
    output ≈ input by design at init, but training drives weights away from
    identity. The class-level attribute suppresses the probe's warning.
    """

    synthetic_forward_probe_skip = {"identity_collapse"}

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _NaNModel(torch.nn.Module):
    """Forward output is NaN-tainted — exercises the nan_in_forward branch."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * float("nan")


class _InfModel(torch.nn.Module):
    """Forward output is +inf-tainted — exercises the !isfinite branch."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * float("inf")


class _ExplodingModel(torch.nn.Module):
    """Huge initialisation → gradient norm > 1e6 on backward.

    A single Conv2d is NOT enough to cross the probe's 1e6 grad-norm threshold
    under an L1 loss: the L1 output-gradient caps at ``sign(y-t)/N``, so the
    chain rule needs extra amplification. Squaring the output adds a ``2·y``
    factor. This fixture previously scaled by 1e6 without the square, never
    exploded, and the resulting failure was papered over with an xfail blaming
    a possible threshold change — the detection path was fine all along (the
    sibling ``test_forward_probe.py::test_probe_flags_gradient_explosion`` has
    always passed with this shape).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        with torch.no_grad():
            self.conv.weight.mul_(1e8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return y * y


class _TupleReturnModel(torch.nn.Module):
    """Returns ``(principal_tensor, aux)`` — probe should peel index 0."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        y = self.conv(x)
        return y, torch.tensor(0.0)


class _DictReturnModel(torch.nn.Module):
    """Returns a dict — probe should take the first value."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        return {"image": self.conv(x), "aux": torch.tensor(0.0)}


class _NonTensorReturnModel(torch.nn.Module):
    """Returns a plain Python int — probe must classify as shape error."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return 42  # type: ignore[return-value]


class _SmapConditionedModel(torch.nn.Module):
    """Mirrors ``kspace_cold_diffusion``'s internal channel doubling.

    The strategy concatenates S-maps onto the input at runtime, so the
    backbone's first conv is built for ``in_channels * 2``. The probe
    must detect ``self.condition_with_smaps`` and synthesise the extra
    channels, otherwise the conv crashes with "expected N channels,
    got N/2".
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 4, **kwargs: Any):
        super().__init__()
        self.condition_with_smaps = True
        self.conv = torch.nn.Conv2d(in_channels * 2, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _InternalDCModel(torch.nn.Module):
    """Mirrors ``diff_varnet`` / ``diff_varnet_kan``: honours the arm's
    ``condition_with_smaps`` declaration and is STILL built at 1x.

    These backbones run their own data consistency, so the generator resolves
    the declaration to ``expects_smaps_concat = False`` and never concatenates.
    A probe that reads the *declaration* sizes a 2x input for them and crashes
    the first conv (or, before the width contract landed, silently squeezed it
    through an untrained ChannelAdapter). This is the divergence shape that
    ``_SmapConditionedModel`` -- where both flags agree -- cannot detect.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 4, **kwargs: Any):
        super().__init__()
        self.condition_with_smaps = True  # the arm's declaration
        self.expects_smaps_concat = False  # the resolved contract
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _ExplicitChannelsModel(torch.nn.Module):
    """Escape hatch for models with non-2× channel expansions.

    Declares ``synthetic_forward_probe_input_channels`` explicitly so
    the probe builds an input of that exact channel count, regardless
    of the YAML's ``in_channels`` value.
    """

    def __init__(self, in_channels: int = 2, out_channels: int = 2, **kwargs: Any):
        super().__init__()
        # Probe should send 5 channels — the model's first conv is built for that.
        self.synthetic_forward_probe_input_channels = 5
        self.conv = torch.nn.Conv2d(5, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@pytest.fixture
def _patch_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the model registry so the probe resolves our fakes.

    The probe imports ``get_model_class`` lazily inside its body
    (``from mriforge.models.registry import get_model_class``), so we patch
    the function on the registry module directly.
    """
    from mriforge.models import registry as registry_mod

    fake = {
        "good": _GoodModel,
        "shape_break": _ShapeBreakingModel,
        "raising": _RaisingForwardModel,
        "identity": _IdentityModel,
        "identity_opted_out": _IdentityModelOptedOut,
        "nan": _NaNModel,
        "inf": _InfModel,
        "exploding": _ExplodingModel,
        "tuple": _TupleReturnModel,
        "dict": _DictReturnModel,
        "non_tensor": _NonTensorReturnModel,
        "smap_conditioned": _SmapConditionedModel,
        "internal_dc": _InternalDCModel,
        "explicit_channels": _ExplicitChannelsModel,
        "field_conditioned": _FieldConditionedModel,
    }

    def _fake_get_model_class(name: str) -> Any:
        if name not in fake:
            raise KeyError(f"{name!r} not registered (test fake)")
        return fake[name]

    monkeypatch.setattr(registry_mod, "get_model_class", _fake_get_model_class)


# ── Smoke: probe runs end-to-end on a minimal model ────────────────────


class TestProbeSmoke:
    """The probe completes for a trivial model without crashing."""

    def test_passes_on_well_formed_model(self, _patch_registry: None) -> None:
        cfg = _cfg(model_type="good")
        result = synthetic_forward_probe(cfg, device="cpu", backward=True)
        assert result.passed, (
            f"Probe should pass on trivial Conv2d model; got "
            f"{result.category}: {result.message}"
        )
        assert result.category == "forward_pass_shape"
        assert result.elapsed_seconds >= 0.0
        assert result.severity == "info"

    def test_passes_without_backward(self, _patch_registry: None) -> None:
        cfg = _cfg(model_type="good")
        result = synthetic_forward_probe(cfg, device="cpu", backward=False)
        assert result.passed
        assert result.category == "forward_pass_shape"

    def test_passes_with_tuple_return(self, _patch_registry: None) -> None:
        """Models that return ``(tensor, aux)`` are common (GAN discriminators,
        VAE encoders) — the probe must peel the principal tensor."""
        cfg = _cfg(model_type="tuple")
        result = synthetic_forward_probe(cfg, device="cpu", backward=False)
        assert result.passed

    def test_passes_on_field_conditioned_model(self, _patch_registry: None) -> None:
        """The MICCAI cross-field score nets require a continuous field_strength AND
        a source cond_image at forward. The probe must synthesise both, instead of
        TypeError-ing on the required keyword-only field_strength or passing
        cond_image=None (which the model rejects). Covers field_flow / cross_field /
        field_guided_diffusion arms."""
        cfg = _cfg(model_type="field_conditioned")
        result = synthetic_forward_probe(cfg, device="cpu", backward=False)
        assert result.passed, f"{result.category}: {result.message}"
        assert result.category == "forward_pass_shape"
        assert result.category == "forward_pass_shape"

    def test_passes_with_dict_return(self, _patch_registry: None) -> None:
        cfg = _cfg(model_type="dict")
        result = synthetic_forward_probe(cfg, device="cpu", backward=False)
        assert result.passed
        assert result.category == "forward_pass_shape"


# ── Probe time budget ──────────────────────────────────────────────────


class TestProbeTimeBudget:
    """A tier-2 probe is documented as ~30 s. A 16x16 toy model on CPU
    must be MUCH faster — we lock in 5 s as the regression boundary."""

    def test_completes_in_under_five_seconds_on_toy_model(
        self, _patch_registry: None
    ) -> None:
        cfg = _cfg(model_type="good", patch_size=[16, 16], batch_size=1)
        result = synthetic_forward_probe(cfg, device="cpu", backward=True)
        assert result.passed
        assert result.elapsed_seconds < 5.0, (
            f"Probe budget regression: {result.elapsed_seconds:.2f}s "
            f"> 5.0s on a trivial Conv2d / 16x16 input."
        )


# ── Probe failure paths (structured outcomes, never exceptions) ───────


class TestProbeFailurePaths:
    """Each pathology surfaces as ``passed=False`` with a stable category."""

    def test_unregistered_model_fails_with_instantiation_category(
        self, _patch_registry: None
    ) -> None:
        cfg = _cfg(model_type="never_registered")
        result = synthetic_forward_probe(cfg, device="cpu")
        assert not result.passed
        assert result.category == "instantiation"
        assert "never_registered" in result.message
        assert result.fix_hint is not None  # actionable

    def test_shape_mismatch_fails_with_forward_pass_shape_category(
        self, _patch_registry: None
    ) -> None:
        """Strided model → output spatial size 8x8 vs. target 16x16."""
        cfg = _cfg(model_type="shape_break", patch_size=[16, 16])
        result = synthetic_forward_probe(cfg, device="cpu")
        assert not result.passed
        assert result.category == "forward_pass_shape"
        # The error must mention 'shape' so it's grep-able in audit logs.
        assert "shape" in result.message.lower()
        # And it must surface actionable yaml keys.
        assert (
            "data.patch_size" in result.yaml_keys
            or "model.out_channels" in result.yaml_keys
        )
        assert result.fix_hint is not None

    def test_runtime_error_classified_as_shape_error(
        self, _patch_registry: None
    ) -> None:
        """Generic forward RuntimeError → forward_pass_shape category."""
        cfg = _cfg(model_type="raising")
        result = synthetic_forward_probe(cfg, device="cpu")
        assert not result.passed
        assert result.category == "forward_pass_shape"
        # Underlying exception text propagates into the message.
        assert "mat1" in result.message

    def test_non_tensor_return_classified_as_shape_error(
        self, _patch_registry: None
    ) -> None:
        cfg = _cfg(model_type="non_tensor")
        result = synthetic_forward_probe(cfg, device="cpu", backward=False)
        assert not result.passed
        assert result.category == "forward_pass_shape"
        assert (
            "non-tensor" in result.message.lower() or "tensor" in result.message.lower()
        )

    def test_missing_model_section_fails_with_instantiation_category(self) -> None:
        bad = types.SimpleNamespace(
            model=types.SimpleNamespace(), data=types.SimpleNamespace()
        )
        result = synthetic_forward_probe(bad, device="cpu")
        assert not result.passed
        assert result.category == "instantiation"

    def test_model_type_none_is_rejected(self) -> None:
        bad = types.SimpleNamespace(
            model=types.SimpleNamespace(
                model_type=None,
                in_channels=1,
                out_channels=1,
                model_kwargs={},
            ),
            data=types.SimpleNamespace(patch_size=[16, 16], batch_size=1),
        )
        result = synthetic_forward_probe(bad, device="cpu")
        assert not result.passed
        assert result.category == "instantiation"
        assert "model_type" in result.message


# ── NaN / Inf detection in forward output ─────────────────────────────


class TestProbeNaNInfDetection:
    """A NaN/Inf in forward output silently poisons training — probe must catch."""

    def test_nan_output_is_caught(self, _patch_registry: None) -> None:
        cfg = _cfg(model_type="nan")
        result = synthetic_forward_probe(cfg, device="cpu", backward=False)
        assert not result.passed
        assert result.category == "nan_in_forward"
        assert result.severity == "error"
        # Surface actionable yaml keys.
        assert any("model" in k for k in result.yaml_keys)
        assert result.fix_hint is not None

    def test_inf_output_is_caught(self, _patch_registry: None) -> None:
        """``isfinite`` covers both NaN and ±Inf — Inf must fail the same way."""
        cfg = _cfg(model_type="inf")
        result = synthetic_forward_probe(cfg, device="cpu", backward=False)
        assert not result.passed
        assert result.category == "nan_in_forward"
        assert result.severity == "error"


# ── Identity collapse + gradient explosion (warnings, not errors) ─────


class TestProbeWarningChecks:
    """Identity collapse and gradient explosion are ``warning`` severities;
    the audit's strict-mode wrapper still rejects them."""

    def test_identity_collapse_flagged(self, _patch_registry: None) -> None:
        cfg = _cfg(model_type="identity")
        result = synthetic_forward_probe(cfg, device="cpu")
        assert not result.passed
        assert result.category == "identity_collapse"
        assert result.severity == "warning"
        assert "identity" in result.message.lower()
        assert result.fix_hint is not None

    def test_identity_collapse_opt_out_respects_class_attr(
        self,
        _patch_registry: None,
    ) -> None:
        """Models declaring ``synthetic_forward_probe_skip = {"identity_collapse"}``
        must NOT trip the identity-collapse warning, even with a literal
        identity forward.

        Regression for kspace_cold_diffusion and latent_flow on the
        2026-05-18 Tier-2 smoke run: both genuinely produce
        output ≈ input at init (cold-diffusion at t=0, normalising-flow
        at zero-init), and tripping the warning blocked their training
        unnecessarily.
        """
        cfg = _cfg(model_type="identity_opted_out")
        result = synthetic_forward_probe(cfg, device="cpu")
        # Probe should NOT flag identity_collapse for opted-out models.
        # Other categories (NaN, shape) might still fire — only assert
        # that this specific one is silent.
        assert result.category != "identity_collapse", (
            f"opt-out failed: probe still emitted identity_collapse "
            f"(message={result.message!r})"
        )

    def test_smap_conditioned_model_doubles_input_channels(
        self,
        _patch_registry: None,
    ) -> None:
        """Models with ``condition_with_smaps=True`` get a 2× input.

        Regression for the 2026-05-18 12:01 Tier-2 smoke run, where
        three ``kspace_cold_diffusion`` arms
        (``experiment_11_timestep_accelerated_cold_diffusion``,
        ``experiment_11a_swin_diff_rec_standardized``,
        ``experiment_11c_swin_diff_rec_kan``) failed with::

            weight of size [128, 16, 3, 3], expected input[2, 8, 258, 258]
            to have 16 channels, but got 8 channels instead

        The model's first conv was built for ``in_channels * 2``
        because the diffusion strategy concatenates S-maps at runtime;
        the probe didn't mirror that, so the conv crashed. With this
        fix the probe synthesises a tensor of the post-concat shape
        and the forward pass succeeds.
        """
        cfg = _cfg(model_type="smap_conditioned", in_channels=4, out_channels=4)
        result = synthetic_forward_probe(cfg, device="cpu", backward=False)
        assert result.passed, (
            f"smap-conditioned probe should succeed when the probe doubles "
            f"its input channels; got {result.category}: {result.message}"
        )

    def test_internal_dc_model_keeps_the_single_width(
        self,
        _patch_registry: None,
    ) -> None:
        """``condition_with_smaps`` is the declaration, not the width contract.

        The six ``diff_varnet``/``diff_varnet_kan`` arms declare
        ``condition_with_smaps: true`` and are built at 1x, because those
        backbones run their own data consistency. The probe used to read the
        declaration and hand them ``2 x in_channels``; ``audit --probe`` then
        measured a network the arm never trains. Sizing must come from
        :func:`model_expects_smaps_concat` -- the one resolver (CLAUDE.md #17).
        """
        cfg = _cfg(model_type="internal_dc", in_channels=4, out_channels=4)
        result = synthetic_forward_probe(cfg, device="cpu", backward=False)
        assert result.passed, (
            f"probe sized the wrong width for an internal-DC backbone; got "
            f"{result.category}: {result.message}"
        )

    def test_explicit_channels_attr_overrides_in_channels(
        self,
        _patch_registry: None,
    ) -> None:
        """Escape hatch for non-2× channel expansions.

        Models that declare ``synthetic_forward_probe_input_channels = N``
        force the probe to send an N-channel input regardless of the
        YAML's ``in_channels`` value. Useful for any future model whose
        runtime input is neither 1× nor 2× the YAML-declared count.
        """
        # YAML says in_channels=2, model declares it really wants 5.
        cfg = _cfg(model_type="explicit_channels", in_channels=2, out_channels=2)
        result = synthetic_forward_probe(cfg, device="cpu", backward=False)
        assert result.passed, (
            f"explicit-channels probe should succeed; got "
            f"{result.category}: {result.message}"
        )

    def test_gradient_explosion_flagged(self, _patch_registry: None) -> None:
        cfg = _cfg(model_type="exploding", patch_size=[16, 16])
        result = synthetic_forward_probe(cfg, device="cpu", backward=True)
        assert not result.passed
        assert result.category == "gradient_explosion"
        assert result.severity == "warning"
        assert any("gradient" in k or "learning_rate" in k for k in result.yaml_keys)


# ── Phantom-vs-noise input toggle ──────────────────────────────────────


class _CaptureInputModel(torch.nn.Module):
    """Stash the input tensor on the class so we can inspect it post-probe."""

    last_input: torch.Tensor | None = None

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Capture only the FIRST forward's input. The probe's input-invariance
        # check re-forwards with ``randn_like(x)``; overwriting on every call
        # would leave ``last_input`` holding that noise re-forward instead of
        # the phantom these tests assert on. (Tests reset ``last_input=None``
        # before each probe, so "first" is the real synthetic input.)
        if type(self).last_input is None:
            type(self).last_input = x.detach().clone()
        return self.conv(x)


@pytest.fixture
def _patch_registry_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    from mriforge.models import registry as registry_mod

    def _fake_get_model_class(name: str) -> Any:
        assert name == "capture"
        return _CaptureInputModel

    monkeypatch.setattr(registry_mod, "get_model_class", _fake_get_model_class)


class TestProbeInputSelection:
    """``use_phantom`` toggles between Shepp-Logan structure and white noise."""

    def test_phantom_input_by_default(self, _patch_registry_capture: None) -> None:
        _CaptureInputModel.last_input = None
        cfg = _cfg(model_type="capture", patch_size=[32, 32])
        result = synthetic_forward_probe(cfg, device="cpu", backward=False)
        assert result.passed
        x = _CaptureInputModel.last_input
        assert x is not None
        assert x.min().item() >= 0.0
        assert x.max().item() <= 1.0 + 1e-6
        # Phantom has ~10 unique levels at most.
        n_unique = torch.unique(x).numel()
        assert n_unique < 50, (
            f"Default probe input must be a Shepp-Logan phantom, got "
            f"{n_unique} unique values — looks like noise."
        )

    def test_noise_input_when_use_phantom_false(
        self, _patch_registry_capture: None
    ) -> None:
        _CaptureInputModel.last_input = None
        cfg = _cfg(model_type="capture", patch_size=[32, 32])
        result = synthetic_forward_probe(
            cfg, device="cpu", backward=False, use_phantom=False
        )
        assert result.passed
        x = _CaptureInputModel.last_input
        assert x is not None
        # Gaussian noise → > 100 unique values at 1x1x32x32 is virtually certain.
        assert torch.unique(x).numel() > 100


# ── Patch-size normalisation (trailing singleton stripping) ────────────


class TestProbePatchSizeHandling:
    """[H, W, 1] is the canonical 2D-in-3D-pipeline shape (TorchIO patches
    always have a depth axis). The probe strips trailing singletons so a
    2D model doesn't crash conv2d on a 5D tensor."""

    def test_trailing_singleton_in_patch_size_is_stripped(
        self, _patch_registry: None
    ) -> None:
        cfg = _cfg(model_type="good", patch_size=[16, 16, 1])
        result = synthetic_forward_probe(cfg, device="cpu", backward=False)
        assert result.passed, (
            f"Probe should strip trailing singleton: got "
            f"{result.category}: {result.message}"
        )

    def test_missing_patch_size_defaults_to_2d(self, _patch_registry: None) -> None:
        cfg = _cfg(model_type="good")
        # Override patch_size to empty list.
        cfg.data.patch_size = []
        result = synthetic_forward_probe(cfg, device="cpu", backward=False)
        assert result.passed

    def test_batch_size_clamped_to_two(self, _patch_registry_capture: None) -> None:
        """Configs with huge batch sizes must be clamped so the probe stays cheap."""
        _CaptureInputModel.last_input = None
        cfg = _cfg(model_type="capture", patch_size=[16, 16], batch_size=64)
        result = synthetic_forward_probe(cfg, device="cpu", backward=False)
        assert result.passed
        x = _CaptureInputModel.last_input
        assert x is not None
        assert x.shape[0] <= 2, f"Probe must clamp batch_size <= 2; got {x.shape[0]}."


# ── No-torch fallback (probe must never crash the audit CLI) ──────────


class TestProbeNoTorchFallback:
    """When torch is unavailable, the probe returns a structured
    ``no_probe`` result rather than raising ImportError."""

    def test_returns_no_probe_when_torch_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate torch absence inside the probe's lazy import."""
        import builtins
        import importlib

        # Force the probe to re-import torch under our hook.
        import mriforge.infrastructure.validation.forward_probe as fp_mod

        real_import = builtins.__import__

        def _no_torch(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("simulated torch absence")
            return real_import(name, *args, **kwargs)

        # The probe does ``import torch`` inside its body — patch builtins.
        monkeypatch.setattr(builtins, "__import__", _no_torch)
        # Re-resolve from the module to ensure we hit the in-body import path.
        importlib.reload(fp_mod)
        try:
            cfg = _cfg(model_type="good")
            result = fp_mod.synthetic_forward_probe(cfg, device="cpu")
            assert result.category == "no_probe"
            assert result.passed is True
            assert result.severity == "info"
        finally:
            # Restore real import so subsequent tests still work.
            monkeypatch.setattr(builtins, "__import__", real_import)
            importlib.reload(fp_mod)


# ── Optional probe-image dumping ──────────────────────────────────────


class TestProbeImageDump:
    """When ``save_images_dir`` is set, the probe writes diagnostic PNGs."""

    def test_writes_pngs_when_save_dir_supplied(
        self, tmp_path, _patch_registry_capture: None
    ) -> None:
        pytest.importorskip("matplotlib")
        cfg = _cfg(model_type="capture", patch_size=[16, 16])
        result = synthetic_forward_probe(
            cfg,
            device="cpu",
            backward=False,
            save_images_dir=str(tmp_path),
            arm_name="arm42",
        )
        assert result.passed
        import os

        pngs = sorted(p for p in os.listdir(str(tmp_path)) if p.endswith(".png"))
        # The probe writes input + output + target = 3 PNGs by default.
        assert any("arm42_input" in p for p in pngs)
        assert any("arm42_output" in p for p in pngs)


# ── Contract-aware shape (Layer-3 ↔ Layer-1 bridge) ────────────────────


class _Conv3DModel(torch.nn.Module):
    """3D conv — valid ONLY on a 5D ``[B, C, D, H, W]`` input.

    Declares ``spatial_dims=(3,)``. Without contract-aware probing, a
    ``[H, W, 1]`` TorchIO patch is stripped to 2D and this model crashes
    conv3d ("Expected 5D input"). With it, the probe expands to 3D.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv3d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Conv2DDeclaredModel(torch.nn.Module):
    """2D conv that DECLARES ``spatial_dims=(2,)``.

    Fed genuinely-3D data (depth > 1, not a strippable singleton), it MUST
    still crash — the contract must surface the real 2D-model-on-3D-data bug,
    never silently collapse the data to hide it (expand-only reconciliation).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _ComplexRequiredModel(torch.nn.Module):
    """Declares ``accepts_complex=True`` and HARD-REQUIRES a complex input.

    The real-valued phantom would raise ``TypeError`` here; the probe must
    coerce the input to ``torch.complex`` because the model declared it.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError("expected a complex input (accepts_complex=True)")
        return self.conv(x.real)


@pytest.fixture
def _patch_registry_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch BOTH ``get_model_class`` and ``get_model_capabilities`` so the
    probe reads a DECLARED ``ModelCapabilities`` for the contract-aware tests.

    The probe imports both lazily from ``mriforge.models.registry``; patch the
    functions on that module so the fakes resolve.
    """
    from mriforge.models import registry as registry_mod
    from mriforge.models.capabilities import ModelCapabilities

    fakes: dict[str, tuple[Any, Any]] = {
        "conv3d_declared": (_Conv3DModel, ModelCapabilities(spatial_dims=(3,))),
        "conv2d_declared": (_Conv2DDeclaredModel, ModelCapabilities(spatial_dims=(2,))),
        "complex_required": (
            _ComplexRequiredModel,
            ModelCapabilities(spatial_dims=(2,), accepts_complex=True),
        ),
        "bad_sample": (_BadSampleModel, None),
        "good_sample": (_GoodSampleModel, None),
        "unbounded_sample": (_UnboundedSampleModel, None),
        "bad_sample_opted_out": (_BadSampleOptedOutModel, None),
    }

    def _get_cls(name: str) -> Any:
        if name not in fakes:
            raise KeyError(f"{name!r} not registered (test fake)")
        return fakes[name][0]

    def _get_caps(name: str) -> Any:
        return fakes.get(name, (None, None))[1]

    monkeypatch.setattr(registry_mod, "get_model_class", _get_cls)
    monkeypatch.setattr(registry_mod, "get_model_capabilities", _get_caps)


class TestProbeContractAwareShape:
    """The probe reads the model's declared ``spatial_dims`` and steers the
    synthetic shape accordingly — expanding UP to a declared rank, never
    collapsing a genuinely-higher-rank input DOWN."""

    def test_3d_declared_model_underfed_2d_patch_is_expanded(
        self, _patch_registry_caps: None
    ) -> None:
        """A ``spatial_dims=(3,)`` model fed a stripped ``[16, 16, 1]`` patch
        must be probed at 3D (conv3d succeeds), not 2D (conv3d would crash)."""
        cfg = _cfg(model_type="conv3d_declared", patch_size=[16, 16, 1])
        result = synthetic_forward_probe(cfg, device="cpu", backward=True)
        assert result.passed, (
            f"3D-declared model should be probed at 3D after expansion; got "
            f"{result.category}: {result.message}"
        )

    def test_2d_declared_model_on_real_3d_data_still_fails(
        self, _patch_registry_caps: None
    ) -> None:
        """Expand-only: a 2D model fed depth>1 data is a REAL bug; the contract
        must let conv2d crash (forward_pass_shape), not hide it by collapsing."""
        cfg = _cfg(model_type="conv2d_declared", patch_size=[16, 16, 8])
        result = synthetic_forward_probe(cfg, device="cpu", backward=False)
        assert not result.passed, (
            "2D model on genuinely-3D data must surface as a failure, not be "
            "silently collapsed to 2D."
        )
        assert result.category == "forward_pass_shape"

    def test_complex_declared_model_gets_complex_input(
        self, _patch_registry_caps: None
    ) -> None:
        """``accepts_complex=True`` → the probe feeds a torch.complex tensor."""
        cfg = _cfg(model_type="complex_required", patch_size=[16, 16])
        result = synthetic_forward_probe(cfg, device="cpu", backward=True)
        assert result.passed, (
            f"complex-declared model should receive a coerced complex input; got "
            f"{result.category}: {result.message}"
        )


# ── Generative sample() path probe ─────────────────────────────────────


class _BadSampleModel(torch.nn.Module):
    """forward() is a correct 2D map, but sample() emits a wrong-RANK tensor.

    Mirrors a latent-diffusion model whose decoder/projection is mis-wired:
    training (forward) looks fine, generation (sample) collapses the spatial
    grid — invisible until the first validation image.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def sample(self, num_steps: int = 2, batch_size: int = 1) -> torch.Tensor:
        # spatial rank 0 (a [B, C] latent) — must mismatch forward's rank 2.
        return torch.randn(batch_size, 4)


class _GoodSampleModel(torch.nn.Module):
    """sample() returns a 4D image whose spatial rank matches forward()."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def sample(self, num_steps: int = 2) -> torch.Tensor:
        return torch.randn(1, 1, 16, 16)


class _UnboundedSampleModel(torch.nn.Module):
    """sample() exposes NO step-count kwarg — the probe must refuse to run it
    (cost unbounded) and skip with a note, never hang the smoke gate."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def sample(self, cond: torch.Tensor) -> torch.Tensor:  # no step kwarg
        raise AssertionError("sample() must NOT be called without a step bound")


class _BadSampleOptedOutModel(_BadSampleModel):
    """A wrong-rank sample() that opts out of the sample-path probe."""

    synthetic_forward_probe_skip = {"sample_path"}


class TestProbeSamplePath:
    """The probe best-effort-exercises a generative ``sample()`` path, failing
    only on a concrete rank mismatch and only when it can bound the cost."""

    def test_sample_rank_mismatch_fails(self, _patch_registry_caps: None) -> None:
        cfg = _cfg(model_type="bad_sample", patch_size=[16, 16])
        result = synthetic_forward_probe(cfg, device="cpu", backward=False)
        assert not result.passed
        assert result.category == "sample_path_shape"
        assert "sample()" in result.message
        assert result.fix_hint is not None

    def test_good_sample_passes_with_note(self, _patch_registry_caps: None) -> None:
        cfg = _cfg(model_type="good_sample", patch_size=[16, 16])
        result = synthetic_forward_probe(cfg, device="cpu", backward=False)
        assert result.passed
        assert "sample() OK" in result.message

    def test_unbounded_sample_is_skipped_not_run(
        self, _patch_registry_caps: None
    ) -> None:
        """No step-count kwarg → the probe skips (note), never calls sample()
        (the fake raises AssertionError if called). Proves the cost bound."""
        cfg = _cfg(model_type="unbounded_sample", patch_size=[16, 16])
        result = synthetic_forward_probe(cfg, device="cpu", backward=False)
        assert result.passed, f"{result.category}: {result.message}"
        assert "skipped" in result.message

    def test_sample_path_opt_out_respected(self, _patch_registry_caps: None) -> None:
        """A model declaring ``probe_skip={'sample_path'}`` is not sample-probed,
        even with a wrong-rank sample()."""
        cfg = _cfg(model_type="bad_sample_opted_out", patch_size=[16, 16])
        result = synthetic_forward_probe(cfg, device="cpu", backward=False)
        assert result.category != "sample_path_shape"
