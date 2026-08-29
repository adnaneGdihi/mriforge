"""Tests for check_workflow_component_regime.

The per-arm companion to the maturity ledger. The ledger asks the corpus-level
"is regime R backed by >=1 tagged component?"; this asks the arm-level "does every
loss/metric THIS arm uses fit its declared regime?". It is what makes the
multi-valued ``workflows`` capability tag do per-arm work — a FLOW-tagged metric
on an mri_structural arm is a mis-pairing nothing else caught.
"""

from __future__ import annotations

from types import SimpleNamespace

from mriforge.config.schemas.enums import Regime
from mriforge.config.schemas.workflow import WorkflowConfigSchema
from mriforge.infrastructure.validation.config_health_checker import (
    ConfigHealthChecker,
    HealthCheckResult,
)
from tests.utils.config_block_stub import block_stub
from tests.utils.corpus import tracked_yamls


def _check(regime: Regime | None, *, loss_names=(), metric_names=()):
    """Run the check against real registry tags, stating the arm's components."""
    checker = ConfigHealthChecker.__new__(ConfigHealthChecker)
    workflow = WorkflowConfigSchema(regime=regime) if regime is not None else None
    losses = SimpleNamespace(
        image_losses=[SimpleNamespace(name=n, enabled=True) for n in loss_names],
        kspace_losses=[],
        complex_losses=[],
    )
    cfg = SimpleNamespace(
        workflow=workflow,
        losses=losses if loss_names else None,
        validation=block_stub("validation", metrics=list(metric_names)),
    )
    return checker.check_workflow_component_regime(cfg)


def _a_metric_tagged(*regimes: Regime) -> str:
    """A registered metric whose workflows tag is exactly ``regimes`` (skip if none)."""
    import pytest

    from mriforge.core.metrics.registry import MetricsRegistry

    want = frozenset(regimes)
    for name, meta in MetricsRegistry._workflow_tags.items():
        if meta.get("workflows") == want:
            return name
    pytest.skip(f"no metric tagged exactly {sorted(r.value for r in regimes)}")


class TestCatchesRealMismatches:
    def test_a_flow_metric_on_a_structural_arm_is_rejected(self) -> None:
        """THE hole this check closes: a regime-specific metric on a foreign arm.

        ``divergence`` / ``mass_conservation`` grade a velocity field; on a
        magnitude structural arm they are meaningless, and nothing before this
        check matched the metric's FLOW tag against the arm's structural regime.
        """
        flow_metric = _a_metric_tagged(Regime.FLOW)
        r = _check(Regime.STRUCTURAL, metric_names=[flow_metric])
        assert not r.passed
        assert r.severity == "error"
        assert flow_metric in r.message
        assert "mri_flow" in r.message

    def test_the_failure_names_the_regime_and_offers_a_fix(self) -> None:
        flow_metric = _a_metric_tagged(Regime.FLOW)
        r = _check(Regime.STRUCTURAL, metric_names=[flow_metric])
        assert r.yaml_keys == ["workflow.regime", "losses", "validation.metrics"]
        assert "workflows=" in (r.fix_hint or "")


class TestDoesNotRejectLegitimateArms:
    def test_a_flow_metric_on_a_flow_arm_passes(self) -> None:
        flow_metric = _a_metric_tagged(Regime.FLOW)
        assert _check(Regime.FLOW, metric_names=[flow_metric]).passed

    def test_agnostic_metrics_are_skipped_never_guessed(self) -> None:
        # psnr / ssim carry no workflows tag -> agnostic -> skip. The common case.
        r = _check(Regime.STRUCTURAL, metric_names=["psnr", "ssim"])
        assert r.passed

    def test_an_unknown_metric_name_is_skipped(self) -> None:
        # A name not in the registry is untagged by construction -> skip, never a
        # false positive.
        assert _check(Regime.STRUCTURAL, metric_names=["_not_a_metric_"]).passed

    def test_a_multi_regime_component_passes_on_any_of_its_regimes(self) -> None:
        # temporal_fidelity is tagged {FUNCTIONAL, DYNAMIC}; it must pass on both.
        import pytest

        from mriforge.core.metrics.registry import MetricsRegistry

        multi = None
        for name, meta in MetricsRegistry._workflow_tags.items():
            wf = meta.get("workflows")
            if wf and len(wf) >= 2:
                multi = (name, wf)
                break
        if multi is None:
            pytest.skip("no multi-regime-tagged metric registered")
        name, regimes = multi
        for regime in regimes:
            assert _check(regime, metric_names=[name]).passed, f"{name} failed on {regime}"

    def test_no_workflow_declared_is_skipped(self) -> None:
        assert _check(None, metric_names=["psnr"]).passed

    def test_a_disabled_loss_is_ignored(self) -> None:
        # A regime-foreign loss that is explicitly disabled must not trip the check.
        checker = ConfigHealthChecker.__new__(ConfigHealthChecker)
        from mriforge.models.losses.registry import LossRegistry

        foreign = None
        for lname, meta in LossRegistry._loss_domains.items():
            wf = meta.get("workflows")
            if wf and Regime.STRUCTURAL not in wf:
                foreign = lname
                break
        if foreign is None:
            import pytest

            pytest.skip("no non-structural-tagged loss registered")
        cfg = SimpleNamespace(
            workflow=WorkflowConfigSchema(regime=Regime.STRUCTURAL),
            losses=SimpleNamespace(
                image_losses=[SimpleNamespace(name=foreign, enabled=False)],
                kspace_losses=[],
                complex_losses=[],
            ),
            validation=block_stub("validation", metrics=[]),
        )
        assert checker.check_workflow_component_regime(cfg).passed


def test_a_foreign_loss_on_a_structural_arm_is_rejected() -> None:
    """The loss side of the same rule, driven by a real regime-tagged loss."""
    import pytest

    from mriforge.models.losses.registry import LossRegistry

    foreign = None
    for lname, meta in LossRegistry._loss_domains.items():
        wf = meta.get("workflows")
        if wf and Regime.STRUCTURAL not in wf:
            foreign = lname
            break
    if foreign is None:
        pytest.skip("no non-structural-tagged loss registered")
    r = _check(Regime.STRUCTURAL, loss_names=[foreign])
    assert not r.passed
    assert foreign in r.message


def test_the_check_is_wired_into_the_audit_run(monkeypatch) -> None:
    """Asserted BEHAVIOURALLY, by driving the real ``run_all_checks``.

    The previous form grepped ``inspect.getsource`` for the call site, which stays
    green against a COMMENTED-OUT call — the escape
    ``tests/unit/pipelines/test_maturity_gate_wiring.py`` documents having actually
    observed. Mutation-verified: commenting the call in ``run_all_checks`` fails this.
    """
    called: list[bool] = []

    def _spy(self, config):
        called.append(True)
        return HealthCheckResult(
            passed=True,
            check_name="workflow_component_regime",
            message="spy",
            severity="info",
        )

    monkeypatch.setattr(ConfigHealthChecker, "check_workflow_component_regime", _spy)
    report = ConfigHealthChecker().run_all_checks(_runnable_config())

    assert called, "run_all_checks never invoked check_workflow_component_regime"
    assert any(r.check_name == "workflow_component_regime" for r in report.results), (
        "the check ran but its result never reached the report"
    )


def _runnable_config():
    """A minimal real ``TrainingSettings`` that ``run_all_checks`` can walk."""
    from mriforge.config.schemas.base import CANONICAL_CONFIG_VERSION
    from mriforge.config.settings import TrainingSettings

    return TrainingSettings.settings_from_dict(
        {
            "config_version": CANONICAL_CONFIG_VERSION,
            "data": {
                "train_path": "/tmp/train",
                "val_path": "/tmp/val",
                "batch_size": 2,
                "dataset_type": "mrixfields",
            },
            "optimization": {"learning_rate": 1e-4},
            "logging": {},
            "model": {"model_type": "unet"},
            "workflow": {"regime": "mri_structural", "task": "synthesis"},
        }
    )


def test_the_77_annotated_arms_would_all_pass() -> None:
    """Regression guard: this check must not newly break any already-annotated arm.

    A per-arm regime check is only safe to add because 0 of the 77 arms that
    declare a workflow use a component tagged for a foreign regime. If a future
    annotation or re-tag breaks that, this fails here rather than on the cluster.
    """
    import pathlib

    import yaml

    from mriforge.core.metrics.registry import MetricsRegistry
    from mriforge.models.losses.registry import LossRegistry

    root = pathlib.Path(__file__).resolve().parents[4] / "experiments" / "inprogress"
    if not root.is_dir():
        import pytest

        pytest.skip("experiments/inprogress not present")

    by_value = {r.value: r for r in Regime}
    checked = 0
    for path in tracked_yamls(root):
        text = path.read_text()
        if "workflow:" not in text:
            continue
        doc = yaml.safe_load(text)
        wf = (doc or {}).get("workflow") or {}
        regime = by_value.get(wf.get("regime"))
        if regime is None:
            continue
        losses = doc.get("losses") or {}
        for attr in ("image_losses", "kspace_losses", "complex_losses"):
            for entry in losses.get(attr) or []:
                name = isinstance(entry, dict) and entry.get("name")
                tags = (
                    (LossRegistry._loss_domains.get(str(name).lower()) or {}).get("workflows")
                    if name
                    else None
                )
                assert not (tags and regime not in tags), f"{path.name}: loss {name}"
        for name in (doc.get("validation") or {}).get("metrics") or []:
            tags = (MetricsRegistry._workflow_tags.get(str(name).lower()) or {}).get("workflows")
            assert not (tags and regime not in tags), f"{path.name}: metric {name}"
        checked += 1
    assert checked >= 1, "expected at least one annotated arm to exercise this guard"
