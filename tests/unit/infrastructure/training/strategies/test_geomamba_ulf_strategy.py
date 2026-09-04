"""Unit tests for GeoMambaULFStrategy wiring (2026-06 Tier-1 fixes).

Covers two inert-mechanism (#16/#15) repairs:

* The metric-SFC SSOT block (``config.training.geomamba_ulf.metric_sfc``) is
  bridged onto the generator. The generator is built from ``config.model`` and
  never sees the ``geomamba_ulf`` training block, so without the bridge the
  whole ``MetricSFCConfig`` (beta / smoothing_sigma / connectivity / cache_dir)
  was inert — ``abl_no_metric_sfc`` (beta=0) was byte-identical to ``v0`` and
  the beta-HPO sweep was six identical runs.
* ``_extract_contrast_id`` reads ``contrast_idx`` in addition to
  ``contrast_id``. The loaders (m4raw / slice_dataset) and the default
  collation emit ``contrast_idx``; reading only ``contrast_id`` silently
  dropped it, collapsing FiLM conditioning to the same (gamma, beta) for every
  contrast.

The full ``__init__`` builds a DI container + AMP + an optimizer, so the
device-agnostic helpers are exercised on a bare ``__new__`` instance.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")
pytest.importorskip("scipy")

from spectramr.infrastructure.training.strategies.geomamba_ulf_strategy import (  # noqa: E402
    GeoMambaULFStrategy,
)
from spectramr.models.generators.geo_mamba_unet import GeoMambaUNet  # noqa: E402


def _tiny_model() -> GeoMambaUNet:
    return GeoMambaUNet(in_channels=1, out_channels=1, d_model=8, n_blocks=1)


# ---------------------------------------------------------------------------
# 1.4 — contrast_idx is read (FiLM conditioning fires)
# ---------------------------------------------------------------------------


def test_extract_contrast_id_reads_contrast_idx() -> None:
    """The standard ``contrast_idx`` key (m4raw / slice_dataset / collation)
    must be honoured, not silently dropped."""
    cid = GeoMambaULFStrategy._extract_contrast_id(
        {"contrast_idx": torch.tensor([2, 1])}, torch.zeros(2, 1, 4, 4)
    )
    assert cid is not None
    assert cid.tolist() == [2, 1]
    assert cid.dtype == torch.long


def test_extract_contrast_id_prefers_contrast_id_over_idx() -> None:
    """When both keys are present ``contrast_id`` wins (mrixfields convention)."""
    cid = GeoMambaULFStrategy._extract_contrast_id(
        {"contrast_id": torch.tensor([0]), "contrast_idx": torch.tensor([3])},
        torch.zeros(1, 1, 4, 4),
    )
    assert cid.tolist() == [0]


def test_extract_contrast_id_none_when_absent() -> None:
    cid = GeoMambaULFStrategy._extract_contrast_id({}, torch.zeros(1))
    assert cid is None


# ---------------------------------------------------------------------------
# 1.5 — metric_sfc SSOT is bridged onto the generator
# ---------------------------------------------------------------------------


def test_apply_metric_sfc_config_bridges_onto_model() -> None:
    """``geo_cfg.metric_sfc`` values land on the generator before any forward."""
    model = _tiny_model()
    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s.env = SimpleNamespace(generator=model)
    geo_cfg = SimpleNamespace(
        metric_sfc=SimpleNamespace(
            beta=0.0, smoothing_sigma=3.0, connectivity=18, cache_dir="cache/x"
        )
    )
    s._apply_metric_sfc_config(geo_cfg)
    assert model.sfc_beta == 0.0
    assert model.sfc_smoothing_sigma == 3.0
    assert model.sfc_connectivity == 18
    assert model.sfc_cache_dir == "cache/x"


def test_apply_metric_sfc_config_noop_without_block() -> None:
    """No ``metric_sfc`` block → the model keeps its construction defaults."""
    model = _tiny_model()
    before = (model.sfc_beta, model.sfc_smoothing_sigma, model.sfc_connectivity)
    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s.env = SimpleNamespace(generator=model)
    s._apply_metric_sfc_config(SimpleNamespace())  # geo_cfg has no metric_sfc
    s._apply_metric_sfc_config(None)  # geo_cfg absent entirely
    assert (model.sfc_beta, model.sfc_smoothing_sigma, model.sfc_connectivity) == before


def test_apply_metric_sfc_config_unwraps_ddp_module() -> None:
    """A DDP/EMA-wrapped generator (``.module``) is unwrapped before patching."""
    model = _tiny_model()
    wrapper = SimpleNamespace(module=model)
    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s.env = SimpleNamespace(generator=wrapper)
    s._apply_metric_sfc_config(SimpleNamespace(metric_sfc=SimpleNamespace(beta=7.0)))
    assert model.sfc_beta == 7.0


# ---------------------------------------------------------------------------
# Commit 1 — PH loss is fail-loud when the [topology] extra is missing
# ---------------------------------------------------------------------------


def test_get_ph_loss_raises_when_topology_deps_missing(monkeypatch) -> None:
    """A non-zero cubical_ph_w2 weight with gudhi/POT absent must RAISE, not
    warn-and-skip. Skipping silently degrades the PH arm to plain L1 while it
    smoke-PASSes (pitfalls #9/#16). The strategy's whole method claim is PH."""
    import spectramr.models.losses.cubical_ph_w2_loss as ph_mod

    class _NoDeps:
        def __init__(self, *a, **k):
            raise ImportError("gudhi/POT not installed")

    monkeypatch.setattr(ph_mod, "CubicalPHWassersteinLoss", _NoDeps)

    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s._ph_loss = None
    with pytest.raises(RuntimeError, match="topology"):
        s._get_ph_loss()


def test_get_ph_loss_caches_constructed_module(monkeypatch) -> None:
    """When deps are present the module is built once and cached."""
    import spectramr.models.losses.cubical_ph_w2_loss as ph_mod

    calls = {"n": 0}

    class _FakePH:
        def __init__(self, *a, **k):
            calls["n"] += 1

    monkeypatch.setattr(ph_mod, "CubicalPHWassersteinLoss", _FakePH)

    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s._ph_loss = None
    first = s._get_ph_loss()
    second = s._get_ph_loss()
    assert first is second
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Cohort review 2026-09-03 — one weight owner, a truthful Beltrami, declared
# loss ownership
# ---------------------------------------------------------------------------


def _weights_cfg(entries, reconstruction=None):
    """A config double: ``image_losses`` entries and an optional lambda section."""
    return SimpleNamespace(
        losses=SimpleNamespace(
            image_losses=[{"name": n, "weight": w} for n, w in entries],
            kspace_losses=[],
            complex_losses=[],
            reconstruction=reconstruction if reconstruction is not None else SimpleNamespace(),
        ),
        training=SimpleNamespace(geomamba_ulf=None),
    )


def _bare_for_setup(cfg):
    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s.config = cfg
    s.env = SimpleNamespace(generator=_tiny_model())
    s._apply_metric_sfc_config = lambda *_a, **_k: None
    s.logging_service = None
    return s


def test_the_three_weights_come_from_the_loss_weight_table() -> None:
    """``losses.image_losses`` entries are the one owner: the strategy used to read l1
    from ``losses.reconstruction.lambda_l1`` (schema default 10.0) and the topology
    terms through its own resolver."""
    s = _bare_for_setup(_weights_cfg([("l1", 10.0), ("cubical_ph_w2", 0.1)]))
    GeoMambaULFStrategy._setup_strategy_specific_components(s)
    assert (s._lambda_l1, s._lambda_ph_w2, s._lambda_beltrami) == (10.0, 0.1, 0.0)


def test_an_author_written_lambda_is_the_same_owner() -> None:
    """The table reads both surfaces; ``losses.reconstruction.lambda_l1`` written by
    the author resolves the same way an ``image_losses`` entry does."""
    s = _bare_for_setup(_weights_cfg([], reconstruction=SimpleNamespace(lambda_l1=3.0)))
    GeoMambaULFStrategy._setup_strategy_specific_components(s)
    assert (s._lambda_l1, s._lambda_ph_w2, s._lambda_beltrami) == (3.0, 0.0, 0.0)


def test_an_arm_without_an_l1_entry_is_refused() -> None:
    s = _bare_for_setup(_weights_cfg([("cubical_ph_w2", 1.0)]))
    with pytest.raises(ValueError, match=r"losses\.image_losses"):
        GeoMambaULFStrategy._setup_strategy_specific_components(s)


def test_a_weighted_beltrami_on_a_one_channel_output_raises() -> None:
    """Planted violation: 13 arms declared the term at out_channels=1 and it never
    computed (the silent skip this replaces)."""
    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s.env = SimpleNamespace(generator=lambda x, mask=None, contrast_id=None: x)
    s._uncertainty_loss = None
    s._lambda_l1, s._lambda_ph_w2, s._lambda_beltrami = 10.0, 0.0, 0.05
    s._topology_warmup_epochs = 0
    s._ph_loss = None
    s._beltrami_loss = None
    x = torch.rand(1, 1, 8, 8)
    with pytest.raises(ValueError, match="2-channel"):
        GeoMambaULFStrategy._compute_losses_impl(s, x, x, epoch=0)


def test_an_unweighted_beltrami_on_a_one_channel_output_is_simply_absent() -> None:
    """Weight 0 is an absent term, not an error, so every arm that deleted the entry
    trains on a 1-channel head."""
    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s.env = SimpleNamespace(generator=lambda x, mask=None, contrast_id=None: x)
    s._uncertainty_loss = None
    s._lambda_l1, s._lambda_ph_w2, s._lambda_beltrami = 10.0, 0.0, 0.0
    s._topology_warmup_epochs = 0
    s._ph_loss = None
    s._beltrami_loss = None
    x = torch.rand(1, 1, 8, 8)
    losses = GeoMambaULFStrategy._compute_losses_impl(s, x, x, epoch=0)
    assert set(losses) == {"loss_l1", "g_total_loss"}


def test_loss_ownership_is_declared() -> None:
    """The class states what it computes inline and that it folds nothing; the private
    weight resolver that made the same YAML weigh differently here is gone."""
    assert GeoMambaULFStrategy.inline_losses == frozenset(
        {"l1", "cubical_ph_w2", "beltrami_diagnostic"}
    )
    assert GeoMambaULFStrategy.folds_image_losses is False
    assert not hasattr(GeoMambaULFStrategy, "_read_loss_weight")
    assert not hasattr(GeoMambaULFStrategy, "_search_loss_weight")
