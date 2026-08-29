"""Unit tests for DisentangledVAETrainingStrategy."""

from unittest.mock import MagicMock

import pytest
import torch

from mriforge.infrastructure.training.strategies.disentangled_vae_strategy import (
    DisentangledVAETrainingStrategy,
)


@pytest.fixture
def mock_env():
    """Create a mock training environment (NEW API - env only)."""
    env = MagicMock()
    env.device = torch.device("cpu")

    # Mock generator (DisentangledVAE)
    gen = MagicMock()
    # Mock forward: (recon, mu, logvar)
    gen.return_value = (torch.randn(2, 1, 64, 64), torch.randn(2, 4), torch.randn(2, 4))
    # Mock reparameterize, decode
    gen.reparameterize = MagicMock(return_value=torch.randn(2, 4))
    gen.decode = MagicMock(return_value=torch.randn(2, 1, 64, 64))
    del gen.module  # Prevent hasattr(gen, "module") from returning True
    gen.training = True
    gen.train = MagicMock()
    env.generator = gen
    env.models = {"generator": gen}

    # Mock context/losses
    env.context = MagicMock()
    env.context.loss_fn = {}

    # Mock config
    env.config = MagicMock()
    env.config.training.kl_anneal_end = 0.01
    env.config.optimization.optimizer.learning_rate = 1e-4
    # schema defaults; bare MagicMock fails StandardOptimizerStepper's raise-on-unknown
    env.config.optimization.gradient.clip.enabled = False
    env.config.optimization.gradient.clip.method = "norm"
    env.config.optimization.gradient.clip.value = 1.0
    # mixed_precision.py validates amp_dtype against a fixed set; a bare
    # MagicMock fails it (stale-fixture repair, 2026-07-01).
    env.config.optimization.precision.dtype = "float32"
    env.config.model.model_type = "disentangled_vae"
    # Concrete numeric values needed by UnifiedReconstructionLossComputer's
    # `iteration < warmup_iterations` comparison; without them the
    # MagicMock attribute returns a MagicMock and the `<` comparison fails.
    env.config.losses.reconstruction.warmup_iterations = 0
    env.config.losses.l1.warmup_iterations = 0
    env.config.losses.l2.warmup_iterations = 0
    env.config.losses.warmup_iterations = 0
    env.model_type = "disentangled_vae"

    return env


@pytest.fixture
def strategy(mock_env):
    """Instantiate DisentangledVAETrainingStrategy (NEW API - env only)."""
    return DisentangledVAETrainingStrategy(env=mock_env)


def test_initialization(strategy):
    """Test initialization."""
    assert strategy is not None


# ── Full loss flow ────────────────────────────────────────────────────────
#
# This used to be skipped as "requires the full UnifiedReconstructionLossComputer
# stack ... coverage of the actual VAE flow lives in tests/integration/". There is
# no such coverage: nothing under tests/integration/ exercises this strategy, and
# the sibling reason naming ``tests/integration/test_disentangled_vae_validation.py``
# points at a file that does not exist. So the strategy's own loss arithmetic —
# the only thing this unit test can own — went unchecked.
#
# The loss COMPUTER has its own tests; what belongs here is how the strategy
# combines its outputs. Stubbing ``compute`` to return a real L1 total is one
# line, not a reimplementation of the loss layer, and it makes the weighted-sum
# contract assertable instead of merely asserting five dict keys exist.


class _TinyVAE(torch.nn.Module):
    """Minimal generator honouring the strategy's documented API contract:
    returns ``(recon, mu, logvar)`` and exposes ``encode_content`` / ``decode``."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x, phys):
        recon = self.conv(x)
        mu = x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1).expand(-1, 4)
        return recon, mu, torch.zeros_like(mu)

    def encode_content(self, x):
        return self.conv(x)

    def decode(self, z, phys):
        return self.conv(z)


@pytest.fixture
def flow_strategy():
    """A real strategy instance with only the collaborators the flow needs.

    Built via ``__new__`` (the pattern in test_disentangled_vae_strategy_contract)
    so no DI bootstrap runs, but every method under test is the real one.
    """
    from types import SimpleNamespace

    from mriforge.models.losses.registry import create_loss

    s = DisentangledVAETrainingStrategy.__new__(DisentangledVAETrainingStrategy)
    s.device = torch.device("cpu")
    s.env = SimpleNamespace(generator=_TinyVAE(), losses={})
    s.config = SimpleNamespace(
        model=SimpleNamespace(model_type="disentangled_mri"),
        training=SimpleNamespace(kl_anneal_end=0.01),
    )
    s._anatomy_criterion = create_loss("mse")
    s._loss_dict_reuse = {}
    # The loss computer has its own suite; here it only has to return a real
    # scalar so the strategy's weighted sum is exercised on live tensors.
    s.loss_computer = SimpleNamespace(
        compute=lambda pred, target, epoch, iteration, losses_dict: SimpleNamespace(
            total=torch.nn.functional.l1_loss(pred, target)
        )
    )
    s._get_loss_weight = lambda name, epoch: 1.0
    s._compute_training_metrics = lambda **kw: {}
    return s


def test_compute_losses_impl_full_flow(flow_strategy):
    """Every declared term is produced, finite, and differentiable."""
    img_64 = torch.randn(2, 1, 16, 16)
    img_3t = torch.randn(2, 1, 16, 16)

    losses = flow_strategy._compute_losses_impl(img_64, img_3t, epoch=0)

    for key in ("g_total_loss", "l1_self", "l_trans", "l_anat", "kl"):
        assert key in losses, f"missing loss term {key!r}"
        assert torch.isfinite(losses[key]).all(), f"{key} is not finite"
    assert losses["g_total_loss"].requires_grad, "total loss is detached from the graph"


def test_total_loss_is_the_documented_weighted_sum(flow_strategy):
    """``total = l1_self + 10*l_trans + 5*l_anat + kl``.

    The weights are hardcoded in the strategy, so a silent change to any of them
    reweights the objective for every disentangled-VAE arm with no config diff to
    show for it (pitfall #13b).
    """
    img_64 = torch.randn(2, 1, 16, 16)
    img_3t = torch.randn(2, 1, 16, 16)

    losses = flow_strategy._compute_losses_impl(img_64, img_3t, epoch=0)

    expected = (
        losses["l1_self"]
        + 10.0 * losses["l_trans"]
        + 5.0 * losses["l_anat"]
        + losses["kl"]
    )
    assert torch.allclose(losses["g_total_loss"], expected)


def test_kl_weight_follows_the_configured_anneal_end(flow_strategy):
    """``kl`` scales with ``training.kl_anneal_end``; a knob that changed nothing
    would be pitfall #15 in the objective itself."""
    img = torch.randn(2, 1, 16, 16)

    kl_small = flow_strategy._compute_losses_impl(img, img, epoch=0)["kl"].clone()
    flow_strategy.config.training.kl_anneal_end = 0.1  # 10x
    kl_large = flow_strategy._compute_losses_impl(img, img, epoch=0)["kl"]

    assert torch.allclose(kl_large, kl_small * 10.0, rtol=1e-4)


def test_generator_without_the_contract_api_is_rejected(flow_strategy):
    """Guards the same TypeError the contract suite pins, through this fixture."""
    flow_strategy.env.generator = torch.nn.Conv2d(1, 1, 3, padding=1)

    with pytest.raises(TypeError, match=r"requires a generator.*encode_content"):
        flow_strategy._compute_losses_impl(
            torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16), epoch=0
        )


# ── validation_step ───────────────────────────────────────────────────────
#
# Previously skipped as "genuine coverage lives in
# tests/integration/test_disentangled_vae_validation.py" — a file that does not
# exist, so nothing covered this at all. The metrics computer is what the old
# MagicMock fixture choked on and it has its own suite; substituting a stub for
# it leaves the part that belongs here: that validation grades the CROSS-DOMAIN
# synthesis (encode_content -> decode with the 3T physics vector), not the plain
# self-reconstruction. Grading the wrong tensor is pitfall #18, and it is
# invisible in the metric values themselves.


@pytest.fixture
def val_strategy(flow_strategy):
    captured = {}

    class _Computer:
        def compute(self, pred, target):
            captured["pred"] = pred
            captured["target"] = target
            return {"psnr": torch.tensor(31.5)}

    flow_strategy._get_validation_metrics_computer = lambda cfg: _Computer()
    flow_strategy.captured = captured
    return flow_strategy


def test_validation_step_returns_val_prefixed_floats(val_strategy):
    img_64 = torch.randn(2, 1, 16, 16)
    img_3t = torch.randn(2, 1, 16, 16)

    metrics = val_strategy.validation_step(
        {"input": img_64, "target": img_3t}, img_64, img_3t
    )

    assert metrics["val_psnr"] == pytest.approx(31.5)
    assert all(isinstance(v, float) for v in metrics.values())


def test_validation_grades_the_cross_domain_synthesis(val_strategy):
    """The graded prediction must be ``decode(encode_content(img_64), phys_3T)``
    and the reference must be the 3T target — not the 64mT input."""
    img_64 = torch.randn(2, 1, 16, 16)
    img_3t = torch.randn(2, 1, 16, 16)

    val_strategy.validation_step({"input": img_64, "target": img_3t}, img_64, img_3t)

    gen = val_strategy.env.generator
    expected = gen.decode(gen.encode_content(img_64), None)
    assert torch.allclose(val_strategy.captured["pred"], expected)
    assert torch.equal(val_strategy.captured["target"], img_3t)


def test_validation_step_puts_the_generator_in_eval_mode(val_strategy):
    """A VAE validated in train mode samples the latent instead of using mu,
    which makes every validation number noisy for a reason no config records."""
    val_strategy.env.generator.train()
    img = torch.randn(2, 1, 16, 16)

    val_strategy.validation_step({"input": img, "target": img}, img, img)

    assert not val_strategy.env.generator.training
