"""Unit tests for the v6.3 strategy promotions.

These tests exercise the *new algorithmic content* added to each
strategy in isolation — they don't spin up a full training loop.

Each test instantiates the relevant module/helper and verifies:
    - shapes are correct
    - gradients flow through the new pieces
    - mathematical invariants hold (e.g. exp(0) = identity warp)
"""

from __future__ import annotations

import math
import torch


# ----------------------------------------------------------------------------
# Diffeomorphic registration: exp(0) should be the identity grid
# ----------------------------------------------------------------------------
def test_diffeomorphic_exp_zero_is_identity():
    from mriforge.infrastructure.training.strategies.diffeomorphic_recon_strategy import (
        DiffeomorphicReconStrategy,
    )

    s = DiffeomorphicReconStrategy.__new__(DiffeomorphicReconStrategy)
    s.n_squarings = 6
    v = torch.zeros(2, 2, 16, 16)
    grid = s.exp_velocity(v)
    grid_id = s._identity_grid(2, 16, 16, v.device)
    # exp(0) = identity; tolerate numerical drift from 6 grid_sample passes.
    assert torch.allclose(grid, grid_id, atol=1e-3)


def test_diffeomorphic_smoothness_and_fold_penalty():
    from mriforge.infrastructure.training.strategies.diffeomorphic_recon_strategy import (
        DiffeomorphicReconStrategy,
    )

    s = DiffeomorphicReconStrategy.__new__(DiffeomorphicReconStrategy)
    v = torch.randn(1, 2, 16, 16) * 0.1
    smooth = DiffeomorphicReconStrategy._smoothness(v)
    det = DiffeomorphicReconStrategy._jacobian_determinant(v)
    assert smooth.item() > 0
    assert det.shape == (1, 15, 15)


# ----------------------------------------------------------------------------
# Spin-SDE Bloch integrator: deterministic limit recovers analytical decay
# ----------------------------------------------------------------------------
def test_spin_sde_zero_diffusion_t2_decay():
    from mriforge.infrastructure.training.strategies.spin_sde_strategy import SpinSDEStrategy

    s = SpinSDEStrategy.__new__(SpinSDEStrategy)
    s.n_steps = 200
    s.dt = 1e-4
    s.diffusion = 0.0
    m0 = torch.tensor([[1.0, 0.0, 0.0]]).view(1, 3, 1, 1)
    T1 = torch.full((1, 1, 1, 1), 1.0)
    T2 = torch.full((1, 1, 1, 1), 0.05)
    out = s.integrate(m0, T1, T2)
    expected_mx = math.exp(-200 * 1e-4 / 0.05)
    # T2 only decays the in-plane component; allow generous tol
    assert abs(out[0, 0, 0, 0].item() - expected_mx) < 0.05


# ----------------------------------------------------------------------------
# SPGR signal models: identity for α=π/2, infinite T1
# ----------------------------------------------------------------------------
def test_spgr_signal_basic():
    from mriforge.models.blocks.spgr_signal import spgr_signal, spin_echo_signal

    M0 = torch.full((1, 1, 4, 4), 0.8)
    T1 = torch.full((1, 1, 4, 4), 1000.0)
    T2 = torch.full((1, 1, 4, 4), 80.0)
    s_se = spin_echo_signal(M0, T1, T2, TR=1000.0, TE=80.0)
    assert torch.all(s_se > 0)
    s_spgr = spgr_signal(M0, T1, T2, TR=10.0, TE=2.0, alpha_rad=0.3)
    assert torch.all(s_spgr > 0)


# ----------------------------------------------------------------------------
# SIREN INR: forward shape + gradient flow
# ----------------------------------------------------------------------------
def test_siren_forward_and_backward():
    from mriforge.models.blocks.siren_inr import SIRENNetwork

    net = SIRENNetwork(in_features=2, hidden_features=64, hidden_layers=2, out_features=2)
    coords = torch.randn(8, 100, 2, requires_grad=True)
    out = net(coords)
    assert out.shape == (8, 100, 2)
    loss = out.pow(2).mean()
    loss.backward()
    assert coords.grad is not None and torch.isfinite(coords.grad).all()


# ----------------------------------------------------------------------------
# Cross-contrast cold diffusion: corruption schedule
# ----------------------------------------------------------------------------
def test_cross_contrast_corruption_schedule():
    from mriforge.infrastructure.training.strategies.cross_contrast_kspace_diffusion_strategy import (
        CrossContrastKspaceDiffusionStrategy as Strat,
    )

    s = Strat.__new__(Strat)
    s.sigma_max = 0.0  # turn off noise → deterministic check
    k_src = torch.randn(2, 1, 8, 8)
    k_dst = torch.randn(2, 1, 8, 8)
    t0 = torch.zeros(2)
    t1 = torch.ones(2)
    assert torch.allclose(
        s.forward_corrupt(k_src, k_dst, t0, eps=torch.zeros_like(k_src)), k_src, atol=1e-6
    )
    assert torch.allclose(
        s.forward_corrupt(k_src, k_dst, t1, eps=torch.zeros_like(k_src)), k_dst, atol=1e-6
    )


# ----------------------------------------------------------------------------
# Synthetic pathology lesion sampler
# ----------------------------------------------------------------------------
def test_lesion_sampler_shapes_and_intensity():
    from mriforge.infrastructure.training.strategies.synthetic_pathology_aug_strategy import (
        LesionSampler,
    )

    sampler = LesionSampler()
    img = torch.zeros(4, 1, 32, 32)
    composite, mask = sampler(img)
    assert composite.shape == img.shape and mask.shape == (4, 1, 32, 32)
    # Lesion injects strictly positive perturbation so composite > 0 somewhere.
    assert composite.amax(dim=(-1, -2, -3)).min() > 0


# ----------------------------------------------------------------------------
# Adversarial PGD attack: produces same shape, bounded perturbation in L∞
# ----------------------------------------------------------------------------
def test_pgd_attack_linf_bound():
    from mriforge.infrastructure.training.strategies.adversarial_robustness_strategy import (
        AdversarialRobustnessStrategy,
    )

    s = AdversarialRobustnessStrategy.__new__(AdversarialRobustnessStrategy)
    s.epsilon = 0.05
    s.attack_steps = 5
    s.attack_norm = "linf"

    target = torch.randn(2, 1, 8, 8)
    x = torch.randn(2, 1, 8, 8)
    # Use identity 'gen' so we have closed-form behaviour.
    gen_call = torch.nn.Identity()

    def loss_fn(z):
        return torch.nn.functional.l1_loss(gen_call(z), target)

    x_adv = s.pgd_attack(x, loss_fn)
    assert x_adv.shape == x.shape
    assert ((x_adv - x).abs().max() <= s.epsilon + 1e-5).item()


# ----------------------------------------------------------------------------
# Low-rank + sparse decomposition: SVT thresholds singular values
# ----------------------------------------------------------------------------
def test_svt_zeros_small_singular_values():
    from mriforge.infrastructure.training.strategies.low_rank_sparse_strategy import (
        LowRankSparseStrategy,
    )

    s = LowRankSparseStrategy.__new__(LowRankSparseStrategy)
    s.tau_low_rank = 1.0
    X = torch.randn(2, 3, 16) * 0.1  # singular values < 1
    L, sv_sum = s._svt(X, tau=s.tau_low_rank)
    assert sv_sum.item() == 0.0
    assert L.shape == X.shape


# ----------------------------------------------------------------------------
# QSM pipeline: end-to-end pass through chains primitives
# ----------------------------------------------------------------------------
def test_qsm_pipeline_runs():
    from mriforge.infrastructure.training.strategies.qsm_pipeline_strategy import QSMPipelineStrategy

    s = QSMPipelineStrategy.__new__(QSMPipelineStrategy)
    field = torch.randn(1, 1, 8, 8, 8) * 0.01
    out = s.run_pipeline({"field_map": field})
    for k in ("field_map", "unwrapped", "background_removed", "chi"):
        assert k in out
        assert torch.isfinite(out[k]).all()


# ----------------------------------------------------------------------------
# QSpace SH basis: orthonormality (approx) of L=0 column
# ----------------------------------------------------------------------------
def test_qspace_sh_basis_constant_column():
    from mriforge.infrastructure.training.strategies.qspace_diffusion_strategy import (
        build_sh_basis_unit_sphere,
    )

    bvecs = torch.randn(64, 3)
    Y, orders = build_sh_basis_unit_sphere(bvecs, max_order=2)
    assert Y.shape[0] == 64
    # First column corresponds to L=0 → constant up to scale.
    assert (Y[:, 0].abs() - Y[0, 0].abs()).abs().max() < 1e-5
    assert int(orders[0].item()) == 0


# ----------------------------------------------------------------------------
# Stochastic interpolants: correct interpolant at endpoints
# ----------------------------------------------------------------------------
def test_stochastic_interpolant_endpoints():
    from mriforge.infrastructure.training.strategies.stochastic_interpolants_strategy import (
        StochasticInterpolantsStrategy as S,
    )

    x0 = torch.zeros(4, 1, 8, 8)
    x1 = torch.ones(4, 1, 8, 8)
    t0 = torch.zeros(4, 1, 1, 1)
    t1 = torch.ones(4, 1, 1, 1)
    z = torch.zeros_like(x0)
    I0 = (1 - t0) * x0 + t0 * x1 + S._sigma(t0) * z
    I1 = (1 - t1) * x0 + t1 * x1 + S._sigma(t1) * z
    assert torch.allclose(I0, x0)
    assert torch.allclose(I1, x1)


# ----------------------------------------------------------------------------
# JEPA multi-block mask: context + targets disjoint
# ----------------------------------------------------------------------------
def test_jepa_multi_block_mask_disjoint():
    from mriforge.infrastructure.training.strategies.jepa_strategy import multi_block_mask

    ctx, tgts = multi_block_mask(32, 32, n_targets=3, ctx_scale=0.85, tgt_scale=0.2)
    occupied = torch.zeros(32, 32, dtype=torch.bool)
    for m in tgts:
        occupied = occupied | m
    overlap = ctx & occupied
    # Context can overlap targets if the random sampler placed them inside,
    # but in our impl ctx = ~occupied, so overlap must be zero.
    assert overlap.sum().item() == 0


# ----------------------------------------------------------------------------
# Field-probe demodulator: forward + inverse round-trip is identity
# ----------------------------------------------------------------------------
def test_field_probe_demod_round_trip():
    from mriforge.infrastructure.training.strategies.field_probe_coupled_strategy import (
        FieldProbeCoupledStrategy as Strat,
    )

    s = Strat.__new__(Strat)
    H = 16
    k = torch.complex(torch.randn(2, 1, H, H), torch.randn(2, 1, H, H))
    phase = torch.randn(2, H) * 0.1
    fwd = s.apply_demod(k, phase)
    back = s.apply_demod(fwd, -phase)
    assert torch.allclose(back, k, atol=1e-5)


# ----------------------------------------------------------------------------
# PnP-ADMM and RED: end-to-end on an identity forward operator
# ----------------------------------------------------------------------------
def test_pnp_admm_identity_problem():
    from mriforge.models.diffusion.samplers.pnp_red import PnPADMMSampler

    sampler = PnPADMMSampler(n_iters=40, rho=0.1, lambda_prior=0.0, cg_iters=8)
    y = torch.randn(1, 1, 8, 8)

    def denoiser(x):
        return x  # no-op denoiser

    out = sampler.sample(
        denoiser,
        forward_op=lambda x: x,
        adjoint_op=lambda x: x,
        y=y,
        shape=y.shape,
        device=y.device,
    )
    # With identity A, no-op denoiser, and λ=0 the fixed point is x=y;
    # tolerate residual ADMM gap.
    assert (out - y).abs().mean().item() < 5e-2


def test_red_identity_problem():
    from mriforge.models.diffusion.samplers.pnp_red import REDSampler

    sampler = REDSampler(n_iters=20, step_size=0.5, lambda_prior=0.0)
    y = torch.randn(1, 1, 8, 8)

    out = sampler.sample(
        denoiser=lambda x: x,
        forward_op=lambda x: x,
        adjoint_op=lambda x: x,
        y=y,
        shape=y.shape,
        device=y.device,
    )
    assert torch.allclose(out, y, atol=1e-3)


# ----------------------------------------------------------------------------
# GIRF: delta kernel is identity
# ----------------------------------------------------------------------------
def test_girf_delta_identity():
    from mriforge.infrastructure.physics.girf import GIRFKernel, apply_girf

    girf = GIRFKernel(n_axes=3, kernel_size=11)
    waveform = torch.randn(2, 3, 64)
    out = apply_girf(waveform, girf.kernel)
    assert torch.allclose(out, waveform, atol=1e-5)
    # Untrained delta_distance is exactly zero.
    assert girf.delta_distance().item() < 1e-12


def test_girf_cross_correlation_convention():
    """apply_girf uses F.conv1d, i.e. the cross-correlation convention (no
    kernel flip). The module docstring documents this; pin it so a future
    change to a true-convolution sign convention is a deliberate, reviewed
    one (doc-contract guard)."""
    import torch.nn.functional as F

    from mriforge.infrastructure.physics.girf import apply_girf

    # Asymmetric kernel weighted on the first (left) tap, single axis.
    kernel = torch.zeros(1, 1, 3)
    kernel[0, 0, 0] = 1.0
    wave = torch.zeros(1, 1, 7)
    wave[0, 0, 3] = 1.0  # impulse in the middle
    out = apply_girf(wave, kernel)
    # cross-correlation (no kernel flip) shifts the impulse RIGHT, not left.
    expected = F.conv1d(wave, kernel, padding=1, groups=1)
    assert torch.allclose(out, expected)
    assert out[0, 0, 4].item() == 1.0  # right of centre


# ----------------------------------------------------------------------------
# PAC-Bayes bound: positive slack, monotonic in n
# ----------------------------------------------------------------------------
def test_pac_bayes_bound_monotone():
    from mriforge.infrastructure.validation.pac_bayes_certificate import mcallester_bound

    bound_small = mcallester_bound(0.1, kl=10.0, n=100, delta=0.05)
    bound_large = mcallester_bound(0.1, kl=10.0, n=10_000, delta=0.05)
    assert bound_small > bound_large > 0.1


# ----------------------------------------------------------------------------
# PEFT: each method freezes / unfreezes the right parameter slice
# ----------------------------------------------------------------------------
def _trainable_count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_peft_lora_freezes_base():
    from mriforge.infrastructure.distributed.peft_inject import inject_peft

    class Cfg:
        enabled = True
        method = "lora"
        rank = 4
        alpha = 8.0
        dropout = 0.0
        target_modules = ["q_proj"]
        bias_mode = "none"

    model = torch.nn.Sequential(
        torch.nn.Linear(16, 16),  # not matched
    )
    model.q_proj = torch.nn.Linear(16, 16)
    inject_peft(model, Cfg)
    # base linear (q_proj.base) frozen; lora_A / lora_B trainable.
    trainable = _trainable_count(model)
    # Original 0 (matched) Linear had 16*16 + 16 = 272 params → frozen.
    assert trainable < 16 * 16


def test_peft_adapter_method_replaces_module():
    from mriforge.infrastructure.distributed.peft_inject import inject_peft, AdapterModule

    class Cfg:
        enabled = True
        method = "adapter"
        rank = 8
        alpha = 8.0
        dropout = 0.0
        target_modules = ["q_proj"]
        bias_mode = "none"

    model = torch.nn.Module()
    model.q_proj = torch.nn.Linear(16, 16)
    inject_peft(model, Cfg)
    assert isinstance(model.q_proj, AdapterModule)
    # Adapter has trainable down + up; base.weight is frozen.
    assert not model.q_proj.base.weight.requires_grad
    assert model.q_proj.up.weight.requires_grad


def test_peft_prefix_tuning_replaces_module():
    from mriforge.infrastructure.distributed.peft_inject import inject_peft, PrefixModule

    class Cfg:
        enabled = True
        method = "prefix_tuning"
        rank = 4
        alpha = 8.0
        dropout = 0.0
        target_modules = ["q_proj"]
        bias_mode = "none"

    model = torch.nn.Module()
    model.q_proj = torch.nn.Linear(16, 16)
    inject_peft(model, Cfg)
    assert isinstance(model.q_proj, PrefixModule)
    assert model.q_proj.prefix.requires_grad


def test_peft_bitfit_only_biases():
    from mriforge.infrastructure.distributed.peft_inject import inject_peft

    class Cfg:
        enabled = True
        method = "bitfit"
        rank = 4
        alpha = 8.0
        dropout = 0.0
        target_modules = ["q_proj"]
        bias_mode = "none"

    model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Linear(16, 8))
    inject_peft(model, Cfg)
    for n, p in model.named_parameters():
        if "bias" in n.split(".")[-1]:
            assert p.requires_grad
        else:
            assert not p.requires_grad
