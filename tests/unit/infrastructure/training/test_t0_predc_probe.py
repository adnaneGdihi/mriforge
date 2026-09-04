"""Pins for the t=0 pre-DC probe.

Every assertion here exists because a specific way of getting the probe wrong
would still produce a plausible number:

* reading the POST-DC output instead of the pre-DC proposal scores a flawless
  reconstruction, because at t=0 under hard DC every bin is acquired and the
  output is the input by construction;
* letting a generator ignore ``return_pre_dc`` yields a bare tensor that unpacks
  as if nothing were wrong;
* relying on the generator's stashed sensitivity maps produces a number
  conditioned on whatever ran last, correct only by call order;
* a key-set that varies makes the DDP all-reduce misalign metric values onto
  other metrics' names -- never an error.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.t0_predc_probe import (
    T0_PREDC_PREFIX,
    build_t0_timesteps,
    forward_pre_dc,
    generator_exposes_pre_dc,
    rename_to_probe_namespace,
    run_t0_predc_probe,
    t0_predc_key,
)


class StubGenerator:
    """A generator whose pre-DC and post-DC outputs are distinguishable.

    ``post_dc`` mimics hard DC at t=0: it hands the input straight back. So a
    probe wired to the wrong return element measures the input against the
    target and reports a perfect score -- the failure this stub exists to make
    visible.
    """

    exposes_pre_dc = True

    def __init__(self, *, honour_flag: bool = True, pre_dc_fill: float = 7.0):
        self.honour_flag = honour_flag
        self.pre_dc_fill = pre_dc_fill
        self.stashed_smaps: torch.Tensor | None = None
        self.received_kwargs: dict = {}
        self.received_timesteps: torch.Tensor | None = None

    def __call__(self, x, timesteps=None, *, return_pre_dc=False, **kwargs):
        self.received_kwargs = dict(kwargs)
        self.received_timesteps = timesteps
        smaps = kwargs.get("sensitivity_maps")
        if smaps is not None:
            self.stashed_smaps = smaps
        post_dc = x  # hard DC at t=0: output IS the input
        pre_dc = torch.full_like(x, self.pre_dc_fill)
        if not self.honour_flag:
            return post_dc
        if return_pre_dc:
            return post_dc, pre_dc
        return post_dc


def _kwargs(smaps=None):
    return {"sensitivity_maps": torch.ones(1, 2, 4, 4) if smaps is None else smaps}


class TestT0PredcKey:
    def test_val_prefixed_key_gets_the_namespace_inserted_not_prepended(self):
        # Prepending would mint `val_t0_predc_val_psnr` and break the CSV's
        # `val_` grouping.
        assert t0_predc_key("val_psnr") == "val_t0_predc_psnr"

    def test_bare_key_gets_the_whole_prefix(self):
        assert t0_predc_key("psnr") == "val_t0_predc_psnr"

    def test_renaming_twice_is_a_no_op(self):
        once = t0_predc_key("val_psnr")
        assert t0_predc_key(once) == once

    def test_prefix_constant_is_what_the_function_produces(self):
        # Pins the constant to the behaviour, so changing one without the other
        # cannot pass.
        assert t0_predc_key("psnr") == T0_PREDC_PREFIX + "psnr"


class TestRenameToProbeNamespace:
    def test_every_key_moves_and_values_are_untouched(self):
        out = rename_to_probe_namespace({"val_psnr": 31.5, "hfen": 0.9})
        assert out == {"val_t0_predc_psnr": 31.5, "val_t0_predc_hfen": 0.9}

    def test_colliding_sources_raise_rather_than_silently_dropping_one(self):
        # `psnr` and `val_psnr` both map to `val_t0_predc_psnr`. Overwriting is
        # the #1682 shape: a computed value disappearing without a word.
        with pytest.raises(ValueError, match="collision"):
            rename_to_probe_namespace({"psnr": 1.0, "val_psnr": 2.0})

    def test_empty_input_gives_empty_output(self):
        assert rename_to_probe_namespace({}) == {}


class TestGeneratorExposesPreDc:
    def test_true_when_the_class_declares_it(self):
        assert generator_exposes_pre_dc(StubGenerator()) is True

    def test_false_when_the_attribute_is_absent(self):
        assert generator_exposes_pre_dc(object()) is False


class TestBuildT0Timesteps:
    def test_every_sample_is_at_timestep_zero(self):
        t = build_t0_timesteps(3, torch.device("cpu"))
        assert t.shape == (3,)
        assert torch.equal(t, torch.zeros(3, dtype=torch.long))

    def test_dtype_is_long_because_timesteps_index_a_schedule(self):
        assert build_t0_timesteps(2, torch.device("cpu")).dtype == torch.long

    def test_non_positive_batch_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            build_t0_timesteps(0, torch.device("cpu"))


class TestForwardPreDc:
    def test_returns_the_pre_dc_proposal_not_the_post_dc_output(self):
        # THE pin. `post_dc` is the input; `pre_dc` is filled with 7.0. Wiring
        # the probe to element [0] returns the input and scores perfectly.
        gen = StubGenerator(pre_dc_fill=7.0)
        x = torch.zeros(1, 2, 4, 4)
        out = forward_pre_dc(
            gen, x, timesteps=build_t0_timesteps(1, x.device), forward_kwargs=_kwargs()
        )
        assert torch.equal(out, torch.full_like(x, 7.0)), (
            "probe returned the post-DC output; at t=0 that is the input itself"
        )

    def test_missing_sensitivity_maps_raises_instead_of_using_the_stash(self):
        gen = StubGenerator()
        x = torch.zeros(1, 2, 4, 4)
        with pytest.raises(ValueError, match="sensitivity_maps"):
            forward_pre_dc(gen, x, timesteps=build_t0_timesteps(1, x.device), forward_kwargs={})

    def test_passed_maps_win_over_already_stashed_ones(self):
        # The generator falls back to its stash when no kwarg is given, so the
        # probe must overwrite it. Stash one set, pass another, assert the
        # passed one is what the generator saw.
        gen = StubGenerator()
        gen.stashed_smaps = torch.zeros(1, 2, 4, 4)
        passed = torch.full((1, 2, 4, 4), 3.0)
        x = torch.zeros(1, 2, 4, 4)
        forward_pre_dc(
            gen,
            x,
            timesteps=build_t0_timesteps(1, x.device),
            forward_kwargs=_kwargs(passed),
        )
        assert torch.equal(gen.received_kwargs["sensitivity_maps"], passed)
        assert torch.equal(gen.stashed_smaps, passed)

    def test_generator_ignoring_the_flag_raises_rather_than_reporting_post_dc(self):
        gen = StubGenerator(honour_flag=False)
        x = torch.zeros(1, 2, 4, 4)
        with pytest.raises(TypeError, match="return_pre_dc"):
            forward_pre_dc(
                gen,
                x,
                timesteps=build_t0_timesteps(1, x.device),
                forward_kwargs=_kwargs(),
            )

    def test_timesteps_reach_the_generator(self):
        gen = StubGenerator()
        x = torch.zeros(2, 2, 4, 4)
        t = build_t0_timesteps(2, x.device)
        forward_pre_dc(gen, x, timesteps=t, forward_kwargs=_kwargs())
        assert torch.equal(gen.received_timesteps, t)


class TestRunT0PredcProbe:
    def test_scores_the_pre_dc_tensor(self):
        seen = {}

        def score(pred, timesteps):
            seen["pred"] = pred.clone()
            seen["t"] = timesteps.clone()
            return {"val_psnr": 12.0}

        gen = StubGenerator(pre_dc_fill=7.0)
        x = torch.zeros(1, 2, 4, 4)
        out = run_t0_predc_probe(
            generator=gen, model_input=x, forward_kwargs=_kwargs(), score=score
        )
        assert torch.equal(seen["pred"], torch.full_like(x, 7.0))
        assert torch.equal(seen["t"], torch.zeros(1, dtype=torch.long))
        assert out == {"val_t0_predc_psnr": 12.0}

    def test_generator_without_the_capability_emits_nothing_and_never_scores(self):
        class Plain(StubGenerator):
            exposes_pre_dc = False

        called = []
        out = run_t0_predc_probe(
            generator=Plain(),
            model_input=torch.zeros(1, 2, 4, 4),
            forward_kwargs=_kwargs(),
            score=lambda p, t: called.append(1) or {},
        )
        assert out == {}
        assert called == []

    def test_probe_runs_without_grad(self):
        # A probe that builds a graph would hold validation activations alive
        # for the whole sweep.
        def score(pred, timesteps):
            assert not pred.requires_grad
            assert not torch.is_grad_enabled()
            return {"val_psnr": 1.0}

        x = torch.zeros(1, 2, 4, 4, requires_grad=True)
        run_t0_predc_probe(
            generator=StubGenerator(),
            model_input=x,
            forward_kwargs=_kwargs(),
            score=score,
        )


class TestSensitivityMapSpellings:
    """Production supplies `smaps`, not `sensitivity_maps`.

    `_build_generator_kwargs` writes `gen_kwargs["smaps"]`; only its
    prior-model branch uses the long spelling. A guard that accepted just one
    of the two would raise on the path the probe actually runs on.
    """

    @pytest.mark.parametrize("key", ["sensitivity_maps", "smaps"])
    def test_either_spelling_satisfies_the_guard(self, key):
        gen = StubGenerator()
        x = torch.zeros(1, 2, 4, 4)
        out = forward_pre_dc(
            gen,
            x,
            timesteps=build_t0_timesteps(1, x.device),
            forward_kwargs={key: torch.ones(1, 2, 4, 4)},
        )
        assert out.shape == x.shape

    def test_a_present_but_none_valued_key_does_not_satisfy_the_guard(self):
        # `gen_kwargs["mask"] = mask` can legitimately be None upstream; a bare
        # `in` test would accept that and silently fall back to the stash.
        gen = StubGenerator()
        x = torch.zeros(1, 2, 4, 4)
        with pytest.raises(ValueError, match="sensitivity maps"):
            forward_pre_dc(
                gen,
                x,
                timesteps=build_t0_timesteps(1, x.device),
                forward_kwargs={"smaps": None},
            )


def test_a_timesteps_key_in_the_kwargs_does_not_collide_with_the_positional():
    """`accelerator_kwargs` is merged wholesale, so the key can arrive from config.

    Passing it through would be a "multiple values for argument 'timesteps'"
    TypeError -- the strategy's own forward path pops it for this reason.
    """
    gen = StubGenerator()
    x = torch.zeros(1, 2, 4, 4)
    t = build_t0_timesteps(1, x.device)
    out = forward_pre_dc(
        gen,
        x,
        timesteps=t,
        forward_kwargs={"sensitivity_maps": torch.ones(1, 2, 4, 4), "timesteps": 99},
    )
    assert out.shape == x.shape
    assert torch.equal(gen.received_timesteps, t), "the positional t=0 must win"
