"""Regression: multi-stage trainable params must be registered on ``opt_g``.

The pipeline (multi-stage) strategy collects ``self._trainable_params`` across
all stages in ``__init__``, but historically never handed them to the global
generator optimizer (``self.env.opt_g``). Result: any stage driven by the
global optimizer (i.e. without its own per-stage optimizer) had
``requires_grad=True`` params that received no updates — silently never trained.

The fix adds ``_register_global_params()``, called after param collection and
again after the E2E unfreeze in ``on_epoch_start``. It must:

* add the global-optimizer params to ``opt_g.param_groups``;
* skip params already in an ``opt_g`` group (idempotent / safe to re-call);
* NOT add params already owned by a per-stage optimizer (those are stepped
  separately in ``_step_stage_optimizers`` and would otherwise update twice).
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn

from mriforge.infrastructure.training.strategies.pipeline_strategy import (
    MultiTrainingStrategy,
)


def _env_with_opt() -> types.SimpleNamespace:
    """Stub env carrying a real SGD optimizer over a single dummy param."""
    dummy = nn.Parameter(torch.zeros(1))
    return types.SimpleNamespace(opt_g=torch.optim.SGD([dummy], lr=0.1))


def _param_ids(opt: torch.optim.Optimizer) -> set[int]:
    return {id(p) for g in opt.param_groups for p in g["params"]}


def _make_strategy(stage_optimizers: dict | None = None) -> MultiTrainingStrategy:
    """Build a minimally-populated strategy touching only the tested method."""
    s = object.__new__(MultiTrainingStrategy)
    s.env = _env_with_opt()
    s.device = torch.device("cpu")
    s.multi_stages = {"stage_a": nn.Linear(4, 4), "stage_b": nn.Linear(4, 4)}
    s._stage_optimizers = stage_optimizers or {}
    s._trainable_params = s._collect_trainable_params()
    return s


class TestPipelineGlobalParamRegistration:
    def test_trainable_params_land_in_opt_g(self):
        s = _make_strategy()
        before = _param_ids(s.env.opt_g)
        added = s._register_global_params()
        after = _param_ids(s.env.opt_g)

        assert added, "no params were appended to opt_g"
        assert after - before, "opt_g.param_groups gained no new params"
        # Every collected trainable param must now be tracked by opt_g.
        for p in s._trainable_params:
            assert id(p) in after, "a trainable stage param is absent from opt_g"

    def test_register_is_idempotent(self):
        s = _make_strategy()
        s._register_global_params()
        ids_once = _param_ids(s.env.opt_g)
        added_second = s._register_global_params()
        ids_twice = _param_ids(s.env.opt_g)

        assert not added_second, "second call re-added params (not deduped)"
        assert ids_once == ids_twice, "idempotent re-registration changed opt_g"

    def test_stage_optimizer_params_not_double_added(self):
        # stage_b is driven by its OWN optimizer → must stay out of opt_g.
        s = _make_strategy()
        stage_b_params = list(s.multi_stages["stage_b"].parameters())
        s._stage_optimizers = {
            "stage_b": torch.optim.SGD(stage_b_params, lr=0.01)
        }

        added = s._register_global_params()
        added_ids = {id(p) for p in added}
        opt_g_ids = _param_ids(s.env.opt_g)

        # stage_b params are stepped separately — never handed to opt_g.
        for p in stage_b_params:
            assert id(p) not in added_ids, "per-stage param double-added to opt_g"
            assert id(p) not in opt_g_ids, "per-stage param leaked into opt_g"
        # stage_a (global) params still get registered.
        for p in s.multi_stages["stage_a"].parameters():
            assert id(p) in opt_g_ids, "global stage param missing from opt_g"

    def test_no_opt_g_is_safe(self):
        s = _make_strategy()
        s.env = types.SimpleNamespace()  # no opt_g attribute
        assert s._register_global_params() == []

    def test_param_flows_gradient_through_global_optimizer(self):
        # End-to-end: a step on opt_g must actually move a registered param.
        s = _make_strategy()
        s._register_global_params()

        param = next(s.multi_stages["stage_a"].parameters())
        before = param.detach().clone()

        x = torch.randn(2, 4)
        out = s.multi_stages["stage_a"](x)
        s.env.opt_g.zero_grad(set_to_none=True)
        out.pow(2).mean().backward()
        assert param.grad is not None, "no gradient reached the registered param"
        s.env.opt_g.step()

        assert not torch.equal(before, param), "registered param did not update"
