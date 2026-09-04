"""Tests for the ``make_*`` config-driven factory helpers.

Historically these took ``config_path: Path`` only and called
``TrainingSettings.from_yaml`` internally, forcing a pure-Python caller to
round-trip an in-memory config through a temp file. The scripting-API work
adds an optional ``config: TrainingSettings`` parameter so the helpers accept
an already-built (in-memory) SSOT directly, while keeping the path form
backward-compatible. Passing neither must raise (no silent default).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.config.settings import TrainingSettings
from spectramr.pipelines.make import make_optimizer


def _in_memory_settings() -> TrainingSettings:
    return TrainingSettings.settings_from_dict(
        {
            "model": {"model_type": "unet"},
            "data": {},
            "optimization": {"learning_rate": 1e-4},
            "logging": {},
        }
    )


def test_make_optimizer_accepts_in_memory_config():
    """An in-memory ``TrainingSettings`` builds an optimizer with no YAML file."""
    settings = _in_memory_settings()
    model = torch.nn.Linear(4, 4)

    optimizer, metadata = make_optimizer(config=settings, model=model)

    assert isinstance(optimizer, torch.optim.Optimizer)
    assert metadata["learning_rate"] == 1e-4
    # Provenance reflects the in-memory origin (no path).
    assert metadata["config_path"] == "<in-memory>"


def test_make_optimizer_requires_path_or_config():
    """Passing neither ``config_path`` nor ``config`` must raise (no default)."""
    model = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match="config_path or config"):
        make_optimizer(model=model)


def test_make_optimizer_still_accepts_path(tmp_path):
    """Backward compat: the original ``config_path`` form keeps working."""
    import yaml as _yaml

    cfg = {
        "config_version": "1.0",
        "model": {"model_type": "unet"},
        "data": {},
        "optimization": {"learning_rate": 2e-4},
        "logging": {},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(_yaml.safe_dump(cfg), encoding="utf-8")
    model = torch.nn.Linear(4, 4)

    optimizer, metadata = make_optimizer(config_path=path, model=model)

    assert isinstance(optimizer, torch.optim.Optimizer)
    assert metadata["learning_rate"] == 2e-4
    assert metadata["config_path"] == str(path)


def test_make_reads_a_declared_dataset_field() -> None:
    """`make_dataset` / `make_dataloader` read `config.data.dataset_name` on
    their first line. It was never a DataConfigSchema field, so both public
    entry points raised AttributeError for every config; no test covered them.
    They now read `dataset_type`, and the emitted metadata key matches."""
    import inspect

    from spectramr.config.schemas.data import DataConfigSchema
    from spectramr.pipelines import make

    assert DataConfigSchema().dataset_type  # readable, unlike dataset_name
    src = inspect.getsource(make)
    assert "config.data.dataset_name" not in src
    assert '"dataset_type": config.data.dataset_type' in src


class TestMakeBuildsThroughTheCanonicalBuilder:
    """``make`` advertises itself as "SSOT: TrainingSettings" and did not honour it.

    It called ``ModelFactory.create_model(config.model)`` -- passing only the
    model sub-schema, which structurally cannot reach
    ``config.data.processing`` or ``config.undersampling``. So the helper whose
    whole contract is "the config is the single source of truth" built a model
    from a strict subset of it.

    It was also the narrowest checkpoint reader in the repository: it knew only
    ``model_state_dict`` and never stripped wrapper prefixes at all, so a DDP or
    ``torch.compile`` checkpoint loaded nothing and reported success.
    """

    @staticmethod
    def _make_src() -> str:
        from pathlib import Path

        return (
            Path(__file__).resolve().parents[3] / "src" / "spectramr" / "pipelines" / "make.py"
        ).read_text()

    def test_no_longer_calls_the_config_sniffing_factory(self) -> None:
        import ast

        tree = ast.parse(self._make_src())
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute | ast.Name)
        }
        assert "create_model" not in called
        assert "get_model_factory" not in called

    def test_builds_through_model_builder(self) -> None:
        assert "ModelBuilder(config, device_obj)" in self._make_src()

    def test_binds_the_shared_checkpoint_reader(self) -> None:
        import spectramr.pipelines.make as make_mod

        assert hasattr(make_mod, "resolve_state_dict"), (
            "make.py still knows only `model_state_dict` and never strips "
            "wrapper prefixes; a DDP/compile checkpoint loads nothing"
        )


# ---------------------------------------------------------------------------
# Split routing. Both helpers used to route with an ``if/elif`` whose ``else``
# was the TRAIN loader, so an unrecognised split silently returned training data
# while the metadata echoed the bogus name back (non-negotiable 3). Two copies of
# that chain is how the same bug got written twice, so the mapping now has one
# owner and these tests pin it.
# ---------------------------------------------------------------------------

from spectramr.pipelines.make import (  # noqa: E402
    _SPLIT_TO_LOADER,
    _loader_shuffles,
    _resolve_split,
)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [("train", "train"), ("val", "val"), ("validation", "val"), ("test", "val")],
)
def test_resolve_split_maps_known_names(requested, expected):
    assert _resolve_split(requested) == expected


@pytest.mark.parametrize("bogus", ["not_a_split", "TRAIN", "", "traon", "holdout"])
def test_resolve_split_raises_on_anything_else(bogus):
    """The regression: an unknown split must RAISE, never fall through to train.

    ``"TRAIN"`` is in here deliberately -- a case-only typo used to sail through
    the ``else`` branch and return training data under a plausible-looking name.
    """
    with pytest.raises(ValueError, match="Unknown split"):
        _resolve_split(bogus)


def test_resolve_split_error_names_the_valid_options():
    with pytest.raises(ValueError) as exc:
        _resolve_split("nope")
    message = str(exc.value)
    for name in _SPLIT_TO_LOADER:
        assert name in message


def test_test_split_is_declared_as_resolving_to_val():
    """``test`` has no held-out loader yet (#665); the mapping must say so.

    Pinning this makes the Wave-5 change visible: when a real test split lands,
    this test is what fails and forces the mapping to be revisited rather than
    silently continuing to serve validation data as 'test'.
    """
    assert _SPLIT_TO_LOADER["test"] == "val"


# ---------------------------------------------------------------------------
# shuffle is READ, not guessed
# ---------------------------------------------------------------------------


class _FakeQueue:
    """Stands in for ``tio.Queue`` — the loader's dataset on the TorchIO path."""

    def __init__(self, shuffle_subjects: bool, shuffle_patches: bool) -> None:
        self.shuffle_subjects = shuffle_subjects
        self.shuffle_patches = shuffle_patches


class _FakeLoader:
    def __init__(self, dataset=None, sampler=None) -> None:
        self.dataset = dataset
        self.sampler = sampler


def test_shuffle_reads_the_torchio_queue_not_the_sampler():
    """The shape that would have made this a NEW lie rather than a fixed one.

    ``tio.Queue`` requires a ``SequentialSampler`` because the Queue itself
    randomises. Asking the sampler alone reports ``False`` for a training loader
    that genuinely shuffles — measured, confident, and wrong.
    """
    from torch.utils.data import SequentialSampler

    loader = _FakeLoader(
        dataset=_FakeQueue(shuffle_subjects=True, shuffle_patches=True),
        sampler=SequentialSampler([1, 2, 3]),
    )
    assert _loader_shuffles(loader) is True


def test_shuffle_false_when_the_queue_shuffles_nothing():
    loader = _FakeLoader(dataset=_FakeQueue(shuffle_subjects=False, shuffle_patches=False))
    assert _loader_shuffles(loader) is False


def test_shuffle_falls_back_to_the_sampler_for_a_plain_dataloader():
    from torch.utils.data import RandomSampler, SequentialSampler

    assert _loader_shuffles(_FakeLoader(sampler=RandomSampler([1, 2, 3]))) is True
    assert _loader_shuffles(_FakeLoader(sampler=SequentialSampler([1, 2, 3]))) is False


def test_shuffle_is_none_when_undeterminable():
    """An order this cannot characterise is reported unknown, never guessed."""
    assert _loader_shuffles(_FakeLoader()) is None


# ---------------------------------------------------------------------------
# batch_size is gone, on purpose
# ---------------------------------------------------------------------------


def test_make_dataloader_has_no_batch_size_parameter():
    """It accepted ``batch_size`` and silently dropped it (``999`` -> config's 4).

    Batch size is the director's, and an override here is not harmless: the
    director exposes only a val-side target, ``strided_validation_subset``
    derives the validation STRIDE from it (so it changes which records the val
    set contains), and ``dataset_type='cine'`` has a guard it would route around.
    """
    import inspect

    from spectramr.pipelines.make import make_dataloader

    assert "batch_size" not in inspect.signature(make_dataloader).parameters


def test_make_dataset_device_is_a_real_parameter():
    """``device`` was accepted and never referenced; it now reaches the director."""
    import inspect

    from spectramr.pipelines.make import make_dataset

    assert "device" in inspect.signature(make_dataset).parameters


# ---------------------------------------------------------------------------
# A DECLARED held-out test set must not be answered with the validation loader
# (cohort review 2026-09-02, T0.3). The ``test -> val`` mapping above stays
# honest only while no test set exists.
# ---------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from spectramr.pipelines.make import (  # noqa: E402
    _refuse_substitution_for_a_declared_test_set,
    make_dataset,
)


def _cfg_declaring_test(*, via: str) -> SimpleNamespace:
    source = SimpleNamespace(test_index_path="test.json" if via == "manifest" else None)
    return SimpleNamespace(
        data=SimpleNamespace(
            source=source, enable_test_split=(via == "fraction"), dataset_type="kspace"
        )
    )


@pytest.mark.parametrize("via", ["manifest", "fraction"])
def test_declared_test_set_refuses_the_validation_substitution(via):
    """The planted violation: asking for 'test' on an arm that declares one."""
    with pytest.raises(RuntimeError, match="Refusing to hand back the VALIDATION loader"):
        _refuse_substitution_for_a_declared_test_set(_cfg_declaring_test(via=via), "test")


def test_undeclared_test_set_keeps_the_documented_substitution():
    cfg = SimpleNamespace(
        data=SimpleNamespace(source=SimpleNamespace(test_index_path=None), enable_test_split=False)
    )
    _refuse_substitution_for_a_declared_test_set(cfg, "test")  # no raise
    _refuse_substitution_for_a_declared_test_set(_cfg_declaring_test(via="manifest"), "val")


def test_make_dataset_applies_the_refusal_before_building_anything(monkeypatch):
    import spectramr.pipelines.make as make_mod

    cfg = _cfg_declaring_test(via="manifest")
    monkeypatch.setattr(make_mod, "_resolve_settings", lambda _p, _c: cfg)
    with pytest.raises(RuntimeError, match="held-out test set"):
        make_dataset(config=cfg, split="test")
