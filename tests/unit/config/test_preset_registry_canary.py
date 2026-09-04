"""Canary + parametrised "every advertised preset loads cleanly" tests.

These tests exercise the *registry contract* — not the YAML schema itself
(see :mod:`tests.unit.config.test_preset_system` for that). Specifically:

* A ``ConfigPreset`` registered under any name must be retrievable via
  ``PresetRegistry.get(name)``.
* ``PresetRegistry.list_all()`` and ``PresetRegistry.filter()`` return
  consistent results.
* An unknown name raises ``KeyError`` with the preset name in the message
  (the error is actionable, not a bare ``KeyError: 'x'``).
* ``PresetRegistry.clear()`` is the test-isolation contract: after calling
  it the registry is empty.
* ``ConfigPreset.to_dict()`` round-trips all metadata fields.

No real YAML loading or ``TrainingSettings`` construction happens here —
the preset ``yaml_path`` is just a string stored in the dataclass; only
``load_config()`` / ``to_settings()`` would touch the filesystem and those
are tested separately in :mod:`tests.unit.config.test_preset_system`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectramr.config.presets import ConfigPreset, PresetRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_registry() -> None:
    """Every test starts and ends with an empty PresetRegistry."""
    PresetRegistry.clear()
    yield
    PresetRegistry.clear()


def _make_preset(
    name: str,
    *,
    category: str = "gan",
    tags: list[str] | None = None,
    description: str = "unit-test preset",
    yaml_path: str = "fake/path.yaml",
) -> ConfigPreset:
    return ConfigPreset(
        name=name,
        category=category,
        description=description,
        yaml_path=yaml_path,
        version="6.0",
        tags=tags or [],
    )


# ---------------------------------------------------------------------------
# Canary: the registry itself is usable right after a clear()
# ---------------------------------------------------------------------------


class TestRegistryCanary:
    def test_empty_after_clear(self) -> None:
        assert PresetRegistry.list_all() == []

    def test_register_and_retrieve(self) -> None:
        p = _make_preset("baseline_gan")
        PresetRegistry.register(p)
        retrieved = PresetRegistry.get("baseline_gan")
        assert retrieved is p

    def test_unknown_preset_raises_key_error(self) -> None:
        with pytest.raises(KeyError) as exc_info:
            PresetRegistry.get("does_not_exist")
        # The error message must contain the sought name so the caller can
        # act on it without reading a traceback.
        assert "does_not_exist" in str(exc_info.value)

    def test_unknown_preset_error_lists_available(self) -> None:
        PresetRegistry.register(_make_preset("alpha"))
        PresetRegistry.register(_make_preset("beta"))
        with pytest.raises(KeyError) as exc_info:
            PresetRegistry.get("gamma")
        msg = str(exc_info.value)
        assert "alpha" in msg or "beta" in msg  # available list shown

    def test_duplicate_registration_raises_value_error(self) -> None:
        PresetRegistry.register(_make_preset("dup"))
        with pytest.raises(ValueError, match="already registered"):
            PresetRegistry.register(_make_preset("dup"))

    def test_list_all_sorted_by_name(self) -> None:
        PresetRegistry.register(_make_preset("zebra"))
        PresetRegistry.register(_make_preset("alpha"))
        names = [p.name for p in PresetRegistry.list_all()]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# Every advertised preset name loads cleanly (parametrised)
# ---------------------------------------------------------------------------

_ADVERTISED = [
    ("baseline_gan",   "gan",         ["baseline", "recommended"]),
    ("baseline_diff",  "diffusion",   ["baseline"]),
    ("recon_unet",     "reconstruction", ["stable"]),
    ("vae_basic",      "vae",         []),
    ("ssl_mae",        "ssl",         ["experimental"]),
]


@pytest.mark.parametrize("name,category,tags", _ADVERTISED, ids=[n for n, _, _ in _ADVERTISED])
def test_advertised_preset_registers_and_retrieves(
    name: str, category: str, tags: list[str]
) -> None:
    """Every advertised (name, category, tags) triple must survive a
    register → get round-trip and expose consistent metadata via
    ``to_dict()``."""
    preset = _make_preset(name, category=category, tags=tags)
    PresetRegistry.register(preset)

    retrieved = PresetRegistry.get(name)
    d = retrieved.to_dict()

    assert d["name"] == name
    assert d["category"] == category
    assert d["tags"] == tags
    assert d["version"] == "6.0"
    assert "yaml_path" in d


# ---------------------------------------------------------------------------
# filter() consistency
# ---------------------------------------------------------------------------


class TestRegistryFilter:
    def test_filter_by_category(self) -> None:
        PresetRegistry.register(_make_preset("gan_a", category="gan"))
        PresetRegistry.register(_make_preset("diff_a", category="diffusion"))
        PresetRegistry.register(_make_preset("gan_b", category="gan"))

        results = PresetRegistry.filter(category="gan")
        names = {p.name for p in results}
        assert names == {"gan_a", "gan_b"}

    def test_filter_by_tag(self) -> None:
        PresetRegistry.register(_make_preset("a", tags=["baseline"]))
        PresetRegistry.register(_make_preset("b", tags=["ablation"]))
        PresetRegistry.register(_make_preset("c", tags=["baseline", "ablation"]))

        baseline_names = {p.name for p in PresetRegistry.filter(tag="baseline")}
        assert baseline_names == {"a", "c"}

    def test_filter_by_name_pattern(self) -> None:
        PresetRegistry.register(_make_preset("recon_v1"))
        PresetRegistry.register(_make_preset("recon_v2"))
        PresetRegistry.register(_make_preset("gan_v1"))

        results = PresetRegistry.filter(name_pattern="recon")
        names = {p.name for p in results}
        assert names == {"recon_v1", "recon_v2"}

    def test_filter_combined(self) -> None:
        PresetRegistry.register(
            _make_preset("gan_base", category="gan", tags=["baseline"])
        )
        PresetRegistry.register(
            _make_preset("gan_abl", category="gan", tags=["ablation"])
        )
        PresetRegistry.register(
            _make_preset("diff_base", category="diffusion", tags=["baseline"])
        )

        results = PresetRegistry.filter(category="gan", tag="baseline")
        assert len(results) == 1
        assert results[0].name == "gan_base"

    def test_filter_no_match_returns_empty(self) -> None:
        PresetRegistry.register(_make_preset("x", category="gan"))
        assert PresetRegistry.filter(category="ssl") == []


# ---------------------------------------------------------------------------
# list_categories / list_tags
# ---------------------------------------------------------------------------


class TestListCategoriesAndTags:
    def test_list_categories_sorted(self) -> None:
        PresetRegistry.register(_make_preset("a", category="vae"))
        PresetRegistry.register(_make_preset("b", category="gan"))
        PresetRegistry.register(_make_preset("c", category="diffusion"))
        cats = PresetRegistry.list_categories()
        assert cats == sorted(cats)
        assert set(cats) == {"vae", "gan", "diffusion"}

    def test_list_tags_sorted(self) -> None:
        PresetRegistry.register(_make_preset("a", tags=["z_tag", "a_tag"]))
        tags = PresetRegistry.list_tags()
        assert tags == sorted(tags)
        assert "z_tag" in tags and "a_tag" in tags

    def test_empty_registry_returns_empty_lists(self) -> None:
        assert PresetRegistry.list_categories() == []
        assert PresetRegistry.list_tags() == []


# ---------------------------------------------------------------------------
# ConfigPreset.to_dict() round-trip
# ---------------------------------------------------------------------------


class TestConfigPresetToDict:
    def test_all_fields_present(self) -> None:
        p = _make_preset(
            "test_preset",
            category="reconstruction",
            tags=["stable"],
            description="a test",
            yaml_path="some/path.yaml",
        )
        d = p.to_dict()
        assert d["name"] == "test_preset"
        assert d["category"] == "reconstruction"
        assert d["tags"] == ["stable"]
        assert d["description"] == "a test"
        assert d["yaml_path"] == "some/path.yaml"
        assert d["version"] == "6.0"
        assert "parent_presets" in d

    def test_repr_contains_name_and_category(self) -> None:
        p = _make_preset("my_preset", category="ssl")
        r = repr(p)
        assert "my_preset" in r
        assert "ssl" in r

    def test_parent_presets_default_empty(self) -> None:
        p = _make_preset("standalone")
        assert p.parent_presets == []
        assert p.to_dict()["parent_presets"] == []

    def test_parent_presets_stored(self) -> None:
        p = ConfigPreset(
            name="child",
            category="gan",
            description="inheriting",
            yaml_path="x.yaml",
            parent_presets=["parent_a", "parent_b"],
        )
        assert p.parent_presets == ["parent_a", "parent_b"]
        d = p.to_dict()
        assert d["parent_presets"] == ["parent_a", "parent_b"]
