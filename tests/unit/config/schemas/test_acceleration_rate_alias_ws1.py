"""WS-1 physics-03: ``rate`` is a documented alias for ``base_acceleration`` —
wire it so the advertised behaviour is real, with explicit ``base_acceleration``
taking precedence."""

from __future__ import annotations

from spectramr.config.schemas.acceleration import AccelerationConfigSchema


def test_rate_aliases_base_acceleration() -> None:
    assert AccelerationConfigSchema(rate=8.0).base_acceleration == 8.0


def test_explicit_base_acceleration_wins_over_rate() -> None:
    c = AccelerationConfigSchema(rate=8.0, base_acceleration=4.0)
    assert c.base_acceleration == 4.0


def test_default_base_acceleration_without_rate() -> None:
    assert AccelerationConfigSchema().base_acceleration == 4.0
