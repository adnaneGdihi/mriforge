import pytest
from pydantic import ValidationError

from mriforge.config.schemas.enums import LogLevel
from mriforge.config.schemas.logging import LoggingConfigSchema


class TestLoggingConfigSchema:
    def test_defaults(self):
        # `level` folded to `sinks.level`, `experiment_name` to
        # `identity.experiment`. Both canonical fields carry the SAME default the
        # legacy spelling had, so asserting the default alone would pass without
        # the fold ever running. Each is paired with a legacy-in / canonical-out
        # line, which is the only part that exercises the rename.
        schema = LoggingConfigSchema()
        assert schema.sinks.level == LogLevel.INFO
        assert LoggingConfigSchema(level=LogLevel.DEBUG).sinks.level == LogLevel.DEBUG
        assert schema.sinks.silent is False
        assert schema.sinks.dir == "./logs"
        assert schema.identity.experiment == "default_experiment"
        assert LoggingConfigSchema(experiment_name="e").identity.experiment == "e"

    def test_custom_values(self):
        schema = LoggingConfigSchema(
            level=LogLevel.DEBUG,
            silent=True,
            log_dir="/tmp/logs",
            experiment_name="test_exp",
            # `wandb` is REFUSED by `TrackingService` -- see
            # TestTrackingServiceIsClosed. `wandb_project` / `wandb_entity`
            # are REFUSED too, not merely inert -- see
            # tests/unit/config/schemas/test_logging.py::test_wandb_project_is_refused_not_ignored
            # (#675, 2026-08-12).
            tracking_service="none",
        )
        assert schema.sinks.level == LogLevel.DEBUG
        assert schema.sinks.silent is True
        assert schema.sinks.dir == "/tmp/logs"
        assert schema.identity.experiment == "test_exp"
        assert schema.tracking.service == "none"

    def test_validation(self):
        with pytest.raises(ValidationError):
            LoggingConfigSchema(log_interval=0)

    def test_extra_fields(self):
        # extra="ignore"
        schema = LoggingConfigSchema(extra_field="value")
        assert not hasattr(schema, "extra_field")


class TestTrackingServiceIsClosed:
    """`service` used to be a bare `str` compared against `"tensorboard"`.

    Any other value fell off that branch in `pipelines/train.py`, so the run
    trained to completion with no tracking, no warning and a zero exit -- the
    silent-fallback shape non-negotiable #3 exists to forbid. The failure was
    invisible precisely because the config was ACCEPTED.
    """

    def test_wandb_is_refused_rather_than_accepted_and_ignored(self) -> None:
        """It was ADVERTISED in the old field description, which is worse than
        undocumented: `wandb_project`/`wandb_entity` are inert (#675), so a W&B
        run configured through this framework is configured through nothing."""
        with pytest.raises(ValidationError):
            LoggingConfigSchema(tracking={"service": "wandb"})

    def test_a_typo_is_refused(self) -> None:
        """The real-world case: `tensorbaord` silently disabled tracking."""
        with pytest.raises(ValidationError):
            LoggingConfigSchema(tracking={"service": "tensorbaord"})

    @pytest.mark.parametrize("value", ["tensorboard", "none"])
    def test_the_two_live_values_are_accepted(self, value: str) -> None:
        """Anti-vacuity. `none` is a member because 6 committed arms declare it
        and mean it; refusing it would break them for no gain."""
        schema = LoggingConfigSchema(tracking={"service": value})
        assert schema.tracking.service == value
