from mriforge.infrastructure.di import (
    DIContainer,
    get_global_container,
    register_service,
    resolve_service,
)


def test_di_init_exports():
    """Verify that all required DI exports are available."""
    assert DIContainer is not None
    assert get_global_container is not None
    assert register_service is not None
    assert resolve_service is not None
