import pytest

import src.config
from src.config import (
    ArgonSettings,
    AuthorizationSettings,
    LoggerSettings,
    ServiceSettings,
    Settings,
)


@pytest.fixture(scope="session", autouse=True)
def mock_settings():
    """
    Override the get_settings function for the entire test session.

    This fixture manually patches the settings to prevent ValidationErrors
    during pytest collection in CI/CD environments. It uses a session scope
    to ensure the patch is active before any tests are collected and runs.
    """
    original_get_settings = src.config.get_settings

    def get_mock_settings():
        """This function will replace the real get_settings()."""
        return Settings(
            service=ServiceSettings(
                db_url="postgresql+asyncpg://test:test@localhost:5432/testdb"
            ),
            authorization=AuthorizationSettings(
                authjwt_secret_key="test-secret-key"
            ),
            logger_settings=LoggerSettings(),
            argon_settings=ArgonSettings(),
        )

    try:
        src.config.get_settings = get_mock_settings
        yield
    finally:
        src.config.get_settings = original_get_settings
