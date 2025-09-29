import src.config
from src.config import (
    ArgonSettings,
    AuthorizationSettings,
    LoggerSettings,
    ServiceSettings,
    Settings,
)
from src.users.models import users  # noqa
from src.simulations.models import simulations  # noqa


def pytest_configure(config):
    """
    Allow plugins and conftest files to perform initial configuration.

    This hook is called for every plugin and initial conftest file
    after command line options have been parsed but before test collection.
    """

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

    src.config.get_settings = get_mock_settings
