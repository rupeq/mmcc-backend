import os
from functools import lru_cache
from typing import Literal, Any

from another_fastapi_jwt_auth import AuthJWT
from pydantic import Field, AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.authorization.config import (
    Settings as AuthorizationSettings,
    get_config as get_authorization_config,
)
from src.core.config import (
    get_logger_settings,
    LoggerSettings,
    get_argon_settings,
    ArgonSettings,
)


class ServiceSettings(BaseSettings):
    """
    Service-specific settings, loaded from a .service.env file.

    Attributes:
        env (Literal["dev", "prod"]): The environment the service is running in. Defaults to "dev".
        app_name (str): The name of the application. Defaults to "SMO Loss Simulator API".
        api_prefix (str): The API prefix. Defaults to "/api".
        cors_origins (list[AnyHttpUrl]): A list of allowed CORS origins. Defaults to an empty list.
        cors_allow_credentials (bool): Whether to allow credentials for CORS. Defaults to True.
        db_url (str): The database URL.
        model_config (SettingsConfigDict): Pydantic settings configuration.
    """

    env: Literal["dev", "prod"] = "dev"

    app_name: str = "SMO Loss Simulator API"
    api_prefix: str = "/api"
    cors_origins: list[AnyHttpUrl] = []
    cors_allow_credentials: bool = True

    db_url: str

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".service.env")
    )

    @property
    def debug(self) -> bool:
        """
        Determine if the application is in debug mode based on the environment.

        Returns:
            bool: True if the environment is "dev", False otherwise.
        """

        return self.env == "dev"


@lru_cache()
def get_service_settings() -> ServiceSettings:
    """
    Get the cached ServiceSettings instance.

    Returns:
        ServiceSettings: The service settings.
    """
    return ServiceSettings()


class Settings(BaseSettings):
    """
    Aggregate all application settings.

    Attributes:
        service (ServiceSettings): Service-specific settings.
        authorization (AuthorizationSettings): Authorization-specific settings.
        logger_settings (LoggerSettings): Logger-specific settings.
        argon_settings (ArgonSettings): Argon2 hashing-specific settings.
    """

    service: ServiceSettings = Field(default_factory=get_service_settings)
    authorization: AuthorizationSettings = Field(
        default_factory=get_authorization_config
    )
    logger_settings: LoggerSettings = Field(default_factory=get_logger_settings)
    argon_settings: ArgonSettings = Field(default_factory=get_argon_settings)

    def model_post_init(self, context: Any, /) -> None:
        """Load JWT config after the model is initialized."""
        super().model_post_init(context)
        AuthJWT.load_config(get_authorization_config)  # type:ignore


@lru_cache
def get_settings() -> Settings:
    """
    Get the cached Settings instance.

    Returns:
        Settings: The aggregated application settings.
    """

    return Settings()
