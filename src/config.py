import os
from functools import lru_cache
from typing import Literal, Any

from another_fastapi_jwt_auth import AuthJWT
from pydantic import AnyHttpUrl, field_validator

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

from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    env: Literal["dev", "prod"] = "dev"

    app_name: str = "SMO Loss Simulator API"
    api_prefix: str = "/api"
    cors_origins: list[AnyHttpUrl] = []
    cors_allow_credentials: bool = True

    db_url: str

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".service.env")
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]

        return v

    @property
    def debug(self) -> bool:
        return self.env == "dev"


@lru_cache()
def get_service_settings() -> ServiceSettings:
    return ServiceSettings()


class Settings(BaseSettings):
    service: ServiceSettings = get_service_settings()
    authorization: AuthorizationSettings = get_authorization_config()
    logger_settings: LoggerSettings = get_logger_settings()
    argon_settings: ArgonSettings = get_argon_settings()

    _auth_jwt = AuthJWT.load_config(get_authorization_config)  # type:ignore


@lru_cache
def get_settings() -> Settings:
    return Settings()
