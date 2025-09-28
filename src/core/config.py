import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggerSettings(BaseSettings):
    log_level: str = "INFO"
    log_json: bool = True

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".logger.env")
    )


class ArgonSettings(BaseSettings):
    argon_time_cost: int = 3
    argon_memory_cost: int = 2**16
    argon_parallelism: int = 2

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".argon.env")
    )


@lru_cache
def get_logger_settings() -> LoggerSettings:
    return LoggerSettings()


@lru_cache
def get_argon_settings() -> ArgonSettings:
    return ArgonSettings()
