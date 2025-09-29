import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggerSettings(BaseSettings):
    """
    Logger-specific settings, loaded from a .logger.env file.

    Attributes:
        log_level (str): The minimum level for log messages. Defaults to "INFO".
        log_json (bool): Whether to output logs in JSON format. Defaults to True.
        model_config (SettingsConfigDict): Pydantic settings configuration.
    """

    log_level: str = "INFO"
    log_json: bool = True

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".logger.env")
    )


class ArgonSettings(BaseSettings):
    """
    Argon2 hashing settings, loaded from a .argon.env file.

    Attributes:
        argon_time_cost (int): The time cost parameter for Argon2. Defaults to 3.
        argon_memory_cost (int): The memory cost parameter for Argon2. Defaults to 2**16.
        argon_parallelism (int): The parallelism parameter for Argon2. Defaults to 2.
        model_config (SettingsConfigDict): Pydantic settings configuration.
    """

    argon_time_cost: int = 3
    argon_memory_cost: int = 2**16
    argon_parallelism: int = 2

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".argon.env")
    )


@lru_cache()
def get_logger_settings() -> LoggerSettings:
    """
    Get the cached LoggerSettings instance.

    Returns:
        LoggerSettings: The logger settings.
    """
    return LoggerSettings()


@lru_cache()
def get_argon_settings() -> ArgonSettings:
    """
    Get the cached ArgonSettings instance.

    Returns:
        ArgonSettings: The Argon2 hashing settings.
    """
    return ArgonSettings()
