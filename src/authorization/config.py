import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Authorization settings, loaded from a .authorization.env file.

    Attributes:
        authjwt_secret_key (str): The secret key for JWT.
        authjwt_token_location (set): Location for JWT tokens. Defaults to {"cookies"}.
        authjwt_cookie_secure (bool): Whether JWT cookies should be secure. Defaults to False.
        access_token_max_age_in_minutes (int): Maximum age for access tokens in minutes. Defaults to 15.
        refresh_token_max_age_in_minutes (int): Maximum age for refresh tokens in minutes. Defaults to 60 * 24 * 7.
        model_config (SettingsConfigDict): Pydantic settings configuration.
    """

    authjwt_secret_key: str
    authjwt_token_location: set = {"cookies"}
    authjwt_cookie_secure: bool = False
    access_token_max_age_in_minutes: int = 15
    refresh_token_max_age_in_minutes: int = 60 * 24 * 7

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".authorization.env")
    )


@lru_cache
def get_config() -> Settings:
    """
    Get the cached AuthorizationSettings instance.

    Returns:
        Settings: The authorization settings.
    """

    settings = Settings()
    return settings
