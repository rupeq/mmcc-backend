import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
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
    settings = Settings()
    return settings
