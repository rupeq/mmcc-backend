from datetime import timedelta

from another_fastapi_jwt_auth import AuthJWT
from starlette.responses import Response

from src.config import get_settings


def create_tokens(authorize: AuthJWT, *, email: str) -> tuple[str, str]:
    return (
        authorize.create_access_token(
            subject=email,
            expires_time=timedelta(
                minutes=get_settings().authorization.access_token_max_age_in_minutes
            ),
            fresh=True,
        ),
        authorize.create_refresh_token(
            subject=email,
            expires_time=timedelta(
                minutes=get_settings().authorization.refresh_token_max_age_in_minutes
            ),
        ),
    )


def get_response_with_tokens(
    authorize: AuthJWT, *, response: Response, token: str, refresh_token: str
) -> Response:
    authorize.set_access_cookies(
        token,
        response=response,
        max_age=get_settings().authorization.access_token_max_age_in_minutes
        * 60,
    )
    authorize.set_refresh_cookies(
        refresh_token,
        response=response,
        max_age=get_settings().authorization.refresh_token_max_age_in_minutes
        * 60,
    )
    return response
