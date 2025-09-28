from datetime import timedelta

from another_fastapi_jwt_auth import AuthJWT
from starlette.responses import Response

from src.config import get_settings


def create_tokens(authorize: AuthJWT, *, email: str) -> tuple[str, str]:
    """
    Create access and refresh tokens for a given user email.

    Args:
        authorize (AuthJWT): The AuthJWT instance for token creation.
        email (str): The email of the user.

    Returns:
        tuple[str, str]: A tuple containing the access token and the refresh token.
    """
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
    """
    Sets access and refresh tokens as cookies in the response.

    Args:
        authorize (AuthJWT): The AuthJWT instance for setting cookies.
        response (Response): The Starlette response object.
        token (str): The access token to set.
        refresh_token (str): The refresh token to set.

    Returns:
        Response: The modified Starlette response object with tokens set as cookies.
    """
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
