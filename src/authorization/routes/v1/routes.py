from another_fastapi_jwt_auth import AuthJWT
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status
from starlette.responses import Response, JSONResponse

from src.authorization.routes.v1.schemas import (
    SignInRequestSchema,
    SignUpRequestSchema,
    SignUpResponseSchema,
)
from src.authorization.routes.v1.utils import (
    create_tokens,
    get_response_with_tokens,
)
from src.core.db_session import get_session
from src.users.db_utils.exceptions import (
    UserNotFound,
    PasswordDoesNotMatch,
    UserAlreadyExists,
    UserIsNotActive,
)
from src.users.db_utils.users import (
    get_user_by_credentials,
    create_user,
    get_user_by_id,
)

router = APIRouter(tags=["v1", "authorization"], prefix="/v1/authorization")


@router.post("/signin")
async def signin(
    body: SignInRequestSchema,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """
    Handle user sign-in.

    Args:
        body (SignInRequestSchema): The sign-in request body containing email and password.
        authorize (AuthJWT): Dependency for JWT authorization.
        session (AsyncSession): Dependency for asynchronous database session.

    Returns:
        Response: A Starlette response with JWT cookies set on successful sign-in,
                  or a JSONResponse with an error if sign-in fails.
    """
    try:
        user = await get_user_by_credentials(
            session, email=str(body.email), password=body.password
        )
    except (UserNotFound, PasswordDoesNotMatch):
        response = JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "User not found"},
        )
        authorize.unset_jwt_cookies(response=response)
        return response

    token, refresh_token = create_tokens(authorize, user_id=str(user.id))
    return get_response_with_tokens(
        authorize,
        response=Response(status_code=status.HTTP_204_NO_CONTENT),
        token=token,
        refresh_token=refresh_token,
    )


@router.post(
    "/signup",
    response_model=SignUpResponseSchema,
    status_code=status.HTTP_201_CREATED,
)
async def signup(
    body: SignUpRequestSchema,
    session: AsyncSession = Depends(get_session),
):
    """
    Handle user sign-up.

    Args:
        body (SignUpRequestSchema): The sign-up request body containing email and password.
        session (AsyncSession): Dependency for asynchronous database session.

    Returns:
        SignUpResponseSchema: The response containing the email of the newly created user.
    """
    try:
        return await create_user(
            session, email=str(body.email), password=body.password
        )
    except UserAlreadyExists:
        response = JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "User with this email already exists."},
        )
        return response
    except UserIsNotActive:
        response = JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "User is not active. Reactivate your account."},
        )
        return response


@router.put("/access-token")
async def refresh_access_token(
    session: AsyncSession = Depends(get_session),
    authorize: AuthJWT = Depends(),
):
    """
    Refresh the access token using a valid refresh token.

    Args:
        session (AsyncSession): Dependency for asynchronous database session.
        authorize (AuthJWT): Dependency for JWT authorization.

    Returns:
        Response: A Starlette response with new JWT cookies set,
                  or a JSONResponse with an error if the user is not found.
    """
    authorize.jwt_refresh_token_required()
    user_id = authorize.get_jwt_subject()

    try:
        await get_user_by_id(session, user_id=user_id)
    except UserNotFound:
        response = JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "User not found"},
        )
        authorize.unset_jwt_cookies(response=response)
        return response

    token, refresh_token = create_tokens(authorize, user_id=user_id)
    return get_response_with_tokens(
        authorize,
        response=Response(status_code=status.HTTP_204_NO_CONTENT),
        token=token,
        refresh_token=refresh_token,
    )
