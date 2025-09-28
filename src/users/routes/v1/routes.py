from another_fastapi_jwt_auth import AuthJWT
from fastapi import Depends, APIRouter
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import JSONResponse
from starlette import status

from src.core.db_session import get_session
from src.users.db_utils.exceptions import UserNotFound
from src.users.db_utils.users import (
    get_user_by_email,
    delete_user,
)
from src.users.routes.v1.schemas import GetMeResponse

router = APIRouter(tags=["v1", "users"], prefix="/v1/users")


@router.get("/me", response_model=GetMeResponse, status_code=status.HTTP_200_OK)
async def get_current_user(
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """
    Retrieve the details of the currently authenticated user.

    Requires a valid JWT access token.

    Args:
        authorize (AuthJWT): Dependency for JWT authorization.
        session (AsyncSession): Dependency for asynchronous database session.

    Returns:
        GetMeResponse | JSONResponse: The user details if found,
            otherwise a JSONResponse with status 404 and an error message.
    """
    authorize.jwt_required()
    current_user_email = authorize.get_jwt_subject()

    try:
        user = await get_user_by_email(session, email=current_user_email)
    except UserNotFound:
        response = JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "User not found"},
        )
        authorize.unset_jwt_cookies(response=response)
        return response

    return user


@router.delete("/me", status_code=status.HTTP_200_OK)
async def delete_current_user(
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """
    Delete the currently authenticated user's account.

    Requires a valid JWT access token.

    Args:
        authorize (AuthJWT): Dependency for JWT authorization.
        session (AsyncSession): Dependency for asynchronous database session.

    Returns:
        JSONResponse: A JSONResponse indicating the user has been deleted.
    """
    authorize.jwt_required()
    current_user_email = authorize.get_jwt_subject()
    await delete_user(session, email=current_user_email)
    response = JSONResponse(content={"detail": "User deleted"})
    authorize.unset_jwt_cookies(response=response)
    return response
