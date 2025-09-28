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
from src.users.db_utils.exceptions import UserNotFound, PasswordDoesNotMatch
from src.users.db_utils.users import (
    get_user_by_credentials,
    create_user,
    get_user_by_email,
)

router = APIRouter(tags=["v1", "authorization"], prefix="/v1/authorization")


@router.post("/signin")
async def signin(
    body: SignInRequestSchema,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
):
    try:
        user = await get_user_by_credentials(
            session, email=str(body.email), password=body.password
        )
    except (UserNotFound, PasswordDoesNotMatch):
        response = JSONResponse(
            status_code=404, content={"detail": "User not found"}
        )
        authorize.unset_jwt_cookies(response=response)
        return response

    token, refresh_token = create_tokens(authorize, email=user.email)
    return get_response_with_tokens(
        authorize,
        response=Response(status_code=status.HTTP_204_NO_CONTENT),
        token=token,
        refresh_token=refresh_token,
    )


@router.post("/signup", response_model=SignUpResponseSchema)
async def signup(
    body: SignUpRequestSchema,
    session: AsyncSession = Depends(get_session),
):
    return await create_user(
        session, email=str(body.email), password=body.password
    )


@router.put("/access-token")
async def refresh_access_token(
    session: AsyncSession = Depends(get_session),
    authorize: AuthJWT = Depends(),
):
    authorize.jwt_refresh_token_required()
    current_user_email = authorize.get_jwt_subject()

    try:
        await get_user_by_email(session, email=current_user_email)
    except UserNotFound:
        response = JSONResponse(
            status_code=404, content={"detail": "User not found"}
        )
        authorize.unset_jwt_cookies(response=response)
        return response

    token, refresh_token = create_tokens(authorize, email=current_user_email)
    return get_response_with_tokens(
        authorize,
        response=Response(status_code=status.HTTP_204_NO_CONTENT),
        token=token,
        refresh_token=refresh_token,
    )
