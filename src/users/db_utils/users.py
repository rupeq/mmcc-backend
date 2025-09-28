import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import verify_password, hash_password
from src.users.db_utils.exceptions import (
    PasswordDoesNotMatch,
    UserNotFound,
    UserAlreadyExists,
)
from src.users.models.users import Users


logger = logging.getLogger(__name__)


async def get_user_by_email(session: AsyncSession, *, email: str) -> Users:
    logger.debug("Trying to get user by email: %s", email)
    maybe_user = (
        await session.execute(
            select(Users).where(Users.email == email, Users.is_active == True)
        )
    ).scalar_one_or_none()

    if maybe_user is None:
        logger.debug("User not found (email: %s)", email)
        raise UserNotFound()

    logger.debug("Found user by email: %s", email)
    return maybe_user


async def get_user_by_credentials(
    session: AsyncSession, *, email: str, password: str
) -> Users:
    logger.debug("Trying to verify user by credentials (email: %s)", email)

    user = await get_user_by_email(session, email=email)

    if not verify_password(password, password_hash=user.password_hash):
        logger.debug(
            "Found user by email (email: %s), but unable to verify them.", email
        )
        raise PasswordDoesNotMatch()

    logger.debug("Verified user using credentials (email: %s)", email)
    return user


async def create_user(
    session: AsyncSession, *, email: str, password: str
) -> Users:
    logger.debug("Trying to create user (email: %s)", email)

    try:
        await get_user_by_email(session, email=email)
        logger.debug("Found user by email (email: %s), creation failed.", email)
        raise UserAlreadyExists()
    except UserNotFound:
        user = Users(
            email=email,
            password_hash=hash_password(password),
            is_active=True,
        )
        session.add(user)
        await session.commit()
        logger.debug("Created user by email (email: %s).", email)
        return user
