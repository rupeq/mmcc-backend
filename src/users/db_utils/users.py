import logging

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import verify_password, hash_password
from src.users.db_utils.exceptions import (
    PasswordDoesNotMatch,
    UserNotFound,
    UserAlreadyExists,
    UserIsNotActive,
)
from src.users.models.users import User


logger = logging.getLogger(__name__)


async def get_user_by_email(session: AsyncSession, *, email: str) -> User:
    """
    Retrieve a user from the database by their email address.

    Args:
        session (AsyncSession): The asynchronous database session.
        email (str): The email address of the user to retrieve.

    Returns:
        User: The user object if found and active.

    Raises:
        UserNotFound: If no active user with the given email is found.
    """
    logger.debug("Trying to get user by email: %s", email)
    maybe_user = (
        await session.execute(
            select(User).where(User.email == email, User.is_active == True)
        )
    ).scalar_one_or_none()

    if maybe_user is None:
        logger.debug("User not found (email: %s)", email)
        raise UserNotFound()

    logger.debug("Found user by email: %s", email)
    return maybe_user


async def get_user_by_credentials(
    session: AsyncSession, *, email: str, password: str
) -> User:
    """
    Retrieve and verify a user based on their email and password.

    Args:
        session (AsyncSession): The asynchronous database session.
        email (str): The email address of the user.
        password (str): The plaintext password to verify.

    Returns:
        User: The user object if credentials are valid.

    Raises:
        UserNotFound: If no active user with the given email is found.
        PasswordDoesNotMatch: If the provided password does not match the stored hash.
    """
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
) -> User:
    """
    Create a new user in the database.

    Args:
        session (AsyncSession): The asynchronous database session.
        email (str): The email address for the new user.
        password (str): The plaintext password for the new user.

    Returns:
        User: The newly created user object.

    Raises:
        UserAlreadyExists: If a user with the given email already exists.
        UserIsNotActive: If a user exists with a is_active=False.
    """
    logger.debug("Trying to create user (email: %s)", email)

    try:
        await get_user_by_email(session, email=email)
        logger.debug("Found user by email (email: %s), creation failed.", email)
        raise UserAlreadyExists()
    except UserNotFound:
        pass

    user = User(
        email=email,
        password_hash=hash_password(password),
        is_active=True,
    )
    session.add(user)

    try:
        await session.commit()
    except IntegrityError:
        logger.debug("Found inactive user (email: %s)", email)
        raise UserIsNotActive()

    logger.debug("Created user by email (email: %s).", email)
    return user


async def delete_user(session: AsyncSession, *, email: str) -> None:
    """
    Delete a user by setting their 'is_active' status to False.

    Args:
        session (AsyncSession): The asynchronous database session.
        email (str): The email address of the user to delete.
    """
    logger.debug("Trying to delete user (email: %s)", email)
    await session.execute(
        update(User).where(User.email == email).values(is_active=False)
    )
    await session.commit()
    logger.debug("Deleted user by email (email: %s).", email)
