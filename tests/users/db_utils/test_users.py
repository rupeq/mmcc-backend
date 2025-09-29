import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy.exc import IntegrityError

from src.users.db_utils.users import (
    get_user_by_email,
    get_user_by_credentials,
    create_user,
    delete_user,
)
from src.users.db_utils.exceptions import (
    UserNotFound,
    PasswordDoesNotMatch,
    UserAlreadyExists,
    UserIsNotActive,
)
from src.users.models.users import Users

TEST_EMAIL = "test@example.com"
TEST_PASSWORD = "password123"
HASHED_PASSWORD = "hashed_password"


def create_mock_user():
    """Helper function to create a mock user object."""
    user = Users()
    user.email = TEST_EMAIL
    user.password_hash = HASHED_PASSWORD
    user.is_active = True
    return user


@pytest.mark.asyncio
async def test_get_user_by_email_found():
    """Test successfully finding an active user by email."""
    mock_session = AsyncMock()
    mock_user = create_mock_user()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_user
    mock_session.execute.return_value = mock_result

    user = await get_user_by_email(mock_session, email=TEST_EMAIL)

    assert user == mock_user
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_user_by_email_not_found():
    """Test that UserNotFound is raised when no user is found."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    with pytest.raises(UserNotFound):
        await get_user_by_email(mock_session, email=TEST_EMAIL)


@pytest.mark.asyncio
@patch("src.users.db_utils.users.get_user_by_email")
@patch("src.users.db_utils.users.verify_password")
async def test_get_user_by_credentials_success(
    mock_verify_password, mock_get_user_by_email
):
    """Test successful user authentication with correct credentials."""
    mock_user = create_mock_user()
    mock_get_user_by_email.return_value = mock_user
    mock_verify_password.return_value = True
    mock_session = AsyncMock()

    user = await get_user_by_credentials(
        mock_session, email=TEST_EMAIL, password=TEST_PASSWORD
    )

    assert user == mock_user
    mock_get_user_by_email.assert_called_once_with(
        mock_session, email=TEST_EMAIL
    )
    mock_verify_password.assert_called_once_with(
        TEST_PASSWORD, password_hash=HASHED_PASSWORD
    )


@pytest.mark.asyncio
@patch("src.users.db_utils.users.get_user_by_email", side_effect=UserNotFound)
async def test_get_user_by_credentials_not_found(_):
    """Test that UserNotFound from get_user_by_email is propagated."""
    mock_session = AsyncMock()
    with pytest.raises(UserNotFound):
        await get_user_by_credentials(
            mock_session, email=TEST_EMAIL, password=TEST_PASSWORD
        )


@pytest.mark.asyncio
@patch("src.users.db_utils.users.get_user_by_email")
@patch("src.users.db_utils.users.verify_password", return_value=False)
async def test_get_user_by_credentials_password_mismatch(
    mock_verify_password, mock_get_user_by_email
):
    """Test that PasswordDoesNotMatch is raised for incorrect passwords."""
    mock_get_user_by_email.return_value = create_mock_user()
    mock_session = AsyncMock()

    with pytest.raises(PasswordDoesNotMatch):
        await get_user_by_credentials(
            mock_session, email=TEST_EMAIL, password="wrongpassword"
        )


@pytest.mark.asyncio
@patch(
    "src.users.db_utils.users.get_user_by_email",
    side_effect=UserNotFound,
)
@patch("src.users.db_utils.users.hash_password", return_value=HASHED_PASSWORD)
async def test_create_user_success(mock_hash_password, mock_get_user_by_email):
    """Test successful user creation."""
    mock_session = AsyncMock()
    mock_session.add = MagicMock()

    user = await create_user(
        mock_session, email=TEST_EMAIL, password=TEST_PASSWORD
    )

    mock_get_user_by_email.assert_called_once_with(
        mock_session, email=TEST_EMAIL
    )
    mock_hash_password.assert_called_once_with(TEST_PASSWORD)
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()
    assert user.email == TEST_EMAIL
    assert user.password_hash == HASHED_PASSWORD
    assert user.is_active is True


@pytest.mark.asyncio
@patch("src.users.db_utils.users.get_user_by_email")
async def test_create_user_already_exists(mock_get_user_by_email):
    """Test that UserAlreadyExists is raised if an active user exists."""
    mock_get_user_by_email.return_value = create_mock_user()
    mock_session = AsyncMock()

    with pytest.raises(UserAlreadyExists):
        await create_user(
            mock_session, email=TEST_EMAIL, password=TEST_PASSWORD
        )


@pytest.mark.asyncio
@patch(
    "src.users.db_utils.users.get_user_by_email",
    side_effect=UserNotFound,
)
@patch("src.users.db_utils.users.hash_password")
async def test_create_user_inactive_user_exists(
    mock_hash_password, mock_get_user_by_email
):
    """Test raising UserIsNotActive when a unique constraint fails."""
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit.side_effect = IntegrityError(None, None, None)  # type:ignore

    with pytest.raises(UserIsNotActive):
        await create_user(
            mock_session, email=TEST_EMAIL, password=TEST_PASSWORD
        )


@pytest.mark.asyncio
async def test_delete_user():
    """Test that a user is marked as inactive."""
    mock_session = AsyncMock()
    await delete_user(mock_session, email=TEST_EMAIL)

    mock_session.execute.assert_called_once()
    mock_session.commit.assert_called_once()
