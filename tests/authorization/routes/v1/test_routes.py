import uuid

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from another_fastapi_jwt_auth import AuthJWT
from src.main import app
from src.core.db_session import get_session
from src.users.db_utils.exceptions import (
    UserNotFound,
    PasswordDoesNotMatch,
    UserAlreadyExists,
    UserIsNotActive,
)
from src.users.models.users import User

BASE_URL = "/api/v1/authorization"
TEST_EMAIL = "test@example.com"
TEST_ID = uuid.uuid4()
TEST_PASSWORD = "password123"
ACCESS_TOKEN = "fake_access_token"
REFRESH_TOKEN = "fake_refresh_token"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_user():
    user = User()
    user.email = TEST_EMAIL
    user.id = TEST_ID
    return user


@pytest.fixture
def mock_auth_dependency():
    mock_authorize = MagicMock()
    mock_authorize.get_jwt_subject.return_value = str(TEST_ID)
    mock_authorize.jwt_required.return_value = None
    mock_authorize.jwt_refresh_token_required.return_value = None
    mock_authorize.unset_jwt_cookies.return_value = None

    def get_mock_authorize():
        return mock_authorize

    app.dependency_overrides[AuthJWT] = get_mock_authorize
    yield mock_authorize
    app.dependency_overrides = {}


@pytest.fixture(autouse=True)
def override_db_dependency():
    mock_session = MagicMock()
    app.dependency_overrides[get_session] = lambda: mock_session
    yield mock_session
    app.dependency_overrides = {}


def test_signin_success(client, mocker, mock_user, mock_auth_dependency):
    """Test successful sign-in with correct credentials."""
    mocker.patch(
        "src.authorization.routes.v1.routes.get_user_by_credentials",
        return_value=mock_user,
    )
    mock_create_tokens = mocker.patch(
        "src.authorization.routes.v1.routes.create_tokens",
        return_value=(ACCESS_TOKEN, REFRESH_TOKEN),
    )

    response = client.post(
        f"{BASE_URL}/signin",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )

    assert response.status_code == status.HTTP_204_NO_CONTENT
    mock_create_tokens.assert_called_once_with(
        mock_auth_dependency, user_id=str(mock_user.id)
    )


@pytest.mark.parametrize("exception", [UserNotFound, PasswordDoesNotMatch])
def test_signin_failure(client, mocker, mock_auth_dependency, exception):
    """Test failed sign-in due to user not found or wrong password."""
    mocker.patch(
        "src.authorization.routes.v1.routes.get_user_by_credentials",
        side_effect=exception,
    )

    response = client.post(
        f"{BASE_URL}/signin",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == {"detail": "User not found"}
    mock_auth_dependency.unset_jwt_cookies.assert_called_once()


def test_signup_success(client, mocker, mock_user):
    """Test successful user registration."""
    mock_create_user = mocker.patch(
        "src.authorization.routes.v1.routes.create_user",
        return_value=mock_user,
    )

    response = client.post(
        f"{BASE_URL}/signup",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json() == {"email": TEST_EMAIL}
    mock_create_user.assert_called_once()


@pytest.mark.parametrize(
    "exception, expected_status, expected_detail",
    [
        (
            UserAlreadyExists,
            status.HTTP_400_BAD_REQUEST,
            "User with this email already exists.",
        ),
        (
            UserIsNotActive,
            status.HTTP_400_BAD_REQUEST,
            "User is not active. Reactivate your account.",
        ),
    ],
)
def test_signup_failure(
    client, mocker, exception, expected_status, expected_detail
):
    """Test failed sign-up due to existing user or other db issues."""
    mocker.patch(
        "src.authorization.routes.v1.routes.create_user",
        side_effect=exception,
    )

    response = client.post(
        f"{BASE_URL}/signup",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}


def test_refresh_access_token_success(
    client, mocker, mock_user, mock_auth_dependency
):
    """Test successful refresh of an access token."""
    mocker.patch(
        "src.authorization.routes.v1.routes.get_user_by_id",
        return_value=mock_user,
    )
    mock_create_tokens = mocker.patch(
        "src.authorization.routes.v1.routes.create_tokens",
        return_value=(ACCESS_TOKEN, REFRESH_TOKEN),
    )

    response = client.put(f"{BASE_URL}/access-token")

    assert response.status_code == status.HTTP_204_NO_CONTENT
    mock_auth_dependency.jwt_refresh_token_required.assert_called_once()
    mock_create_tokens.assert_called_once_with(
        mock_auth_dependency, user_id=str(TEST_ID)
    )


def test_refresh_access_token_user_not_found(
    client, mocker, mock_auth_dependency
):
    """Test token refresh failure when the user no longer exists."""
    mocker.patch(
        "src.authorization.routes.v1.routes.get_user_by_id",
        side_effect=UserNotFound,
    )

    response = client.put(f"{BASE_URL}/access-token")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == {"detail": "User not found"}
    mock_auth_dependency.unset_jwt_cookies.assert_called_once()
