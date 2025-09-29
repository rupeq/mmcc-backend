import uuid
import datetime
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from another_fastapi_jwt_auth import AuthJWT

from src.main import app
from src.core.db_session import get_session
from src.users.db_utils.exceptions import UserNotFound
from src.users.models.users import User

BASE_URL = "/api/v1/users"
TEST_EMAIL = "test@example.com"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_user():
    user = User()
    user.id = uuid.uuid4()
    user.email = TEST_EMAIL
    user.is_active = True
    user.created_at = datetime.datetime.now(datetime.timezone.utc)
    user.updated_at = datetime.datetime.now(datetime.timezone.utc)
    return user


@pytest.fixture
def mock_auth_dependency():
    mock_authorize = MagicMock()
    mock_authorize.get_jwt_subject.return_value = TEST_EMAIL
    mock_authorize.jwt_required.return_value = None
    mock_authorize.unset_jwt_cookies.return_value = None

    def get_mock_authorize():
        return mock_authorize

    app.dependency_overrides[AuthJWT] = get_mock_authorize

    yield mock_authorize

    app.dependency_overrides = {}


@pytest.fixture(autouse=True)
def override_db_dependency(mocker):
    """Separate fixture for the database session to keep things clean."""
    mock_session = MagicMock()
    app.dependency_overrides[get_session] = lambda: mock_session
    yield mock_session
    app.dependency_overrides = {}


def test_get_current_user_success(
    client, mocker, mock_user, mock_auth_dependency
):
    """
    Test the GET /me endpoint for a successful scenario where the user is found.
    """
    mock_get_user = mocker.patch(
        "src.users.routes.v1.routes.get_user_by_email",
        return_value=mock_user,
    )

    response = client.get(f"{BASE_URL}/me")

    assert response.status_code == 200
    mock_auth_dependency.jwt_required.assert_called_once()
    mock_get_user.assert_called_once_with(mocker.ANY, email=TEST_EMAIL)
    data = response.json()
    assert data["email"] == mock_user.email
    assert data["id"] == str(mock_user.id)
    assert data["is_active"] is True


def test_get_current_user_not_found(client, mocker, mock_auth_dependency):
    """
    Test the GET /me endpoint for a scenario where the user does not exist in the db.
    """
    mocker.patch(
        "src.users.routes.v1.routes.get_user_by_email",
        side_effect=UserNotFound,
    )

    response = client.get(f"{BASE_URL}/me")

    assert response.status_code == 404
    assert response.json() == {"detail": "User not found"}
    mock_auth_dependency.unset_jwt_cookies.assert_called_once()


def test_delete_current_user_success(client, mocker, mock_auth_dependency):
    """
    Test the DELETE /me endpoint for a successful user deletion.
    """
    mock_delete_user = mocker.patch("src.users.routes.v1.routes.delete_user")

    response = client.delete(f"{BASE_URL}/me")
    assert response.status_code == 200
    assert response.json() == {"detail": "User deleted"}
    mock_delete_user.assert_called_once_with(mocker.ANY, email=TEST_EMAIL)
    mock_auth_dependency.unset_jwt_cookies.assert_called_once()
