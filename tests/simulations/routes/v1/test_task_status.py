"""Tests for task status endpoint."""
import uuid
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from another_fastapi_jwt_auth import AuthJWT

from src.main import app

TEST_USER_ID = uuid.uuid4()
BASE_URL = "/api/v1/simulations"


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def setup_mocks(mocker):
    """Set up common mocks for authentication."""
    mock_authorize = MagicMock()
    mock_authorize.get_jwt_subject.return_value = str(TEST_USER_ID)
    mocker.patch.object(AuthJWT, "jwt_required", return_value=None)
    app.dependency_overrides[AuthJWT] = lambda: mock_authorize

    yield {"authorize": mock_authorize}
    app.dependency_overrides = {}


class TestGetTaskStatus:
    """Test suite for GET /tasks/{task_id}/status endpoint."""

    @patch("src.simulations.routes.v1.routes.AsyncResult")
    def test_get_pending_task_status(self, mock_async_result, client):
        """Test retrieving status of a pending task."""
        mock_result = MagicMock()
        mock_result.state = "PENDING"
        mock_result.info = None
        mock_async_result.return_value = mock_result

        task_id = "test-task-123"
        response = client.get(f"{BASE_URL}/tasks/{task_id}/status")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["task_id"] == task_id
        assert data["state"] == "PENDING"
        assert "queued" in data["status"].lower()

    @patch("src.simulations.routes.v1.routes.AsyncResult")
    def test_get_started_task_status(self, mock_async_result, client):
        """Test retrieving status of a started task."""
        mock_result = MagicMock()
        mock_result.state = "STARTED"
        mock_result.info = {"current": 50, "total": 100}
        mock_async_result.return_value = mock_result

        task_id = "test-task-456"
        response = client.get(f"{BASE_URL}/tasks/{task_id}/status")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["state"] == "STARTED"
        assert "running" in data["status"].lower()
        assert data["progress"] == {"current": 50, "total": 100}

    @patch("src.simulations.routes.v1.routes.AsyncResult")
    def test_get_success_task_status(self, mock_async_result, client):
        """Test retrieving status of a successful task."""
        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_result.result = {
            "status": "success",
            "metrics": {
                "processed_requests": 100,
                "rejected_requests": 20,
            }
        }
        mock_async_result.return_value = mock_result

        task_id = "test-task-789"
        response = client.get(f"{BASE_URL}/tasks/{task_id}/status")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["state"] == "SUCCESS"
        assert "completed" in data["status"].lower()
        assert data["result"] is not None
        assert data["result"]["metrics"]["processed_requests"] == 100

    @patch("src.simulations.routes.v1.routes.AsyncResult")
    def test_get_failure_task_status(self, mock_async_result, client):
        """Test retrieving status of a failed task."""
        mock_result = MagicMock()
        mock_result.state = "FAILURE"
        mock_result.info = Exception("Simulation crashed")
        mock_async_result.return_value = mock_result

        task_id = "test-task-fail"
        response = client.get(f"{BASE_URL}/tasks/{task_id}/status")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["state"] == "FAILURE"
        assert "failed" in data["status"].lower()
        assert "Simulation crashed" in data["error"]

    @patch("src.simulations.routes.v1.routes.AsyncResult")
    def test_get_retry_task_status(self, mock_async_result, client):
        """Test retrieving status of a retrying task."""
        mock_result = MagicMock()
        mock_result.state = "RETRY"
        mock_result.info = {"exc": "Temporary failure"}
        mock_async_result.return_value = mock_result

        task_id = "test-task-retry"
        response = client.get(f"{BASE_URL}/tasks/{task_id}/status")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["state"] == "RETRY"
        assert "retried" in data["status"].lower()
        assert "Temporary failure" in data["error"]

    @patch("src.simulations.routes.v1.routes.AsyncResult")
    def test_get_revoked_task_status(self, mock_async_result, client):
        """Test retrieving status of a cancelled task."""
        mock_result = MagicMock()
        mock_result.state = "REVOKED"
        mock_result.info = None
        mock_async_result.return_value = mock_result

        task_id = "test-task-cancel"
        response = client.get(f"{BASE_URL}/tasks/{task_id}/status")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["state"] == "REVOKED"
        assert "cancelled" in data["status"].lower()

    def test_requires_authentication(self, client):
        """Test that endpoint requires authentication."""
        app.dependency_overrides = {}

        from another_fastapi_jwt_auth.exceptions import JWTDecodeError
        with patch.object(
                AuthJWT,
                "jwt_required",
                side_effect=JWTDecodeError(
                    status.HTTP_401_UNAUTHORIZED, "Unauthorized"
                ),
        ):
            response = client.get(f"{BASE_URL}/tasks/test-task/status")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
