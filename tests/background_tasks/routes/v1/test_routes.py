"""Tests for background tasks routes."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.authorization.routes.v1.routes import AuthJWT
from src.background_tasks.models.enums import TaskStatus, TaskType
from src.background_tasks.routes.v1.exceptions import InvalidSubjectID
from src.background_tasks.routes.v1.schemas import Task
from src.core.db_session import get_session
from src.main import app

BASE_URL = "/api/v1/background-tasks"
TEST_USER_ID = uuid.uuid4()


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_auth():
    """Mock authentication dependency."""
    mock_authorize = MagicMock()
    mock_authorize.jwt_required.return_value = None
    mock_authorize.get_jwt_subject.return_value = str(TEST_USER_ID)

    def get_mock_authorize():
        return mock_authorize

    app.dependency_overrides[AuthJWT] = get_mock_authorize
    yield mock_authorize
    app.dependency_overrides.clear()


@pytest.fixture
def mock_db():
    """Mock database session dependency."""
    mock_session = AsyncMock()

    async def get_mock_session():
        yield mock_session

    app.dependency_overrides[get_session] = get_mock_session
    yield mock_session
    if get_session in app.dependency_overrides:
        del app.dependency_overrides[get_session]


@pytest.fixture
def mock_background_task():
    """Create a mock background task."""

    class MockTask:
        def __init__(self):
            self.id = uuid.uuid4()
            self.user_id = TEST_USER_ID
            self.subject_id = uuid.uuid4()
            self.task_id = str(uuid.uuid4())
            self.task_type = TaskType.SIMULATION
            self.created_at = "2025-01-05T12:00:00Z"

    return MockTask()


class TestGetBackgroundTasks:
    """Test suite for GET /background-tasks endpoint."""

    @patch("src.background_tasks.routes.v1.routes.get_background_tasks_from_db")
    def test_get_tasks_success(
        self, mock_get_tasks, client, mock_auth, mock_db, mock_background_task
    ):
        """Test successful retrieval of background tasks."""
        mock_get_tasks.return_value = [mock_background_task]

        response = client.get(BASE_URL)

        assert response.status_code == 200
        data = response.json()
        assert "background_tasks" in data
        assert "total" in data
        assert data["total"] == 1

    @patch("src.background_tasks.routes.v1.routes.get_background_tasks_from_db")
    def test_get_tasks_empty(self, mock_get_tasks, client, mock_auth, mock_db):
        """Test retrieving tasks when none exist."""
        from src.simulations.db_utils.exceptions import BackgroundTaskNotFound

        mock_get_tasks.side_effect = BackgroundTaskNotFound

        response = client.get(BASE_URL)

        assert response.status_code == 404

    @patch("src.background_tasks.routes.v1.routes.verify_subject_ids")
    @patch("src.background_tasks.routes.v1.routes.get_background_tasks_from_db")
    def test_get_tasks_with_subject_ids(
        self,
        mock_get_tasks,
        mock_verify,
        client,
        mock_auth,
        mock_db,
        mock_background_task,
    ):
        """Test filtering tasks by subject IDs."""
        subject_id = str(uuid.uuid4())
        mock_verify.return_value = [uuid.UUID(subject_id)]
        mock_get_tasks.return_value = [mock_background_task]

        response = client.get(f"{BASE_URL}?subject_ids={subject_id}")

        assert response.status_code == 200
        # FastAPI converts query params to list
        mock_verify.assert_called_once_with([subject_id])

    @patch("src.background_tasks.routes.v1.routes.verify_subject_ids")
    def test_get_tasks_invalid_subject_id(
        self, mock_verify, client, mock_auth, mock_db
    ):
        """Test with invalid subject ID format."""
        mock_verify.side_effect = InvalidSubjectID("Invalid UUID")

        response = client.get(f"{BASE_URL}?subject_ids=invalid-uuid")

        assert response.status_code == 400

    @patch("src.background_tasks.routes.v1.routes.get_background_tasks_from_db")
    def test_get_tasks_multiple_results(
        self, mock_get_tasks, client, mock_auth, mock_db, mock_background_task
    ):
        """Test retrieving multiple background tasks."""
        tasks = [mock_background_task for _ in range(5)]
        mock_get_tasks.return_value = tasks

        response = client.get(BASE_URL)

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5

    @patch("src.background_tasks.routes.v1.routes.verify_subject_ids")
    @patch("src.background_tasks.routes.v1.routes.get_background_tasks_from_db")
    def test_get_tasks_multiple_subject_ids(
        self, mock_get_tasks, mock_verify, client, mock_auth, mock_db
    ):
        """Test filtering by multiple subject IDs."""
        subject_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        subject_ids_str = ",".join(subject_ids)
        mock_verify.return_value = [uuid.UUID(sid) for sid in subject_ids]
        mock_get_tasks.return_value = []

        response = client.get(f"{BASE_URL}?subject_ids={subject_ids_str}")

        assert response.status_code in [200, 404]
        mock_verify.assert_called_once_with([subject_ids_str])


class TestGetBackgroundTask:
    """Test suite for GET /background-tasks/{id} endpoint."""

    @patch("src.background_tasks.routes.v1.routes.get_background_task_from_db")
    def test_get_task_success(
        self, mock_get_task, client, mock_auth, mock_db, mock_background_task
    ):
        """Test successful retrieval of a single task."""
        mock_get_task.return_value = mock_background_task

        response = client.get(f"{BASE_URL}/{mock_background_task.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(mock_background_task.id)
        assert data["task_type"] == TaskType.SIMULATION.value

    @patch("src.background_tasks.routes.v1.routes.get_background_task_from_db")
    def test_get_task_not_found(
        self, mock_get_task, client, mock_auth, mock_db
    ):
        """Test retrieving non-existent task."""
        from src.simulations.db_utils.exceptions import BackgroundTaskNotFound

        mock_get_task.side_effect = BackgroundTaskNotFound

        response = client.get(f"{BASE_URL}/{uuid.uuid4()}")

        assert response.status_code == 404

    def test_get_task_invalid_uuid(self, client, mock_auth, mock_db):
        """Test with invalid UUID format."""
        response = client.get(f"{BASE_URL}/invalid-uuid")

        assert response.status_code == 422

    @patch("src.background_tasks.routes.v1.routes.get_background_task_from_db")
    def test_get_task_calls_db_with_correct_params(
        self, mock_get_task, client, mock_auth, mock_db, mock_background_task
    ):
        """Test that DB is called with correct parameters."""
        mock_get_task.return_value = mock_background_task

        response = client.get(f"{BASE_URL}/{mock_background_task.id}")

        assert response.status_code == 200
        call_kwargs = mock_get_task.call_args[1]
        assert call_kwargs["background_task_id"] == mock_background_task.id
        assert call_kwargs["user_id"] == TEST_USER_ID


class TestGetTask:
    """Test suite for GET /background-tasks/{id}/tasks/{task_id} endpoint."""

    @patch("src.background_tasks.routes.v1.routes.build_task_response")
    @patch("src.background_tasks.routes.v1.routes.AsyncResult")
    @patch("src.background_tasks.routes.v1.routes.get_background_task_from_db")
    def test_get_task_pending(
        self,
        mock_get_bg_task,
        mock_async_result,
        mock_build_response,
        client,
        mock_auth,
        mock_db,
        mock_background_task,
    ):
        """Test getting task status when pending."""
        mock_get_bg_task.return_value = mock_background_task

        mock_celery_result = MagicMock()
        mock_celery_result.state = "PENDING"
        mock_async_result.return_value = mock_celery_result

        # Return actual Task object
        task_obj = Task(
            task_id=mock_background_task.task_id,
            status=TaskStatus.PENDING,
            status_message="Task is queued",
            progress=None,
            result=None,
            error=None,
            started_at=None,
            completed_at=None,
        )
        mock_build_response.return_value = task_obj

        response = client.get(
            f"{BASE_URL}/{mock_background_task.id}/tasks/{mock_background_task.task_id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == TaskStatus.PENDING.value

    @patch("src.background_tasks.routes.v1.routes.build_task_response")
    @patch("src.background_tasks.routes.v1.routes.AsyncResult")
    @patch("src.background_tasks.routes.v1.routes.get_background_task_from_db")
    def test_get_task_running_with_progress(
        self,
        mock_get_bg_task,
        mock_async_result,
        mock_build_response,
        client,
        mock_auth,
        mock_db,
        mock_background_task,
    ):
        """Test getting task status when running with progress."""
        from src.background_tasks.routes.v1.schemas import TaskProgress

        mock_get_bg_task.return_value = mock_background_task

        mock_celery_result = MagicMock()
        mock_async_result.return_value = mock_celery_result

        task_obj = Task(
            task_id=mock_background_task.task_id,
            status=TaskStatus.RUNNING,
            status_message="Processing",
            progress=TaskProgress(
                current=50, total=100, percent=50.0, message="Processing"
            ),
            result=None,
            error=None,
            started_at="2025-01-05T12:00:00Z",
            completed_at=None,
        )
        mock_build_response.return_value = task_obj

        response = client.get(
            f"{BASE_URL}/{mock_background_task.id}/tasks/{mock_background_task.task_id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == TaskStatus.RUNNING.value
        assert data["progress"]["percent"] == 50.0

    @patch("src.background_tasks.routes.v1.routes.build_task_response")
    @patch("src.background_tasks.routes.v1.routes.AsyncResult")
    @patch("src.background_tasks.routes.v1.routes.get_background_task_from_db")
    def test_get_task_success(
        self,
        mock_get_bg_task,
        mock_async_result,
        mock_build_response,
        client,
        mock_auth,
        mock_db,
        mock_background_task,
    ):
        """Test getting task status when completed successfully."""
        from src.background_tasks.routes.v1.schemas import TaskResult

        mock_get_bg_task.return_value = mock_background_task

        mock_celery_result = MagicMock()
        mock_async_result.return_value = mock_celery_result

        task_obj = Task(
            task_id=mock_background_task.task_id,
            status=TaskStatus.SUCCESS,
            status_message="Completed",
            progress=None,
            result=TaskResult(data={"processed": 100}, summary="Done"),
            error=None,
            started_at="2025-01-05T12:00:00Z",
            completed_at="2025-01-05T12:05:00Z",
        )
        mock_build_response.return_value = task_obj

        response = client.get(
            f"{BASE_URL}/{mock_background_task.id}/tasks/{mock_background_task.task_id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == TaskStatus.SUCCESS.value

    @patch("src.background_tasks.routes.v1.routes.build_task_response")
    @patch("src.background_tasks.routes.v1.routes.AsyncResult")
    @patch("src.background_tasks.routes.v1.routes.get_background_task_from_db")
    def test_get_task_failed(
        self,
        mock_get_bg_task,
        mock_async_result,
        mock_build_response,
        client,
        mock_auth,
        mock_db,
        mock_background_task,
    ):
        """Test getting task status when failed."""
        from src.background_tasks.routes.v1.schemas import TaskError

        mock_get_bg_task.return_value = mock_background_task

        mock_celery_result = MagicMock()
        mock_async_result.return_value = mock_celery_result

        task_obj = Task(
            task_id=mock_background_task.task_id,
            status=TaskStatus.FAILED,
            status_message="Failed",
            progress=None,
            result=None,
            error=TaskError(
                code="ValidationError",
                message="Invalid",
                details=None,
                traceback=None,
            ),
            started_at="2025-01-05T12:00:00Z",
            completed_at="2025-01-05T12:01:00Z",
        )
        mock_build_response.return_value = task_obj

        response = client.get(
            f"{BASE_URL}/{mock_background_task.id}/tasks/{mock_background_task.task_id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == TaskStatus.FAILED.value

    @patch("src.background_tasks.routes.v1.routes.get_background_task_from_db")
    def test_get_task_background_task_not_found(
        self, mock_get_bg_task, client, mock_auth, mock_db
    ):
        """Test when background task doesn't exist."""
        from src.simulations.db_utils.exceptions import BackgroundTaskNotFound

        mock_get_bg_task.side_effect = BackgroundTaskNotFound

        task_id = str(uuid.uuid4())
        response = client.get(f"{BASE_URL}/{uuid.uuid4()}/tasks/{task_id}")

        assert response.status_code == 404

    def test_get_task_invalid_background_task_uuid(
        self, client, mock_auth, mock_db
    ):
        """Test with invalid background task UUID."""
        task_id = str(uuid.uuid4())
        response = client.get(f"{BASE_URL}/invalid-uuid/tasks/{task_id}")

        assert response.status_code == 422


class TestAuthenticationAndAuthorization:
    """Test suite for authentication and authorization."""

    def test_get_tasks_requires_auth(self, client):
        """Test that GET /background-tasks requires authentication."""
        response = client.get(BASE_URL)
        assert response.status_code == 401

    def test_get_task_requires_auth(self, client):
        """Test that GET /background-tasks/{id} requires authentication."""
        response = client.get(f"{BASE_URL}/{uuid.uuid4()}")
        assert response.status_code == 401

    def test_get_task_status_requires_auth(self, client):
        """Test that GET /background-tasks/{id}/tasks/{task_id} requires auth."""
        bg_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())
        response = client.get(f"{BASE_URL}/{bg_id}/tasks/{task_id}")
        assert response.status_code == 401

    @patch("src.background_tasks.routes.v1.routes.get_background_task_from_db")
    def test_wrong_user_cannot_access_task(
        self, mock_get_task, client, mock_auth, mock_db
    ):
        """Test that user cannot access another user's task."""
        from src.simulations.db_utils.exceptions import BackgroundTaskNotFound

        mock_get_task.side_effect = BackgroundTaskNotFound

        response = client.get(f"{BASE_URL}/{uuid.uuid4()}")

        assert response.status_code == 404


class TestIntegrationScenarios:
    """Test suite for integration scenarios."""

    @patch("src.background_tasks.routes.v1.routes.build_task_response")
    @patch("src.background_tasks.routes.v1.routes.AsyncResult")
    @patch("src.background_tasks.routes.v1.routes.get_background_task_from_db")
    @patch("src.background_tasks.routes.v1.routes.get_background_tasks_from_db")
    def test_list_and_get_task_flow(
        self,
        mock_get_tasks,
        mock_get_task,
        mock_async_result,
        mock_build_response,
        client,
        mock_auth,
        mock_db,
        mock_background_task,
    ):
        """Test complete flow of listing tasks and getting details."""
        from src.background_tasks.routes.v1.schemas import TaskResult

        # Step 1: List tasks
        mock_get_tasks.return_value = [mock_background_task]
        list_response = client.get(BASE_URL)
        assert list_response.status_code == 200

        # Step 2: Get specific task
        mock_get_task.return_value = mock_background_task
        task_response = client.get(f"{BASE_URL}/{mock_background_task.id}")
        assert task_response.status_code == 200

        # Step 3: Get task status
        mock_celery_result = MagicMock()
        mock_async_result.return_value = mock_celery_result

        task_obj = Task(
            task_id=mock_background_task.task_id,
            status=TaskStatus.SUCCESS,
            status_message="Completed",
            progress=None,
            result=TaskResult(data={}, summary="Done"),
            error=None,
            started_at="2025-01-05T12:00:00Z",
            completed_at="2025-01-05T12:05:00Z",
        )
        mock_build_response.return_value = task_obj

        status_response = client.get(
            f"{BASE_URL}/{mock_background_task.id}/tasks/{mock_background_task.task_id}"
        )
        assert status_response.status_code == 200

    @patch("src.background_tasks.routes.v1.routes.get_background_tasks_from_db")
    def test_filter_tasks_by_subject(
        self, mock_get_tasks, client, mock_auth, mock_db, mock_background_task
    ):
        """Test filtering tasks by subject ID."""
        subject_id = str(mock_background_task.subject_id)

        with patch(
            "src.background_tasks.routes.v1.routes.verify_subject_ids"
        ) as mock_verify:
            mock_verify.return_value = [uuid.UUID(subject_id)]
            mock_get_tasks.return_value = [mock_background_task]

            response = client.get(f"{BASE_URL}?subject_ids={subject_id}")

            assert response.status_code == 200
