# tests/simulations/worker/test_integration.py
"""Integration tests for end-to-end task flow."""
import uuid
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from src.main import app
from src.simulations.models.enums import ReportStatus


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_celery_task():
    """Mock Celery task for integration testing."""
    with patch("src.simulations.tasks.simulations.run_simulation_task") as mock:
        mock_task_result = MagicMock()
        mock_task_result.id = "test-task-id"
        mock.delay.return_value = mock_task_result
        yield mock


class TestEndToEndTaskFlow:
    """Test complete task flow from API to completion."""

    @patch("src.simulations.routes.v1.routes.create_simulation_configuration")
    def test_api_dispatches_task_successfully(
            self, mock_create, mock_celery_task, client
    ):
        """Test that API successfully dispatches task to Celery."""
        from another_fastapi_jwt_auth import AuthJWT

        # Mock authentication
        mock_auth = MagicMock()
        mock_auth.get_jwt_subject.return_value = str(uuid.uuid4())
        app.dependency_overrides[AuthJWT] = lambda: mock_auth

        # Mock database creation
        config_id = uuid.uuid4()
        report_id = uuid.uuid4()
        mock_create.return_value = (
            MagicMock(id=config_id),
            MagicMock(id=report_id),
        )

        # Make request
        response = client.post(
            "/api/v1/simulations",
            json={
                "name": "Test Simulation",
                "simulationParameters": {
                    "numChannels": 2,
                    "simulationTime": 100.0,
                    "numReplications": 1,
                    "arrivalProcess": {
                        "distribution": "exponential",
                        "rate": 1.0,
                    },
                    "serviceProcess": {
                        "distribution": "exponential",
                        "rate": 2.0,
                    },
                },
            },
        )

        assert response.status_code == 202
        assert mock_celery_task.delay.called

        # Clean up
        app.dependency_overrides = {}

    @pytest.mark.asyncio
    @patch("src.simulations.tasks.simulations.run_replications")
    @patch("src.simulations.tasks.simulations.update_simulation_report_results")
    async def test_task_updates_database_on_completion(
            self, mock_update, mock_run
    ):
        """Test that task updates database with results."""
        from src.simulations.tasks.simulations import SimulationTask

        report_id = str(uuid.uuid4())
        params = {
            "numChannels": 2,
            "simulationTime": 10.0,
            "numReplications": 1,
            "arrivalProcess": {"distribution": "exponential", "rate": 1.0},
            "serviceProcess": {"distribution": "exponential", "rate": 2.0},
        }

        # Mock simulation results
        mock_response = MagicMock()
        mock_response.aggregated_metrics.total_requests = 50
        mock_response.aggregated_metrics.processed_requests = 40
        mock_response.aggregated_metrics.rejected_requests = 10
        mock_response.aggregated_metrics.rejection_probability = 0.2
        mock_response.aggregated_metrics.num_replications = 1
        mock_response.model_dump.return_value = {"test": "results"}
        mock_run.return_value = mock_response

        # Mock session
        mock_session = AsyncMock()
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__.return_value = mock_session
        mock_factory.return_value.__aexit__.return_value = None

        with patch(
                "src.simulations.tasks.simulations.get_worker_session_factory",
                return_value=mock_factory,
        ):
            task = SimulationTask()
            result = await task.execute_simulation(report_id, params)

            # Verify database update was called
            mock_update.assert_called_once()
            call_args = mock_update.call_args.kwargs
            assert call_args["status"] == ReportStatus.COMPLETED
            assert call_args["results"] == {"test": "results"}


class TestConcurrentTaskExecution:
    """Test handling of multiple concurrent tasks."""

    @pytest.mark.asyncio
    async def test_handles_multiple_concurrent_tasks(self):
        """Test that multiple tasks can execute concurrently."""
        from src.simulations.tasks.simulations import SimulationTask

        # Create multiple tasks
        tasks = []
        for i in range(5):
            report_id = str(uuid.uuid4())
            params = {
                "numChannels": 2,
                "simulationTime": 10.0,
                "numReplications": 1,
                "arrivalProcess": {"distribution": "exponential", "rate": 1.0},
                "serviceProcess": {"distribution": "exponential", "rate": 2.0},
            }
            tasks.append((report_id, params))

        # Execute concurrently (mocked)
        with patch(
                "src.simulations.tasks.simulations.run_replications"
        ) as mock_run, patch(
            "src.simulations.tasks.simulations.get_worker_session_factory"
        ) as mock_factory:
            mock_response = MagicMock()
            mock_response.aggregated_metrics.total_requests = 50
            mock_response.aggregated_metrics.processed_requests = 40
            mock_response.aggregated_metrics.rejected_requests = 10
            mock_response.aggregated_metrics.rejection_probability = 0.2
            mock_response.aggregated_metrics.num_replications = 1
            mock_response.model_dump.return_value = {}
            mock_run.return_value = mock_response

            mock_session = AsyncMock()
            mock_factory.return_value.return_value.__aenter__.return_value = mock_session
            mock_factory.return_value.return_value.__aexit__.return_value = None

            task_obj = SimulationTask()
            results = []
            for report_id, params in tasks:
                result = await task_obj.execute_simulation(report_id, params)
                results.append(result)

            # All tasks should complete
            assert len(results) == 5
            assert all(r["processed_requests"] == 40 for r in results)


class TestTaskErrorRecovery:
    """Test error recovery in task execution."""

    @pytest.mark.asyncio
    async def test_recovers_from_transient_errors(self):
        """Test that tasks recover from transient errors."""
        from src.simulations.tasks.simulations import SimulationTask

        report_id = str(uuid.uuid4())
        params = {
            "numChannels": 1,
            "simulationTime": 10.0,
            "numReplications": 1,
            "arrivalProcess": {"distribution": "exponential", "rate": 1.0},
            "serviceProcess": {"distribution": "exponential", "rate": 2.0},
        }

        # First call fails, second succeeds
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("Temporary failure")
            mock_response = MagicMock()
            mock_response.aggregated_metrics.total_requests = 50
            mock_response.aggregated_metrics.processed_requests = 40
            mock_response.aggregated_metrics.rejected_requests = 10
            mock_response.aggregated_metrics.rejection_probability = 0.2
            mock_response.aggregated_metrics.num_replications = 1
            mock_response.model_dump.return_value = {}
            return mock_response

        with patch(
                "src.simulations.tasks.simulations.run_replications",
                side_effect=side_effect,
        ), patch(
            "src.simulations.tasks.simulations.get_worker_session_factory"
        ) as mock_factory:
            mock_session = AsyncMock()
            mock_factory.return_value.return_value.__aenter__.return_value = mock_session
            mock_factory.return_value.return_value.__aexit__.return_value = None

            task = SimulationTask()

            # First attempt fails
            with pytest.raises(ConnectionError):
                await task.execute_simulation(report_id, params)

            # Second attempt succeeds
            result = await task.execute_simulation(report_id, params)
            assert result["processed_requests"] == 40