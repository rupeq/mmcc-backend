# tests/simulations/worker/test_task_execution.py (FIXED)
"""Comprehensive tests for simulation task execution."""
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.simulations.tasks.simulations import (
    run_simulation_task,
    calculate_task_timeout,
    SimulationTask,
)
from src.simulations.models.enums import ReportStatus


@pytest.fixture
def valid_simulation_params():
    """Valid simulation parameters for testing."""
    return {
        "numChannels": 2,
        "simulationTime": 100.0,
        "numReplications": 5,
        "arrivalProcess": {"distribution": "exponential", "rate": 1.0},
        "serviceProcess": {"distribution": "exponential", "rate": 2.0},
        "randomSeed": 42,
    }


@pytest.fixture
def mock_session_factory():
    """Mock session factory that returns a mock session."""
    mock_session = AsyncMock()
    mock_factory = MagicMock()
    mock_factory.return_value.__aenter__.return_value = mock_session
    mock_factory.return_value.__aexit__.return_value = None
    return mock_factory, mock_session


class TestTaskTimeoutCalculation:
    """Test task timeout calculation logic."""

    def test_calculates_reasonable_timeout(self):
        """Test timeout calculation for typical simulation."""
        params = {
            "numChannels": 2,
            "simulationTime": 1000.0,
            "numReplications": 10,
        }
        timeout = calculate_task_timeout(params)

        # 1000 * 10 * 2 / 100 = 200 seconds, but min is 300
        assert timeout == 300  # Enforces minimum

    def test_enforces_minimum_timeout(self):
        """Test minimum timeout of 5 minutes."""
        params = {
            "numChannels": 1,
            "simulationTime": 10.0,
            "numReplications": 1,
        }
        timeout = calculate_task_timeout(params)
        assert timeout == 300

    def test_enforces_maximum_timeout(self):
        """Test maximum timeout of 2 hours."""
        params = {
            "numChannels": 100,
            "simulationTime": 100000.0,
            "numReplications": 1000,
        }
        timeout = calculate_task_timeout(params)
        assert timeout == 7200

    def test_calculates_timeout_above_minimum(self):
        """Test timeout calculation for larger simulations."""
        params = {
            "numChannels": 10,
            "simulationTime": 10000.0,
            "numReplications": 50,
        }
        timeout = calculate_task_timeout(params)
        # 10000 * 50 * 10 / 100 = 50000, clamped to max 7200
        assert timeout == 7200


class TestSimulationTaskExecution:
    """Test SimulationTask.execute_simulation method."""

    @pytest.mark.asyncio
    async def test_successful_simulation_execution(
            self, valid_simulation_params, mock_session_factory
    ):
        """Test successful simulation execution and result storage."""
        mock_factory, mock_session = mock_session_factory
        report_id = str(uuid.uuid4())

        task = SimulationTask()

        with patch(
                "src.simulations.tasks.simulations.get_worker_session_factory",
                return_value=mock_factory,
        ), patch(
            "src.simulations.tasks.simulations.run_replications"
        ) as mock_run, patch(
            "src.simulations.tasks.simulations.update_simulation_report_results"
        ) as mock_update:
            # Mock successful simulation
            mock_response = MagicMock()
            mock_response.aggregated_metrics.total_requests = 100
            mock_response.aggregated_metrics.processed_requests = 80
            mock_response.aggregated_metrics.rejected_requests = 20
            mock_response.aggregated_metrics.rejection_probability = 0.2
            mock_response.aggregated_metrics.num_replications = 5
            mock_response.model_dump.return_value = {"test": "data"}
            mock_run.return_value = mock_response

            result = await task.execute_simulation(
                report_id, valid_simulation_params
            )

            # Verify simulation was executed
            mock_run.assert_called_once()

            # Verify results were stored
            mock_update.assert_called_once()
            call_args = mock_update.call_args
            assert call_args.kwargs["report_id"] == uuid.UUID(report_id)
            assert call_args.kwargs["status"] == ReportStatus.COMPLETED
            assert call_args.kwargs["results"] == {"test": "data"}
            assert "completed_at" in call_args.kwargs

            # Verify return value
            assert result["total_requests"] == 100
            assert result["processed_requests"] == 80
            assert result["rejected_requests"] == 20

    @pytest.mark.asyncio
    async def test_validation_error_handling(
            self, mock_session_factory
    ):
        """Test handling of invalid simulation parameters."""
        mock_factory, mock_session = mock_session_factory
        report_id = str(uuid.uuid4())
        invalid_params = {
            "numChannels": -1,  # Invalid
            "simulationTime": 100.0,
        }

        task = SimulationTask()

        with patch(
                "src.simulations.tasks.simulations.get_worker_session_factory",
                return_value=mock_factory,
        ), patch(
            "src.simulations.tasks.simulations.update_simulation_report_status"
        ) as mock_update:
            result = await task.execute_simulation(
                report_id, invalid_params
            )

            # Should mark as FAILED
            mock_update.assert_called_once()
            call_args = mock_update.call_args
            assert call_args.kwargs["status"] == ReportStatus.FAILED
            assert "Invalid parameters" in call_args.kwargs["error_message"]

            # Should return zero metrics
            assert result["total_requests"] == 0
            assert result["processed_requests"] == 0


class TestRunSimulationTask:
    """Test the Celery task wrapper function."""

    def test_task_is_registered(self):
        """Test that task is properly registered with Celery."""
        from src.simulations.worker import celery_app
        assert "simulations.run_simulation" in celery_app.tasks

    def test_task_configuration(self):
        """Test task configuration settings."""
        task = run_simulation_task

        # Check retry configuration
        assert task.max_retries == 3
        assert task.retry_backoff is True
        assert task.retry_backoff_max == 600
        assert task.retry_jitter is True

        # Check execution settings
        assert task.acks_late is True
        assert task.reject_on_worker_lost is True
        assert task.track_started is True


class TestTaskRetryLogic:
    """Test task retry behavior."""

    def test_task_retry_configuration(self):
        """Test that task has correct retry configuration."""
        task = run_simulation_task

        assert hasattr(task, 'max_retries')
        assert task.max_retries == 3
        assert hasattr(task, 'autoretry_for')
        assert ConnectionError in task.autoretry_for
        assert TimeoutError in task.autoretry_for


class TestTaskLogging:
    """Test task logging behavior."""

    @patch("src.simulations.tasks.simulations.SimulationTask.execute_simulation")
    def test_logs_task_start(self, mock_execute, caplog):
        """Test that task logs start event."""
        import logging
        caplog.set_level(logging.INFO)

        mock_execute.return_value = {
            "total_requests": 0,
            "processed_requests": 0,
            "rejected_requests": 0,
            "rejection_probability": 0.0,
            "num_replications": 1,
        }

        report_id = str(uuid.uuid4())
        params = {"numChannels": 1, "simulationTime": 10.0}

        run_simulation_task(report_id, params)

        assert any(
            "Starting simulation task" in record.message
            for record in caplog.records
        )

    @patch("src.simulations.tasks.simulations.SimulationTask.execute_simulation")
    def test_logs_task_completion(self, mock_execute, caplog):
        """Test that task logs completion event."""
        import logging
        caplog.set_level(logging.INFO)

        mock_execute.return_value = {
            "total_requests": 100,
            "processed_requests": 80,
            "rejected_requests": 20,
            "rejection_probability": 0.2,
            "num_replications": 1,
        }

        report_id = str(uuid.uuid4())
        params = {"numChannels": 1, "simulationTime": 10.0}

        run_simulation_task(report_id, params)

        assert any(
            "Simulation task completed" in record.message
            for record in caplog.records
        )


class TestDeadLetterQueue:
    """Test DLQ handler integration."""

    def test_dlq_task_exists(self):
        """Test that DLQ task module exists."""
        from src.simulations.tasks import dlq
        assert hasattr(dlq, 'dead_letter_handler')