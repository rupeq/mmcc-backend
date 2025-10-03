"""Comprehensive tests for worker lifecycle management."""

from unittest.mock import patch, MagicMock, AsyncMock

from src.simulations.worker import (
    on_worker_ready,
    on_worker_shutdown,
    on_worker_process_init,
    on_task_failure,
    on_task_success,
    on_task_retry,
    on_task_prerun,
    on_task_postrun,
    check_worker_health,
    _TASK_START_TIME,
)


class TestWorkerStartup:
    """Test worker startup behavior."""

    @patch("src.simulations.worker.get_worker_engine")
    @patch("src.simulations.worker.get_worker_session_factory")
    def test_initializes_database_on_ready(
        self, mock_session_factory, mock_engine
    ):
        """Test database initialization on worker ready."""
        mock_engine_instance = MagicMock()
        mock_engine_instance.pool.size.return_value = 10
        mock_engine_instance.pool._max_overflow = 5
        mock_engine.return_value = mock_engine_instance

        on_worker_ready()

        mock_engine.assert_called_once()
        mock_session_factory.assert_called_once()

    @patch("src.simulations.worker.get_worker_engine")
    def test_logs_worker_configuration(self, mock_engine, caplog):
        """Test that worker configuration is logged on startup."""
        import logging

        caplog.set_level(logging.INFO)

        mock_engine_instance = MagicMock()
        mock_engine_instance.pool.size.return_value = 10
        mock_engine_instance.pool._max_overflow = 5
        mock_engine.return_value = mock_engine_instance

        on_worker_ready()

        assert any(
            "Worker ready" in record.message for record in caplog.records
        )


class TestWorkerShutdown:
    """Test worker shutdown behavior."""

    @patch("src.simulations.worker.dispose_worker_engine")
    def test_disposes_database_on_shutdown(self, mock_dispose):
        """Test database disposal on worker shutdown."""
        mock_dispose.return_value = AsyncMock()

        on_worker_shutdown()

        # Verify dispose was called
        assert mock_dispose.called


class TestProcessInitialization:
    """Test worker process initialization (fork handling)."""

    @patch("src.simulations.tasks.db_session.reset_worker_db")
    def test_resets_database_on_process_init(self, mock_reset):
        """Test database reset on process initialization."""
        on_worker_process_init()

        mock_reset.assert_called_once()

    @patch("src.simulations.tasks.db_session.reset_worker_db")
    def test_logs_process_initialization(self, mock_reset, caplog):
        """Test that process initialization is logged."""
        import logging

        caplog.set_level(logging.INFO)

        on_worker_process_init()

        assert any(
            "Worker process initialized" in record.message
            for record in caplog.records
        )


class TestTaskLifecycleEvents:
    """Test task-level lifecycle events."""

    def test_records_task_start_time(self):
        """Test that task start time is recorded."""
        task_id = "test-task-123"
        task = MagicMock()
        task.name = "simulations.run_simulation"

        # Clear previous entries
        _TASK_START_TIME.clear()

        on_task_prerun(task_id=task_id, task=task)

        assert task_id in _TASK_START_TIME
        start_time, last_access = _TASK_START_TIME[task_id]
        assert start_time > 0
        assert last_access > 0

    def test_calculates_task_duration_on_completion(self):
        """Test that task duration is calculated on completion."""
        import time

        task_id = "test-task-456"
        task = MagicMock()
        task.name = "simulations.run_simulation"

        # Record start
        _TASK_START_TIME[task_id] = (time.time() - 5.0, time.time())

        on_task_postrun(task_id=task_id, task=task, state="SUCCESS")

        # Start time should be removed after completion
        assert task_id not in _TASK_START_TIME


class TestTaskEventLogging:
    """Test task event logging."""

    def test_logs_task_failure(self, caplog):
        """Test that task failures are logged."""
        import logging

        caplog.set_level(logging.ERROR)

        task = MagicMock()
        task.name = "simulations.run_simulation"

        # Call with proper parameter names that don't conflict with logging
        on_task_failure(
            sender=task,
            task_id="failed-task",
            exception=Exception("Test failure"),
            args=(),  # Empty tuple instead of None
            kwargs={},  # Empty dict instead of None
        )

        assert any("Task failed" in record.message for record in caplog.records)

    def test_logs_task_success(self, caplog):
        """Test that task successes are logged."""
        import logging

        caplog.set_level(logging.INFO)

        task = MagicMock()
        task.name = "simulations.run_simulation"

        result = {
            "metrics": {
                "processed_requests": 100,
                "rejected_requests": 20,
            }
        }

        on_task_success(sender=task, result=result)

        assert any(
            "Task succeeded" in record.message for record in caplog.records
        )

    def test_logs_task_retry(self, caplog):
        """Test that task retries are logged."""
        import logging

        caplog.set_level(logging.WARNING)

        task = MagicMock()
        task.name = "simulations.run_simulation"

        on_task_retry(
            sender=task,
            task_id="retry-task",
            reason="Connection timeout",
        )

        assert any(
            "Task retrying" in record.message for record in caplog.records
        )


class TestWorkerHealthCheck:
    """Test worker health check functionality."""

    @patch("src.simulations.worker.celery_app.control.inspect")
    @patch("src.simulations.worker.get_worker_engine")
    def test_reports_healthy_status(self, mock_engine, mock_inspect):
        """Test health check with healthy workers."""
        mock_stats = {"worker1": {"pool": {}}}
        mock_inspect.return_value.stats.return_value = mock_stats

        mock_engine_instance = MagicMock()
        mock_engine_instance.pool.size.return_value = 10
        mock_engine.return_value = mock_engine_instance

        health = check_worker_health()

        assert health["status"] == "healthy"
        assert health["workers"] == 1
        assert health["pool_size"] == 10

    @patch("src.simulations.worker.celery_app.control.inspect")
    def test_reports_unhealthy_when_no_workers(self, mock_inspect):
        """Test health check with no active workers."""
        mock_inspect.return_value.stats.return_value = None

        health = check_worker_health()

        assert health["status"] == "unhealthy"
        assert health["workers"] == 0
