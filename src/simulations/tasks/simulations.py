"""
Simulation task execution with comprehensive error handling.

This module defines the main simulation task that runs in Celery workers.
It handles task lifecycle, error recovery, result persistence, and monitoring.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from pydantic import ValidationError

from src.simulations.core.engine import run_replications
from src.simulations.core.schemas import SimulationRequest
from src.simulations.db_utils.simulation_reports import (
    update_simulation_report_results,
    update_simulation_report_status,
)
from src.simulations.models.enums import ReportStatus
from src.simulations.tasks.db_session import get_worker_session_factory
from src.simulations.worker import celery_app, worker_settings


logger = logging.getLogger(__name__)


def calculate_task_timeout(simulation_params: dict) -> int:
    """Calculate appropriate timeout based on simulation size"""
    num_channels = simulation_params.get("numChannels", 1)
    sim_time = simulation_params.get("simulationTime", 100)
    num_reps = simulation_params.get("numReplications", 1)

    # Rough estimate: 1 second per 1000 simulation time units
    base_timeout = int(sim_time * num_reps * num_channels / 100)

    # Minimum 5 minutes, maximum 2 hours
    return max(300, min(base_timeout, 7200))


class SimulationTask(Task):
    """Custom Celery task class for simulation execution.

    Provide automatic retry logic, failure handling, and lifecycle
    hooks for simulation tasks. Implement best practices for long-running
    computational tasks.

    Architecture:
        1. Task received → validate parameters
        2. Execute simulation → catch exceptions
        3. Update database → persist results
        4. On transient failure → retry with exponential backoff
        5. On permanent failure → mark as FAILED

    Retry Strategy:
        - Max retries: 3 (configurable)
        - Backoff: Exponential with jitter
        - Backoff max: 10 minutes
        - Retry only on transient errors (DB, network, timeout)

    Non-Retryable Errors:
        - ValidationError: Invalid parameters
        - SimulationReportNotFound: Report doesn't exist

    Attributes:
        autoretry_for: Tuple of exception types to automatically retry.
        retry_kwargs: Configuration for retry behavior.
        retry_backoff: Enable exponential backoff between retries.
        retry_backoff_max: Maximum backoff delay (seconds).
        retry_jitter: Add randomness to backoff to prevent thundering herd.
    """

    autoretry_for = (
        ConnectionError,
        TimeoutError,
        SoftTimeLimitExceeded,
    )

    retry_kwargs = {"max_retries": worker_settings.task_max_retries}
    retry_backoff = True
    retry_backoff_max = 60 * 10
    retry_jitter = True

    async def execute_simulation(
        self,
        report_id: str,
        simulation_params: dict,
    ) -> dict[str, Any]:
        """Execute simulation and persist results.

        Run the discrete-event simulation with the provided parameters,
        handle all errors, and update the database with results or
        error information.

        Args:
            report_id: Simulation report UUID.
            simulation_params: Simulation configuration parameters.

        Returns:
            Dictionary with execution summary:
                - total_requests: Total arrivals
                - processed_requests: Successfully served
                - rejected_requests: Rejected due to capacity
                - rejection_probability: Rejection rate
                - num_replications: Number of replications run

        Raises:
            ValidationError: If simulation parameters are invalid (not retried).
            SoftTimeLimitExceeded: If execution exceeds soft time limit (retried).
            ConnectionError: If database connection fails (retried).

        Example:
            >>> result = await task.execute_simulation(
            ...     report_id="123e4567-e89b-12d3-a456-426614174000",
            ...     simulation_params={"numChannels": 2, ...}
            ... )
            >>> print(f"Processed: {result['processed_requests']}")
        """
        session_factory = get_worker_session_factory()

        try:
            sim_request = SimulationRequest.model_validate(simulation_params)
        except ValidationError as e:
            logger.error(
                "Invalid simulation parameters for report %s: %s",
                report_id,
                e,
            )

            async with session_factory() as session:
                await update_simulation_report_status(
                    session,
                    report_id=uuid.UUID(report_id),
                    status=ReportStatus.FAILED,
                    error_message=f"Invalid parameters: {str(e)[:1000]}",
                    completed_at=datetime.now(timezone.utc),
                )

            return {
                "total_requests": 0,
                "processed_requests": 0,
                "rejected_requests": 0,
                "rejection_probability": 0.0,
                "num_replications": 0,
            }

        logger.info(
            "Running simulation for report %s: channels=%d, time=%.2f, replications=%d",
            report_id,
            sim_request.num_channels,
            sim_request.simulation_time,
            sim_request.num_replications,
        )

        try:
            simulation_response = run_replications(sim_request)
            logger.info(
                "✅ Simulation complete for report %s: processed=%d, rejected=%d, "
                "rejection_prob=%.4f",
                report_id,
                simulation_response.aggregated_metrics.processed_requests,
                simulation_response.aggregated_metrics.rejected_requests,
                simulation_response.aggregated_metrics.rejection_probability,
            )
        except SoftTimeLimitExceeded:
            logger.warning(
                "⏱️  Simulation soft time limit exceeded for report %s",
                report_id,
            )
            async with session_factory() as session:
                await update_simulation_report_status(
                    session,
                    report_id=uuid.UUID(report_id),
                    status=ReportStatus.FAILED,
                    error_message="Simulation exceeded time limit",
                    completed_at=datetime.now(timezone.utc),
                )
            raise
        except Exception as e:
            logger.exception(
                "❌ Simulation execution failed for report %s: %s",
                report_id,
                e,
            )

            try:
                async with session_factory() as session:
                    await update_simulation_report_status(
                        session,
                        report_id=uuid.UUID(report_id),
                        status=ReportStatus.FAILED,
                        error_message=f"Simulation error: {str(e)[:1000]}",
                        completed_at=datetime.now(timezone.utc),
                    )
            except Exception as db_error:
                logger.error(
                    "Failed to update report status after simulation error: %s",
                    db_error,
                )
                raise e from db_error
            raise

        async with session_factory() as session:
            try:
                await update_simulation_report_results(
                    session,
                    report_id=uuid.UUID(report_id),
                    status=ReportStatus.COMPLETED,
                    results=simulation_response.model_dump(mode="json"),
                    completed_at=datetime.now(timezone.utc),
                )
                logger.info(
                    "💾 Successfully updated report %s with results", report_id
                )
            except Exception as e:
                logger.exception(
                    "❌ Failed to update report %s with results: %s",
                    report_id,
                    e,
                )
                raise

        return {
            "total_requests": simulation_response.aggregated_metrics.total_requests,
            "processed_requests": simulation_response.aggregated_metrics.processed_requests,
            "rejected_requests": simulation_response.aggregated_metrics.rejected_requests,
            "rejection_probability": simulation_response.aggregated_metrics.rejection_probability,
            "num_replications": simulation_response.aggregated_metrics.num_replications,
        }


@celery_app.task(
    bind=True,
    base=SimulationTask,
    name="simulations.run_simulation",
    max_retries=worker_settings.task_max_retries,
    default_retry_delay=worker_settings.task_default_retry_delay,
    acks_late=True,
    reject_on_worker_lost=True,
    track_started=True,
    on_failure=lambda self, exc, task_id, args, kwargs, _: (
        celery_app.send_task(
            "simulations.dlq",
            args=[task_id, args, kwargs, str(exc)],
            queue="simulations_dlq",
        )
    ),
    soft_time_limit=None,
    time_limit=None,
)
def run_simulation_task(
    self: Task,
    report_id: str,
    simulation_params: dict,
) -> dict[str, Any]:
    """
    Run simulation task in Celery worker.

    This is the main entry point for background simulation execution. It
    orchestrates the entire simulation lifecycle from validation to result
    persistence.

    Args:
        self: Task instance (automatically bound).
        report_id: Simulation report UUID as string.
        simulation_params: Dictionary with simulation configuration.

    Returns:
        Dictionary with execution status and metrics summary.

    Raises:
        ValidationError: If simulation parameters are invalid.
        SoftTimeLimitExceeded: If execution exceeds soft time limit.
        Exception: For any other execution errors (triggers retry).

    Example:
        >>> task = run_simulation_task.delay(
        ...     report_id="123e4567-e89b-12d3-a456-426614174000",
        ...     simulation_params={...}
        ... )
        >>> result = task.get(timeout=3600)
        >>> print(result["status"])
        success
    """
    logger.info(
        "🚀 Starting simulation task for report %s (task_id=%s)",
        report_id,
        self.request.id,
        extra={
            "task_id": self.request.id,
            "report_id": report_id,
            "retry_count": self.request.retries,
        },
    )

    timeout = calculate_task_timeout(simulation_params)

    self.time_limit = timeout + 60 * 5
    self.soft_time_limit = timeout

    try:
        import asyncio

        result = asyncio.run(
            SimulationTask.execute_simulation(self, report_id, simulation_params)
        )

        logger.info(
            "✅ Simulation task completed for report %s: %d requests processed",
            report_id,
            result.get("processed_requests", 0),
            extra={
                "task_id": self.request.id,
                "report_id": report_id,
                "metrics": result,
            },
        )

        return {
            "status": "success",
            "report_id": report_id,
            "task_id": self.request.id,
            "metrics": result,
        }

    except Exception as exc:
        logger.exception(
            "❌ Simulation task failed for report %s: %s",
            report_id,
            exc,
            extra={
                "task_id": self.request.id,
                "report_id": report_id,
                "exception_type": type(exc).__name__,
            },
        )

        if self.request.retries < self.max_retries:
            logger.warning(
                "🔄 Retrying task for report %s (attempt %d/%d)",
                report_id,
                self.request.retries + 1,
                self.max_retries,
            )
            raise self.retry(exc=exc, countdown=self.default_retry_delay)

        raise
