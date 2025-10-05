"""
Simulation task execution.

This module defines the main simulation task that runs in Celery workers.
It handles task lifecycle, error recovery, result persistence, and monitoring.
"""

import logging
import uuid
from datetime import datetime, timezone

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
from src.simulations.tasks.state_manager import create_task_manager
from src.simulations.worker import celery_app, worker_settings

logger = logging.getLogger(__name__)


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
    ) -> dict:
        """Execute simulation and persist results.

        Args:
            report_id: Simulation report UUID.
            simulation_params: Simulation configuration parameters.

        Returns:
            Dictionary with execution summary.
        """
        session_factory = get_worker_session_factory()
        state_manager = create_task_manager(self, "simulation")

        state_manager.report_started("Validating simulation parameters...")

        async with session_factory() as session:
            await update_simulation_report_status(
                session,
                report_id=uuid.UUID(report_id),
                status=ReportStatus.RUNNING,
            )

        try:
            sim_request = SimulationRequest.model_validate(simulation_params)
        except ValidationError as e:
            error_msg = f"Invalid parameters: {str(e)[:1000]}"
            state_manager.report_failure(e, error_code="ValidationError")

            async with session_factory() as session:
                await update_simulation_report_status(
                    session,
                    report_id=uuid.UUID(report_id),
                    status=ReportStatus.FAILED,
                    error_message=error_msg,
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
            simulation_response = run_replications(sim_request, task=self)

            logger.info(
                "Simulation complete for report %s: processed=%d, rejected=%d",
                report_id,
                simulation_response.aggregated_metrics.processed_requests,
                simulation_response.aggregated_metrics.rejected_requests,
            )

        except SoftTimeLimitExceeded:
            logger.warning(
                "Simulation soft time limit exceeded for report %s", report_id
            )
            state_manager.report_failure(
                "Simulation exceeded time limit", "TimeoutError"
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
                "Simulation execution failed for report %s", report_id
            )
            state_manager.report_failure(e)

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
                logger.error("Failed to update report status: %s", db_error)
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
                    "Successfully updated report %s with results", report_id
                )
            except Exception as e:
                logger.exception(
                    "Failed to update report %s with results", report_id
                )
                state_manager.report_failure(e, error_code="DatabaseError")
                raise

        metrics = simulation_response.aggregated_metrics
        result = {
            "total_requests": metrics.total_requests,
            "processed_requests": metrics.processed_requests,
            "rejected_requests": metrics.rejected_requests,
            "rejection_probability": metrics.rejection_probability,
            "num_replications": metrics.num_replications,
        }

        return state_manager.report_success(
            result=result,
            summary=f"Simulation completed: {metrics.processed_requests} requests processed",
        )


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
):
    """Run simulation task in Celery worker.

    Args:
        self: Task instance (automatically bound).
        report_id: Simulation report UUID as string.
        simulation_params: Dictionary with simulation configuration.

    Returns:
        Dictionary with execution status and metrics summary.
    """
    logger.info(
        "Starting simulation task for report %s (task_id=%s)",
        report_id,
        self.request.id,
        extra={
            "task_id": self.request.id,
            "report_id": report_id,
            "retry_count": self.request.retries,
        },
    )

    try:
        import asyncio

        result = asyncio.run(
            SimulationTask.execute_simulation(
                self, report_id, simulation_params
            )
        )

        logger.info(
            "✅ Simulation task completed for report %s",
            report_id,
            extra={
                "task_id": self.request.id,
                "report_id": report_id,
                "metrics": result,
            },
        )

        return result

    except Exception as exc:
        logger.exception("Simulation task failed for report %s", report_id)

        if self.request.retries < self.max_retries:
            logger.warning(
                "Retrying task for report %s (attempt %d/%d)",
                report_id,
                self.request.retries + 1,
                self.max_retries,
            )
            raise self.retry(exc=exc, countdown=self.default_retry_delay)

        raise
