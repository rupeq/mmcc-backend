"""Standardized task state management for Celery workers.

This module provides utilities for consistent task state reporting,
progress tracking, and result formatting across all background tasks.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from celery import Task


logger = logging.getLogger(__name__)


class TaskStateManager:
    """Manage task state updates with standardized formatting.

    This class provides a consistent interface for Celery tasks to report
    their status, progress, and results. It ensures that all tasks use the
    same data structure, making frontend integration easier.

    Attributes:
        task: Celery task instance.
        task_type: Type of task (simulation, animation, etc.).

    Example:
        >>> task_manager = TaskStateManager(self, "simulation")
        >>> task_manager.report_started()
        >>> task_manager.report_progress(50, 100, "Processing data...")
        >>> task_manager.report_success({"result": "data"})
    """

    def __init__(self, task: Task, task_type: str):
        """Initialize task state manager.

        Args:
            task: Celery task instance (self in task methods).
            task_type: Human-readable task type identifier.
        """
        self.task = task
        self.task_type = task_type
        self.task_id = task.request.id
        self.started_at: datetime | None = None

    def report_started(self, message: str | None = None) -> None:
        """Report that task execution has started.

        Args:
            message: Optional custom status message.
        """
        self.started_at = datetime.now(timezone.utc)

        default_message = f"{self.task_type.capitalize()} task started"

        self.task.update_state(
            state="STARTED",
            meta={
                "status": message or default_message,
                "task_type": self.task_type,
                "started_at": self.started_at.isoformat(),
            },
        )

        logger.info(
            "Task started: task_id=%s, type=%s",
            self.task_id,
            self.task_type,
        )

    def report_progress(
        self,
        current: int | float,
        total: int | float,
        message: str | None = None,
        **extra_data: Any,
    ) -> None:
        """Report task execution progress.

        Args:
            current: Current progress value.
            total: Total progress value.
            message: Optional progress message.
            **extra_data: Additional data to include in progress update.
        """
        percent = (current / total * 100) if total > 0 else 0

        meta = {
            "status": message or f"Processing {self.task_type}...",
            "task_type": self.task_type,
            "progress": {
                "current": current,
                "total": total,
                "percent": round(percent, 2),
                "message": message,
            },
        }

        # Include any extra data
        meta.update(extra_data)

        self.task.update_state(state="PROGRESS", meta=meta)

        logger.debug(
            "Task progress: task_id=%s, progress=%.2f%%",
            self.task_id,
            percent,
        )

    def report_success(
        self,
        result: dict[str, Any],
        summary: str | None = None,
    ) -> dict[str, Any]:
        """Report successful task completion.

        Args:
            result: Task result data.
            summary: Optional result summary.

        Returns:
            Standardized result dictionary for Celery.
        """
        completed_at = datetime.now(timezone.utc)

        formatted_result = {
            "status": "success",
            "task_type": self.task_type,
            "data": result,
            "summary": summary
            or f"{self.task_type.capitalize()} completed successfully",
            "completed_at": completed_at.isoformat(),
        }

        if self.started_at:
            duration = (completed_at - self.started_at).total_seconds()
            formatted_result["duration_seconds"] = round(duration, 2)

        logger.info(
            "Task completed successfully: task_id=%s, type=%s",
            self.task_id,
            self.task_type,
        )

        return formatted_result

    def report_failure(
        self,
        error: Exception | str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Report task failure.

        This method should be called before raising the exception.
        Celery will handle the actual failure state.

        Args:
            error: Exception or error message.
            error_code: Optional error code.
            details: Optional additional error details.
        """
        error_message = str(error)
        error_type = error_code or (
            type(error).__name__ if isinstance(error, Exception) else "Error"
        )

        logger.error(
            "Task failed: task_id=%s, type=%s, error=%s: %s",
            self.task_id,
            self.task_type,
            error_type,
            error_message,
            exc_info=isinstance(error, Exception),
        )

        # Update state before raising
        self.task.update_state(
            state="FAILURE",
            meta={
                "status": f"{self.task_type.capitalize()} failed",
                "task_type": self.task_type,
                "error": {
                    "code": error_type,
                    "message": error_message,
                    "details": details,
                },
            },
        )


def create_task_manager(task: Task, task_type: str) -> TaskStateManager:
    """Factory function to create a task state manager.

    Args:
        task: Celery task instance.
        task_type: Task type identifier.

    Returns:
        Configured TaskStateManager instance.

    Example:
        >>> @celery_app.task(bind=True)
        >>> def my_task(self, data):
        ...     manager = create_task_manager(self, "my_task")
        ...     manager.report_started()
        ...     # ... do work ...
        ...     return manager.report_success({"result": result})
    """
    return TaskStateManager(task, task_type)
