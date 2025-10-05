"""Utilities for converting between task and report statuses.

This module provides bidirectional conversion between TaskStatus
(used by background tasks) and ReportStatus (used by simulation reports).
"""

from src.background_tasks.models.enums import TaskStatus
from src.simulations.models.enums import ReportStatus


def task_status_to_report_status(task_status: TaskStatus) -> ReportStatus:
    """Convert TaskStatus to ReportStatus.

    Args:
        task_status: Task status from background task system.

    Returns:
        Corresponding report status for database storage.

    Example:
        >>> status = task_status_to_report_status(TaskStatus.SUCCESS)
        >>> assert status == ReportStatus.COMPLETED
    """
    mapping = {
        TaskStatus.PENDING: ReportStatus.PENDING,
        TaskStatus.RUNNING: ReportStatus.RUNNING,
        TaskStatus.SUCCESS: ReportStatus.COMPLETED,
        TaskStatus.FAILED: ReportStatus.FAILED,
        TaskStatus.RETRY: ReportStatus.RUNNING,
        TaskStatus.CANCELLED: ReportStatus.CANCELLED,
    }

    return mapping.get(task_status, ReportStatus.PENDING)


def report_status_to_task_status(report_status: ReportStatus) -> TaskStatus:
    """Convert ReportStatus to TaskStatus.

    Args:
        report_status: Report status from database.

    Returns:
        Corresponding task status for API responses.

    Example:
        >>> status = report_status_to_task_status(ReportStatus.COMPLETED)
        >>> assert status == TaskStatus.SUCCESS
    """
    mapping = {
        ReportStatus.PENDING: TaskStatus.PENDING,
        ReportStatus.RUNNING: TaskStatus.RUNNING,
        ReportStatus.COMPLETED: TaskStatus.SUCCESS,
        ReportStatus.FAILED: TaskStatus.FAILED,
        ReportStatus.CANCELLED: TaskStatus.CANCELLED,
    }

    return mapping.get(report_status, TaskStatus.PENDING)
