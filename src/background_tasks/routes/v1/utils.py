import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from celery.result import AsyncResult

from src.background_tasks.models.enums import TaskStatus
from src.background_tasks.routes.v1.exceptions import InvalidSubjectID
from src.background_tasks.routes.v1.schemas import (
    Task,
    TaskProgress,
    TaskResult,
    TaskError,
)


logger = logging.getLogger(__name__)


def verify_subject_ids(
    subject_ids: list[Any] | None,
) -> list[uuid.UUID] | None:
    """Validate and normalize subject IDs into UUIDs.

    Convert each provided subject ID to a UUID instance. Return None if
    no IDs are provided.

    Args:
        subject_ids: Optional list of subject identifiers (UUIDs or
            UUID-like strings).

    Returns:
        List of verified UUIDs or None if input is None.

    Raises:
        InvalidSubjectID: If any subject ID is not a valid UUID.
    """
    if subject_ids is None:
        return None

    verified = []
    for subject_id in subject_ids:
        try:
            verified_subject_id = uuid.UUID(subject_id)
            verified.append(verified_subject_id)
        except ValueError:
            raise InvalidSubjectID

    return verified


def celery_state_to_task_status(celery_state: str) -> TaskStatus:
    """Map Celery task state to standardized TaskStatus.

    Args:
        celery_state: Celery task state (e.g., 'PENDING', 'SUCCESS').

    Returns:
        Standardized TaskStatus enum value.
    """
    state_mapping = {
        "PENDING": TaskStatus.PENDING,
        "RECEIVED": TaskStatus.PENDING,
        "STARTED": TaskStatus.RUNNING,
        "PROGRESS": TaskStatus.RUNNING,
        "SUCCESS": TaskStatus.SUCCESS,
        "FAILURE": TaskStatus.FAILED,
        "RETRY": TaskStatus.RETRY,
        "REVOKED": TaskStatus.CANCELLED,
    }

    return state_mapping.get(celery_state, TaskStatus.PENDING)


def get_status_message(status: TaskStatus, task_type: str | None = None) -> str:
    """Generate human-readable status message.

    Args:
        status: Standardized task status.
        task_type: Optional task type for context-specific messages.

    Returns:
        Human-readable status message.
    """
    base_messages = {
        TaskStatus.PENDING: "Task is queued and waiting to start",
        TaskStatus.RUNNING: "Task is currently executing",
        TaskStatus.SUCCESS: "Task completed successfully",
        TaskStatus.FAILED: "Task failed with an error",
        TaskStatus.RETRY: "Task is being retried after an error",
        TaskStatus.CANCELLED: "Task was cancelled",
    }

    message = base_messages.get(status, "Unknown status")

    if task_type and status == TaskStatus.RUNNING:
        type_messages = {
            "simulation": "Simulation is running",
            "animation": "Animation is being generated",
            "optimization": "Optimization is in progress",
        }
        message = type_messages.get(task_type.lower(), message)

    return message


def parse_celery_progress(info: dict) -> TaskProgress | None:
    """Parse Celery task progress information.

    Args:
        info: Celery task info dictionary.

    Returns:
        TaskProgress object if progress data is available, None otherwise.
    """
    if not info or not isinstance(info, dict):
        return None

    if "current" in info and "total" in info:
        current = info["current"]
        total = info["total"]
        percent = (current / total * 100) if total > 0 else 0

        return TaskProgress(
            current=current,
            total=total,
            percent=round(percent, 2),
            message=info.get("message") or info.get("status"),
        )

    if "progress" in info:
        percent = info["progress"]
        return TaskProgress(
            current=percent,
            total=100,
            percent=round(percent, 2),
            message=info.get("message") or info.get("status"),
        )

    return None


def parse_celery_result(result: Any) -> TaskResult | None:
    """Parse Celery task result.

    Args:
        result: Celery task result.

    Returns:
        TaskResult object if result is available, None otherwise.
    """
    if result is None:
        return None

    if isinstance(result, dict):
        return TaskResult(
            data=result,
            summary=result.get("summary"),
        )

    return TaskResult(
        data={"value": result},
        summary=None,
    )


def parse_celery_error(
    exception: Exception | dict | str,
    include_traceback: bool = False,
) -> TaskError:
    """Parse Celery task error information.

    Args:
        exception: Exception object, error dict, or error string.
        include_traceback: Whether to include traceback in response.

    Returns:
        TaskError object with error details.
    """
    if isinstance(exception, dict):
        return TaskError(
            code=exception.get("exc_type", "UnknownError"),
            message=exception.get("exc_message", str(exception)),
            details=exception.get("exc_details"),
            traceback=exception.get("traceback") if include_traceback else None,
        )

    if isinstance(exception, Exception):
        return TaskError(
            code=type(exception).__name__,
            message=str(exception),
            details=None,
            traceback=None,
        )

    # String or unknown type
    return TaskError(
        code="UnknownError",
        message=str(exception),
        details=None,
        traceback=None,
    )


def build_task_response(
    task_id: str,
    celery_result: AsyncResult,
    task_type: str | None = None,
    include_traceback: bool = False,
) -> Task:
    """Build standardized Task response from Celery AsyncResult.

    This function centralizes the logic for converting Celery task states
    into our standardized Task schema.

    Args:
        task_id: Task identifier.
        celery_result: Celery AsyncResult object.
        task_type: Optional task type for context-specific messages.
        include_traceback: Whether to include error tracebacks.

    Returns:
        Standardized Task object.

    Example:
        >>> result = AsyncResult(task_id, app=celery_app)
        >>> task = build_task_response(task_id, result, task_type="simulation")
        >>> print(task.status)  # TaskStatus.RUNNING
    """
    celery_state = celery_result.state
    status = celery_state_to_task_status(celery_state)
    status_message = get_status_message(status, task_type)

    progress = None
    result = None
    error = None
    started_at = None
    completed_at = None

    if status == TaskStatus.PENDING:
        status_message = "Task is queued or does not exist"

    elif status == TaskStatus.RUNNING:
        if celery_result.info:
            progress = parse_celery_progress(celery_result.info)
            if progress and progress.message:
                status_message = progress.message

    elif status == TaskStatus.SUCCESS:
        result = parse_celery_result(celery_result.result)
        completed_at = datetime.now(timezone.utc)  # Approximate
        logger.debug("Task %s completed successfully", task_id)

    elif status == TaskStatus.FAILED:
        error = parse_celery_error(
            celery_result.info or "Unknown error",
            include_traceback=include_traceback,
        )
        completed_at = datetime.now(timezone.utc)  # Approximate
        logger.warning("Task %s failed: %s", task_id, error.message)

    elif status == TaskStatus.RETRY:
        if celery_result.info:
            if isinstance(celery_result.info, Exception):
                error = parse_celery_error(celery_result.info)
            elif isinstance(celery_result.info, dict):
                error = parse_celery_error(
                    celery_result.info.get("exc", "Unknown retry reason")
                )
        status_message = "Task is being retried after an error"

    elif status == TaskStatus.CANCELLED:
        status_message = "Task was cancelled"
        completed_at = datetime.now(timezone.utc)

    return Task(
        task_id=task_id,
        status=status,
        status_message=status_message,
        progress=progress,
        result=result,
        error=error,
        started_at=started_at,
        completed_at=completed_at,
    )
