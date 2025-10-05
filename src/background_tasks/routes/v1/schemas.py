import datetime
import uuid
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from src.background_tasks.models.enums import TaskType, TaskStatus


class BackgroundTask(BaseModel):
    """Represent a persisted background task record.

    Attributes:
        id: Unique identifier of the background task.
        user_id: Owner user UUID.
        subject_id: Subject UUID associated with the task (e.g., report ID).
        task_id: Celery task identifier.
        task_type: Type of the background task.
        created_at: UTC timestamp when the task was created.
    """

    id: uuid.UUID
    user_id: uuid.UUID
    subject_id: uuid.UUID
    task_id: str
    task_type: TaskType
    created_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


class TaskProgress(BaseModel):
    """Represent task execution progress information.

    Attributes:
        current: Current progress value.
        total: Total progress value.
        percent: Progress percentage (0-100).
        message: Optional human-readable progress message.
    """

    current: int | float = Field(..., description="Current progress value")
    total: int | float = Field(..., description="Total progress value")
    percent: float = Field(..., ge=0, le=100, description="Progress percentage")
    message: str | None = Field(None, description="Progress message")


class TaskResult(BaseModel):
    """Represent the result payload of a successful task.

    Attributes:
        data: Task-specific result data.
        summary: Optional human-readable result summary.
    """

    data: dict[str, Any] = Field(..., description="Task result data")
    summary: str | None = Field(None, description="Result summary")


class TaskError(BaseModel):
    """Represent task error information.

    Attributes:
        code: Error code or exception type.
        message: Human-readable error message.
        details: Optional additional error details.
        traceback: Optional error traceback (dev only).
    """

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: dict[str, Any] | None = Field(None, description="Error details")
    traceback: str | None = Field(None, description="Error traceback")


class Task(BaseModel):
    """Represent a standardized Celery task status payload.

    This schema provides a consistent interface for task status across
    all task types, abstracting away Celery-specific implementation details.

    Attributes:
        task_id: Celery task identifier.
        status: Standardized task status.
        status_message: Human-readable status description.
        progress: Optional progress information (if task is running).
        result: Optional result payload (if task succeeded).
        error: Optional error information (if task failed).
        started_at: UTC timestamp when task started execution.
        completed_at: UTC timestamp when task completed.
    """

    task_id: str = Field(..., description="Celery task identifier")
    status: TaskStatus = Field(..., description="Task status")
    status_message: str = Field(..., description="Status description")
    progress: TaskProgress | None = Field(
        None, description="Progress information"
    )
    result: TaskResult | None = Field(None, description="Task result")
    error: TaskError | None = Field(None, description="Error information")
    started_at: datetime.datetime | None = Field(
        None, description="Task start timestamp"
    )
    completed_at: datetime.datetime | None = Field(
        None, description="Task completion timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


class GetBackgroundTasksResponse(BaseModel):
    """Response containing a collection of background tasks.

    Attributes:
        background_tasks: List of background task records.
        total: Total number of tasks.
    """

    background_tasks: list[BackgroundTask]
    total: int = Field(..., description="Total number of tasks")


class GetBackgroundTaskResponse(BackgroundTask):
    """Response containing a single background task record."""

    pass


class GetTaskResponse(Task):
    """Response containing a single task status."""

    pass
