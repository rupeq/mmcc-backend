import datetime
import uuid

from pydantic import BaseModel, ConfigDict

from src.background_tasks.models.enums import TaskType


class BackgroundTask(BaseModel):
    """Represent a persisted background task.

    Attributes:
        id: Unique identifier of the background task.
        user_id: Owner user UUID.
        subject_id: Subject UUID associated with the task.
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

    model_config = ConfigDict(
        from_attributes=True,
    )


class Task(BaseModel):
    """Represent a Celery task status payload.

    Attributes:
        task_id: Celery task identifier.
        state: Celery task state (e.g., PENDING, STARTED, SUCCESS, FAILURE).
        status: Human-readable status message.
        result: Optional result payload when the task succeeds.
        progress: Optional progress data reported during execution.
        error: Optional error message when the task fails.
    """

    task_id: str
    state: str
    status: str
    result: dict | None = None
    progress: dict | None = None
    error: str | None = None

    model_config = ConfigDict(
        from_attributes=True,
    )


class GetBackgroundTasksResponse(BaseModel):
    """Wrap background tasks list in a response model.

    Attributes:
        background_tasks: Collection of background task records.
    """

    background_tasks: list[BackgroundTask]


class GetBackgroundTaskResponse(BackgroundTask):
    """Represent a single background task response."""

    pass


class GetTaskResponse(Task):
    """Represent a single task status response."""

    pass
