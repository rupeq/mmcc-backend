import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.simulations.db_utils.exceptions import BackgroundTaskNotFound
from src.background_tasks.models.enums import TaskType
from src.background_tasks.models.background_tasks import BackgroundTask


async def create_background_task(
    session: AsyncSession,
    *,
    task_id: str,
    task_type: TaskType,
    subject_id: uuid.UUID,
    user_id: uuid.UUID,
) -> BackgroundTask:
    """Create and persist a background task record.

    Args:
        session: Async SQLAlchemy session.
        task_id: Celery task identifier.
        task_type: Enumerated type of the background task.
        subject_id: Subject UUID the task relates to.
        user_id: Owner user UUID.

    Returns:
        BackgroundTask: Newly created background task entity.
    """
    task = BackgroundTask(
        task_id=task_id,
        task_type=task_type,
        subject_id=subject_id,
        user_id=user_id,
    )
    session.add(task)
    await session.commit()
    return task


async def get_background_task(
    session: AsyncSession,
    *,
    background_task_id: uuid.UUID,
    user_id: uuid.UUID,
    task_id: str | None = None,
) -> BackgroundTask:
    """Retrieve a single background task by ID and owner.

    Optionally restrict by Celery task ID.

    Args:
        session: Async SQLAlchemy session.
        background_task_id: Background task UUID.
        user_id: Owner user UUID.
        task_id: Optional Celery task identifier to further filter.

    Returns:
        BackgroundTask: Matching background task.

    Raises:
        BackgroundTaskNotFound: If no matching task exists.
    """
    filters = [
        BackgroundTask.id == background_task_id,
        BackgroundTask.user_id == user_id,
    ]

    if task_id is not None:
        filters.append(BackgroundTask.task_id == task_id)

    query = await session.execute(
        select(
            BackgroundTask,
        ).where(*filters)
    )
    background_task = query.scalars().one_or_none()

    if background_task is None:
        raise BackgroundTaskNotFound()

    return background_task


async def get_background_tasks(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    subject_ids: list[uuid.UUID] | None = None,
) -> list[BackgroundTask]:
    """List background tasks for a user, optionally filtered by subjects.

    Args:
        session: Async SQLAlchemy session.
        user_id: Owner user UUID.
        subject_ids: Optional list of subject UUIDs to filter.

    Returns:
        list[BackgroundTask]: Collection of background tasks.

    Raises:
        BackgroundTaskNotFound: If no tasks match the criteria.
    """
    filters = [BackgroundTask.user_id == user_id]

    if subject_ids is not None:
        filters.append(BackgroundTask.subject_id.in_(subject_ids))

    query = await session.execute(
        select(
            BackgroundTask,
        ).where(*filters)
    )
    background_tasks = query.scalars().all()

    if not background_tasks:
        raise BackgroundTaskNotFound()

    return list(background_tasks)
