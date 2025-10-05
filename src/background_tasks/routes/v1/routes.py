import logging
import uuid

from another_fastapi_jwt_auth import AuthJWT
from celery.result import AsyncResult
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from src.background_tasks.routes.v1.exceptions import InvalidSubjectID
from src.background_tasks.routes.v1.schemas import (
    GetBackgroundTasksResponse,
    GetBackgroundTaskResponse,
    GetTaskResponse,
)
from src.background_tasks.routes.v1.utils import (
    verify_subject_ids,
    build_task_response,
)
from src.core.db_session import get_session
from src.background_tasks.db_utils.background_tasks import (
    get_background_task as get_background_task_from_db,
    get_background_tasks as get_background_tasks_from_db,
)
from src.simulations.db_utils.exceptions import BackgroundTaskNotFound
from src.simulations.worker import celery_app


router = APIRouter(
    tags=["v1", "background_tasks"],
    prefix="/v1/background-tasks",
)
logger = logging.getLogger(__name__)


@router.get(
    "",
    status_code=status.HTTP_200_OK,
    response_model=GetBackgroundTasksResponse,
)
async def get_background_tasks(
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
    subject_ids: list[str] | None = Query(
        None, description="Filter by subject IDs (comma-separated)."
    ),
):
    """List background tasks for the authenticated user.

    Retrieve all background tasks belonging to the authenticated user,
    with optional filtering by subject IDs (e.g., report IDs).

    Args:
        authorize: JWT authorization dependency that enforces authentication.
        session: Async SQLAlchemy database session.
        subject_ids: Optional list of subject IDs to filter tasks.

    Returns:
        GetBackgroundTasksResponse containing:
            - background_tasks: List of task records
            - total: Total count of tasks

    Raises:
        HTTPException: 400 if a subject ID is invalid.
        HTTPException: 404 if no background tasks are found.

    Example:
        GET /api/v1/background-tasks?subject_ids=uuid1,uuid2
    """
    authorize.jwt_required()
    user_id = uuid.UUID(authorize.get_jwt_subject())

    try:
        subject_ids = verify_subject_ids(subject_ids)
    except InvalidSubjectID:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid subject ID format. Must be valid UUIDs.",
        )

    try:
        tasks = await get_background_tasks_from_db(
            session, user_id=user_id, subject_ids=subject_ids
        )
        return GetBackgroundTasksResponse(
            background_tasks=tasks,
            total=len(tasks),
        )
    except BackgroundTaskNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No background tasks found for the specified filters.",
        )


@router.get(
    "/{background_task_id}",
    status_code=status.HTTP_200_OK,
    response_model=GetBackgroundTaskResponse,
)
async def get_background_task(
    background_task_id: uuid.UUID,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """Get a single background task record by ID.

    Retrieve metadata about a background task, including its task ID,
    type, creation time, and associated subject.

    Args:
        background_task_id: UUID of the background task to retrieve.
        authorize: JWT authorization dependency.
        session: Async SQLAlchemy database session.

    Returns:
        GetBackgroundTaskResponse containing task metadata.

    Raises:
        HTTPException: 404 if the background task is not found.

    Example:
        GET /api/v1/background-tasks/{uuid}
    """
    authorize.jwt_required()
    user_id = uuid.UUID(authorize.get_jwt_subject())

    try:
        return await get_background_task_from_db(
            session, user_id=user_id, background_task_id=background_task_id
        )
    except BackgroundTaskNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Background task not found or does not belong to you.",
        )


@router.get(
    "/{background_task_id}/tasks/{task_id}",
    status_code=status.HTTP_200_OK,
    response_model=GetTaskResponse,
)
async def get_task(
    background_task_id: uuid.UUID,
    task_id: str,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """Get Celery task execution status and results.

    Query the Celery result backend for the current status, progress,
    results, or errors associated with a background task. This endpoint
    provides real-time task execution status.

    Args:
        background_task_id: Parent background task UUID.
        task_id: Celery task identifier (string UUID).
        authorize: JWT authorization dependency.
        session: Async SQLAlchemy database session.

    Returns:
        GetTaskResponse containing:
            - task_id: Celery task identifier
            - status: Standardized task status (pending, running, etc.)
            - status_message: Human-readable status description
            - progress: Progress information (if running)
            - result: Task results (if succeeded)
            - error: Error details (if failed)
            - timestamps: Start and completion times

    Raises:
        HTTPException: 404 if the background task is not found.
        HTTPException: 500 if task status retrieval fails.

    Example:
        GET /api/v1/background-tasks/{bg_uuid}/tasks/{task_uuid}

        Response (running):
        {
            "task_id": "abc123...",
            "status": "running",
            "status_message": "Simulation is running",
            "progress": {
                "current": 50,
                "total": 100,
                "percent": 50.0,
                "message": "Processing replication 5/10"
            }
        }

        Response (success):
        {
            "task_id": "abc123...",
            "status": "success",
            "status_message": "Task completed successfully",
            "result": {
                "data": { "processed_requests": 100, ... },
                "summary": "Simulation completed"
            },
            "completed_at": "2025-01-05T12:00:00Z"
        }
    """
    authorize.jwt_required()
    user_id = uuid.UUID(authorize.get_jwt_subject())

    try:
        background_task = await get_background_task_from_db(
            session,
            user_id=user_id,
            background_task_id=background_task_id,
            task_id=task_id,
        )
    except BackgroundTaskNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Background task not found or does not belong to you.",
        )

    try:
        celery_result = AsyncResult(background_task.task_id, app=celery_app)

        task_response = build_task_response(
            task_id=task_id,
            celery_result=celery_result,
            task_type=background_task.task_type.value,
            include_traceback=False,
        )

        logger.debug(
            "Task status retrieved: task_id=%s, status=%s",
            task_id,
            task_response.status,
        )

        return task_response

    except Exception as e:
        logger.error(
            "Failed to get task status for task_id=%s: %s",
            task_id,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve task status: {str(e)}",
        )
