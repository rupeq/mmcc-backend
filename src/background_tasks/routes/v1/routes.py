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
    get_status_message,
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
        None, description="Comma-separated list of subjects to return."
    ),
):
    """List background tasks for the authenticated user.

    Validate JWT, optionally filter by subject IDs, and return the
    user's background tasks.

    Args:
        authorize: Dependency that enforces JWT authorization.
        session: Async SQLAlchemy session.
        subject_ids: Optional list of subject IDs to filter tasks.

    Returns:
        GetBackgroundTasksResponse: Collection of background tasks.

    Raises:
        HTTPException: 400 if a subject ID is invalid.
        HTTPException: 404 if no background tasks are found.
    """
    authorize.jwt_required()
    user_id = uuid.UUID(authorize.get_jwt_subject())

    try:
        subject_ids = verify_subject_ids(subject_ids)
    except InvalidSubjectID:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid subject ID",
        )

    try:
        return {
            "background_tasks": await get_background_tasks_from_db(
                session, user_id=user_id, subject_ids=subject_ids
            )
        }
    except BackgroundTaskNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Background tasks not found",
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
    """Get a single background task by ID.

    Validate JWT and return the background task owned by the user.

    Args:
        background_task_id: Background task UUID.
        authorize: Dependency that enforces JWT authorization.
        session: Async SQLAlchemy session.

    Returns:
        GetBackgroundTaskResponse: Background task details.

    Raises:
        HTTPException: 404 if the background task is not found.
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
            detail="Background task not found",
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
    """Get Celery task status for a background task.

    Validate JWT, verify ownership of the background task, and query the
    Celery backend for task status, progress, and results.

    Args:
        background_task_id: Parent background task UUID.
        task_id: Celery task identifier.
        authorize: Dependency that enforces JWT authorization.
        session: Async SQLAlchemy session.

    Returns:
        GetTaskResponse: Task status, progress, result, or error.

    Raises:
        HTTPException: 404 if the background task is not found.
        HTTPException: 500 on failure to retrieve task status.
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
            detail="Background task not found",
        )

    try:
        result = AsyncResult(background_task.task_id, app=celery_app)

        state = result.state
        status_msg = get_status_message(state)

        response = GetTaskResponse(
            task_id=task_id,
            state=state,
            status=status_msg,
        )

        if state == "PENDING":
            response.status = "Task is queued or does not exist"

        elif state == "STARTED":
            response.status = "Simulation is running"
            if result.info:
                response.progress = result.info

        elif state == "SUCCESS":
            response.status = "Simulation completed successfully"
            response.result = result.result if result.result else None

        elif state == "FAILURE":
            response.status = "Simulation failed"
            response.error = (
                str(result.info) if result.info else "Unknown error"
            )
            logger.warning(
                "Task %s failed: %s",
                task_id,
                response.error,
            )

        elif state == "RETRY":
            response.status = "Simulation is being retried after an error"
            if result.info:
                response.error = (
                    str(result.info)
                    if isinstance(result.info, Exception)
                    else str(result.info.get("exc", "Unknown error"))
                )

        elif state == "REVOKED":
            response.status = "Simulation was cancelled"

        logger.debug(
            "Task status retrieved: task_id=%s, state=%s",
            task_id,
            state,
        )

        return response

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
