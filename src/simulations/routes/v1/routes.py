"""API routes for simulation configurations and reports.

This module provides RESTful endpoints for creating, retrieving, updating,
and deleting simulation configurations and their associated reports.
"""

import logging
import uuid

from another_fastapi_jwt_auth import AuthJWT
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from src.core.db_session import get_session
from src.simulations.db_utils.exceptions import (
    IdColumnRequiredException,
    SimulationNotFound,
    SimulationReportsNotFound,
    SimulationReportNotFound,
)
from src.simulations.routes.v1.exceptions import (
    BadFilterFormat,
    InvalidColumn,
    InvalidReportStatus,
)
from src.simulations.routes.v1.schemas import (
    GetSimulationsResponse,
    CreateSimulationResponse,
    CreateSimulationRequest,
    GetSimulationConfigurationResponse,
    GetSimulationConfigurationReportsResponse,
    GetSimulationConfigurationReportResponse,
)
from src.simulations.db_utils.simulation_configurations import (
    get_simulation_configurations,
    get_simulation_configuration as get_simulation_configuration_from_db,
    create_simulation_configuration,
    delete_simulation_configuration as delete_simulation_configuration_from_db,
)
from src.simulations.db_utils.simulation_reports import (
    get_simulation_configuration_report as get_simulation_configuration_report_from_db,
    get_simulation_configuration_reports as get_simulation_configuration_reports_from_db,
    delete_simulation_configuration_report as delete_simulation_configuration_report_from_db,
)
from src.simulations.routes.v1.utils import (
    parse_search_query,
    validate_simulation_columns,
    verify_report_status_value,
    get_simulations_response,
)

router = APIRouter(tags=["v1", "simulations"], prefix="/v1/simulations")
logger = logging.getLogger(__name__)


@router.get(
    "",
    response_model=GetSimulationsResponse,
    status_code=status.HTTP_200_OK,
    response_model_exclude_none=True,
)
async def get_simulations(
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
    columns: list[str] | None = Query(
        None, description="Comma-separated list of columns to return."
    ),
    filters: str | None = Query(
        None,
        description="Search filters in key:value format, comma-separated.",
    ),
    page: int | None = Query(None, ge=1, description="Page number"),
    limit: int | None = Query(None, ge=1, le=100, description="Items per page"),
):
    """Retrieve simulation configurations for the authenticated user.

    Fetch a paginated list of simulation configurations belonging to the
    authenticated user, with optional filtering and column selection.

    Args:
        authorize: JWT authentication dependency.
        session: Asynchronous database session.
        columns: Optional list of column names to include in response.
        filters: Optional search filters in "key:value,key:value" format.
        page: Optional page number for pagination (1-indexed).
        limit: Optional maximum number of items per page (1-100).

    Returns:
        GetSimulationsResponse containing:
            - items: List of simulation configurations
            - total_items: Total count of matching configurations
            - total_pages: Total number of pages (if paginated)
            - page: Current page number
            - limit: Items per page

    Raises:
        HTTPException 400: If filters or columns are invalid.
        HTTPException 401: If user is not authenticated.
        HTTPException 500: If an internal server error occurs.

    Example:
        GET /api/v1/simulations?page=1&limit=10&filters=name:Test
    """
    authorize.jwt_required()

    try:
        filters = parse_search_query(filters)
        verify_report_status_value(filters)
    except BadFilterFormat:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid search format. Use comma-separated key:value pairs.",
        )
    except InvalidReportStatus:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid report_status.",
        )

    try:
        columns = validate_simulation_columns(columns)
    except InvalidColumn:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid columns. Use comma-separated valid column names.",
        )

    try:
        configs, total_items = await get_simulation_configurations(
            session,
            user_id=authorize.get_jwt_subject(),
            columns=columns,
            filters=filters,
            page=page,
            limit=limit,
        )
    except IdColumnRequiredException:
        logger.exception(
            msg="Unexpected error: id must be in the columns list."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return get_simulations_response(
        configs, total_items, page=page, limit=limit, columns=columns
    )


@router.post(
    "",
    response_model=CreateSimulationResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_simulation(
    request: CreateSimulationRequest,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """Create a new simulation configuration with an initial pending report.

    Atomically create a simulation configuration and its associated initial
    report in a single database transaction. The report is created with
    PENDING status.

    Args:
        request: Simulation creation request containing:
            - name: Configuration name (required)
            - description: Optional description
            - simulation_parameters: Simulation parameters (required)
        authorize: JWT authentication dependency.
        session: Asynchronous database session.

    Returns:
        CreateSimulationResponse containing:
            - simulation_configuration_id: UUID of created configuration
            - simulation_report_id: UUID of created report

    Raises:
        HTTPException 401: If user is not authenticated.
        HTTPException 422: If request validation fails.

    Example:
        POST /api/v1/simulations
        {
            "name": "Test Simulation",
            "description": "A test simulation",
            "simulationParameters": {
                "numChannels": 2,
                "simulationTime": 100.0,
                ...
            }
        }
    """
    authorize.jwt_required()

    configuration, report = await create_simulation_configuration(
        session,
        user_id=authorize.get_jwt_subject(),
        name=request.name,
        description=request.description,
        simulation_parameters=request.simulation_parameters.model_dump(
            by_alias=True
        ),
    )

    return CreateSimulationResponse(
        simulation_configuration_id=configuration.id,
        simulation_report_id=report.id,
    )


@router.get(
    "/{simulation_configuration_id}",
    response_model=GetSimulationConfigurationResponse,
    status_code=status.HTTP_200_OK,
)
async def get_simulation_configuration(
    simulation_configuration_id: uuid.UUID,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """Retrieve a specific simulation configuration by ID.

    Fetch detailed information about a simulation configuration belonging
    to the authenticated user.

    Args:
        simulation_configuration_id: UUID of the configuration to retrieve.
        authorize: JWT authentication dependency.
        session: Asynchronous database session.

    Returns:
        GetSimulationConfigurationResponse containing configuration details.

    Raises:
        HTTPException 401: If user is not authenticated.
        HTTPException 404: If configuration not found or doesn't belong to user.

    Example:
        GET /api/v1/simulations/{uuid}
    """
    authorize.jwt_required()
    try:
        return await get_simulation_configuration_from_db(
            session,
            user_id=uuid.UUID(authorize.get_jwt_subject()),
            simulation_configuration_id=simulation_configuration_id,
        )
    except SimulationNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation configuration not found",
        )


@router.get(
    "/{simulation_configuration_id}/reports/{report_id}",
    response_model=GetSimulationConfigurationReportResponse,
    status_code=status.HTTP_200_OK,
)
async def get_simulation_configuration_report(
    simulation_configuration_id: uuid.UUID,
    report_id: uuid.UUID,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """Retrieve a specific simulation report by ID.

    Fetch detailed information about a simulation report, including its
    status, results, and any error messages.

    Args:
        simulation_configuration_id: UUID of the parent configuration.
        report_id: UUID of the report to retrieve.
        authorize: JWT authentication dependency.
        session: Asynchronous database session.

    Returns:
        GetSimulationConfigurationReportResponse containing:
            - id: Report UUID
            - status: Report status (PENDING, COMPLETED, FAILED)
            - results: Simulation results (if completed)
            - error_message: Error details (if failed)
            - created_at: Creation timestamp
            - completed_at: Completion timestamp (if applicable)

    Raises:
        HTTPException 401: If user is not authenticated.
        HTTPException 404: If report not found or doesn't belong to user.

    Example:
        GET /api/v1/simulations/{config_uuid}/reports/{report_uuid}
    """
    authorize.jwt_required()
    try:
        return await get_simulation_configuration_report_from_db(
            session,
            user_id=uuid.UUID(authorize.get_jwt_subject()),
            report_id=report_id,
            simulation_configuration_id=simulation_configuration_id,
        )
    except SimulationReportNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation report not found",
        )


@router.get(
    "/{simulation_configuration_id}/reports",
    response_model=GetSimulationConfigurationReportsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_simulation_configuration_reports(
    simulation_configuration_id: uuid.UUID,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """Retrieve all reports for a specific simulation configuration.

    Fetch all simulation reports associated with a given configuration,
    including historical runs and their results.

    Args:
        simulation_configuration_id: UUID of the configuration.
        authorize: JWT authentication dependency.
        session: Asynchronous database session.

    Returns:
        GetSimulationConfigurationReportsResponse containing:
            - reports: List of all reports for this configuration

    Raises:
        HTTPException 401: If user is not authenticated.
        HTTPException 404: If configuration not found or has no reports.

    Example:
        GET /api/v1/simulations/{config_uuid}/reports
    """
    authorize.jwt_required()
    try:
        return {
            "reports": await get_simulation_configuration_reports_from_db(
                session,
                user_id=uuid.UUID(authorize.get_jwt_subject()),
                simulation_configuration_id=simulation_configuration_id,
            )
        }
    except SimulationReportsNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation reports not found",
        )


@router.delete(
    "/{simulation_configuration_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_simulation_configuration(
    simulation_configuration_id: uuid.UUID,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """Delete a simulation configuration and all associated reports.

    Soft delete (mark as inactive) a simulation configuration and cascade
    the deletion to all associated reports. This operation is irreversible.

    Args:
        simulation_configuration_id: UUID of the configuration to delete.
        authorize: JWT authentication dependency.
        session: Asynchronous database session.

    Returns:
        None (204 No Content on success).

    Raises:
        HTTPException 401: If user is not authenticated.
        HTTPException 404: If configuration not found or doesn't belong to user.

    Example:
        DELETE /api/v1/simulations/{uuid}
    """
    authorize.jwt_required()
    try:
        await delete_simulation_configuration_from_db(
            session,
            user_id=uuid.UUID(authorize.get_jwt_subject()),
            simulation_configuration_id=simulation_configuration_id,
        )
    except SimulationNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation configuration not found",
        )
    return None


@router.delete(
    "/{simulation_configuration_id}/reports/{report_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_simulation_configuration_report(
    simulation_configuration_id: uuid.UUID,
    report_id: uuid.UUID,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """Delete a specific simulation report.

    Soft delete (mark as inactive) a simulation report. The parent
    configuration remains active. This operation is irreversible.

    Args:
        simulation_configuration_id: UUID of the parent configuration.
        report_id: UUID of the report to delete.
        authorize: JWT authentication dependency.
        session: Asynchronous database session.

    Returns:
        None (204 No Content on success).

    Raises:
        HTTPException 401: If user is not authenticated.
        HTTPException 404: If report not found or doesn't belong to user.

    Example:
        DELETE /api/v1/simulations/{config_uuid}/reports/{report_uuid}
    """
    authorize.jwt_required()
    try:
        await delete_simulation_configuration_report_from_db(
            session,
            report_id=report_id,
            user_id=uuid.UUID(authorize.get_jwt_subject()),
            simulation_configuration_id=simulation_configuration_id,
        )
    except SimulationReportNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation report not found",
        )
    return None
