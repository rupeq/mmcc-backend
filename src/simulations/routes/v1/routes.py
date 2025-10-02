import logging

from another_fastapi_jwt_auth import AuthJWT
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from src.core.db_session import get_session
from src.simulations.db_utils.exceptions import IdColumnRequiredException
from src.simulations.routes.v1.exceptions import (
    BadFilterFormat,
    InvalidColumn,
    InvalidReportStatus,
)
from src.simulations.routes.v1.schemas import (
    GetSimulationsResponse,
    CreateSimulationResponse,
    CreateSimulationRequest,
)
from src.simulations.db_utils.simulation_configurations import (
    get_simulation_configurations,
    create_simulation_configuration,
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
    """Get a list of simulation configurations.

    Args:
        authorize: Dependency for JWT authentication.
        session: Asynchronous database session.
        columns: Comma-separated list of columns to return.
        filters: Search filters in key:value format, comma-separated.
        page: Page number for pagination.
        limit: Number of items per page.

    Returns:
        A response containing a list of simulation configurations.

    Raises:
        HTTPException: If authentication fails, filters are invalid, columns are invalid,
                       or an internal server error occurs.
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
    """Atomically create a SimulationConfiguration and its initial SimulationReport.

    This function attempts to run a simulation and then persists both the
    configuration and an associated report (even if the simulation failed)
    within a single database transaction.

    Args:
        authorize: Dependency for JWT authentication.
        session: The asynchronous database session.
        request: Request object.

    Returns:
        A response containing a SimulationConfiguration.
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
