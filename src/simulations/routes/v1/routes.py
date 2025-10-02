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
)
from src.simulations.db_utils.simulation_configurations import (
    get_simulation_configurations,
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
    authorize.jwt_required()
    user_id = authorize.get_jwt_subject()

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
            user_id=user_id,
            columns=columns,
            filters=filters,
            page=page,
            limit=limit,
        )
    except IdColumnRequiredException:
        logger.exception(msg="Unexpected error: id must be in the columns list.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return get_simulations_response(
        configs, total_items, page=page, limit=limit, columns=columns
    )
