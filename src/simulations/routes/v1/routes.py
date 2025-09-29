import math

from another_fastapi_jwt_auth import AuthJWT
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from src.core.db_session import get_session
from src.simulations.routes.v1.exceptions import BadFilterFormat, InvalidColumn, \
    InvalidReportStatus
from src.simulations.routes.v1.schemas import (
    GetSimulationsResponse,
    SimulationConfigurationInfo,
)
from src.simulations.db_utils.simulation_configurations import (
    get_simulation_configurations,
)
from src.simulations.routes.v1.utils import (
    parse_search_query,
    validate_simulation_columns,
    verify_report_status_value,
)

router = APIRouter(tags=["v1", "simulations"], prefix="/v1/simulations")


@router.get(
    "",
    response_model=GetSimulationsResponse,
    status_code=status.HTTP_200_OK,
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
        example="name:My Test Sim,description:scenario",
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
        validate_simulation_columns(columns)
    except InvalidColumn:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid columns. Use comma-separated valid column names.",
        )

    configs, total_items = await get_simulation_configurations(
        session,
        user_id=user_id,
        columns=columns,
        filters=filters,
        page=page,
        limit=limit,
    )

    items_data = []
    fields_to_extract = (
        columns if columns else SimulationConfigurationInfo.model_fields.keys()
    )
    for config in configs:
        items_data.append(
            {field: getattr(config, field) for field in fields_to_extract}
        )

    return GetSimulationsResponse(
        items=items_data,
        total_items=total_items,
        total_pages=math.ceil(total_items / limit)
        if limit is not None
        else None,
        page=page,
        limit=limit,
    )
