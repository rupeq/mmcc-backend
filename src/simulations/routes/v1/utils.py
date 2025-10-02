import math

from sqlalchemy import Sequence, RowMapping, Row

from src.simulations.models.enums import ReportStatus
from src.simulations.models.simulations import SimulationConfiguration
from src.simulations.routes.v1.exceptions import (
    BadFilterFormat,
    InvalidColumn,
    InvalidReportStatus,
)
from src.simulations.routes.v1.schemas import (
    GetSimulationsResponse,
    SimulationConfigurationInfo,
)


def parse_search_query(search: str | None) -> dict[str, str]:
    """Parse a search query string into a dictionary of filters.

    Args:
        search: The search query string, e.g., "key1:value1,key2:value2".

    Returns:
        A dictionary of filters.

    Raises:
        BadFilterFormat: If the search query format is invalid.
    """
    if not search:
        return {}

    filters = {}

    try:
        for item in search.split(","):
            key, value = item.split(":", 1)
            filters[key.strip()] = value.strip()
    except ValueError:
        raise BadFilterFormat()

    return filters


def verify_report_status_value(filters: dict[str, str]) -> None:
    """Verify if the report_status filter has a valid value.

    Args:
        filters: A dictionary of filters.

    Raises:
        InvalidReportStatus: If the report_status value is invalid.
    """
    if "report_status" not in filters:
        return None

    status_value = filters["report_status"].lower()
    valid_statuses = {s.value for s in ReportStatus}
    if status_value not in valid_statuses:
        raise InvalidReportStatus()

    filters["report_status"] = status_value
    return None


def validate_simulation_columns(columns: list[str] | None) -> list[str] | None:
    """Validate the requested simulation columns.

    Ensures that all requested columns are valid and adds 'id' if not present.

    Args:
        columns: A list of column names to validate.

    Returns:
        A list of valid column names, including 'id', or None if no columns were specified.

    Raises:
        InvalidColumn: If any of the requested columns are invalid.
    """
    if not columns:
        return None

    valid_columns = {c.name for c in SimulationConfiguration.__table__.columns}

    all_cols = []
    for col_group in columns:
        all_cols.extend(c.strip() for c in col_group.split(","))

    if "id" not in all_cols:
        all_cols.insert(0, "id")

    for col in all_cols:
        if col not in valid_columns:
            raise InvalidColumn()

    return all_cols


def get_simulations_response(
    configs: Sequence[Row | RowMapping],
    total_items: int,
    *,
    limit: int | None,
    page: int | None,
    columns: list[str] | None = None,
) -> GetSimulationsResponse:
    """Construct the GetSimulationsResponse object.

    Args:
        configs: A sequence of simulation configuration rows.
        total_items: The total number of items.
        limit: The number of items per page.
        page: The current page number.
        columns: A list of columns to include in the response.

    Returns:
        A GetSimulationsResponse object.
    """
    items_data = []
    fields_to_extract = (
        columns if columns else SimulationConfigurationInfo.model_fields.keys()
    )
    for config in configs:
        items_data.append(
            {
                field: getattr(config, field)
                for field in fields_to_extract
                if hasattr(config, field)
            }
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
