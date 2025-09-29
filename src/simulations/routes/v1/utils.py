from src.simulations.models.enums import ReportStatus
from src.simulations.models.simulations import SimulationConfiguration
from src.simulations.routes.v1.exceptions import (
    BadFilterFormat,
    InvalidColumn,
    InvalidReportStatus,
)


def parse_search_query(search: str | None) -> dict[str, str]:
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
    if "report_status" not in filters:
        return None

    status_value = filters["report_status"].lower()
    valid_statuses = {s.value for s in ReportStatus}
    if status_value not in valid_statuses:
        raise InvalidReportStatus()

    filters["report_status"] = status_value
    return None


def validate_simulation_columns(columns: list[str] | None) -> None:
    if not columns:
        return None

    valid_columns = {c.name for c in SimulationConfiguration.__table__.columns}

    for col in columns:
        if col not in valid_columns:
            raise InvalidColumn()

    return None
