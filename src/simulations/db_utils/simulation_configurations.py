import uuid
from typing import Any, Sequence

from sqlalchemy import select, func, Row, RowMapping
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import load_only

from src.simulations.db_utils.exceptions import IdColumnRequiredException
from src.simulations.db_utils.simulation_reports import create_simulation_report
from src.simulations.models.enums import ReportStatus
from src.simulations.models.simulations import (
    SimulationConfiguration,
    SimulationReport,
)


async def get_simulation_configurations(
    session: AsyncSession,
    *,
    user_id: str,
    columns: list[str] | None = None,
    filters: dict[str, str] | None = None,
    page: int | None = None,
    limit: int | None = None,
) -> tuple[Sequence[Row[Any] | RowMapping | Any], Any]:
    """Retrieve simulation configurations from the database.

    Args:
        session: The asynchronous database session.
        user_id: The ID of the user whose configurations are to be retrieved.
        columns: A list of columns to load. If provided, 'id' must be included.
        filters: A dictionary of filters to apply to the query.
                 Supported filters include:
                 - report_status: Filters by the status of associated reports.
                 - Other columns in SimulationConfiguration can be used for ilike filtering.
        page: The page number for pagination.
        limit: The number of items per page for pagination.

    Returns:
        A tuple containing:
        - A sequence of simulation configurations.
        - The total number of items matching the query (before pagination).

    Raises:
        IdColumnRequiredException: If 'columns' is provided and does not include 'id'.
    """
    stmt = select(SimulationConfiguration).where(
        SimulationConfiguration.user_id == user_id,
        SimulationConfiguration.is_active == True,
    )

    if filters:
        report_status_filter = filters.pop("report_status", None)

        for key, value in filters.items():
            if hasattr(SimulationConfiguration, key):
                stmt = stmt.where(
                    getattr(SimulationConfiguration, key).ilike(f"%{value}%")
                )

        if report_status_filter:
            try:
                status_enum = ReportStatus(report_status_filter)
                stmt = stmt.where(
                    SimulationConfiguration.reports.any(
                        SimulationReport.status == status_enum
                    )
                )
            except ValueError:
                pass

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_items = (await session.execute(count_stmt)).scalar_one()

    if columns:
        if "id" not in columns:
            raise IdColumnRequiredException()

        valid_load_columns = [
            getattr(SimulationConfiguration, col)
            for col in columns
            if hasattr(SimulationConfiguration, col)
        ]
        stmt = stmt.options(load_only(*valid_load_columns))

    if page is not None and limit is not None:
        offset = (page - 1) * limit
        stmt = stmt.offset(offset).limit(limit)

    stmt = stmt.order_by(SimulationConfiguration.updated_at.desc())
    result = await session.execute(stmt)
    configurations = result.scalars().all()

    return configurations, total_items


async def create_simulation_configuration(
    session: AsyncSession, *, user_id: str, **params
) -> tuple[SimulationConfiguration, SimulationReport]:
    """Create a new simulation configuration and an associated initial report.

    This function constructs a `SimulationConfiguration` and a `SimulationReport`
    in memory, links them, adds them to the session, and then commits the transaction.
    The report is initially created with a `PENDING` status.

    Args:
        session: The asynchronous database session.
        user_id: The UUID of the user associated with this configuration.
        **params: Additional parameters for creating the configuration:
                  - `name` (str): The name of the simulation configuration.
                  - `description` (str | None): An optional description.
                  - `simulation_parameters` (dict): The dictionary of simulation settings.

    Returns:
        A tuple containing the newly created and committed `SimulationConfiguration`
        and `SimulationReport` objects.

    Raises:
        Exception: Any exception raised during session operations (e.g., IntegrityError)
                   will be propagated.
    """
    configuration = SimulationConfiguration(
        name=params.get("name"),
        description=params.get("description"),
        simulation_parameters=params.get("simulation_parameters"),
        user_id=uuid.UUID(user_id),
    )
    report = await create_simulation_report(
        session, configuration=configuration, should_commit=False
    )
    session.add_all([configuration, report])
    await session.commit()
    return configuration, report
