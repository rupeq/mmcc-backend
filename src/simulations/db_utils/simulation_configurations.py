"""Database operations for simulation configurations.

This module provides CRUD operations for managing simulation configurations
in the database.
"""

import logging
import uuid
from typing import Any, Sequence

from sqlalchemy import select, func, Row, RowMapping, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import load_only

from src.simulations.db_utils.exceptions import (
    IdColumnRequiredException,
    SimulationNotFound,
)
from src.simulations.db_utils.simulation_reports import (
    create_simulation_report,
)
from src.simulations.models.enums import ReportStatus
from src.simulations.models.simulations import (
    SimulationConfiguration,
    SimulationReport,
)


logger = logging.getLogger(__name__)


async def get_simulation_configurations(
    session: AsyncSession,
    *,
    user_id: str,
    columns: list[str] | None = None,
    filters: dict[str, str] | None = None,
    page: int | None = None,
    limit: int | None = None,
) -> tuple[Sequence[Row[Any] | RowMapping | Any], Any]:
    """Retrieve simulation configurations for a user with optional filtering.

    Fetch simulation configurations from the database with support for column
    selection, filtering by attributes, and pagination.

    Args:
        session: Async database session.
        user_id: ID of the user whose configurations to retrieve.
        columns: List of column names to return. If None, returns all columns.
        filters: Dict of attribute filters (key=column, value=search string).
        page: Page number for pagination (1-indexed).
        limit: Number of results per page.

    Returns:
        Tuple of (list of configurations, total count).

    Raises:
        IdColumnRequiredException: If 'id' column not in columns list.
    """
    logger.debug("Retrieving simulation configurations for user %s", user_id)
    stmt = select(SimulationConfiguration).where(
        SimulationConfiguration.user_id == user_id,
        SimulationConfiguration.is_active == True,
    )

    if filters:
        logger.debug(
            "Filtering simulation configurations for user %s, filters include: %s",
            user_id,
            filters,
        )
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
        logger.debug("Limiting columns to %s for user: %s", columns, user_id)
        if "id" not in columns:
            raise IdColumnRequiredException()

        valid_load_columns = [
            getattr(SimulationConfiguration, col)
            for col in columns
            if hasattr(SimulationConfiguration, col)
        ]
        stmt = stmt.options(load_only(*valid_load_columns))

    if page is not None and limit is not None:
        logger.debug(
            "Limiting page to %s, limit set to %s, for user: %s",
            page,
            limit,
            user_id,
        )
        offset = (page - 1) * limit
        stmt = stmt.offset(offset).limit(limit)

    stmt = stmt.order_by(SimulationConfiguration.updated_at.desc())
    result = await session.execute(stmt)
    configurations = result.scalars().all()

    return configurations, total_items


async def get_simulation_configuration(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    simulation_configuration_id: uuid.UUID,
) -> SimulationConfiguration:
    logger.debug("Retrieving simulation configuration for user %s", user_id)
    query = await session.execute(
        select(SimulationConfiguration).where(
            SimulationConfiguration.id == simulation_configuration_id,
            SimulationConfiguration.is_active == True,
            SimulationConfiguration.user_id == user_id,
        )
    )
    simulation = query.scalars().one_or_none()

    if simulation is None:
        logger.debug("No simulation found for user %s", user_id)
        raise SimulationNotFound()

    logger.debug("Retrieved simulation configuration for user %s", user_id)
    return simulation


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
    logger.debug("Creating new simulation configuration for user %s", user_id)
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
    logger.debug("Created new simulation configuration for user %s", user_id)
    return configuration, report


async def delete_simulation_configuration(
    session: AsyncSession,
    *,
    simulation_configuration_id: uuid.UUID,
    user_id: uuid.UUID,
) -> None:
    logger.debug(
        "Deleting simulation configuration %s for user %s",
        simulation_configuration_id,
        user_id,
    )
    await get_simulation_configuration(
        session,
        user_id=user_id,
        simulation_configuration_id=simulation_configuration_id,
    )
    result = await session.execute(
        update(SimulationConfiguration)
        .where(
            SimulationConfiguration.id == simulation_configuration_id,
            SimulationConfiguration.user_id == user_id,
        )
        .values(is_active=False)
    )
    nested_result = await session.execute(
        update(SimulationReport)
        .where(
            SimulationReport.configuration_id == simulation_configuration_id,
        )
        .values(is_active=False)
    )
    await session.commit()
    logger.debug(
        "Successfully deleted simulation configuration with id %s, rowcount %s",
        simulation_configuration_id,
        result.rowcount + nested_result.rowcount,
    )
