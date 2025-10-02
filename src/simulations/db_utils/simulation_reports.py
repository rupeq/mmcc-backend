import logging
import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.simulations.db_utils.exceptions import (
    SimulationReportNotFound,
    SimulationReportsNotFound,
)
from src.simulations.models.enums import ReportStatus
from src.simulations.models.simulations import (
    SimulationReport,
    SimulationConfiguration,
)


logger = logging.getLogger(__name__)


async def get_simulation_configuration_report(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    report_id: uuid.UUID,
    simulation_configuration_id: uuid.UUID,
) -> SimulationReport:
    logger.debug("Getting simulation report with id %s", report_id)
    query = await session.execute(
        select(SimulationReport)
        .join(
            SimulationConfiguration,
            SimulationConfiguration.id == SimulationReport.configuration_id,
        )
        .where(
            SimulationReport.id == report_id,
            SimulationReport.is_active == True,
            SimulationConfiguration.user_id == user_id,
            SimulationConfiguration.id == simulation_configuration_id,
            SimulationConfiguration.is_active == True,
        )
    )
    report = query.scalars().one_or_none()

    if report is None:
        logger.debug("No report found for id %s", report_id)
        raise SimulationReportNotFound()

    logger.debug("Found report for id %s", report_id)
    return report


async def get_simulation_configuration_reports(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    simulation_configuration_id: uuid.UUID,
) -> list[SimulationReport]:
    logger.debug(
        "Getting simulation reports for configuration id %s",
        simulation_configuration_id,
    )
    query = await session.execute(
        select(SimulationReport)
        .join(
            SimulationConfiguration,
            SimulationConfiguration.id == SimulationReport.configuration_id,
        )
        .where(
            SimulationReport.configuration_id == simulation_configuration_id,
            SimulationReport.is_active == True,
            SimulationConfiguration.user_id == user_id,
        )
    )
    reports = query.scalars().all()

    if not reports:
        logger.debug(
            "No report found for configuration %s", simulation_configuration_id
        )
        raise SimulationReportsNotFound()

    logger.debug("Found %s reports", len(reports))
    return list(reports)


async def create_simulation_report(
    session: AsyncSession,
    *,
    configuration: SimulationConfiguration,
    should_commit: bool = True,
) -> SimulationReport:
    """Create a new simulation report associated with a configuration.

    This function instantiates a new `SimulationReport` with a `PENDING` status
    and links it to the provided `SimulationConfiguration`. It optionally
    commits the session.

    Args:
        session: The asynchronous database session.
        configuration: The `SimulationConfiguration` object to which this
                       report will be associated.
        should_commit: If True, the session will be committed after adding the
                       report. Defaults to True.
                       (Note: Consider handling commits at a higher level
                       for better transaction management.)

    Returns:
        The newly created `SimulationReport` instance, which will be in a
        pending or persistent state depending on `should_commit`.
    """
    logger.debug(
        "Creating new simulation report for user %s", configuration.user_id
    )
    report = SimulationReport(
        status=ReportStatus.PENDING,
        configuration=configuration,
        is_active=True,
    )

    if should_commit:
        logger.debug("Committing session since should_commit set to True")
        await session.commit()

    logger.debug(
        "Successfully created simulation report for user %s",
        configuration.user_id,
    )
    return report


async def delete_simulation_configuration_report(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    report_id: uuid.UUID,
    simulation_configuration_id: uuid.UUID,
) -> None:
    logger.debug("Deleting simulation report with id %s", report_id)
    await get_simulation_configuration_report(
        session=session,
        user_id=user_id,
        report_id=report_id,
        simulation_configuration_id=simulation_configuration_id,
    )
    report_ids_to_update = (
        select(SimulationReport.id)
        .join(
            SimulationConfiguration,
            SimulationReport.configuration_id == SimulationConfiguration.id,
        )
        .where(
            SimulationReport.id == report_id,
            SimulationReport.is_active == True,
            SimulationConfiguration.user_id == user_id,
            SimulationConfiguration.id == simulation_configuration_id,
        )
    )
    result = await session.execute(
        update(SimulationReport)
        .where(SimulationReport.id.in_(report_ids_to_update))
        .values(is_active=False)
    )
    await session.commit()
    logger.debug(
        "Successfully deleted simulation report with id %s, rowcount %s",
        report_id,
        result.rowcount,
    )
