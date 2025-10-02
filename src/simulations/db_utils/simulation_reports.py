from sqlalchemy.ext.asyncio import AsyncSession

from src.simulations.models.enums import ReportStatus
from src.simulations.models.simulations import (
    SimulationReport,
    SimulationConfiguration,
)


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
    report = SimulationReport(
        status=ReportStatus.PENDING,
        configuration=configuration,
    )

    if should_commit:
        await session.commit()

    return report
