from typing import Any, Sequence

from sqlalchemy import select, func, Row, RowMapping
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import load_only

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
            columns.insert(0, "id")

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
