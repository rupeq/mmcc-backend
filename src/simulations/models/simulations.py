import uuid
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Text, func, Boolean
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.core.model_base import Base
from src.simulations.models.enums import ReportStatus
from sqlalchemy.types import Enum as SQLAlchemyEnum

if TYPE_CHECKING:
    from src.users.models.users import User
    from src.background_tasks.models.background_tasks import BackgroundTask


class SimulationConfiguration(Base):
    """Represent a simulation configuration.

    Attributes:
        id: The unique identifier of the simulation configuration.
        name: The name of the simulation configuration.
        description: The description of the simulation configuration.
        simulation_parameters: The parameters for the simulation.
        user_id: The ID of the user who owns this configuration.
        user: The user who owns this configuration.
        reports: The list of simulation reports associated with this configuration.
        created_at: The timestamp when the configuration was created.
        updated_at: The timestamp when the configuration was last updated.
        is_active: Indicates if the configuration is active.
    """

    __tablename__ = "simulation_configurations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    simulation_parameters: Mapped[dict] = mapped_column(JSONB, nullable=False)

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    user: Mapped["User"] = relationship(
        "User", back_populates="simulation_configurations"
    )
    reports: Mapped[list["SimulationReport"]] = relationship(
        back_populates="configuration", cascade="all, delete-orphan"
    )

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class SimulationReport(Base):
    """Represent a simulation report.

    Attributes:
        id: The unique identifier of the simulation report.
        status: The current status of the report (pending, completed, failed).
        results: The results of the simulation.
        error_message: The error message if the simulation failed.
        configuration_id: The ID of the simulation configuration this report belongs to.
        configuration: The simulation configuration this report belongs to.
        created_at: The timestamp when the report was created.
        completed_at: The timestamp when the report was completed.
        is_active: Indicates if the report is active.
    """

    __tablename__ = "simulation_reports"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    status: Mapped[ReportStatus] = mapped_column(
        SQLAlchemyEnum(ReportStatus),
        nullable=False,
        default=ReportStatus.PENDING,
        index=True,
    )

    results: Mapped[dict | None] = mapped_column(JSONB)

    error_message: Mapped[str | None] = mapped_column(Text)

    configuration_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("simulation_configurations.id", ondelete="CASCADE"),
        index=True,
    )
    configuration: Mapped["SimulationConfiguration"] = relationship(
        back_populates="reports"
    )
    tasks: Mapped[list["BackgroundTask"]] = relationship(
        primaryjoin="BackgroundTask.subject_id == foreign(SimulationReport.id)",
    )

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[DateTime | None] = mapped_column(
        DateTime(timezone=True)
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
