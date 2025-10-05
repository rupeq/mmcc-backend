import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Enum as SQLAlchemyEnum

from src.core.model_base import Base
from src.background_tasks.models.enums import TaskType


class BackgroundTask(Base):
    """Represent a dispatched background task record.

    Store metadata that links a Celery task to a user and subject for
    auditing and status tracking.

    Attributes:
        id: Primary key UUID of the background task.
        task_id: External Celery task identifier (string UUID).
        task_type: Type/category of the background task.
        subject_id: Subject UUID associated with the task.
        user_id: Owner user UUID (FK to users.id).
        created_at: UTC timestamp when the record was created.
    """

    __tablename__ = "background_tasks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    task_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    task_type: Mapped[TaskType] = mapped_column(
        SQLAlchemyEnum(TaskType), index=True
    )

    subject_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
