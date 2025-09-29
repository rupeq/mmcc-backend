import uuid

from sqlalchemy import Boolean, String, func, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from src.core.model_base import Base


class User(Base):
    """
    SQLAlchemy model for the 'users' table.

    Attributes:
        id (Mapped[uuid.UUID]): The unique identifier for the user (primary key).
        email (Mapped[str]): The user's email address, unique and indexed.
        password_hash (Mapped[str]): The hashed password of the user.
        is_active (Mapped[bool]): Flag indicating if the user account is active.
        created_at (Mapped["DateTime"]): Timestamp of when the user was created.
        updated_at (Mapped["DateTime"]): Timestamp of the last update to the user's record.
    """
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)

    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(512))

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped["DateTime"] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped["DateTime"] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
