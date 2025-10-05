from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, declared_attr
from sqlalchemy import MetaData


convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
metadata = MetaData(naming_convention=convention)


class Base(AsyncAttrs, DeclarativeBase):
    """
    Base class for SQLAlchemy declarative models, providing naming conventions and __tablename__.
    """

    metadata = metadata

    @declared_attr.directive
    def __tablename__(cls) -> str:
        """
        Generate the table name for the model based on its class name.

        Returns:
            str: The lowercase class name as the table name.
        """
        return cls.__name__.lower()
