from functools import lru_cache
from typing import Any, AsyncGenerator

from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
    AsyncSession,
    AsyncEngine,
)

from src.config import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    """Get a cached instance of the SQLAlchemy async engine."""
    return create_async_engine(
        get_settings().service.db_url,
        future=True,
        echo=False,
        pool_pre_ping=True,
    )


@lru_cache
def get_session_local() -> async_sessionmaker[AsyncSession]:
    """Get a cached instance of the async session maker."""
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, Any]:
    """
    Provide an asynchronous database session.

    Yields:
        AsyncSession: An asynchronous SQLAlchemy session.
    """
    session_local = get_session_local()
    async with session_local() as session:
        yield session
