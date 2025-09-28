from typing import Any, AsyncGenerator

from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
    AsyncSession,
)

from src.config import get_settings

engine = create_async_engine(
    get_settings().service.db_url, future=True, echo=False, pool_pre_ping=True
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, Any]:
    async with SessionLocal() as session:
        yield session
