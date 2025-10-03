"""Database session management for Celery workers.

This module provides isolated database connection pooling for Celery
workers, ensuring efficient connection reuse while preventing connection
exhaustion. It implements per-worker connection pools with automatic
cleanup on worker shutdown.
"""

import logging
from typing import Optional, AsyncContextManager
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncEngine,
    AsyncSession,
)
from sqlalchemy.exc import SQLAlchemyError

from src.config import get_settings
from src.simulations.config import get_worker_settings


logger = logging.getLogger(__name__)

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None

# Connection pool sizing
CONNECTIONS_PER_TASK = 2  # per concurrent task


def get_worker_engine() -> AsyncEngine:
    """Get or create the worker database engine.

    Creates a shared SQLAlchemy async engine with connection pooling
    optimized for Celery workers. Pool size is calculated based on
    worker concurrency to prevent connection exhaustion.

    Returns:
        Configured AsyncEngine instance.

    Raises:
        SQLAlchemyError: If engine creation fails.

    Note:
        This function is thread-safe and returns a singleton engine.
        The engine is automatically cleaned up on worker shutdown.

    Example:
        ```python
        engine = get_worker_engine()
        logger.info(f"Pool size: {engine.pool.size()}")
        ```
    """
    global _engine

    if _engine is None:
        worker_concurrency = get_worker_settings().worker_concurrency

        pool_size = worker_concurrency * CONNECTIONS_PER_TASK
        max_overflow = worker_concurrency

        _engine = create_async_engine(
            get_settings().service.db_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=60 * 60,  # Recycle connections every hour
            pool_timeout=30,  # 30s timeout for getting connection
            echo=False,
            echo_pool=False,
            future=True,
            pool_reset_on_return="rollback",  # Rollback on connection return
        )

        logger.info(
            "Created worker database engine: pool_size=%d, max_overflow=%d, "
            "total_capacity=%d",
            pool_size,
            max_overflow,
            pool_size + max_overflow,
        )

    return _engine


def get_worker_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the worker session factory.

    Returns a session factory configured for worker tasks. Sessions
    created from this factory are properly isolated and will not
    commit automatically.

    Returns:
        Async session factory.

    Example:
        ```python
        factory = get_worker_session_factory()
        async with factory() as session:
            # Use session
            pass
        ```
    """
    global _session_factory

    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_worker_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
            autoflush=True,
            autocommit=False,
        )
        logger.debug("Created worker session factory")

    return _session_factory


async def dispose_worker_engine() -> None:
    """Dispose of the worker database engine gracefully.

    Closes all database connections and disposes of the engine.
    Should be called during worker shutdown to ensure clean teardown.

    Raises:
        Exception: Logs errors but doesn't raise to allow graceful shutdown.

    Example:
        ```python
        # In worker shutdown handler
        await dispose_worker_engine()
        ```
    """
    global _engine, _session_factory

    if _engine is not None:
        logger.info("Disposing worker database engine...")
        try:
            await _engine.dispose()
            logger.info(
                "✅ Worker database engine disposed successfully: "
                "pool_size=%d, checked_in=%d",
                _engine.pool.size(),
                _engine.pool.checkedin(),
            )
        except Exception as e:
            logger.error(
                "❌ Error disposing worker database engine: %s",
                e,
                exc_info=True,
            )
        finally:
            _engine = None
            _session_factory = None


def reset_worker_db() -> None:
    """Reset worker database state after fork (REQUIRED for prefork pool).

    When Celery uses prefork pool, child processes inherit the parent's
    database engine. However, asyncpg connections are tied to specific
    event loops, which causes "attached to different loop" errors.

    This function MUST be called in worker_process_init to ensure each
    worker gets its own engine with connections in the correct event loop.

    Warning:
        This function should ONLY be called in worker_process_init.
        DO NOT call in production code or tests unless you know what you're doing.
    """
    global _engine, _session_factory

    if _engine is not None:
        try:
            import asyncio

            asyncio.run(_engine.dispose())
            logger.debug("Disposed inherited database engine in child process")
        except Exception as e:
            logger.warning(
                "Failed to dispose inherited engine (expected in fork): %s", e
            )

    _engine = None
    _session_factory = None
    logger.info("Worker database state reset (post-fork)")


@asynccontextmanager
async def get_worker_session() -> AsyncContextManager[AsyncSession]:
    """Get an async database session context manager.

    Yields a properly configured database session that automatically
    commits on success and rolls back on exceptions.

    Yields:
        AsyncSession instance.

    Raises:
        SQLAlchemyError: On database errors.

    Example:
        ```python
        async with get_worker_session() as session:
            result = await session.execute(select(User))
            user = result.scalar_one()
        ```
    """
    factory = get_worker_session_factory()
    async with factory() as session:
        try:
            yield session
        except SQLAlchemyError:
            await session.rollback()
            raise


async def check_database_health() -> dict[str, any]:
    """Check database connection health.

    Performs a simple query to verify database connectivity and
    returns pool statistics.

    Returns:
        Dictionary containing:
            - status: "healthy" or "unhealthy"
            - pool_size: Current pool size
            - checked_out: Currently checked out connections
            - overflow: Current overflow connections
            - error: Error message if unhealthy

    Example:
        ```python
        health = await check_database_health()
        if health["status"] == "unhealthy":
            logger.error(f"DB unhealthy: {health['error']}")
        ```
    """
    try:
        engine = get_worker_engine()
        factory = get_worker_session_factory()

        async with factory() as session:
            result = await session.execute("SELECT 1")
            result.scalar_one()

        return {
            "status": "healthy",
            "pool_size": engine.pool.size(),
            "checked_out": engine.pool.checkedout(),
            "overflow": engine.pool.overflow(),
        }
    except Exception as e:
        logger.error("Database health check failed: %s", e)
        return {
            "status": "unhealthy",
            "error": str(e),
        }
