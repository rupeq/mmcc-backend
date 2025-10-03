"""Tests for worker database session management."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from src.simulations.tasks.db_session import (
    get_worker_engine,
    get_worker_session_factory,
    dispose_worker_engine,
    reset_worker_db,
    get_worker_session,
    check_database_health,
)


class TestWorkerEngineCreation:
    """Test worker engine creation and configuration."""

    def test_creates_engine_singleton(self):
        """Test that engine is created as singleton."""
        # Reset global state
        import src.simulations.tasks.db_session as db_module

        db_module._engine = None

        engine1 = get_worker_engine()
        engine2 = get_worker_engine()

        assert engine1 is engine2
        assert isinstance(engine1, AsyncEngine)

    def test_configures_connection_pool(self):
        """Test connection pool configuration."""
        import src.simulations.tasks.db_session as db_module

        db_module._engine = None

        engine = get_worker_engine()

        # Verify engine was created with pool
        assert hasattr(engine, "pool")
        assert engine.pool.size() > 0

    def test_enables_pool_pre_ping(self):
        """Test that pool pre-ping is enabled."""
        import src.simulations.tasks.db_session as db_module

        db_module._engine = None

        engine = get_worker_engine()

        # Check that pool_pre_ping is enabled
        assert engine.pool._pre_ping is True


class TestSessionFactory:
    """Test session factory creation."""

    def test_creates_session_factory(self):
        """Test session factory creation."""
        import src.simulations.tasks.db_session as db_module

        db_module._session_factory = None

        factory = get_worker_session_factory()

        assert factory is not None
        assert hasattr(factory, "__call__")

    def test_factory_creates_sessions(self):
        """Test that factory creates session instances."""
        factory = get_worker_session_factory()

        session = factory()
        assert hasattr(session, "__aenter__")
        assert hasattr(session, "__aexit__")


class TestEngineDisposal:
    """Test engine disposal and cleanup."""

    @pytest.mark.asyncio
    async def test_disposes_engine_successfully(self):
        """Test successful engine disposal."""
        import src.simulations.tasks.db_session as db_module

        # Create engine
        engine = get_worker_engine()

        # Dispose
        await dispose_worker_engine()

        # Verify engine is cleared
        assert db_module._engine is None
        assert db_module._session_factory is None

    @pytest.mark.asyncio
    async def test_handles_disposal_errors_gracefully(self):
        """Test graceful error handling during disposal."""
        import src.simulations.tasks.db_session as db_module

        # Create mock engine that raises error on dispose
        mock_engine = MagicMock()
        mock_engine.pool.size.return_value = 10
        mock_engine.pool.checkedin.return_value = 5
        mock_engine.dispose = AsyncMock(side_effect=Exception("Dispose failed"))
        db_module._engine = mock_engine

        # Should not raise - just log
        await dispose_worker_engine()

        # Engine should still be cleared
        assert db_module._engine is None


class TestDatabaseReset:
    """Test database state reset for forked workers."""

    def test_resets_global_state(self):
        """Test that reset clears global state."""
        import src.simulations.tasks.db_session as db_module

        # Set up initial state
        db_module._engine = MagicMock()
        db_module._session_factory = MagicMock()

        # Reset
        reset_worker_db()

        # Verify state is cleared
        assert db_module._engine is None
        assert db_module._session_factory is None


class TestSessionContextManager:
    """Test session context manager."""

    @pytest.mark.asyncio
    async def test_provides_session(self):
        """Test that context manager provides session."""
        async with get_worker_session() as session:
            assert isinstance(session, AsyncSession)

    @pytest.mark.asyncio
    async def test_rolls_back_on_error(self):
        """Test that session rolls back on error."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.rollback = AsyncMock()
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__.return_value = mock_session
        mock_factory.return_value.__aexit__.return_value = None

        with patch(
            "src.simulations.tasks.db_session.get_worker_session_factory",
            return_value=mock_factory,
        ):
            with pytest.raises(SQLAlchemyError):
                async with get_worker_session() as session:
                    raise SQLAlchemyError("Test error")


class TestDatabaseHealthCheck:
    """Test database health check functionality."""

    @pytest.mark.asyncio
    async def test_healthy_database(self):
        """Test health check with healthy database."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 1
        mock_session.execute.return_value = mock_result

        mock_engine = MagicMock()
        mock_engine.pool.size.return_value = 10
        mock_engine.pool.checkedout.return_value = 2
        mock_engine.pool.overflow.return_value = 0

        with (
            patch(
                "src.simulations.tasks.db_session.get_worker_engine",
                return_value=mock_engine,
            ),
            patch(
                "src.simulations.tasks.db_session.get_worker_session_factory"
            ) as mock_factory,
        ):
            mock_factory.return_value.return_value.__aenter__.return_value = (
                mock_session
            )
            mock_factory.return_value.return_value.__aexit__.return_value = None

            health = await check_database_health()

            assert health["status"] == "healthy"
            assert health["pool_size"] == 10
            assert health["checked_out"] == 2
            assert health["overflow"] == 0

    @pytest.mark.asyncio
    async def test_unhealthy_database(self):
        """Test health check with unhealthy database."""
        with patch(
            "src.simulations.tasks.db_session.get_worker_engine",
            side_effect=Exception("Connection failed"),
        ):
            health = await check_database_health()

            assert health["status"] == "unhealthy"
            assert "error" in health
            assert "Connection failed" in health["error"]


class TestConnectionPooling:
    """Test connection pool behavior."""

    def test_pool_configured_correctly(self):
        """Test that connection pool is configured."""
        import src.simulations.tasks.db_session as db_module

        db_module._engine = None

        engine = get_worker_engine()

        # Verify pool exists and has reasonable settings
        assert hasattr(engine, "pool")
        pool_size = engine.pool.size()
        assert pool_size > 0
        assert pool_size <= 100  # Reasonable upper bound
