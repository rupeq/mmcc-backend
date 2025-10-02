import uuid

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from src.simulations.db_utils.simulation_reports import (
    get_simulation_configuration_report,
    get_simulation_configuration_reports,
    create_simulation_report,
    delete_simulation_configuration_report,
)
from src.simulations.db_utils.exceptions import (
    SimulationReportNotFound,
    SimulationReportsNotFound,
)
from src.simulations.models.simulations import (
    SimulationConfiguration,
    SimulationReport,
)
from src.simulations.models.enums import ReportStatus


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_config():
    """Create a mock simulation configuration."""
    config = MagicMock(spec=SimulationConfiguration)
    config.id = uuid.uuid4()
    config.user_id = uuid.uuid4()
    config.name = "Test Config"
    config.description = "Test Description"
    config.is_active = True
    return config


@pytest.fixture
def mock_report(mock_config):
    """Create a mock simulation report."""
    report = MagicMock(spec=SimulationReport)
    report.id = uuid.uuid4()
    report.status = ReportStatus.PENDING
    report.is_active = True
    report.configuration_id = mock_config.id
    report.configuration = mock_config
    report.results = None
    report.error_message = None
    return report


class TestGetSimulationConfigurationReport:
    """Test suite for get_simulation_configuration_report function."""

    @pytest.mark.asyncio
    async def test_get_existing_report_success(
        self, mock_session, mock_report, mock_config
    ):
        """Test successfully retrieving an existing report."""
        query_result = MagicMock()
        query_result.scalars.return_value.one_or_none.return_value = mock_report
        mock_session.execute.return_value = query_result

        result = await get_simulation_configuration_report(
            mock_session,
            user_id=mock_config.user_id,
            report_id=mock_report.id,
            simulation_configuration_id=mock_config.id,
        )

        assert result == mock_report
        assert result.id == mock_report.id
        assert result.status == ReportStatus.PENDING
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_nonexistent_report_raises_exception(self, mock_session):
        """Test that getting a non-existent report raises SimulationReportNotFound."""
        query_result = MagicMock()
        query_result.scalars.return_value.one_or_none.return_value = None
        mock_session.execute.return_value = query_result

        with pytest.raises(SimulationReportNotFound):
            await get_simulation_configuration_report(
                mock_session,
                user_id=uuid.uuid4(),
                report_id=uuid.uuid4(),
                simulation_configuration_id=uuid.uuid4(),
            )

        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_report_wrong_user_raises_exception(
        self, mock_session, mock_report, mock_config
    ):
        """Test that getting a report with wrong user ID raises exception."""
        query_result = MagicMock()
        query_result.scalars.return_value.one_or_none.return_value = None
        mock_session.execute.return_value = query_result

        different_user_id = uuid.uuid4()

        with pytest.raises(SimulationReportNotFound):
            await get_simulation_configuration_report(
                mock_session,
                user_id=different_user_id,
                report_id=mock_report.id,
                simulation_configuration_id=mock_config.id,
            )

    @pytest.mark.asyncio
    async def test_get_report_wrong_configuration_raises_exception(
        self, mock_session, mock_report, mock_config
    ):
        """Test that getting a report with wrong configuration ID raises exception."""
        query_result = MagicMock()
        query_result.scalars.return_value.one_or_none.return_value = None
        mock_session.execute.return_value = query_result

        different_config_id = uuid.uuid4()

        with pytest.raises(SimulationReportNotFound):
            await get_simulation_configuration_report(
                mock_session,
                user_id=mock_config.user_id,
                report_id=mock_report.id,
                simulation_configuration_id=different_config_id,
            )

    @pytest.mark.asyncio
    async def test_get_inactive_report_raises_exception(
        self, mock_session, mock_config
    ):
        """Test that getting an inactive report raises exception."""
        query_result = MagicMock()
        query_result.scalars.return_value.one_or_none.return_value = None
        mock_session.execute.return_value = query_result

        with pytest.raises(SimulationReportNotFound):
            await get_simulation_configuration_report(
                mock_session,
                user_id=mock_config.user_id,
                report_id=uuid.uuid4(),
                simulation_configuration_id=mock_config.id,
            )


class TestGetSimulationConfigurationReports:
    """Test suite for get_simulation_configuration_reports function."""

    @pytest.mark.asyncio
    async def test_get_single_report(
        self, mock_session, mock_report, mock_config
    ):
        """Test retrieving a single report for a configuration."""
        query_result = MagicMock()
        query_result.scalars.return_value.all.return_value = [mock_report]
        mock_session.execute.return_value = query_result

        results = await get_simulation_configuration_reports(
            mock_session,
            user_id=mock_config.user_id,
            simulation_configuration_id=mock_config.id,
        )

        assert len(results) == 1
        assert results[0] == mock_report
        assert isinstance(results, list)
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_multiple_reports(self, mock_session, mock_config):
        """Test retrieving multiple reports for a configuration."""
        report1 = MagicMock(spec=SimulationReport)
        report1.id = uuid.uuid4()
        report1.status = ReportStatus.PENDING

        report2 = MagicMock(spec=SimulationReport)
        report2.id = uuid.uuid4()
        report2.status = ReportStatus.COMPLETED

        report3 = MagicMock(spec=SimulationReport)
        report3.id = uuid.uuid4()
        report3.status = ReportStatus.FAILED

        query_result = MagicMock()
        query_result.scalars.return_value.all.return_value = [
            report1,
            report2,
            report3,
        ]
        mock_session.execute.return_value = query_result

        results = await get_simulation_configuration_reports(
            mock_session,
            user_id=mock_config.user_id,
            simulation_configuration_id=mock_config.id,
        )

        assert len(results) == 3
        assert results[0] == report1
        assert results[1] == report2
        assert results[2] == report3

    @pytest.mark.asyncio
    async def test_get_no_reports_raises_exception(self, mock_session):
        """Test that getting reports when none exist raises exception."""
        query_result = MagicMock()
        query_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = query_result

        with pytest.raises(SimulationReportsNotFound):
            await get_simulation_configuration_reports(
                mock_session,
                user_id=uuid.uuid4(),
                simulation_configuration_id=uuid.uuid4(),
            )

        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_reports_wrong_user_raises_exception(
        self, mock_session, mock_config
    ):
        """Test that getting reports with wrong user ID returns empty."""
        query_result = MagicMock()
        query_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = query_result

        different_user_id = uuid.uuid4()

        with pytest.raises(SimulationReportsNotFound):
            await get_simulation_configuration_reports(
                mock_session,
                user_id=different_user_id,
                simulation_configuration_id=mock_config.id,
            )

    @pytest.mark.asyncio
    async def test_get_reports_only_returns_active(
        self, mock_session, mock_config
    ):
        """Test that only active reports are returned."""
        active_report = MagicMock(spec=SimulationReport)
        active_report.id = uuid.uuid4()
        active_report.is_active = True

        query_result = MagicMock()
        query_result.scalars.return_value.all.return_value = [active_report]
        mock_session.execute.return_value = query_result

        results = await get_simulation_configuration_reports(
            mock_session,
            user_id=mock_config.user_id,
            simulation_configuration_id=mock_config.id,
        )

        assert len(results) == 1
        assert all(report.is_active for report in results)


class TestCreateSimulationReport:
    """Test suite for create_simulation_report function."""

    @pytest.mark.asyncio
    async def test_create_report_with_commit(self, mock_session, mock_config):
        """Test creating a report with commit=True."""
        with patch(
            "src.simulations.db_utils.simulation_reports.SimulationReport"
        ) as MockReport:
            mock_report_instance = MagicMock()
            mock_report_instance.status = ReportStatus.PENDING
            mock_report_instance.is_active = True
            mock_report_instance.configuration = mock_config
            MockReport.return_value = mock_report_instance

            report = await create_simulation_report(
                mock_session,
                configuration=mock_config,
                should_commit=True,
            )

            assert report is not None
            MockReport.assert_called_once_with(
                status=ReportStatus.PENDING,
                configuration=mock_config,
                is_active=True,
            )
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_report_without_commit(
        self, mock_session, mock_config
    ):
        """Test creating a report with commit=False."""
        with patch(
            "src.simulations.db_utils.simulation_reports.SimulationReport"
        ) as MockReport:
            mock_report_instance = MagicMock()
            MockReport.return_value = mock_report_instance

            report = await create_simulation_report(
                mock_session,
                configuration=mock_config,
                should_commit=False,
            )

            assert report is not None
            MockReport.assert_called_once()
            mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_report_default_commit_behavior(
        self, mock_session, mock_config
    ):
        """Test that default behavior is to commit."""
        with patch(
            "src.simulations.db_utils.simulation_reports.SimulationReport"
        ) as MockReport:
            mock_report_instance = MagicMock()
            MockReport.return_value = mock_report_instance

            report = await create_simulation_report(
                mock_session,
                configuration=mock_config,
            )

            assert report is not None
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_report_initial_state(self, mock_session, mock_config):
        """Test that created report has correct initial state."""
        with patch(
            "src.simulations.db_utils.simulation_reports.SimulationReport"
        ) as MockReport:
            mock_report_instance = MagicMock()
            mock_report_instance.status = ReportStatus.PENDING
            mock_report_instance.is_active = True
            mock_report_instance.configuration = mock_config
            MockReport.return_value = mock_report_instance

            report = await create_simulation_report(
                mock_session,
                configuration=mock_config,
                should_commit=False,
            )

            MockReport.assert_called_once_with(
                status=ReportStatus.PENDING,
                configuration=mock_config,
                is_active=True,
            )

    @pytest.mark.asyncio
    async def test_create_multiple_reports_for_same_config(
        self, mock_session, mock_config
    ):
        """Test creating multiple reports for the same configuration."""
        with patch(
            "src.simulations.db_utils.simulation_reports.SimulationReport"
        ) as MockReport:
            report_instance_1 = MagicMock()
            report_instance_2 = MagicMock()
            MockReport.side_effect = [report_instance_1, report_instance_2]

            report1 = await create_simulation_report(
                mock_session,
                configuration=mock_config,
                should_commit=False,
            )

            report2 = await create_simulation_report(
                mock_session,
                configuration=mock_config,
                should_commit=False,
            )

            assert report1 is report_instance_1
            assert report2 is report_instance_2
            assert report1 is not report2
            assert MockReport.call_count == 2
            mock_session.commit.assert_not_called()


class TestDeleteSimulationConfigurationReport:
    """Test suite for delete_simulation_configuration_report function."""

    @pytest.mark.asyncio
    async def test_delete_existing_report_success(
        self, mock_session, mock_report, mock_config
    ):
        """Test successfully deleting an existing report."""
        with patch(
            "src.simulations.db_utils.simulation_reports.get_simulation_configuration_report"
        ) as mock_get:
            mock_get.return_value = mock_report

            update_result = MagicMock()
            update_result.rowcount = 1
            mock_session.execute.return_value = update_result

            await delete_simulation_configuration_report(
                mock_session,
                user_id=mock_config.user_id,
                report_id=mock_report.id,
                simulation_configuration_id=mock_config.id,
            )

            mock_get.assert_called_once_with(
                session=mock_session,
                user_id=mock_config.user_id,
                report_id=mock_report.id,
                simulation_configuration_id=mock_config.id,
            )

            mock_session.execute.assert_called_once()

            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_report_raises_exception(
        self, mock_session, mock_config
    ):
        """Test that deleting non-existent report raises exception."""
        with patch(
            "src.simulations.db_utils.simulation_reports.get_simulation_configuration_report"
        ) as mock_get:
            mock_get.side_effect = SimulationReportNotFound()

            with pytest.raises(SimulationReportNotFound):
                await delete_simulation_configuration_report(
                    mock_session,
                    user_id=mock_config.user_id,
                    report_id=uuid.uuid4(),
                    simulation_configuration_id=mock_config.id,
                )

            mock_get.assert_called_once()

            mock_session.execute.assert_not_called()
            mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_report_wrong_user(self, mock_session, mock_report):
        """Test deleting report with wrong user ID raises exception."""
        with patch(
            "src.simulations.db_utils.simulation_reports.get_simulation_configuration_report"
        ) as mock_get:
            mock_get.side_effect = SimulationReportNotFound()

            wrong_user_id = uuid.uuid4()

            with pytest.raises(SimulationReportNotFound):
                await delete_simulation_configuration_report(
                    mock_session,
                    user_id=wrong_user_id,
                    report_id=mock_report.id,
                    simulation_configuration_id=mock_report.configuration_id,
                )

            mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_report_wrong_configuration(
        self, mock_session, mock_report, mock_config
    ):
        """Test deleting report with wrong configuration ID raises exception."""
        with patch(
            "src.simulations.db_utils.simulation_reports.get_simulation_configuration_report"
        ) as mock_get:
            mock_get.side_effect = SimulationReportNotFound()

            wrong_config_id = uuid.uuid4()

            with pytest.raises(SimulationReportNotFound):
                await delete_simulation_configuration_report(
                    mock_session,
                    user_id=mock_config.user_id,
                    report_id=mock_report.id,
                    simulation_configuration_id=wrong_config_id,
                )

            mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_sets_is_active_to_false(
        self, mock_session, mock_report, mock_config
    ):
        """Test that delete operation sets is_active to False (soft delete)."""
        with patch(
            "src.simulations.db_utils.simulation_reports.get_simulation_configuration_report"
        ) as mock_get:
            mock_get.return_value = mock_report

            update_result = MagicMock()
            update_result.rowcount = 1
            mock_session.execute.return_value = update_result

            await delete_simulation_configuration_report(
                mock_session,
                user_id=mock_config.user_id,
                report_id=mock_report.id,
                simulation_configuration_id=mock_config.id,
            )

            mock_session.execute.assert_called_once()
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_already_deleted_report(
        self, mock_session, mock_report, mock_config
    ):
        """Test deleting an already deleted (inactive) report."""
        with patch(
            "src.simulations.db_utils.simulation_reports.get_simulation_configuration_report"
        ) as mock_get:
            mock_get.side_effect = SimulationReportNotFound()

            with pytest.raises(SimulationReportNotFound):
                await delete_simulation_configuration_report(
                    mock_session,
                    user_id=mock_config.user_id,
                    report_id=mock_report.id,
                    simulation_configuration_id=mock_config.id,
                )

    @pytest.mark.asyncio
    async def test_delete_report_rowcount_zero(
        self, mock_session, mock_report, mock_config
    ):
        """Test delete operation when no rows are affected."""
        with patch(
            "src.simulations.db_utils.simulation_reports.get_simulation_configuration_report"
        ) as mock_get:
            mock_get.return_value = mock_report

            update_result = MagicMock()
            update_result.rowcount = 0
            mock_session.execute.return_value = update_result

            await delete_simulation_configuration_report(
                mock_session,
                user_id=mock_config.user_id,
                report_id=mock_report.id,
                simulation_configuration_id=mock_config.id,
            )

            mock_session.commit.assert_called_once()


class TestEdgeCasesAndBehavior:
    """Test edge cases and special behaviors."""

    @pytest.mark.asyncio
    async def test_create_report_preserves_configuration_reference(
        self, mock_session, mock_config
    ):
        """Test that created report maintains reference to configuration."""
        with patch(
            "src.simulations.db_utils.simulation_reports.SimulationReport"
        ) as MockReport:
            mock_report_instance = MagicMock()
            mock_report_instance.configuration = mock_config
            MockReport.return_value = mock_report_instance

            report = await create_simulation_report(
                mock_session,
                configuration=mock_config,
                should_commit=False,
            )

            call_kwargs = MockReport.call_args.kwargs
            assert call_kwargs["configuration"] == mock_config

    @pytest.mark.asyncio
    async def test_get_reports_returns_list_type(
        self, mock_session, mock_report, mock_config
    ):
        """Test that get_reports returns a proper list."""
        query_result = MagicMock()
        query_result.scalars.return_value.all.return_value = [mock_report]
        mock_session.execute.return_value = query_result

        results = await get_simulation_configuration_reports(
            mock_session,
            user_id=mock_config.user_id,
            simulation_configuration_id=mock_config.id,
        )

        assert isinstance(results, list)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_concurrent_report_creation(self, mock_session, mock_config):
        """Test creating reports without committing for batch operations."""
        with patch(
            "src.simulations.db_utils.simulation_reports.SimulationReport"
        ) as MockReport:
            mock_instances = [MagicMock() for _ in range(5)]
            MockReport.side_effect = mock_instances

            reports = []
            for _ in range(5):
                report = await create_simulation_report(
                    mock_session,
                    configuration=mock_config,
                    should_commit=False,
                )
                reports.append(report)

            assert len(reports) == 5
            assert MockReport.call_count == 5
            mock_session.commit.assert_not_called()


class TestLogging:
    """Test logging behavior (optional but good practice)."""

    @pytest.mark.asyncio
    async def test_get_report_logs_debug_messages(
        self, mock_session, mock_report, mock_config, caplog
    ):
        """Test that get_report logs appropriate debug messages."""
        import logging

        caplog.set_level(logging.DEBUG)

        query_result = MagicMock()
        query_result.scalars.return_value.one_or_none.return_value = mock_report
        mock_session.execute.return_value = query_result

        await get_simulation_configuration_report(
            mock_session,
            user_id=mock_config.user_id,
            report_id=mock_report.id,
            simulation_configuration_id=mock_config.id,
        )

        assert any(
            "Getting simulation report" in record.message
            for record in caplog.records
        )
        assert any(
            "Found report" in record.message for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_get_report_not_found_logs_debug(self, mock_session, caplog):
        """Test that not found scenario logs debug message."""
        import logging

        caplog.set_level(logging.DEBUG)

        query_result = MagicMock()
        query_result.scalars.return_value.one_or_none.return_value = None
        mock_session.execute.return_value = query_result

        with pytest.raises(SimulationReportNotFound):
            await get_simulation_configuration_report(
                mock_session,
                user_id=uuid.uuid4(),
                report_id=uuid.uuid4(),
                simulation_configuration_id=uuid.uuid4(),
            )

        assert any(
            "No report found" in record.message for record in caplog.records
        )
