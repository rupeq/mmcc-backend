import uuid

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from src.simulations.db_utils.simulation_configurations import (
    get_simulation_configurations,
    get_simulation_configuration,
    create_simulation_configuration,
    delete_simulation_configuration,
)
from src.simulations.db_utils.exceptions import (
    IdColumnRequiredException,
    SimulationNotFound,
)
from src.simulations.models.simulations import (
    SimulationConfiguration,
    SimulationReport,
)
from src.simulations.models.enums import ReportStatus


@pytest.fixture
def mock_session():
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_config():
    config = MagicMock(spec=SimulationConfiguration)
    config.id = uuid.uuid4()
    config.name = "Test Simulation"
    config.description = "Test Description"
    config.user_id = uuid.uuid4()
    config.is_active = True
    config.simulation_parameters = {"numChannels": 2}
    return config


@pytest.fixture
def mock_report():
    report = MagicMock(spec=SimulationReport)
    report.id = uuid.uuid4()
    report.status = ReportStatus.PENDING
    report.is_active = True
    return report


class TestGetSimulationConfigurations:
    @pytest.mark.asyncio
    async def test_empty_results(self, mock_session):
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0

        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = []

        mock_session.execute.side_effect = [count_result, data_result]

        configs, total = await get_simulation_configurations(
            mock_session,
            user_id=str(uuid.uuid4()),
        )

        assert len(configs) == 0
        assert total == 0
        assert mock_session.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_with_results(self, mock_session, mock_config):
        count_result = MagicMock()
        count_result.scalar_one.return_value = 1

        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = [mock_config]

        mock_session.execute.side_effect = [count_result, data_result]

        configs, total = await get_simulation_configurations(
            mock_session,
            user_id=str(mock_config.user_id),
        )

        assert len(configs) == 1
        assert total == 1
        assert configs[0].id == mock_config.id

    @pytest.mark.asyncio
    async def test_with_name_filter(self, mock_session, mock_config):
        count_result = MagicMock()
        count_result.scalar_one.return_value = 1

        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = [mock_config]

        mock_session.execute.side_effect = [count_result, data_result]

        configs, total = await get_simulation_configurations(
            mock_session,
            user_id=str(mock_config.user_id),
            filters={"name": "Test"},
        )

        assert len(configs) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_with_report_status_filter(self, mock_session, mock_config):
        count_result = MagicMock()
        count_result.scalar_one.return_value = 1

        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = [mock_config]

        mock_session.execute.side_effect = [count_result, data_result]

        configs, total = await get_simulation_configurations(
            mock_session,
            user_id=str(mock_config.user_id),
            filters={"report_status": "pending"},
        )

        assert len(configs) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_with_invalid_report_status_filter(
        self, mock_session, mock_config
    ):
        count_result = MagicMock()
        count_result.scalar_one.return_value = 1

        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = [mock_config]

        mock_session.execute.side_effect = [count_result, data_result]

        configs, total = await get_simulation_configurations(
            mock_session,
            user_id=str(mock_config.user_id),
            filters={"report_status": "invalid_status"},
        )

        assert len(configs) == 1

    @pytest.mark.asyncio
    async def test_with_pagination(self, mock_session, mock_config):
        count_result = MagicMock()
        count_result.scalar_one.return_value = 10

        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = [mock_config]

        mock_session.execute.side_effect = [count_result, data_result]

        configs, total = await get_simulation_configurations(
            mock_session,
            user_id=str(mock_config.user_id),
            page=2,
            limit=5,
        )

        assert len(configs) == 1
        assert total == 10

    @pytest.mark.asyncio
    async def test_with_columns(self, mock_session, mock_config):
        count_result = MagicMock()
        count_result.scalar_one.return_value = 1

        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = [mock_config]

        mock_session.execute.side_effect = [count_result, data_result]

        configs, total = await get_simulation_configurations(
            mock_session,
            user_id=str(mock_config.user_id),
            columns=["id", "name"],
        )

        assert len(configs) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_columns_without_id_raises_error(self, mock_session):
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0

        mock_session.execute.return_value = count_result

        with pytest.raises(IdColumnRequiredException):
            await get_simulation_configurations(
                mock_session,
                user_id=str(uuid.uuid4()),
                columns=["name", "description"],
            )

    @pytest.mark.asyncio
    async def test_multiple_filters(self, mock_session, mock_config):
        count_result = MagicMock()
        count_result.scalar_one.return_value = 1

        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = [mock_config]

        mock_session.execute.side_effect = [count_result, data_result]

        configs, total = await get_simulation_configurations(
            mock_session,
            user_id=str(mock_config.user_id),
            filters={"name": "Test", "description": "Desc"},
        )

        assert len(configs) == 1


class TestGetSimulationConfiguration:
    @pytest.mark.asyncio
    async def test_get_existing_configuration(self, mock_session, mock_config):
        query_result = MagicMock()
        query_result.scalars.return_value.one_or_none.return_value = mock_config

        mock_session.execute.return_value = query_result

        result = await get_simulation_configuration(
            mock_session,
            user_id=mock_config.user_id,
            simulation_configuration_id=mock_config.id,
        )

        assert result == mock_config
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_nonexistent_configuration(self, mock_session):
        query_result = MagicMock()
        query_result.scalars.return_value.one_or_none.return_value = None

        mock_session.execute.return_value = query_result

        with pytest.raises(SimulationNotFound):
            await get_simulation_configuration(
                mock_session,
                user_id=uuid.uuid4(),
                simulation_configuration_id=uuid.uuid4(),
            )

    @pytest.mark.asyncio
    async def test_get_configuration_wrong_user(self, mock_session):
        query_result = MagicMock()
        query_result.scalars.return_value.one_or_none.return_value = None

        mock_session.execute.return_value = query_result

        with pytest.raises(SimulationNotFound):
            await get_simulation_configuration(
                mock_session,
                user_id=uuid.uuid4(),
                simulation_configuration_id=uuid.uuid4(),
            )


class TestCreateSimulationConfiguration:
    @pytest.mark.asyncio
    async def test_create_with_all_parameters(self, mock_session):
        user_id = str(uuid.uuid4())
        params = {
            "name": "Test Simulation",
            "description": "Test Description",
            "simulation_parameters": {
                "numChannels": 2,
                "simulationTime": 100.0,
            },
        }

        with patch(
            "src.simulations.db_utils.simulation_configurations.create_simulation_report"
        ) as mock_create_report:
            mock_report = MagicMock(spec=SimulationReport)
            mock_create_report.return_value = mock_report

            config, report = await create_simulation_configuration(
                mock_session,
                user_id=user_id,
                **params,
            )

            assert config is not None
            assert config.name == params["name"]
            assert config.description == params["description"]
            assert report is not None

            mock_session.add_all.assert_called_once()
            mock_session.commit.assert_called_once()
            mock_create_report.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_without_description(self, mock_session):
        user_id = str(uuid.uuid4())
        params = {
            "name": "Test Simulation",
            "simulation_parameters": {"numChannels": 1},
        }

        with patch(
            "src.simulations.db_utils.simulation_configurations.create_simulation_report"
        ) as mock_create_report:
            mock_create_report.return_value = MagicMock()

            config, report = await create_simulation_configuration(
                mock_session,
                user_id=user_id,
                **params,
            )

            assert config.name == params["name"]
            assert config.description is None
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_calls_report_without_commit(self, mock_session):
        user_id = str(uuid.uuid4())
        params = {
            "name": "Test",
            "simulation_parameters": {},
        }

        with patch(
            "src.simulations.db_utils.simulation_configurations.create_simulation_report"
        ) as mock_create_report:
            mock_create_report.return_value = MagicMock()

            await create_simulation_configuration(
                mock_session,
                user_id=user_id,
                **params,
            )

            call_args = mock_create_report.call_args
            assert call_args.kwargs.get("should_commit") is False


class TestDeleteSimulationConfiguration:
    @pytest.mark.asyncio
    async def test_delete_existing_configuration(
        self, mock_session, mock_config
    ):
        with patch(
            "src.simulations.db_utils.simulation_configurations.get_simulation_configuration"
        ) as mock_get:
            mock_get.return_value = mock_config

            config_update_result = MagicMock()
            config_update_result.rowcount = 1

            report_update_result = MagicMock()
            report_update_result.rowcount = 1

            mock_session.execute.side_effect = [
                config_update_result,
                report_update_result,
            ]

            await delete_simulation_configuration(
                mock_session,
                simulation_configuration_id=mock_config.id,
                user_id=mock_config.user_id,
            )

            mock_get.assert_called_once_with(
                mock_session,
                user_id=mock_config.user_id,
                simulation_configuration_id=mock_config.id,
            )

            assert mock_session.execute.call_count == 2

            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_configuration(self, mock_session):
        with patch(
            "src.simulations.db_utils.simulation_configurations.get_simulation_configuration"
        ) as mock_get:
            mock_get.side_effect = SimulationNotFound()

            with pytest.raises(SimulationNotFound):
                await delete_simulation_configuration(
                    mock_session,
                    simulation_configuration_id=uuid.uuid4(),
                    user_id=uuid.uuid4(),
                )

            mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_wrong_user(self, mock_session):
        with patch(
            "src.simulations.db_utils.simulation_configurations.get_simulation_configuration"
        ) as mock_get:
            mock_get.side_effect = SimulationNotFound()

            with pytest.raises(SimulationNotFound):
                await delete_simulation_configuration(
                    mock_session,
                    simulation_configuration_id=uuid.uuid4(),
                    user_id=uuid.uuid4(),
                )

    @pytest.mark.asyncio
    async def test_delete_cascades_to_reports(self, mock_session, mock_config):
        with patch(
            "src.simulations.db_utils.simulation_configurations.get_simulation_configuration"
        ) as mock_get:
            mock_get.return_value = mock_config

            config_result = MagicMock()
            config_result.rowcount = 1

            report_result = MagicMock()
            report_result.rowcount = 3

            mock_session.execute.side_effect = [config_result, report_result]

            await delete_simulation_configuration(
                mock_session,
                simulation_configuration_id=mock_config.id,
                user_id=mock_config.user_id,
            )

            assert mock_session.execute.call_count == 2
            mock_session.commit.assert_called_once()


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_get_configs_with_zero_limit(self, mock_session):
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0

        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = []

        mock_session.execute.side_effect = [count_result, data_result]

        configs, total = await get_simulation_configurations(
            mock_session,
            user_id=str(uuid.uuid4()),
            page=1,
            limit=0,
        )

        assert len(configs) == 0

    @pytest.mark.asyncio
    async def test_get_configs_with_large_page_number(self, mock_session):
        count_result = MagicMock()
        count_result.scalar_one.return_value = 5

        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = []

        mock_session.execute.side_effect = [count_result, data_result]

        configs, total = await get_simulation_configurations(
            mock_session,
            user_id=str(uuid.uuid4()),
            page=1000,
            limit=10,
        )

        assert len(configs) == 0
        assert total == 5

    @pytest.mark.asyncio
    async def test_create_with_empty_parameters(self, mock_session):
        user_id = str(uuid.uuid4())

        with patch(
            "src.simulations.db_utils.simulation_configurations.create_simulation_report"
        ) as mock_create_report:
            mock_create_report.return_value = MagicMock()

            config, report = await create_simulation_configuration(
                mock_session,
                user_id=user_id,
            )

            assert config.name is None
            assert config.description is None
            assert config.simulation_parameters is None

    @pytest.mark.asyncio
    async def test_filter_with_invalid_column(self, mock_session):
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0

        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = []

        mock_session.execute.side_effect = [count_result, data_result]

        configs, total = await get_simulation_configurations(
            mock_session,
            user_id=str(uuid.uuid4()),
            filters={"invalid_column": "value"},
        )

        assert len(configs) == 0
