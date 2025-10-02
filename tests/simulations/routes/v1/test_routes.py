import uuid
import pytest
from another_fastapi_jwt_auth.exceptions import JWTDecodeError
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch

from another_fastapi_jwt_auth import AuthJWT
from src.main import app
from src.core.db_session import get_session
from src.simulations.core.schemas import SimulationRequest, ExponentialParams

from src.simulations.routes.v1.exceptions import (
    BadFilterFormat,
    InvalidColumn,
    InvalidReportStatus,
)

TEST_USER_ID = uuid.uuid4()
BASE_URL = "/api/v1/simulations"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def setup_mocks(mocker):
    mock_authorize = MagicMock()
    mock_authorize.get_jwt_subject.return_value = str(TEST_USER_ID)
    mocker.patch.object(AuthJWT, "jwt_required", return_value=None)
    app.dependency_overrides[AuthJWT] = lambda: mock_authorize

    mock_session = AsyncMock()
    app.dependency_overrides[get_session] = lambda: mock_session

    yield {"authorize": mock_authorize, "session": mock_session}

    app.dependency_overrides = {}


@pytest.fixture
def mock_simulation_configs():
    class MockSimConfig:
        def __init__(self, id, name, description):
            self.id = id
            self.name = name
            self.description = description
            self.created_at = "2025-10-01T12:00:00Z"
            self.updated_at = "2025-10-01T12:00:00Z"

    return [
        MockSimConfig(uuid.uuid4(), "Test Sim 1", "Description for sim 1"),
        MockSimConfig(uuid.uuid4(), "Another Sim 2", "Details about sim 2"),
    ]


def test_get_simulations_unauthorized(client):
    app.dependency_overrides = {}
    with patch.object(
        AuthJWT,
        "jwt_required",
        side_effect=JWTDecodeError(
            status.HTTP_401_UNAUTHORIZED, "Missing cookie"
        ),
    ):
        response = client.get(BASE_URL)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Missing cookie" in response.json()["detail"]


@patch("src.simulations.routes.v1.routes.get_simulation_configurations")
def test_get_simulations_success_basic(
    mock_get_configs, client, mock_simulation_configs
):
    mock_get_configs.return_value = (
        mock_simulation_configs,
        len(mock_simulation_configs),
    )
    response = client.get(BASE_URL)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    mock_get_configs.assert_called_once()
    assert data["total_items"] == 2
    assert len(data["items"]) == 2
    assert data["items"][0]["name"] == "Test Sim 1"


@patch("src.simulations.routes.v1.routes.get_simulation_configurations")
def test_get_simulations_with_pagination(
    mock_get_configs, client, mock_simulation_configs
):
    mock_get_configs.return_value = ([mock_simulation_configs[0]], 10)
    response = client.get(f"{BASE_URL}?page=2&limit=5")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    call_args = mock_get_configs.call_args[1]
    assert call_args["page"] == 2
    assert call_args["limit"] == 5
    assert data["total_pages"] == 2


@patch("src.simulations.routes.v1.routes.get_simulation_configurations")
def test_get_simulations_with_column_selection(
    mock_get_configs, client, mock_simulation_configs
):
    mock_get_configs.return_value = (mock_simulation_configs, 2)
    response = client.get(f"{BASE_URL}?columns=name,description")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    item = data["items"][0]
    assert "name" in item
    assert "description" in item
    assert "created_at" not in item


@patch(
    "src.simulations.routes.v1.routes.validate_simulation_columns",
    side_effect=InvalidColumn,
)
def test_get_simulations_with_invalid_column(mock_validator, client):
    response = client.get(f"{BASE_URL}?columns=non_existent_field")
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid columns" in response.json()["detail"]


@patch("src.simulations.routes.v1.routes.get_simulation_configurations")
def test_get_simulations_with_filters(
    mock_get_configs, client, mock_simulation_configs
):
    mock_get_configs.return_value = ([mock_simulation_configs[0]], 1)
    response = client.get(
        f"{BASE_URL}?filters=name:Test,report_status:completed"
    )
    assert response.status_code == status.HTTP_200_OK
    call_args = mock_get_configs.call_args[1]
    assert call_args["filters"] == {
        "name": "Test",
        "report_status": "completed",
    }


@pytest.mark.parametrize(
    "invalid_filter, mock_path, exception, detail",
    [
        (
            "name-Test",
            "src.simulations.routes.v1.routes.parse_search_query",
            BadFilterFormat,
            "Invalid search format.",
        ),
        (
            "report_status:wrong",
            "src.simulations.routes.v1.routes.verify_report_status_value",
            InvalidReportStatus,
            "Invalid report_status.",
        ),
        (
            "name:Test:Value",
            "src.simulations.routes.v1.routes.parse_search_query",
            BadFilterFormat,
            "Invalid search format.",
        ),
    ],
)
def test_get_simulations_with_invalid_filters(
    client, mocker, invalid_filter, mock_path, exception, detail
):
    mocker.patch(mock_path, side_effect=exception)
    response = client.get(f"{BASE_URL}?filters={invalid_filter}")
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert detail in response.json()["detail"]


@patch("src.simulations.routes.v1.routes.get_simulation_configurations")
def test_get_simulations_no_results(mock_get_configs, client):
    mock_get_configs.return_value = ([], 0)
    response = client.get(f"{BASE_URL}?filters=name:DoesNotExist&limit=10")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["total_items"] == 0
    assert data["items"] == []
    assert data["total_pages"] == 0


def make_body(
    *,
    name: str = "My simulation",
    description: str | None = "Test description",
    num_channels: int = 2,
    simulation_time: float = 10.0,
    num_replications: int = 3,
    arrival_rate: float = 1.2,
    service_rate: float = 2.5,
    random_seed: int | None = 42,
):
    return {
        "name": name,
        "description": description,
        "simulationParameters": {
            "numChannels": num_channels,
            "simulationTime": simulation_time,
            "numReplications": num_replications,
            "arrivalProcess": {
                "distribution": "exponential",
                "rate": arrival_rate,
            },
            "serviceProcess": {
                "distribution": "exponential",
                "rate": service_rate,
            },
            "arrivalSchedule": None,
            "randomSeed": random_seed,
        },
    }


@patch("src.simulations.routes.v1.routes.create_simulation_configuration")
def test_create_simulation_success(mock_create_config, client, setup_mocks):
    config_id = uuid.uuid4()
    report_id = uuid.uuid4()

    class Cfg:
        id = config_id

    class Rep:
        id = report_id

    mock_create_config.return_value = (Cfg(), Rep())

    body = make_body()
    resp = client.post(BASE_URL, json=body)

    assert resp.status_code == status.HTTP_202_ACCEPTED
    data = resp.json()
    assert data["simulation_configuration_id"] == str(config_id)
    assert data["simulation_report_id"] == str(report_id)

    kwargs = mock_create_config.call_args.kwargs
    assert (
        "user_id" in kwargs
        and kwargs["user_id"] == setup_mocks["authorize"].get_jwt_subject()
    )
    assert kwargs["name"] == body["name"]
    assert kwargs["description"] == body["description"]

    sim_params = kwargs["simulation_parameters"]
    assert "numChannels" in sim_params
    assert "simulationTime" in sim_params
    assert "numReplications" in sim_params
    assert "arrivalProcess" in sim_params
    assert "serviceProcess" in sim_params
    assert sim_params["arrivalProcess"]["distribution"] == "exponential"
    assert sim_params["serviceProcess"]["distribution"] == "exponential"


def test_create_simulation_requires_auth(client):
    app.dependency_overrides = {}
    with patch.object(
        AuthJWT,
        "jwt_required",
        side_effect=JWTDecodeError(
            status.HTTP_401_UNAUTHORIZED, "Missing cookie"
        ),
    ):
        resp = client.post(BASE_URL, json=make_body())
    assert resp.status_code in (
        status.HTTP_401_UNAUTHORIZED,
        status.HTTP_422_UNPROCESSABLE_ENTITY,
    )
    app.dependency_overrides[AuthJWT] = lambda: MagicMock(
        jwt_required=MagicMock(side_effect=None)
    )
    app.dependency_overrides[get_session] = lambda: AsyncMock()


@patch("src.simulations.routes.v1.routes.create_simulation_configuration")
def test_create_simulation_validates_pydantic_aliases(
    mock_create_config, client
):
    class Cfg:
        id = uuid.uuid4()

    class Rep:
        id = uuid.uuid4()

    mock_create_config.return_value = (Cfg(), Rep())

    body = make_body()
    body["simulationParameters"]["arrivalProcess"] = {
        "distribution": "exponential",
        "rate": 3.33,
    }
    body["simulationParameters"]["serviceProcess"] = {
        "distribution": "exponential",
        "rate": 5.55,
    }

    resp = client.post(BASE_URL, json=body)
    assert resp.status_code == status.HTTP_202_ACCEPTED

    passed = mock_create_config.call_args.kwargs["simulation_parameters"]
    model = SimulationRequest.model_validate(passed)
    assert isinstance(model.arrival_process, ExponentialParams)
    assert isinstance(model.service_process, ExponentialParams)
    assert model.arrival_process.rate == 3.33
    assert model.service_process.rate == 5.55


@pytest.mark.parametrize(
    "invalid_body, expected_status",
    [
        (
            {"simulationParameters": make_body()["simulationParameters"]},
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ),
        (
            {
                "name": "",
                "simulationParameters": make_body()["simulationParameters"],
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ),
        (
            {
                "name": "X",
                "simulationParameters": {
                    **make_body()["simulationParameters"],
                    "numChannels": 0,
                },
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ),
        ({"name": "X"}, status.HTTP_422_UNPROCESSABLE_ENTITY),
        (
            {
                "name": "X",
                "simulationParameters": {
                    **make_body()["simulationParameters"],
                    "arrivalProcess": {"distribution": "unknown", "rate": 1.0},
                },
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ),
    ],
)
@patch(
    "src.simulations.db_utils.simulation_configurations.create_simulation_configuration"
)
def test_create_simulation_validation_errors(
    mock_create_config, client, invalid_body, expected_status
):
    resp = client.post(BASE_URL, json=invalid_body)
    assert resp.status_code == expected_status
    assert not mock_create_config.called
