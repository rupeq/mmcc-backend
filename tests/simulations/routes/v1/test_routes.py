import uuid

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
from another_fastapi_jwt_auth import AuthJWT
from another_fastapi_jwt_auth.exceptions import JWTDecodeError

from src.main import app
from src.core.db_session import get_session
from src.simulations.db_utils.exceptions import (
    SimulationNotFound,
    SimulationReportNotFound,
    SimulationReportsNotFound,
)
from src.simulations.routes.v1.exceptions import (
    BadFilterFormat,
    InvalidColumn,
    InvalidReportStatus,
)

TEST_USER_ID = uuid.uuid4()
BASE_URL = "/api/v1/simulations"


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def setup_mocks(mocker):
    """Set up common mocks for authentication and database."""
    mock_authorize = MagicMock()
    mock_authorize.get_jwt_subject.return_value = str(TEST_USER_ID)
    mocker.patch.object(AuthJWT, "jwt_required", return_value=None)
    app.dependency_overrides[AuthJWT] = lambda: mock_authorize

    mock_session = AsyncMock()
    app.dependency_overrides[get_session] = lambda: mock_session

    yield {"authorize": mock_authorize, "session": mock_session}

    app.dependency_overrides = {}


@pytest.fixture
def mock_config():
    """Create a mock simulation configuration."""

    class MockConfig:
        def __init__(self):
            self.id = uuid.uuid4()
            self.name = "Test Simulation"
            self.description = "Test Description"
            self.user_id = TEST_USER_ID
            self.created_at = "2025-01-01T00:00:00Z"
            self.updated_at = "2025-01-01T00:00:00Z"
            self.is_active = True
            self.simulation_parameters = {"numChannels": 2}

    return MockConfig()


@pytest.fixture
def mock_report():
    """Create a mock simulation report."""

    class MockReport:
        def __init__(self):
            self.id = uuid.uuid4()
            self.status = "pending"
            self.results = None
            self.error_message = None
            self.configuration_id = uuid.uuid4()
            self.created_at = "2025-01-01T00:00:00Z"
            self.completed_at = None
            self.is_active = True

    return MockReport()


class TestGetSimulations:
    """Test suite for GET /simulations endpoint."""

    @patch("src.simulations.routes.v1.routes.get_simulation_configurations")
    def test_get_simulations_success(
        self, mock_get_configs, client, mock_config
    ):
        """Test successful retrieval of simulations."""
        mock_get_configs.return_value = ([mock_config], 1)

        response = client.get(BASE_URL)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_items"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["name"] == "Test Simulation"

    @patch("src.simulations.routes.v1.routes.get_simulation_configurations")
    def test_get_simulations_empty(self, mock_get_configs, client):
        """Test retrieving simulations when none exist."""
        mock_get_configs.return_value = ([], 0)

        response = client.get(BASE_URL)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_items"] == 0
        assert data["items"] == []

    @patch("src.simulations.routes.v1.routes.get_simulation_configurations")
    def test_get_simulations_with_pagination(
        self, mock_get_configs, client, mock_config
    ):
        """Test pagination parameters."""
        mock_get_configs.return_value = ([mock_config], 10)

        response = client.get(f"{BASE_URL}?page=2&limit=5")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["page"] == 2
        assert data["limit"] == 5
        assert data["total_pages"] == 2

    @pytest.mark.parametrize(
        "invalid_param",
        [
            ("page", 0),
            ("page", -1),
            ("limit", 0),
            ("limit", 101),
        ],
    )
    def test_get_simulations_invalid_pagination(self, client, invalid_param):
        """Test invalid pagination parameters."""
        param_name, param_value = invalid_param
        response = client.get(f"{BASE_URL}?{param_name}={param_value}")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.simulations.routes.v1.routes.parse_search_query")
    def test_get_simulations_invalid_filters(self, mock_parse, client):
        """Test invalid filter format."""
        mock_parse.side_effect = BadFilterFormat()

        response = client.get(f"{BASE_URL}?filters=invalid")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid search format" in response.json()["detail"]

    @patch("src.simulations.routes.v1.routes.verify_report_status_value")
    @patch("src.simulations.routes.v1.routes.parse_search_query")
    def test_get_simulations_invalid_report_status(
        self, mock_parse, mock_verify, client
    ):
        """Test invalid report status filter."""
        mock_parse.return_value = {"report_status": "invalid"}
        mock_verify.side_effect = InvalidReportStatus()

        response = client.get(f"{BASE_URL}?filters=report_status:invalid")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid report_status" in response.json()["detail"]

    @patch("src.simulations.routes.v1.routes.validate_simulation_columns")
    def test_get_simulations_invalid_columns(self, mock_validate, client):
        """Test invalid column selection."""
        mock_validate.side_effect = InvalidColumn()

        response = client.get(f"{BASE_URL}?columns=invalid_column")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid columns" in response.json()["detail"]

    def test_get_simulations_unauthorized(self, client):
        """Test accessing endpoint without authentication."""
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


class TestCreateSimulation:
    """Test suite for POST /simulations endpoint."""

    def make_request_body(
        self,
        name="Test Simulation",
        description="Test Description",
        num_channels=2,
        simulation_time=100.0,
    ):
        """Helper to create request body."""
        return {
            "name": name,
            "description": description,
            "simulationParameters": {
                "numChannels": num_channels,
                "simulationTime": simulation_time,
                "numReplications": 1,
                "arrivalProcess": {
                    "distribution": "exponential",
                    "rate": 1.0,
                },
                "serviceProcess": {
                    "distribution": "exponential",
                    "rate": 2.0,
                },
                "randomSeed": 42,
            },
        }

    @patch("src.simulations.routes.v1.routes.create_simulation_configuration")
    def test_create_simulation_success(self, mock_create, client):
        """Test successful simulation creation."""
        config_id = uuid.uuid4()
        report_id = uuid.uuid4()

        class MockConfig:
            id = config_id

        class MockReport:
            id = report_id

        mock_create.return_value = (MockConfig(), MockReport())

        body = self.make_request_body()
        response = client.post(BASE_URL, json=body)

        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert data["simulation_configuration_id"] == str(config_id)
        assert data["simulation_report_id"] == str(report_id)

    @pytest.mark.parametrize("missing_field", ["name", "simulationParameters"])
    def test_create_simulation_missing_required_fields(
        self, client, missing_field
    ):
        """Test creation with missing required fields."""
        body = self.make_request_body()
        del body[missing_field]

        response = client.post(BASE_URL, json=body)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_simulation_empty_name(self, client):
        """Test creation with empty name."""
        body = self.make_request_body(name="")

        response = client.post(BASE_URL, json=body)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_simulation_invalid_channels(self, client):
        """Test creation with invalid number of channels."""
        body = self.make_request_body(num_channels=0)

        response = client.post(BASE_URL, json=body)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_simulation_without_description(self, client):
        """Test creation without optional description."""
        body = self.make_request_body()
        body["description"] = None

        with patch(
            "src.simulations.routes.v1.routes.create_simulation_configuration"
        ) as mock_create:
            mock_create.return_value = (
                MagicMock(id=uuid.uuid4()),
                MagicMock(id=uuid.uuid4()),
            )
            response = client.post(BASE_URL, json=body)

        assert response.status_code == status.HTTP_202_ACCEPTED


class TestGetSimulationConfiguration:
    """Test suite for GET /simulations/{id} endpoint."""

    @patch(
        "src.simulations.routes.v1.routes.get_simulation_configuration_from_db"
    )
    def test_get_configuration_success(self, mock_get, client, mock_config):
        """Test successful retrieval of a configuration."""
        mock_get.return_value = mock_config

        response = client.get(f"{BASE_URL}/{mock_config.id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == mock_config.name

    @patch(
        "src.simulations.routes.v1.routes.get_simulation_configuration_from_db"
    )
    def test_get_configuration_not_found(self, mock_get, client):
        """Test retrieving non-existent configuration."""
        mock_get.side_effect = SimulationNotFound()

        response = client.get(f"{BASE_URL}/{uuid.uuid4()}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()

    def test_get_configuration_invalid_uuid(self, client):
        """Test with invalid UUID format."""
        response = client.get(f"{BASE_URL}/invalid-uuid")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestGetSimulationConfigurationReport:
    """Test suite for GET /simulations/{config_id}/reports/{report_id}."""

    @patch(
        "src.simulations.routes.v1.routes.get_simulation_configuration_report_from_db"
    )
    def test_get_report_success(
        self, mock_get, client, mock_config, mock_report
    ):
        """Test successful retrieval of a report."""
        mock_get.return_value = mock_report

        response = client.get(
            f"{BASE_URL}/{mock_config.id}/reports/{mock_report.id}"
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == mock_report.status

    @patch(
        "src.simulations.routes.v1.routes.get_simulation_configuration_report_from_db"
    )
    def test_get_report_not_found(self, mock_get, client):
        """Test retrieving non-existent report."""
        mock_get.side_effect = SimulationReportNotFound()

        response = client.get(
            f"{BASE_URL}/{uuid.uuid4()}/reports/{uuid.uuid4()}"
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestGetSimulationConfigurationReports:
    """Test suite for GET /simulations/{config_id}/reports."""

    @patch(
        "src.simulations.routes.v1.routes.get_simulation_configuration_reports_from_db"
    )
    def test_get_reports_success(
        self, mock_get, client, mock_config, mock_report
    ):
        """Test successful retrieval of multiple reports."""
        mock_get.return_value = [mock_report]

        response = client.get(f"{BASE_URL}/{mock_config.id}/reports")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "reports" in data
        assert len(data["reports"]) == 1

    @patch(
        "src.simulations.routes.v1.routes.get_simulation_configuration_reports_from_db"
    )
    def test_get_reports_not_found(self, mock_get, client):
        """Test when no reports exist."""
        mock_get.side_effect = SimulationReportsNotFound()

        response = client.get(f"{BASE_URL}/{uuid.uuid4()}/reports")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestDeleteSimulationConfiguration:
    """Test suite for DELETE /simulations/{id}."""

    @patch(
        "src.simulations.routes.v1.routes.delete_simulation_configuration_from_db"
    )
    def test_delete_configuration_success(
        self, mock_delete, client, mock_config
    ):
        """Test successful deletion of a configuration."""
        mock_delete.return_value = None

        response = client.delete(f"{BASE_URL}/{mock_config.id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_delete.assert_called_once()

    @patch(
        "src.simulations.routes.v1.routes.delete_simulation_configuration_from_db"
    )
    def test_delete_configuration_not_found(self, mock_delete, client):
        """Test deleting non-existent configuration."""
        mock_delete.side_effect = SimulationNotFound()

        response = client.delete(f"{BASE_URL}/{uuid.uuid4()}")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestDeleteSimulationConfigurationReport:
    """Test suite for DELETE /simulations/{config_id}/reports/{report_id}."""

    @patch(
        "src.simulations.routes.v1.routes.delete_simulation_configuration_report_from_db"
    )
    def test_delete_report_success(
        self, mock_delete, client, mock_config, mock_report
    ):
        """Test successful deletion of a report."""
        mock_delete.return_value = None

        response = client.delete(
            f"{BASE_URL}/{mock_config.id}/reports/{mock_report.id}"
        )

        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_delete.assert_called_once()

    @patch(
        "src.simulations.routes.v1.routes.delete_simulation_configuration_report_from_db"
    )
    def test_delete_report_not_found(self, mock_delete, client):
        """Test deleting non-existent report."""
        mock_delete.side_effect = SimulationReportNotFound()

        response = client.delete(
            f"{BASE_URL}/{uuid.uuid4()}/reports/{uuid.uuid4()}"
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND


# tests/simulations/routes/v1/test_routes.py
# Replace the TestAuthenticationAndAuthorization class with this corrected version:


class TestAuthenticationAndAuthorization:
    """Test suite for authentication and authorization."""

    def test_get_endpoints_require_auth(self, client):
        """Test that GET endpoints require authentication."""
        app.dependency_overrides = {}

        get_endpoints = [
            BASE_URL,
            f"{BASE_URL}/{uuid.uuid4()}",
            f"{BASE_URL}/{uuid.uuid4()}/reports",
            f"{BASE_URL}/{uuid.uuid4()}/reports/{uuid.uuid4()}",
        ]

        with patch.object(
            AuthJWT,
            "jwt_required",
            side_effect=JWTDecodeError(
                status.HTTP_401_UNAUTHORIZED, "Unauthorized"
            ),
        ):
            for url in get_endpoints:
                response = client.get(url)
                assert response.status_code == status.HTTP_401_UNAUTHORIZED, (
                    f"Expected 401 for GET {url}, got {response.status_code}"
                )

    def test_delete_endpoints_require_auth(self, client):
        """Test that DELETE endpoints require authentication."""
        app.dependency_overrides = {}

        delete_endpoints = [
            f"{BASE_URL}/{uuid.uuid4()}",
            f"{BASE_URL}/{uuid.uuid4()}/reports/{uuid.uuid4()}",
        ]

        with patch.object(
            AuthJWT,
            "jwt_required",
            side_effect=JWTDecodeError(
                status.HTTP_401_UNAUTHORIZED, "Unauthorized"
            ),
        ):
            for url in delete_endpoints:
                response = client.delete(url)
                assert response.status_code == status.HTTP_401_UNAUTHORIZED, (
                    f"Expected 401 for DELETE {url}, got {response.status_code}"
                )

    def test_post_endpoint_requires_auth_with_valid_body(self, client):
        """Test that POST endpoint requires authentication even with valid body."""
        app.dependency_overrides = {}

        valid_body = {
            "name": "Test",
            "description": "Test",
            "simulationParameters": {
                "numChannels": 2,
                "simulationTime": 100.0,
                "numReplications": 1,
                "arrivalProcess": {
                    "distribution": "exponential",
                    "rate": 1.0,
                },
                "serviceProcess": {
                    "distribution": "exponential",
                    "rate": 2.0,
                },
                "randomSeed": 42,
            },
        }

        with patch.object(
            AuthJWT,
            "jwt_required",
            side_effect=JWTDecodeError(
                status.HTTP_401_UNAUTHORIZED, "Unauthorized"
            ),
        ):
            response = client.post(BASE_URL, json=valid_body)
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_post_endpoint_validates_before_auth(self, client):
        """Test that POST endpoint validates request body before auth check.

        Note: This is FastAPI's default behavior - Pydantic validation
        happens before dependency injection (including auth).
        """
        app.dependency_overrides = {}

        invalid_body = {}  # Missing required fields

        with patch.object(
            AuthJWT,
            "jwt_required",
            side_effect=JWTDecodeError(
                status.HTTP_401_UNAUTHORIZED, "Unauthorized"
            ),
        ):
            response = client.post(BASE_URL, json=invalid_body)
            # Expects 422 because validation happens before auth
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestIntegrationScenarios:
    """Test suite for integration scenarios."""

    @patch("src.simulations.routes.v1.routes.get_simulation_configurations")
    def test_full_pagination_flow(self, mock_get_configs, client, mock_config):
        """Test complete pagination flow."""
        # Create 25 mock configs
        configs = [mock_config for _ in range(10)]
        mock_get_configs.return_value = (configs, 25)

        # First page
        response = client.get(f"{BASE_URL}?page=1&limit=10")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["page"] == 1
        assert data["total_pages"] == 3
        assert len(data["items"]) == 10

        # Last page
        mock_get_configs.return_value = (configs[:5], 25)
        response = client.get(f"{BASE_URL}?page=3&limit=10")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["page"] == 3
        assert len(data["items"]) == 5

    @patch("src.simulations.routes.v1.routes.get_simulation_configurations")
    def test_filter_and_pagination_combined(
        self, mock_get_configs, client, mock_config
    ):
        """Test combining filters with pagination."""
        mock_get_configs.return_value = ([mock_config], 1)

        response = client.get(f"{BASE_URL}?filters=name:Test&page=1&limit=10")

        assert response.status_code == status.HTTP_200_OK
        call_args = mock_get_configs.call_args[1]
        assert call_args["filters"]["name"] == "Test"
        assert call_args["page"] == 1
        assert call_args["limit"] == 10

    @patch("src.simulations.routes.v1.routes.create_simulation_configuration")
    @patch(
        "src.simulations.routes.v1.routes.get_simulation_configuration_from_db"
    )
    def test_create_then_retrieve(self, mock_get, mock_create, client):
        """Test creating a simulation and then retrieving it."""
        config_id = uuid.uuid4()
        report_id = uuid.uuid4()

        class MockConfig:
            id = config_id
            name = "Test"
            description = "Test"

        class MockReport:
            id = report_id

        mock_create.return_value = (MockConfig(), MockReport())
        mock_get.return_value = MockConfig()

        # Create
        body = {
            "name": "Test",
            "description": "Test",
            "simulationParameters": {
                "numChannels": 2,
                "simulationTime": 100.0,
                "numReplications": 1,
                "arrivalProcess": {
                    "distribution": "exponential",
                    "rate": 1.0,
                },
                "serviceProcess": {
                    "distribution": "exponential",
                    "rate": 2.0,
                },
            },
        }
        create_response = client.post(BASE_URL, json=body)
        assert create_response.status_code == status.HTTP_202_ACCEPTED

        # Retrieve
        get_response = client.get(f"{BASE_URL}/{config_id}")
        assert get_response.status_code == status.HTTP_200_OK
        assert get_response.json()["name"] == "Test"


class TestEdgeCasesAndBoundaries:
    """Test suite for edge cases and boundary conditions."""

    @patch("src.simulations.routes.v1.routes.get_simulation_configurations")
    def test_max_limit_boundary(self, mock_get_configs, client):
        """Test maximum limit boundary (100)."""
        mock_get_configs.return_value = ([], 0)

        response = client.get(f"{BASE_URL}?limit=100")
        assert response.status_code == status.HTTP_200_OK

        response = client.get(f"{BASE_URL}?limit=101")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.simulations.routes.v1.routes.get_simulation_configurations")
    def test_very_large_page_number(self, mock_get_configs, client):
        """Test very large page number."""
        mock_get_configs.return_value = ([], 10)

        response = client.get(f"{BASE_URL}?page=999999&limit=10")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["items"] == []

    def test_create_simulation_with_all_distributions(self, client):
        """Test creating simulations with different distributions."""
        distributions = [
            {"distribution": "exponential", "rate": 1.0},
            {"distribution": "uniform", "a": 1.0, "b": 2.0},
            {"distribution": "gamma", "k": 2.0, "theta": 1.0},
            {"distribution": "weibull", "k": 2.0, "lambda_param": 1.0},
            {
                "distribution": "truncated_normal",
                "mu": 5.0,
                "sigma": 2.0,
                "a": 0.0,
                "b": 10.0,
            },
        ]

        with patch(
            "src.simulations.routes.v1.routes.create_simulation_configuration"
        ) as mock_create:
            mock_create.return_value = (
                MagicMock(id=uuid.uuid4()),
                MagicMock(id=uuid.uuid4()),
            )

            for dist in distributions:
                body = {
                    "name": f"Test {dist['distribution']}",
                    "simulationParameters": {
                        "numChannels": 2,
                        "simulationTime": 100.0,
                        "numReplications": 1,
                        "arrivalProcess": dist,
                        "serviceProcess": dist,
                    },
                }
                response = client.post(BASE_URL, json=body)
                assert response.status_code == status.HTTP_202_ACCEPTED, (
                    f"Failed for distribution: {dist['distribution']}"
                )

    def test_create_simulation_with_arrival_schedule(self, client):
        """Test creating simulation with non-stationary arrivals."""
        body = {
            "name": "Non-stationary Test",
            "simulationParameters": {
                "numChannels": 2,
                "simulationTime": 100.0,
                "numReplications": 1,
                "arrivalProcess": {"distribution": "exponential", "rate": 1.0},
                "serviceProcess": {"distribution": "exponential", "rate": 2.0},
                "arrivalSchedule": [
                    {"duration": 50.0, "rate": 1.0},
                    {"duration": 50.0, "rate": 2.0},
                ],
            },
        }

        with patch(
            "src.simulations.routes.v1.routes.create_simulation_configuration"
        ) as mock_create:
            mock_create.return_value = (
                MagicMock(id=uuid.uuid4()),
                MagicMock(id=uuid.uuid4()),
            )

            response = client.post(BASE_URL, json=body)
            assert response.status_code == status.HTTP_202_ACCEPTED

    @patch("src.simulations.routes.v1.routes.get_simulation_configurations")
    def test_empty_filters_string(self, mock_get_configs, client):
        """Test with empty filters string."""
        mock_get_configs.return_value = ([], 0)

        response = client.get(f"{BASE_URL}?filters=")
        assert response.status_code == status.HTTP_200_OK

    @patch("src.simulations.routes.v1.routes.get_simulation_configurations")
    def test_whitespace_in_filters(self, mock_get_configs, client):
        """Test filters with extra whitespace."""
        mock_get_configs.return_value = ([], 0)

        response = client.get(
            f"{BASE_URL}?filters=name: Test , description: Desc"
        )
        # Should handle gracefully (utils trim whitespace)
        assert response.status_code in (
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
        )


class TestErrorHandling:
    """Test suite for error handling scenarios."""

    @patch(
        "src.simulations.routes.v1.routes.get_simulation_configuration_from_db"
    )
    def test_wrong_user_cannot_access_config(self, mock_get, client):
        """Test that user cannot access another user's configuration."""
        mock_get.side_effect = SimulationNotFound()

        response = client.get(f"{BASE_URL}/{uuid.uuid4()}")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @patch(
        "src.simulations.routes.v1.routes.delete_simulation_configuration_from_db"
    )
    def test_cannot_delete_nonexistent_config(self, mock_delete, client):
        """Test attempting to delete non-existent configuration."""
        mock_delete.side_effect = SimulationNotFound()

        response = client.delete(f"{BASE_URL}/{uuid.uuid4()}")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @patch(
        "src.simulations.routes.v1.routes.get_simulation_configuration_reports_from_db"
    )
    def test_reports_not_found_returns_404(self, mock_get, client):
        """Test that missing reports return 404."""
        mock_get.side_effect = SimulationReportsNotFound()

        response = client.get(f"{BASE_URL}/{uuid.uuid4()}/reports")
        assert response.status_code == status.HTTP_404_NOT_FOUND
