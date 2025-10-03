import uuid
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from another_fastapi_jwt_auth import AuthJWT

from src.main import app
from src.simulations.core.optimization import OptimizationResult

TEST_USER_ID = uuid.uuid4()
BASE_URL = "/api/v1/simulations/optimize"


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def setup_mocks(mocker):
    """Set up common mocks for authentication."""
    mock_authorize = MagicMock()
    mock_authorize.get_jwt_subject.return_value = str(TEST_USER_ID)
    mocker.patch.object(AuthJWT, "jwt_required", return_value=None)
    app.dependency_overrides[AuthJWT] = lambda: mock_authorize

    yield {"authorize": mock_authorize}

    app.dependency_overrides = {}


@pytest.fixture
def mock_optimization_result():
    """Create a mock optimization result."""
    return OptimizationResult(
        optimal_channels=5,
        achieved_rejection_prob=0.04,
        achieved_utilization=0.75,
        throughput=8.5,
        total_cost=150.0,
        iterations=10,
        convergence_history=[
            {"iteration": 1, "channels": 3, "rejection_prob": 0.08},
            {"iteration": 2, "channels": 5, "rejection_prob": 0.04},
        ],
    )


def make_base_request():
    """Helper to create base simulation request."""
    return {
        "numChannels": 1,
        "simulationTime": 100.0,
        "numReplications": 10,
        "arrivalProcess": {"distribution": "exponential", "rate": 5.0},
        "serviceProcess": {"distribution": "exponential", "rate": 10.0},
    }


class TestBinarySearchOptimization:
    """Test binary search optimization endpoint."""

    @patch("src.simulations.routes.v1.utils.binary_search_channels")
    def test_binary_search_success(
        self, mock_optimize, client, mock_optimization_result
    ):
        """Test successful binary search optimization."""
        mock_optimize.return_value = mock_optimization_result

        request_body = {
            "base_request": make_base_request(),
            "optimizationType": "binary_search",
            "targetRejectionProb": 0.05,
            "maxChannels": 20,
        }

        response = client.post(BASE_URL, json=request_body)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["optimal_channels"] == 5
        assert data["achieved_rejection_prob"] == 0.04
        assert data["iterations"] == 10

        mock_optimize.assert_called_once()

    def test_binary_search_missing_target(self, client):
        """Test binary search without target rejection probability."""
        request_body = {
            "base_request": make_base_request(),
            "optimizationType": "binary_search",
            "maxChannels": 20,
        }

        response = client.post(BASE_URL, json=request_body)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "target_rejection_prob" in response.json()["detail"].lower()


class TestCostMinimization:
    """Test cost minimization optimization endpoint."""

    @patch("src.simulations.routes.v1.utils.minimize_cost")
    def test_cost_minimization_success(
        self, mock_optimize, client, mock_optimization_result
    ):
        """Test successful cost minimization."""
        mock_optimize.return_value = mock_optimization_result

        request_body = {
            "base_request": make_base_request(),
            "optimizationType": "cost_minimization",
            "channelCost": 10.0,
            "rejectionPenalty": 100.0,
            "maxChannels": 30,
        }

        response = client.post(BASE_URL, json=request_body)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["optimal_channels"] == 5
        assert data["total_cost"] == 150.0

    def test_cost_minimization_missing_costs(self, client):
        """Test cost minimization without required cost parameters."""
        request_body = {
            "base_request": make_base_request(),
            "optimizationType": "cost_minimization",
            "channelCost": 10.0,
            # Missing rejectionPenalty
        }

        response = client.post(BASE_URL, json=request_body)

        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestMultiObjectiveOptimization:
    """Test multi-objective optimization endpoint."""

    @patch("src.simulations.routes.v1.utils.multi_objective_optimization")
    def test_multi_objective_success(
        self, mock_optimize, client, mock_optimization_result
    ):
        """Test successful multi-objective optimization."""
        mock_optimize.return_value = mock_optimization_result

        request_body = {
            "base_request": make_base_request(),
            "optimizationType": "multi_objective",
            "rejectionWeight": 0.5,
            "utilizationWeight": 0.3,
            "costWeight": 0.2,
            "channelCost": 10.0,
            "rejectionPenalty": 100.0,
        }

        response = client.post(BASE_URL, json=request_body)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["optimal_channels"] == 5


class TestRateLimiting:
    """Test rate limiting for optimization endpoint."""

    @patch("src.simulations.routes.v1.utils.binary_search_channels")
    @patch("src.simulations.routes.v1.routes.check_rate_limit")
    def test_rate_limit_enforced(self, mock_rate_limit, mock_optimize, client):
        """Test that rate limiting is enforced."""
        from fastapi import HTTPException

        mock_rate_limit.side_effect = HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )

        request_body = {
            "base_request": make_base_request(),
            "optimizationType": "binary_search",
            "targetRejectionProb": 0.05,
        }

        response = client.post(BASE_URL, json=request_body)

        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS


class TestErrorHandling:
    """Test error handling scenarios."""

    @patch("src.simulations.routes.v1.utils.binary_search_channels")
    @patch(
        "src.simulations.routes.v1.routes.check_rate_limit",
        return_value=None,
    )
    def test_timeout_handling(self, _, mock_optimize, client):
        """Test handling of optimization timeout."""
        mock_optimize.side_effect = TimeoutError("Optimization timed out")

        request_body = {
            "base_request": make_base_request(),
            "optimizationType": "binary_search",
            "targetRejectionProb": 0.05,
        }

        response = client.post(BASE_URL, json=request_body)

        assert response.status_code == status.HTTP_408_REQUEST_TIMEOUT

    @patch("src.simulations.routes.v1.utils.binary_search_channels")
    @patch(
        "src.simulations.routes.v1.routes.check_rate_limit",
        return_value=None,
    )
    def test_validation_error_handling(self, _, mock_optimize, client):
        """Test handling of validation errors."""
        mock_optimize.side_effect = ValueError("Invalid parameter")

        request_body = {
            "base_request": make_base_request(),
            "optimizationType": "binary_search",
            "targetRejectionProb": 0.05,
        }

        response = client.post(BASE_URL, json=request_body)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid" in response.json()["detail"]
