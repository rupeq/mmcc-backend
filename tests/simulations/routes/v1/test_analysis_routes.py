"""Tests for service time analysis routes."""

import base64
import io
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

from src.authorization.routes.v1.utils import AuthJWT
from src.core.db_session import get_session
from src.main import app
from src.simulations.core.service_time_analysis import (
    GoodnessOfFitResult,
    QuantileStatistics,
    ServiceTimeAnalysis,
)
from src.simulations.db_utils.exceptions import SimulationReportNotFound
from src.simulations.models.enums import ReportStatus

TEST_USER_ID = uuid.uuid4()
BASE_URL = "/api/v1/simulations"


@pytest.fixture
def mock_authorize():
    """Create a mock AuthJWT instance."""
    mock = MagicMock()
    mock.jwt_required = MagicMock()
    mock.get_jwt_subject = MagicMock(return_value=str(TEST_USER_ID))
    return mock


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    return AsyncMock()


@pytest.fixture
def client(mock_authorize, mock_session):
    """Create FastAPI test client with mocked dependencies."""

    def override_authorize():
        return mock_authorize

    async def override_session():
        yield mock_session

    # Override dependencies
    app.dependency_overrides[AuthJWT] = override_authorize
    app.dependency_overrides[get_session] = override_session

    client = TestClient(app)
    yield client

    # Cleanup
    app.dependency_overrides.clear()


@pytest.fixture
def mock_report():
    """Create a mock simulation report with service times."""
    report = MagicMock()
    report.id = uuid.uuid4()
    report.status = ReportStatus.COMPLETED
    report.configuration_id = uuid.uuid4()
    report.results = {
        "replications": [
            {
                "metrics": {},
                "gantt_chart": [],
                "raw_service_times": [1.2, 1.5, 2.1, 1.8, 2.3, 1.9, 2.0],
            },
            {
                "metrics": {},
                "gantt_chart": [],
                "raw_service_times": [1.3, 1.6, 2.0, 1.7, 2.4, 2.1, 1.9],
            },
        ]
    }

    # Mock awaitable_attrs for configuration
    mock_config = MagicMock()
    mock_config.simulation_parameters = {
        "numChannels": 2,
        "serviceProcess": {
            "distribution": "exponential",
            "rate": 2.0,
        },
    }
    report.awaitable_attrs.configuration = AsyncMock(return_value=mock_config)

    return report


@pytest.fixture
def mock_report_no_service_times():
    """Create a mock report without service times."""
    report = MagicMock()
    report.id = uuid.uuid4()
    report.status = ReportStatus.COMPLETED
    report.configuration_id = uuid.uuid4()
    report.results = {
        "replications": [
            {
                "metrics": {},
                "gantt_chart": [],
            }
        ]
    }
    return report


@pytest.fixture
def mock_analysis():
    """Create a mock ServiceTimeAnalysis result."""
    quantiles = QuantileStatistics(
        p50=1.9,
        p90=2.3,
        p95=2.35,
        p99=2.39,
        min=1.2,
        max=2.4,
    )

    goodness_of_fit = GoodnessOfFitResult(
        ks_statistic=0.023,
        ks_pvalue=0.342,
        chi2_statistic=15.67,
        chi2_pvalue=0.478,
        test_passed=True,
        warnings=[],
    )

    return ServiceTimeAnalysis(
        sample_size=14,
        mean=1.9,
        std=0.35,
        variance=0.12,
        skewness=0.1,
        kurtosis=-0.5,
        quantiles=quantiles,
        theoretical_mean=0.5,
        goodness_of_fit=goodness_of_fit,
    )


class TestGetServiceTimeAnalysis:
    """Test suite for GET service-time-analysis endpoint."""

    @patch(
        "src.simulations.routes.v1.analysis_routes.get_simulation_configuration_report"
    )
    @patch("src.simulations.routes.v1.analysis_routes.analyze_service_times")
    @patch(
        "src.simulations.routes.v1.analysis_routes.create_summary_statistics_table"
    )
    def test_successful_analysis(
        self,
        mock_summary,
        mock_analyze,
        mock_get_report,
        client,
        mock_report,
        mock_analysis,
    ):
        """Test successful service time analysis."""
        config_id = mock_report.configuration_id
        report_id = mock_report.id

        mock_get_report.return_value = mock_report
        mock_analyze.return_value = mock_analysis
        mock_summary.return_value = {"Sample Size": "14", "Mean": "1.9"}

        response = client.get(
            f"{BASE_URL}/{config_id}/reports/{report_id}/service-time-analysis"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["sample_size"] == 14
        assert data["mean"] == 1.9
        assert data["std"] == 0.35
        assert data["quantiles"]["p50"] == 1.9
        assert data["quantiles"]["p90"] == 2.3
        assert data["goodness_of_fit"]["ks_statistic"] == 0.023
        assert data["goodness_of_fit"]["test_passed"] is True

    @patch(
        "src.simulations.routes.v1.analysis_routes.get_simulation_configuration_report"
    )
    def test_analysis_report_not_found(self, mock_get_report, client):
        """Test analysis when report doesn't exist."""
        mock_get_report.side_effect = SimulationReportNotFound

        config_id = uuid.uuid4()
        report_id = uuid.uuid4()

        response = client.get(
            f"{BASE_URL}/{config_id}/reports/{report_id}/service-time-analysis"
        )

        assert response.status_code == 404

    @patch(
        "src.simulations.routes.v1.analysis_routes.get_simulation_configuration_report"
    )
    def test_analysis_no_service_times(
        self, mock_get_report, client, mock_report_no_service_times
    ):
        """Test analysis when service times weren't collected."""
        mock_get_report.return_value = mock_report_no_service_times

        config_id = mock_report_no_service_times.configuration_id
        report_id = mock_report_no_service_times.id

        response = client.get(
            f"{BASE_URL}/{config_id}/reports/{report_id}/service-time-analysis"
        )

        assert response.status_code == 400
        data = response.json()
        assert "no service times collected" in data["detail"].lower()

    @patch(
        "src.simulations.routes.v1.analysis_routes.get_simulation_configuration_report"
    )
    @patch("src.simulations.routes.v1.analysis_routes.analyze_service_times")
    @patch(
        "src.simulations.routes.v1.analysis_routes.create_summary_statistics_table"
    )
    def test_analysis_with_custom_significance_level(
        self,
        mock_summary,
        mock_analyze,
        mock_get_report,
        client,
        mock_report,
        mock_analysis,
    ):
        """Test analysis with custom significance level."""
        mock_get_report.return_value = mock_report
        mock_analyze.return_value = mock_analysis
        mock_summary.return_value = {"Sample Size": "14"}

        config_id = mock_report.configuration_id
        report_id = mock_report.id

        response = client.get(
            f"{BASE_URL}/{config_id}/reports/{report_id}/service-time-analysis"
            "?significance_level=0.01"
        )

        assert response.status_code == 200
        # Verify analyze_service_times was called with custom significance level
        mock_analyze.assert_called_once()
        call_kwargs = mock_analyze.call_args[1]
        assert call_kwargs["significance_level"] == 0.01


class TestGetServiceTimeVisualizations:
    """Test suite for GET service-time-visualizations endpoint."""

    @patch(
        "src.simulations.routes.v1.analysis_routes.get_simulation_configuration_report"
    )
    @patch(
        "src.simulations.routes.v1.analysis_routes.plot_service_time_histogram"
    )
    @patch("src.simulations.routes.v1.analysis_routes.plot_service_time_ecdf")
    def test_successful_visualizations_without_qq_plot(
        self,
        mock_ecdf,
        mock_histogram,
        mock_get_report,
        client,
        mock_report,
    ):
        """Test successful generation of visualizations when distribution is not available."""
        mock_get_report.return_value = mock_report

        # Mock matplotlib figures with proper savefig behavior
        def create_mock_fig():
            mock_fig = MagicMock()

            def mock_savefig(buf, format=None, **kwargs):
                # Write some fake PNG data
                buf.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

            mock_fig.savefig = mock_savefig
            return mock_fig

        mock_histogram.return_value = create_mock_fig()
        mock_ecdf.return_value = create_mock_fig()

        config_id = mock_report.configuration_id
        report_id = mock_report.id

        response = client.get(
            f"{BASE_URL}/{config_id}/reports/{report_id}/service-time-visualizations"
        )

        assert response.status_code == 200
        data = response.json()

        assert "histogram" in data
        assert "ecdf" in data
        assert "qq_plot" in data

        assert data["histogram"]["plot_type"] == "histogram"
        assert data["histogram"]["content_type"] == "image/png"
        assert len(data["histogram"]["image_base64"]) > 0

        # QQ plot should be None when distribution is not available
        assert data["qq_plot"] is None

        # Verify histogram and ecdf were called
        mock_histogram.assert_called_once()
        mock_ecdf.assert_called_once()

    @patch(
        "src.simulations.routes.v1.analysis_routes.get_simulation_configuration_report"
    )
    @patch(
        "src.simulations.routes.v1.analysis_routes.get_service_distribution_from_report"
    )
    @patch(
        "src.simulations.routes.v1.analysis_routes.plot_service_time_histogram"
    )
    @patch("src.simulations.routes.v1.analysis_routes.plot_service_time_ecdf")
    @patch("src.simulations.routes.v1.analysis_routes.plot_qq_plot")
    def test_successful_visualizations_with_qq_plot(
        self,
        mock_qq_plot,
        mock_ecdf,
        mock_histogram,
        mock_get_dist,
        mock_get_report,
        client,
        mock_report,
    ):
        """Test successful generation of all visualizations including QQ plot."""
        from src.simulations.core.distributions import ExponentialDistribution

        mock_get_report.return_value = mock_report

        # Mock distribution so QQ plot can be generated
        mock_dist = ExponentialDistribution(rate=2.0)
        mock_get_dist.return_value = mock_dist

        # Mock matplotlib figures with proper savefig behavior
        def create_mock_fig():
            mock_fig = MagicMock()

            def mock_savefig(buf, format=None, **kwargs):
                buf.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

            mock_fig.savefig = mock_savefig
            return mock_fig

        mock_histogram.return_value = create_mock_fig()
        mock_ecdf.return_value = create_mock_fig()
        mock_qq_plot.return_value = create_mock_fig()

        config_id = mock_report.configuration_id
        report_id = mock_report.id

        response = client.get(
            f"{BASE_URL}/{config_id}/reports/{report_id}/service-time-visualizations"
        )

        assert response.status_code == 200
        data = response.json()

        # All plots should have data
        assert len(data["histogram"]["image_base64"]) > 0
        assert len(data["ecdf"]["image_base64"]) > 0
        assert data["qq_plot"] is not None
        assert len(data["qq_plot"]["image_base64"]) > 0

        # All plot functions should be called
        mock_histogram.assert_called_once()
        mock_ecdf.assert_called_once()
        mock_qq_plot.assert_called_once()


class TestAuthenticationRequirement:
    """Test that endpoints require authentication."""

    def test_analysis_requires_auth(self):
        """Test that analysis endpoint requires authentication."""
        # Create client without auth override
        client = TestClient(app)

        config_id = uuid.uuid4()
        report_id = uuid.uuid4()

        response = client.get(
            f"{BASE_URL}/{config_id}/reports/{report_id}/service-time-analysis"
        )

        # Should fail with 401 or 422 (validation error for missing auth)
        assert response.status_code in [401, 422]
