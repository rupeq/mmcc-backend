"""Tests for theoretical queueing formulas."""

import pytest
from src.simulations.core.theoretical import (
    calculate_erlang_b,
    calculate_theoretical_metrics,
    compare_with_theory,
)
from src.simulations.core.schemas import (
    SimulationRequest,
    ExponentialParams,
    UniformParams,
)


class TestErlangB:
    """Test Erlang B blocking probability calculation."""

    def test_zero_traffic(self):
        """Test with zero traffic intensity."""
        result = calculate_erlang_b(rho=0.0, c=5)
        assert result == pytest.approx(0.0)

    def test_single_channel_low_load(self):
        """Test M/M/1/1 with low load."""
        result = calculate_erlang_b(rho=0.5, c=1)
        # B(1, 0.5) = 0.5 / (1 + 0.5) = 1/3
        assert result == pytest.approx(1/3, rel=0.01)

    def test_single_channel_high_load(self):
        """Test M/M/1/1 with high load."""
        result = calculate_erlang_b(rho=2.0, c=1)
        # B(1, 2.0) = 2.0 / (1 + 2.0) = 2/3
        assert result == pytest.approx(2/3, rel=0.01)

    def test_multi_channel(self):
        """Test M/M/3/3 system."""
        result = calculate_erlang_b(rho=2.0, c=3)
        # Known value from Erlang B tables
        assert 0.2 < result < 0.25

    def test_high_channels(self):
        """Test with many channels."""
        result = calculate_erlang_b(rho=10.0, c=15)
        assert 0 < result < 0.1  # Low blocking with excess capacity

    def test_invalid_negative_rho(self):
        """Test that negative traffic intensity raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            calculate_erlang_b(rho=-1.0, c=3)

    def test_invalid_zero_channels(self):
        """Test that zero channels raises error."""
        with pytest.raises(ValueError, match="positive"):
            calculate_erlang_b(rho=1.0, c=0)

    def test_monotonicity_with_load(self):
        """Test that blocking increases with load."""
        c = 5
        b1 = calculate_erlang_b(rho=2.0, c=c)
        b2 = calculate_erlang_b(rho=4.0, c=c)
        b3 = calculate_erlang_b(rho=6.0, c=c)

        assert b1 < b2 < b3


class TestTheoreticalMetrics:
    """Test theoretical metrics calculation."""

    def test_mm_c_c_system(self):
        """Test M/M/c/c system metrics."""
        request = SimulationRequest(
            num_channels=3,
            simulation_time=100,
            arrival_process=ExponentialParams(rate=5.0),
            service_process=ExponentialParams(rate=10.0),
        )

        metrics = calculate_theoretical_metrics(request)

        assert metrics.is_applicable is True
        assert metrics.system_type == "M/M/3/3"
        assert metrics.traffic_intensity == pytest.approx(0.5)
        assert metrics.blocking_probability is not None
        assert 0 <= metrics.blocking_probability <= 1
        assert metrics.utilization is not None
        assert 0 <= metrics.utilization <= 1
        assert metrics.throughput is not None
        assert len(metrics.assumptions) > 0

    def test_non_exponential_arrivals(self):
        """Test that non-exponential arrivals are not applicable."""
        request = SimulationRequest(
            num_channels=3,
            simulation_time=100,
            arrival_process=UniformParams(a=0.1, b=0.5),
            service_process=ExponentialParams(rate=10.0),
        )

        metrics = calculate_theoretical_metrics(request)

        assert metrics.is_applicable is False
        assert "uniform/exponential" in metrics.system_type.lower()
        assert len(metrics.warnings) > 0

    def test_high_load_warning(self):
        """Test warning for overloaded system."""
        request = SimulationRequest(
            num_channels=2,
            simulation_time=100,
            arrival_process=ExponentialParams(rate=10.0),
            service_process=ExponentialParams(rate=2.0),  # ρ = 5 > c = 2
        )

        metrics = calculate_theoretical_metrics(request)

        assert metrics.traffic_intensity == pytest.approx(5.0)
        assert any("overloaded" in w.lower() for w in metrics.warnings)

    def test_derived_metrics(self):
        """Test that derived metrics are consistent."""
        lambda_rate = 8.0
        mu_rate = 10.0
        c = 2

        request = SimulationRequest(
            num_channels=c,
            simulation_time=100,
            arrival_process=ExponentialParams(rate=lambda_rate),
            service_process=ExponentialParams(rate=mu_rate),
        )

        metrics = calculate_theoretical_metrics(request)

        # Check consistency
        assert metrics.mean_service_time == pytest.approx(1.0 / mu_rate)
        assert metrics.mean_interarrival_time == pytest.approx(1.0 / lambda_rate)

        # Throughput = λ(1 - B)
        expected_throughput = lambda_rate * (1 - metrics.blocking_probability)
        assert metrics.throughput == pytest.approx(expected_throughput, rel=0.01)


class TestComparison:
    """Test comparison between simulation and theory."""

    def test_perfect_match(self):
        """Test comparison when simulation matches theory perfectly."""
        theoretical = calculate_theoretical_metrics(
            SimulationRequest(
                num_channels=3,
                simulation_time=100,
                arrival_process=ExponentialParams(rate=5.0),
                service_process=ExponentialParams(rate=10.0),
            )
        )

        # Simulate perfect match
        sim_metrics = {
            "rejection_probability": theoretical.blocking_probability,
            "avg_channel_utilization": theoretical.utilization,
            "throughput": theoretical.throughput,
        }

        comparison = compare_with_theory(sim_metrics, theoretical)

        assert comparison["applicable"] is True
        assert "blocking_probability" in comparison["metrics"]
        assert comparison["metrics"]["blocking_probability"]["absolute_error"] == pytest.approx(0.0, abs=1e-10)
        assert comparison["metrics"]["blocking_probability"]["relative_error"] == pytest.approx(0.0, abs=1e-10)

    def test_with_confidence_intervals(self):
        """Test that CI coverage is checked."""
        theoretical = calculate_theoretical_metrics(
            SimulationRequest(
                num_channels=3,
                simulation_time=100,
                arrival_process=ExponentialParams(rate=5.0),
                service_process=ExponentialParams(rate=10.0),
            )
        )

        # CI that covers theoretical value
        sim_metrics = {
            "rejection_probability": 0.03,
            "rejection_probability_ci": {
                "lower_bound": 0.01,
                "upper_bound": 0.05,
            },
            "avg_channel_utilization": theoretical.utilization,
            "throughput": theoretical.throughput,
        }

        comparison = compare_with_theory(sim_metrics, theoretical)

        # Check if theoretical value is within CI
        if (
                0.01 <= theoretical.blocking_probability <= 0.05
        ):
            assert comparison["metrics"]["blocking_probability"]["within_ci"] is True

    def test_non_applicable_system(self):
        """Test comparison for non-M/M/c/c systems."""
        theoretical = calculate_theoretical_metrics(
            SimulationRequest(
                num_channels=3,
                simulation_time=100,
                arrival_process=UniformParams(a=0.1, b=0.5),
                service_process=ExponentialParams(rate=10.0),
            )
        )

        sim_metrics = {
            "rejection_probability": 0.05,
        }

        comparison = compare_with_theory(sim_metrics, theoretical)

        assert comparison["applicable"] is False
        assert "not available" in comparison["message"]