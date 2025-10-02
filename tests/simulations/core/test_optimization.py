import pytest
from unittest.mock import patch

from src.simulations.core.optimization import (
    binary_search_channels,
    minimize_cost,
    gradient_descent_channels,
    erlang_b_estimate,
    multi_objective_optimization,
    CostFunction,
    OptimizationResult,
)
from src.simulations.core.schemas import (
    SimulationRequest,
    SimulationResponse,
    AggregatedMetrics,
    ExponentialParams,
)


def create_mock_response(
    rejection_prob: float,
    utilization: float = 0.5,
    total_requests: float = 100.0,
) -> SimulationResponse:
    """Helper to create mock simulation response"""
    return SimulationResponse(
        aggregated_metrics=AggregatedMetrics(
            num_replications=1,
            total_requests=total_requests,
            processed_requests=total_requests * (1 - rejection_prob),
            rejected_requests=total_requests * rejection_prob,
            rejection_probability=rejection_prob,
            avg_channel_utilization=utilization,
            throughput=total_requests * (1 - rejection_prob) / 100.0,
        ),
        replications=[],
    )


@pytest.fixture
def base_request():
    return SimulationRequest(
        num_channels=1,
        simulation_time=100.0,
        num_replications=1,
        arrival_process=ExponentialParams(rate=5.0),
        service_process=ExponentialParams(rate=10.0),
    )


class TestBinarySearchChannels:
    def test_finds_optimal_channels(self, base_request):
        """Test binary search finds minimum channels meeting target"""

        def mock_run(request):
            # Simulate: rejection decreases with more channels
            c = request.num_channels
            rejection = max(0.01, 0.5 / c)
            return create_mock_response(rejection, utilization=0.8)

        with patch(
            "src.simulations.core.optimization.run_replications",
            side_effect=mock_run,
        ):
            result = binary_search_channels(
                base_request=base_request,
                target_rejection_prob=0.05,
                max_channels=20,
            )

        assert isinstance(result, OptimizationResult)
        assert 1 <= result.optimal_channels <= 20
        assert result.achieved_rejection_prob <= 0.06  # Within tolerance
        assert result.iterations > 0
        assert len(result.convergence_history) == result.iterations

    def test_handles_unachievable_target(self, base_request):
        """Test handles case where target cannot be met"""

        def mock_run(request):
            # Always return high rejection
            return create_mock_response(0.5, utilization=0.3)

        with patch(
            "src.simulations.core.optimization.run_replications",
            side_effect=mock_run,
        ):
            result = binary_search_channels(
                base_request=base_request,
                target_rejection_prob=0.01,
                max_channels=10,
            )

        assert result.optimal_channels == 10  # Uses max_channels
        assert result.achieved_rejection_prob == 0.5

    def test_respects_min_channels(self, base_request):
        """Test respects minimum channel constraint"""

        def mock_run(request):
            return create_mock_response(0.0, utilization=0.1)

        with patch(
            "src.simulations.core.optimization.run_replications",
            side_effect=mock_run,
        ):
            result = binary_search_channels(
                base_request=base_request,
                target_rejection_prob=0.05,
                max_channels=20,
                min_channels=5,
            )

        assert result.optimal_channels >= 5


class TestMinimizeCost:
    def test_finds_cost_optimal_channels(self, base_request):
        """Test cost minimization finds optimal balance"""

        def mock_run(request):
            c = request.num_channels
            # Trade-off: more channels = lower rejection but higher cost
            rejection = max(0.01, 0.8 / c)
            return create_mock_response(rejection, total_requests=100.0)

        cost_fn = CostFunction(channel_cost=10.0, rejection_penalty=100.0)

        with patch(
            "src.simulations.core.optimization.run_replications",
            side_effect=mock_run,
        ):
            result = minimize_cost(
                base_request=base_request,
                cost_function=cost_fn,
                max_channels=20,
            )

        assert isinstance(result, OptimizationResult)
        assert result.total_cost is not None
        assert result.optimal_channels > 0
        assert len(result.convergence_history) > 0

    def test_high_channel_cost_favors_fewer_channels(self, base_request):
        """Test high channel cost results in fewer channels"""

        def mock_run(request):
            c = request.num_channels
            rejection = 0.5 / c
            return create_mock_response(rejection, total_requests=100.0)

        cost_fn = CostFunction(channel_cost=1000.0, rejection_penalty=1.0)

        with patch(
            "src.simulations.core.optimization.run_replications",
            side_effect=mock_run,
        ):
            result = minimize_cost(
                base_request=base_request,
                cost_function=cost_fn,
                max_channels=20,
            )

        # With very high channel cost, should prefer fewer channels
        assert result.optimal_channels < 10

    def test_high_rejection_penalty_favors_more_channels(self, base_request):
        """Test high rejection penalty results in more channels"""

        def mock_run(request):
            c = request.num_channels
            rejection = 0.5 / c
            return create_mock_response(rejection, total_requests=100.0)

        cost_fn = CostFunction(channel_cost=1.0, rejection_penalty=1000.0)

        with patch(
            "src.simulations.core.optimization.run_replications",
            side_effect=mock_run,
        ):
            result = minimize_cost(
                base_request=base_request,
                cost_function=cost_fn,
                max_channels=20,
            )

        # With very high rejection penalty, should prefer more channels
        assert result.optimal_channels > 10


class TestGradientDescentChannels:
    def test_rejection_objective(self, base_request):
        """Test gradient descent with rejection objective"""

        def mock_run(request):
            c = request.num_channels
            rejection = max(0.01, 0.5 / c)
            return create_mock_response(rejection)

        with patch(
            "src.simulations.core.optimization.run_replications",
            side_effect=mock_run,
        ):
            result = gradient_descent_channels(
                base_request=base_request,
                objective="rejection",
                max_iterations=10,
            )

        assert result.optimal_channels > 0
        assert result.iterations <= 10

    def test_cost_objective_requires_cost_function(self, base_request):
        """Test cost objective requires cost_function parameter"""
        with pytest.raises(ValueError, match="cost_function required"):
            gradient_descent_channels(
                base_request=base_request,
                objective="cost",
                cost_function=None,
            )

    def test_converges_within_tolerance(self, base_request):
        """Test converges when gradient is within tolerance"""
        call_count = [0]

        def mock_run(request):
            call_count[0] += 1
            # Return same value to force convergence
            return create_mock_response(0.05)

        with patch(
            "src.simulations.core.optimization.run_replications",
            side_effect=mock_run,
        ):
            result = gradient_descent_channels(
                base_request=base_request,
                objective="rejection",
                max_iterations=20,
                tolerance=0.001,
            )

        # Should converge quickly
        assert result.iterations < 5


class TestErlangBEstimate:
    def test_returns_valid_estimate(self):
        """Test Erlang B returns reasonable estimate"""
        channels = erlang_b_estimate(
            arrival_rate=5.0,
            service_rate=10.0,
            target_rejection_prob=0.01,
        )

        assert isinstance(channels, int)
        assert channels > 0
        assert channels < 100

    def test_higher_traffic_needs_more_channels(self):
        """Test higher traffic intensity requires more channels"""
        low_traffic = erlang_b_estimate(
            arrival_rate=1.0,
            service_rate=10.0,
            target_rejection_prob=0.01,
        )

        high_traffic = erlang_b_estimate(
            arrival_rate=5.0,
            service_rate=10.0,
            target_rejection_prob=0.01,
        )

        assert high_traffic > low_traffic

    def test_stricter_target_needs_more_channels(self):
        """Test stricter blocking probability requires more channels"""
        relaxed = erlang_b_estimate(
            arrival_rate=5.0,
            service_rate=10.0,
            target_rejection_prob=0.10,
        )

        strict = erlang_b_estimate(
            arrival_rate=5.0,
            service_rate=10.0,
            target_rejection_prob=0.01,
        )

        assert strict > relaxed


class TestMultiObjectiveOptimization:
    def test_balances_objectives(self, base_request):
        """Test multi-objective finds balanced solution"""

        def mock_run(request):
            c = request.num_channels
            rejection = 0.5 / c
            utilization = min(0.95, 0.3 * c)
            return create_mock_response(
                rejection, utilization, total_requests=100.0
            )

        with patch(
            "src.simulations.core.optimization.run_replications",
            side_effect=mock_run,
        ):
            result = multi_objective_optimization(
                base_request=base_request,
                rejection_weight=0.4,
                utilization_weight=0.3,
                cost_weight=0.3,
                max_channels=20,
            )

        assert result.optimal_channels > 0
        assert result.total_cost is not None
        assert len(result.convergence_history) > 0

    def test_normalizes_weights(self, base_request):
        """Test weight normalization"""

        def mock_run(request):
            return create_mock_response(0.1, 0.5, 100.0)

        with patch(
            "src.simulations.core.optimization.run_replications",
            side_effect=mock_run,
        ):
            # Weights don't sum to 1.0
            result = multi_objective_optimization(
                base_request=base_request,
                rejection_weight=1.0,
                utilization_weight=1.0,
                cost_weight=1.0,
                max_channels=5,
            )

        assert result.optimal_channels > 0  # Should still work

    def test_high_rejection_weight_favors_more_channels(self, base_request):
        """Test high rejection weight leads to more channels"""

        def mock_run(request):
            c = request.num_channels
            rejection = 0.5 / c
            return create_mock_response(rejection, 0.5, 100.0)

        with patch(
            "src.simulations.core.optimization.run_replications",
            side_effect=mock_run,
        ):
            result = multi_objective_optimization(
                base_request=base_request,
                rejection_weight=0.9,  # Very high
                utilization_weight=0.05,
                cost_weight=0.05,
                max_channels=20,
            )

        # Should favor more channels to reduce rejection
        assert result.optimal_channels > 5


class TestCostFunction:
    def test_calculates_cost_correctly(self):
        """Test cost function calculation"""
        cost_fn = CostFunction(channel_cost=10.0, rejection_penalty=100.0)

        cost = cost_fn.calculate(
            num_channels=5,
            rejection_prob=0.1,
            total_requests=100.0,
        )

        # 5 channels * 10 + 100 * 0.1 * 100 = 50 + 1000 = 1050
        assert cost == pytest.approx(1050.0)

    def test_higher_rejection_increases_cost(self):
        """Test higher rejection increases total cost"""
        cost_fn = CostFunction(channel_cost=10.0, rejection_penalty=100.0)

        low_rej = cost_fn.calculate(5, 0.01, 100.0)
        high_rej = cost_fn.calculate(5, 0.10, 100.0)

        assert high_rej > low_rej
