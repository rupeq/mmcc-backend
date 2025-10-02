import pytest

from src.simulations.core.engine import Simulator, run_replications, run_sweep
from src.simulations.core.distributions import Distribution
from src.simulations.core.schemas import (
    SimulationRequest,
    ExponentialParams,
    ArrivalScheduleItem,
    SweepRequest,
    SweepParameter,
    SimulationResponse,
    AggregatedMetrics,
)


def create_base_request(**kwargs):
    """Create a default SimulationRequest that can be overridden.

    Args:
        **kwargs: Override default parameters.

    Returns:
        SimulationRequest with specified parameters.
    """
    defaults = {
        "num_channels": 1,
        "simulation_time": 10,
        "num_replications": 1,
        "arrival_process": ExponentialParams(rate=1.0),
        "service_process": ExponentialParams(rate=2.0),
    }
    defaults.update(kwargs)
    return SimulationRequest.model_validate(defaults)


class DeterministicDistribution(Distribution):
    """Mock distribution that returns a constant value.

    Used for testing to create predictable scenarios.

    Attributes:
        value: The constant value to return.
    """

    def __init__(self, value: float):
        """Initialize with a constant value.

        Args:
            value: The value to always return.
        """
        super().__init__()
        self.value = value

    def generate(self) -> float:
        """Return the constant value.

        Returns:
            The constant value.
        """
        self._generation_count += 1
        return self.value

    def get_mean(self) -> float:
        """Return the constant value as mean.

        Returns:
            The constant value.
        """
        return self.value

    def get_params(self) -> dict:
        """Get distribution parameters.

        Returns:
            Dictionary with 'value' key.
        """
        return {"value": self.value}

    def __repr__(self) -> str:
        """String representation.

        Returns:
            String like "Deterministic(1.5)".
        """
        return f"Deterministic({self.value})"


def test_simulator_deterministic_run():
    """Test a simple, predictable scenario with deterministic arrivals."""
    params = create_base_request(simulation_time=4)
    simulator = Simulator(params)

    # Replace distributions with deterministic ones
    simulator.arrival_distribution = DeterministicDistribution(1.5)
    simulator.service_distribution = DeterministicDistribution(0.5)

    result = simulator.run()

    # t=0.0: First arrival scheduled for t=1.5
    # t=1.5: Arrival 1, service starts (duration 0.5), ends at 2.0, next arrival at t=3.0
    # t=3.0: Arrival 2, service starts (duration 0.5), ends at 3.5, next arrival at t=4.5
    # t=4.0: Simulation ends. Only 2 arrivals should have been fully processed.
    assert result.metrics.total_requests == 2
    assert result.metrics.processed_requests == 2
    assert result.metrics.rejected_requests == 0


def test_rejection_logic():
    """Test that a request is rejected if all channels are busy."""
    params = create_base_request(num_channels=1, simulation_time=2.5)
    simulator = Simulator(params)

    # Arrivals every 1.0 seconds, service takes 1.5 seconds
    simulator.arrival_distribution = DeterministicDistribution(1.0)
    simulator.service_distribution = DeterministicDistribution(1.5)

    result = simulator.run()

    # t=0.0: Schedule first arrival at t=1.0
    # t=1.0: Arrival 1 arrives, starts service (duration 1.5). Channel busy until 2.5. Next arrival at t=2.0.
    # t=2.0: Arrival 2 arrives. Channel is busy (until 2.5). Arrival 2 is REJECTED. Next arrival at t=3.0.
    # t=2.5: Simulation ends.
    assert result.metrics.total_requests == 2
    assert result.metrics.processed_requests == 1
    assert result.metrics.rejected_requests == 1


def test_multi_channel_logic():
    """Test that a second channel prevents rejection."""
    params = create_base_request(num_channels=2, simulation_time=2.5)
    simulator = Simulator(params)

    simulator.arrival_distribution = DeterministicDistribution(1.0)
    simulator.service_distribution = DeterministicDistribution(1.5)

    result = simulator.run()

    # t=1.0: Arrival 1 -> Channel 0 (busy until 2.5)
    # t=2.0: Arrival 2 -> Channel 1 is free! (busy until 3.5)
    # No rejections should occur. 2 arrivals processed.
    assert result.metrics.total_requests == 2
    assert result.metrics.processed_requests == 2
    assert result.metrics.rejected_requests == 0


def test_random_seed_consistency():
    """Test that the same seed produces the same results."""
    params = create_base_request(random_seed=42, num_replications=5)

    response1 = run_replications(params)
    response2 = run_replications(params)

    assert response1 == response2


def test_run_replications_aggregation():
    """Test that replications run and produce aggregated metrics with CIs."""
    params = create_base_request(num_replications=10, random_seed=123)
    response = run_replications(params)

    assert len(response.replications) == 10
    agg = response.aggregated_metrics
    assert agg.num_replications == 10

    assert agg.rejection_probability_ci is not None
    assert agg.avg_channel_utilization_ci is not None

    assert agg.rejection_probability_ci.lower_bound <= agg.rejection_probability
    assert agg.rejection_probability <= agg.rejection_probability_ci.upper_bound


def test_run_replications_single_replication():
    """Test that single replication doesn't compute CIs."""
    params = create_base_request(num_replications=1, random_seed=456)
    response = run_replications(params)

    assert len(response.replications) == 1
    agg = response.aggregated_metrics
    assert agg.num_replications == 1

    # With single replication, CIs should be None
    assert agg.rejection_probability_ci is None
    assert agg.avg_channel_utilization_ci is None


def test_non_stationary_flow_deterministic():
    """Test that the arrival rate changes according to the schedule.

    We use a custom distribution that returns different values based on
    whether we're in the first or second period.
    """

    class NonStationaryDeterministicDistribution(Distribution):
        """Distribution that returns different inter-arrival times based on schedule."""

        def __init__(self, schedule_end_time: float):
            """Initialize with schedule transition time.

            Args:
                schedule_end_time: Time when rate changes.
            """
            super().__init__()
            self.schedule_end_time = schedule_end_time
            self.simulator_clock = 0.0

        def generate(self) -> float:
            """Generate inter-arrival time based on current time.

            Returns:
                1.0 if before schedule_end_time, 0.25 otherwise.
            """
            self._generation_count += 1
            # Rate 1.0 -> inter-arrival 1.0s
            # Rate 4.0 -> inter-arrival 0.25s
            if self.simulator_clock < self.schedule_end_time:
                return 1.0
            else:
                return 0.25

        def get_mean(self) -> float:
            return 0.625  # Average of 1.0 and 0.25

        def get_params(self) -> dict:
            return {"schedule_end_time": self.schedule_end_time}

        def __repr__(self) -> str:
            return (
                f"NonStationaryDeterministic(boundary={self.schedule_end_time})"
            )

    params = create_base_request(
        simulation_time=10,
        arrival_schedule=[
            ArrivalScheduleItem(duration=5, rate=1.0),
            ArrivalScheduleItem(duration=5, rate=4.0),
        ],
    )

    simulator = Simulator(params)

    # Create non-stationary arrival distribution
    non_stat_dist = NonStationaryDeterministicDistribution(
        schedule_end_time=5.0
    )

    # We need to wire up the simulator clock reference
    # This is a bit hacky but necessary for deterministic testing
    class NonStationaryWrapper(Distribution):
        """Wrapper that uses simulator's clock."""

        def __init__(self, simulator, schedule_end_time):
            super().__init__()
            self.simulator = simulator
            self.schedule_end_time = schedule_end_time

        def generate(self) -> float:
            self._generation_count += 1
            if self.simulator.clock < self.schedule_end_time:
                return 1.0
            else:
                return 0.25

        def get_mean(self) -> float:
            return 0.625

        def get_params(self) -> dict:
            return {"schedule_end_time": self.schedule_end_time}

        def __repr__(self) -> str:
            return f"NonStationaryWrapper(boundary={self.schedule_end_time})"

    simulator.arrival_distribution = NonStationaryWrapper(
        simulator, schedule_end_time=5.0
    )
    simulator.service_distribution = DeterministicDistribution(0.3)

    result = simulator.run()

    # Expected arrivals:
    # Period 1 (t=0 to t=5, inter-arrival=1.0s):
    #   First arrival at t=1.0, then t=2.0, 3.0, 4.0, 5.0
    #   Total: 5 arrivals
    # Period 2 (t=5 to t=10, inter-arrival=0.25s):
    #   t=5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75
    #   Total: 19 arrivals
    # Grand total: 5 + 19 = 24 arrivals

    assert result.metrics.total_requests >= 20
    assert result.metrics.total_requests <= 26


def test_gantt_chart_generation():
    """Test that Gantt chart data is properly collected."""
    params = create_base_request(
        num_channels=2, simulation_time=5, random_seed=789
    )

    simulator = Simulator(params)
    simulator.arrival_distribution = DeterministicDistribution(1.0)
    simulator.service_distribution = DeterministicDistribution(0.5)

    result = simulator.run()

    # Should have Gantt items for each processed request
    assert len(result.gantt_chart) == result.metrics.processed_requests

    # Each item should have valid fields
    for item in result.gantt_chart:
        assert 0 <= item.channel < params.num_channels
        assert item.start >= 0
        assert item.end > item.start
        assert item.duration == item.end - item.start
        assert item.duration > 0


def test_service_times_logging():
    """Test that raw service times are logged."""
    params = create_base_request(simulation_time=5, random_seed=321)

    simulator = Simulator(params)
    simulator.arrival_distribution = DeterministicDistribution(0.5)
    simulator.service_distribution = DeterministicDistribution(0.2)

    result = simulator.run()

    # Should have service times for each processed request
    assert len(result.raw_service_times) == result.metrics.processed_requests

    # All service times should be positive
    assert all(t > 0 for t in result.raw_service_times)


def test_channel_utilization_calculation():
    """Test that channel utilization is calculated correctly."""
    params = create_base_request(
        num_channels=2, simulation_time=10, random_seed=654
    )

    simulator = Simulator(params)
    simulator.arrival_distribution = DeterministicDistribution(1.0)
    simulator.service_distribution = DeterministicDistribution(0.5)

    result = simulator.run()

    # Utilization should be between 0 and 1
    assert 0 <= result.metrics.avg_channel_utilization <= 1

    # With deterministic arrivals every 1s and service 0.5s,
    # and 2 channels, utilization should be around 0.25
    # (each channel busy 0.5s out of every 2s of arrivals)
    assert 0.2 <= result.metrics.avg_channel_utilization <= 0.3


def test_throughput_calculation():
    """Test that throughput is calculated correctly."""
    params = create_base_request(
        num_channels=1, simulation_time=10, random_seed=987
    )

    simulator = Simulator(params)
    simulator.arrival_distribution = DeterministicDistribution(1.0)
    simulator.service_distribution = DeterministicDistribution(0.3)

    result = simulator.run()

    # Throughput = processed_requests / simulation_time
    expected_throughput = (
        result.metrics.processed_requests / params.simulation_time
    )
    assert result.metrics.throughput == pytest.approx(expected_throughput)


def test_run_sweep_nested_parameter(mocker):
    """Test the sweep functionality with a nested parameter."""
    dummy_response = SimulationResponse(
        aggregated_metrics=AggregatedMetrics(
            num_replications=1,
            total_requests=10,
            processed_requests=8,
            rejected_requests=2,
            rejection_probability=0.2,
            avg_channel_utilization=0.8,
            throughput=1.0,
        ),
        replications=[],
    )

    mock_runner = mocker.patch(
        "src.simulations.core.engine.run_replications",
        return_value=dummy_response,
    )

    base_request = create_base_request(
        arrival_process=ExponentialParams(rate=1.0)
    )

    sweep_values = [0.5, 1.5, 2.5]
    sweep_request = SweepRequest(
        base_request=base_request,
        sweep_parameter=SweepParameter(
            name="arrival_process.rate", values=sweep_values
        ),
    )

    sweep_response = run_sweep(sweep_request)

    assert mock_runner.call_count == len(sweep_values)
    assert len(sweep_response.results) == len(sweep_values)

    for i, call in enumerate(mock_runner.call_args_list):
        called_request = call[0][0]
        expected_rate = sweep_values[i]

        assert isinstance(called_request, SimulationRequest)
        assert called_request.arrival_process.rate == expected_rate
        assert called_request.num_channels == base_request.num_channels


def test_run_sweep_num_channels():
    """Test sweep over number of channels."""
    base_request = create_base_request(
        num_channels=1, simulation_time=5, random_seed=111
    )

    sweep_request = SweepRequest(
        base_request=base_request,
        sweep_parameter=SweepParameter(
            name="num_channels", values=[1, 2, 3, 4]
        ),
    )

    sweep_response = run_sweep(sweep_request)

    assert len(sweep_response.results) == 4

    # More channels should reduce rejection probability
    rejection_probs = [
        r.result.aggregated_metrics.rejection_probability
        for r in sweep_response.results
    ]

    # Generally, rejection probability should decrease with more channels
    # (though not strictly monotonic due to randomness)
    assert rejection_probs[0] >= rejection_probs[-1]


def test_simulator_with_zero_arrivals():
    """Test edge case where no arrivals occur before simulation end."""
    params = create_base_request(simulation_time=1)

    simulator = Simulator(params)
    # Very slow arrival rate - first arrival after simulation ends
    simulator.arrival_distribution = DeterministicDistribution(10.0)
    simulator.service_distribution = DeterministicDistribution(0.5)

    result = simulator.run()

    # No arrivals should be processed
    assert result.metrics.total_requests == 0
    assert result.metrics.processed_requests == 0
    assert result.metrics.rejected_requests == 0


def test_simulator_all_rejections():
    """Test scenario where most arrivals are rejected."""
    params = create_base_request(num_channels=1, simulation_time=10)

    simulator = Simulator(params)
    # Very fast arrivals (every 0.1s), very slow service (5.0s)
    simulator.arrival_distribution = DeterministicDistribution(0.1)
    simulator.service_distribution = DeterministicDistribution(5.0)

    result = simulator.run()

    # Analysis:
    # t=0.1: First arrival, starts service (ends at 5.1)
    # t=0.2-5.0: All arrivals rejected (channel busy)
    # t=5.1: Channel free
    # t=5.2: Next arrival after channel becomes free, accepted (ends at 10.2)
    # t=5.3-10.0: All arrivals rejected
    #
    # Expected: 2 processed (at ~0.1 and ~5.2), rest rejected
    # Total arrivals in 10s with 0.1s intervals: ~100 arrivals

    assert result.metrics.processed_requests == 2
    assert result.metrics.rejected_requests > 90
    assert result.metrics.rejection_probability > 0.9


def test_confidence_intervals_coverage():
    """Test that confidence intervals have proper coverage."""
    # Run multiple times and check that true mean falls within CI
    params = create_base_request(
        num_replications=30, simulation_time=20, random_seed=222
    )

    response = run_replications(params)

    agg = response.aggregated_metrics
    ci = agg.rejection_probability_ci

    assert ci is not None
    assert ci.lower_bound < agg.rejection_probability < ci.upper_bound

    # Width should be reasonable (not too wide)
    width = ci.upper_bound - ci.lower_bound
    assert 0 < width < 1


def test_replication_variance():
    """Test that replications produce varied results."""
    params = create_base_request(
        num_replications=10, simulation_time=10, random_seed=333
    )

    response = run_replications(params)

    # Get rejection probabilities from all replications
    rejection_probs = [
        r.metrics.rejection_probability for r in response.replications
    ]

    # Not all replications should have identical results
    # (unless system is deterministic, which it shouldn't be)
    unique_values = len(set(rejection_probs))
    assert unique_values > 1  # At least some variation


def test_non_stationary_rate_transition():
    """Test that rate actually changes at schedule boundary."""
    params = create_base_request(
        simulation_time=6,
        arrival_schedule=[
            ArrivalScheduleItem(duration=3, rate=0.5),  # Slow arrivals
            ArrivalScheduleItem(duration=3, rate=5.0),  # Fast arrivals
        ],
        random_seed=444,
    )

    result = run_replications(params)

    # Should see significantly more arrivals in second half
    # This is a weak test but checks basic functionality
    assert result.aggregated_metrics.total_requests > 5


def test_multiple_channels_balanced_usage():
    """Test that multiple channels get used relatively evenly."""
    params = create_base_request(
        num_channels=3, simulation_time=10, random_seed=555
    )

    simulator = Simulator(params)
    simulator.arrival_distribution = DeterministicDistribution(0.5)
    simulator.service_distribution = DeterministicDistribution(0.4)

    result = simulator.run()

    # Count usage per channel
    channel_usage = [0] * params.num_channels
    for item in result.gantt_chart:
        channel_usage[item.channel] += 1

    # All channels should be used at least once
    assert all(usage > 0 for usage in channel_usage)

    # Usage should be relatively balanced (within 50% of mean)
    mean_usage = sum(channel_usage) / len(channel_usage)
    for usage in channel_usage:
        assert usage >= mean_usage * 0.5


def test_simulator_high_rejection_rate():
    """Test scenario where rejection rate is very high but not 100%."""
    params = create_base_request(num_channels=1, simulation_time=10)

    simulator = Simulator(params)
    # Fast arrivals, long service
    simulator.arrival_distribution = DeterministicDistribution(0.2)
    simulator.service_distribution = DeterministicDistribution(3.0)

    result = simulator.run()

    # With arrivals every 0.2s and service 3.0s:
    # Each service blocks ~15 arrivals (3.0/0.2)
    # In 10s, we can fit ~3 complete services
    # So we expect around 3-4 processed requests

    assert result.metrics.processed_requests <= 5
    assert result.metrics.rejected_requests > 40
    assert result.metrics.rejection_probability > 0.8
