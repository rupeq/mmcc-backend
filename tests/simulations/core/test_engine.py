from src.simulations.core.engine import Simulator, run_replications, run_sweep
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
    """Create a default SimulationRequest that can be overridden."""
    defaults = {
        "num_channels": 1,
        "simulation_time": 10,
        "num_replications": 1,
        "arrival_process": ExponentialParams(rate=1.0),
        "service_process": ExponentialParams(rate=2.0),
    }
    defaults.update(kwargs)
    return SimulationRequest.model_validate(defaults)


def test_simulator_deterministic_run(mocker):
    """Test a simple, predictable scenario with mocked random generators."""
    arrival_mock = mocker.patch(
        "src.simulations.core.engine.get_generator", return_value=lambda: 1.5
    )
    service_mock = mocker.patch(
        "src.simulations.core.distributions.get_generator",
        return_value=lambda: 0.5,
    )

    params = create_base_request(simulation_time=4)
    simulator = Simulator(params)

    simulator.arrival_generator = arrival_mock.return_value
    simulator.service_generator = service_mock.return_value

    result = simulator.run()

    # t=0.0: First arrival scheduled for t=1.5
    # t=1.5: Arrival 1, service starts (duration 0.5), next arrival at t=3.0
    # t=3.0: Arrival 2, service starts (duration 0.5), next arrival at t=4.5
    # t=4.0: Simulation ends. Only 2 arrivals should have been fully processed.
    assert result.metrics.total_requests == 2
    assert result.metrics.processed_requests == 2
    assert result.metrics.rejected_requests == 0


def test_rejection_logic(mocker):
    """Test that a request is rejected if all channels are busy."""
    arrival_mock = mocker.patch(
        "src.simulations.core.engine.get_generator", return_value=lambda: 1.0
    )
    service_mock = mocker.patch(
        "src.simulations.core.distributions.get_generator",
        return_value=lambda: 1.5,
    )

    params = create_base_request(num_channels=1, simulation_time=2.5)
    simulator = Simulator(params)
    simulator.arrival_generator = arrival_mock.return_value
    simulator.service_generator = service_mock.return_value

    result = simulator.run()

    # t=0.0: Schedule first arrival at t=1.0
    # t=1.0: Arrival 1 arrives, starts service (duration 1.5). Channel busy until 2.5. Next arrival scheduled at t=2.0.
    # t=2.0: Arrival 2 arrives. Channel is busy (until 2.5). Arrival 2 is REJECTED. Next arrival scheduled at t=3.0.
    # t=2.5: Simulation ends.
    assert result.metrics.total_requests == 2
    assert result.metrics.processed_requests == 1
    assert result.metrics.rejected_requests == 1


def test_multi_channel_logic(mocker):
    """Test that a second channel prevents rejection."""
    arrival_mock = mocker.patch(
        "src.simulations.core.engine.get_generator",
        return_value=lambda: 1.0,
    )
    service_mock = mocker.patch(
        "src.simulations.core.distributions.get_generator",
        return_value=lambda: 1.5,
    )

    params = create_base_request(num_channels=2, simulation_time=2.5)
    simulator = Simulator(params)
    simulator.arrival_generator = arrival_mock.return_value
    simulator.service_generator = service_mock.return_value

    result = simulator.run()

    # t=1.0: Arrival 1 -> Channel 1 (busy until 2.5)
    # t=2.0: Arrival 2 -> Channel 2 is free! (busy until 3.5)
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


def test_non_stationary_flow_deterministic(mocker):
    """Test that the arrival rate changes according to the schedule."""
    mocker.patch("numpy.random.exponential", side_effect=lambda scale: scale)

    params = create_base_request(
        simulation_time=8,
        arrival_schedule=[
            ArrivalScheduleItem(duration=5, rate=1.0),  # Arrivals every 1s
            ArrivalScheduleItem(duration=5, rate=4.0),  # Arrivals every 0.25s
        ],
    )
    simulator = Simulator(params)
    result = simulator.run()

    # t=0 to t=5 (rate=1): arrivals at t=0, 1, 2, 3, 4 (5 arrivals)
    # t=5 to t=8 (rate=4): arrivals at t=5, 5.25, 5.5, ... 7.75 (12 arrivals)
    # Total arrivals = 5 + 12 = 17
    assert result.metrics.total_requests == 17


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
        sweep_parameter=SweepParameter(  # type:ignore
            name="arrival_process.rate", values=sweep_values
        ),
    )  # type:ignore

    sweep_response = run_sweep(sweep_request)

    assert mock_runner.call_count == len(sweep_values)
    assert len(sweep_response.results) == len(sweep_values)

    for i, call in enumerate(mock_runner.call_args_list):
        called_request = call[0][0]
        expected_rate = sweep_values[i]

        assert isinstance(called_request, SimulationRequest)
        assert called_request.arrival_process.rate == expected_rate
        assert called_request.num_channels == base_request.num_channels
