import pytest
import time

from src.simulations.core.engine import run_replications
from src.simulations.core.schemas import ExponentialParams, SimulationRequest


@pytest.mark.benchmark
def test_simulation_performance():
    """Benchmark simulation engine performance"""
    params = SimulationRequest(
        num_channels=10,
        simulation_time=1000,
        num_replications=1,
        arrival_process=ExponentialParams(rate=50.0),
        service_process=ExponentialParams(rate=100.0),
        collect_gantt_data=False,
        collect_service_times=False,
    )

    start = time.time()
    result = run_replications(params)
    duration = time.time() - start

    events_per_second = result.aggregated_metrics.total_requests / duration

    print("\nPerformance metrics:")
    print("  Simulation time: 1000.0s")
    print(f"  Real time: {duration:.2f}s")
    print(f"  Speedup: {1000.0 / duration:.1f}x")
    print(f"  Events/sec: {events_per_second:.0f}")

    assert events_per_second > 10000, "Performance regression detected"
