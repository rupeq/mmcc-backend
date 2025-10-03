import math

from src.simulations.core.engine import run_replications
from src.simulations.core.schemas import ExponentialParams, SimulationRequest


def test_mm_c_c_matches_erlang_b():
    """Validate simulator against theoretical results"""
    lambda_rate = 5.0
    mu_rate = 10.0
    c = 2
    rho = lambda_rate / mu_rate

    def erlang_b(rho: float, c: int) -> float:
        numerator = (rho**c) / math.factorial(c)
        denominator = sum((rho**k) / math.factorial(k) for k in range(c + 1))
        return numerator / denominator

    theoretical_blocking = erlang_b(rho, c)

    params = SimulationRequest(
        num_channels=c,
        simulation_time=10000,
        num_replications=30,
        arrival_process=ExponentialParams(rate=lambda_rate),
        service_process=ExponentialParams(rate=mu_rate),
        random_seed=42,
    )

    response = run_replications(params)
    simulated_blocking = response.aggregated_metrics.rejection_probability
    ci = response.aggregated_metrics.rejection_probability_ci

    assert ci.lower_bound <= theoretical_blocking <= ci.upper_bound, (
        f"Theoretical Erlang B ({theoretical_blocking:.4f}) outside "
        f"simulation CI [{ci.lower_bound:.4f}, {ci.upper_bound:.4f}]"
    )

    relative_error = (
        abs(simulated_blocking - theoretical_blocking) / theoretical_blocking
    )
    assert relative_error < 0.05, f"Relative error {relative_error:.2%} > 5%"
