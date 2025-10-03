"""Theoretical formulas for queueing systems.

This module provides analytical solutions for comparison with simulation results.
Primarily focuses on M/M/c/c (Erlang B) loss systems.
"""

import logging
from dataclasses import dataclass

from src.simulations.core.enums import DistributionType
from src.simulations.core.schemas import SimulationRequest

logger = logging.getLogger(__name__)


@dataclass
class TheoreticalMetrics:
    """Theoretical performance metrics for comparison."""

    is_applicable: bool
    system_type: str
    traffic_intensity: float | None = None
    blocking_probability: float | None = None
    utilization: float | None = None
    throughput: float | None = None
    mean_service_time: float | None = None
    mean_interarrival_time: float | None = None
    assumptions: list[str] | None = None
    warnings: list[str] | None = None


def calculate_erlang_b(rho: float, c: int) -> float:
    """Calculate Erlang B (blocking probability) for M/M/c/c system.

    Uses iterative formula to avoid factorial overflow:
    B(c, ρ) = [ρ^c / c!] / Σ(k=0 to c)[ρ^k / k!]

    Can be computed iteratively:
    B(0, ρ) = 1
    B(n, ρ) = ρ * B(n-1, ρ) / [n + ρ * B(n-1, ρ)]

    Args:
        rho: Traffic intensity (λ/μ in Erlangs).
        c: Number of channels.

    Returns:
        Blocking probability B(c, ρ).

    Raises:
        ValueError: If parameters are invalid.

    Example:
        >>> erlang_b = calculate_erlang_b(rho=2.0, c=3)
        >>> print(f"Blocking probability: {erlang_b:.4f}")
    """
    if rho < 0:
        raise ValueError(f"Traffic intensity must be non-negative, got {rho}")
    if c < 1:
        raise ValueError(f"Number of channels must be positive, got {c}")

    if rho == 0:
        return 0.0

    # Iterative calculation
    b = 1.0
    for n in range(1, c + 1):
        b = (rho * b) / (n + rho * b)

    logger.debug("Erlang B: ρ=%.4f, c=%d, B=%.6f", rho, c, b)
    return b


def calculate_theoretical_metrics(
    params: SimulationRequest,
) -> TheoreticalMetrics:
    """Calculate theoretical metrics if system is analytically tractable.

    Currently supports:
    - M/M/c/c: Exponential arrivals and service (Erlang B)

    Future support could include:
    - M/G/c/c approximations
    - G/G/c/c bounds

    Args:
        params: Simulation parameters.

    Returns:
        TheoreticalMetrics with analytical predictions.

    Example:
        >>> from src.simulations.core.schemas import SimulationRequest, ExponentialParams
        >>> request = SimulationRequest(
        ...     num_channels=3,
        ...     simulation_time=100,
        ...     arrival_process=ExponentialParams(rate=5.0),
        ...     service_process=ExponentialParams(rate=10.0),
        ... )
        >>> metrics = calculate_theoretical_metrics(request)
        >>> print(f"System: {metrics.system_type}")
        >>> print(f"Theoretical blocking: {metrics.blocking_probability:.4f}")
    """
    assumptions = []
    warnings = []

    # Check if system is M/M/c/c (both exponential)
    is_mm_c_c = (
        params.arrival_process.distribution == DistributionType.EXPONENTIAL
        and params.service_process.distribution == DistributionType.EXPONENTIAL
        and params.arrival_schedule is None  # Stationary arrivals
    )

    if not is_mm_c_c:
        # System is not analytically tractable with current methods
        warnings.append(
            "Non-Markovian system - theoretical results not available. "
            "Consider simulation-only analysis."
        )

        return TheoreticalMetrics(
            is_applicable=False,
            system_type=f"{params.arrival_process.distribution.value}/"
            f"{params.service_process.distribution.value}/"
            f"{params.num_channels}/{params.num_channels}",
            assumptions=[],
            warnings=warnings,
        )

    # M/M/c/c system - apply Erlang B
    lambda_rate = params.arrival_process.rate
    mu_rate = params.service_process.rate
    c = params.num_channels

    # Traffic intensity (offered load in Erlangs)
    rho = lambda_rate / mu_rate

    assumptions.extend(
        [
            "Poisson arrival process (exponential inter-arrival times)",
            "Exponential service times",
            "No queue (immediate rejection if all channels busy)",
            "Stationary arrival rate",
            f"c = {c} channels",
            f"λ = {lambda_rate:.4f} arrivals/time",
            f"μ = {mu_rate:.4f} services/time",
            f"ρ = λ/μ = {rho:.4f} Erlangs",
        ]
    )

    # Calculate Erlang B
    try:
        blocking_prob = calculate_erlang_b(rho, c)
    except (ValueError, OverflowError) as e:
        logger.warning("Failed to calculate Erlang B: %s", e)
        warnings.append(f"Erlang B calculation failed: {str(e)}")
        return TheoreticalMetrics(
            is_applicable=False,
            system_type=f"M/M/{c}/{c}",
            traffic_intensity=rho,
            assumptions=assumptions,
            warnings=warnings,
        )

    # Derived metrics
    utilization = rho * (1 - blocking_prob) / c
    throughput = lambda_rate * (1 - blocking_prob)
    mean_service_time = 1.0 / mu_rate
    mean_interarrival_time = 1.0 / lambda_rate

    # Sanity checks
    if blocking_prob < 0 or blocking_prob > 1:
        warnings.append(
            f"Blocking probability out of range: {blocking_prob:.6f}"
        )

    if utilization < 0 or utilization > 1:
        warnings.append(f"Utilization out of range: {utilization:.6f}")

    # High load warning
    if rho > c:
        warnings.append(
            f"High traffic intensity (ρ={rho:.2f} > c={c}). "
            "System is overloaded - expect high blocking."
        )

    logger.info(
        "M/M/%d/%d theoretical: ρ=%.4f, B=%.6f, util=%.4f",
        c,
        c,
        rho,
        blocking_prob,
        utilization,
    )

    return TheoreticalMetrics(
        is_applicable=True,
        system_type=f"M/M/{c}/{c}",
        traffic_intensity=rho,
        blocking_probability=blocking_prob,
        utilization=utilization,
        throughput=throughput,
        mean_service_time=mean_service_time,
        mean_interarrival_time=mean_interarrival_time,
        assumptions=assumptions,
        warnings=warnings,
    )


def compare_with_theory(
    simulation_metrics: dict,
    theoretical_metrics: TheoreticalMetrics,
) -> dict:
    """Compare simulation results with theoretical predictions.

    Args:
        simulation_metrics: Aggregated simulation metrics.
        theoretical_metrics: Theoretical predictions.

    Returns:
        Dictionary containing comparison results.

    Example:
        >>> comparison = compare_with_theory(sim_metrics, theory_metrics)
        >>> print(f"Blocking error: {comparison['blocking_probability']['relative_error']:.2%}")
    """
    if not theoretical_metrics.is_applicable:
        return {
            "applicable": False,
            "message": "Theoretical comparison not available for this system type",
            "warnings": theoretical_metrics.warnings,
        }

    comparison = {
        "applicable": True,
        "system_type": theoretical_metrics.system_type,
        "traffic_intensity": theoretical_metrics.traffic_intensity,
        "assumptions": theoretical_metrics.assumptions,
        "warnings": theoretical_metrics.warnings,
        "metrics": {},
    }

    # Compare blocking probability
    if theoretical_metrics.blocking_probability is not None:
        sim_blocking = simulation_metrics.get("rejection_probability", 0)
        theory_blocking = theoretical_metrics.blocking_probability

        abs_error = abs(sim_blocking - theory_blocking)
        rel_error = abs_error / theory_blocking if theory_blocking > 0 else 0

        comparison["metrics"]["blocking_probability"] = {
            "simulation": sim_blocking,
            "theoretical": theory_blocking,
            "absolute_error": abs_error,
            "relative_error": rel_error,
            "within_ci": False,  # Updated if CI available
        }

        # Check if theoretical value is within confidence interval
        if "rejection_probability_ci" in simulation_metrics:
            ci = simulation_metrics["rejection_probability_ci"]
            if ci and "lower_bound" in ci and "upper_bound" in ci:
                within_ci = (
                    ci["lower_bound"] <= theory_blocking <= ci["upper_bound"]
                )
                comparison["metrics"]["blocking_probability"]["within_ci"] = (
                    within_ci
                )
                comparison["metrics"]["blocking_probability"]["ci_lower"] = ci[
                    "lower_bound"
                ]
                comparison["metrics"]["blocking_probability"]["ci_upper"] = ci[
                    "upper_bound"
                ]

    # Compare utilization
    if theoretical_metrics.utilization is not None:
        sim_util = simulation_metrics.get("avg_channel_utilization", 0)
        theory_util = theoretical_metrics.utilization

        abs_error = abs(sim_util - theory_util)
        rel_error = abs_error / theory_util if theory_util > 0 else 0

        comparison["metrics"]["utilization"] = {
            "simulation": sim_util,
            "theoretical": theory_util,
            "absolute_error": abs_error,
            "relative_error": rel_error,
            "within_ci": False,
        }

        if "avg_channel_utilization_ci" in simulation_metrics:
            ci = simulation_metrics["avg_channel_utilization_ci"]
            if ci and "lower_bound" in ci and "upper_bound" in ci:
                within_ci = (
                    ci["lower_bound"] <= theory_util <= ci["upper_bound"]
                )
                comparison["metrics"]["utilization"]["within_ci"] = within_ci
                comparison["metrics"]["utilization"]["ci_lower"] = ci[
                    "lower_bound"
                ]
                comparison["metrics"]["utilization"]["ci_upper"] = ci[
                    "upper_bound"
                ]

    # Compare throughput
    if theoretical_metrics.throughput is not None:
        sim_throughput = simulation_metrics.get("throughput", 0)
        theory_throughput = theoretical_metrics.throughput

        abs_error = abs(sim_throughput - theory_throughput)
        rel_error = (
            abs_error / theory_throughput if theory_throughput > 0 else 0
        )

        comparison["metrics"]["throughput"] = {
            "simulation": sim_throughput,
            "theoretical": theory_throughput,
            "absolute_error": abs_error,
            "relative_error": rel_error,
            "within_ci": False,
        }

        if "throughput_ci" in simulation_metrics:
            ci = simulation_metrics["throughput_ci"]
            if ci and "lower_bound" in ci and "upper_bound" in ci:
                within_ci = (
                    ci["lower_bound"] <= theory_throughput <= ci["upper_bound"]
                )
                comparison["metrics"]["throughput"]["within_ci"] = within_ci
                comparison["metrics"]["throughput"]["ci_lower"] = ci[
                    "lower_bound"
                ]
                comparison["metrics"]["throughput"]["ci_upper"] = ci[
                    "upper_bound"
                ]

    return comparison
