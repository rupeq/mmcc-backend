import logging
from typing import Literal
from dataclasses import dataclass

from src.simulations.core.engine import run_replications
from src.simulations.core.schemas import (
    SimulationRequest,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of channel optimization"""

    optimal_channels: int
    achieved_rejection_prob: float
    achieved_utilization: float
    throughput: float
    total_cost: float | None = None
    iterations: int = 0
    convergence_history: list[dict] | None = None


@dataclass
class CostFunction:
    """Cost function for optimization"""

    channel_cost: float = 1.0
    rejection_penalty: float = 100.0

    def calculate(
        self,
        num_channels: int,
        rejection_prob: float,
        total_requests: float,
    ) -> float:
        """
        Calculate total cost = channel_cost * channels + rejection_penalty * rejected_requests
        """
        rejected_requests = rejection_prob * total_requests
        return (self.channel_cost * num_channels) + (
            self.rejection_penalty * rejected_requests
        )


def binary_search_channels(
    base_request: SimulationRequest,
    target_rejection_prob: float,
    max_channels: int = 100,
    tolerance: float = 0.01,
    min_channels: int = 1,
) -> OptimizationResult:
    """
    Find minimum channels to meet rejection probability target using binary search.

    Args:
        base_request: Base simulation parameters
        target_rejection_prob: Target maximum rejection probability
        max_channels: Maximum number of channels to consider
        tolerance: Acceptable tolerance above target
        min_channels: Minimum number of channels to start with

    Returns:
        OptimizationResult with optimal configuration

    Example:
        >>> request = SimulationRequest(
        ...     num_channels=1,
        ...     simulation_time=100,
        ...     num_replications=10,
        ...     arrival_process=ExponentialParams(rate=5.0),
        ...     service_process=ExponentialParams(rate=10.0),
        ... )
        >>> result = binary_search_channels(
        ...     base_request=request,
        ...     target_rejection_prob=0.05,
        ...     max_channels=20,
        ... )
        >>> print(f"Optimal channels: {result.optimal_channels}")
    """
    logger.info(
        "Starting binary search optimization: target_rejection=%.4f, "
        "max_channels=%d, tolerance=%.4f",
        target_rejection_prob,
        max_channels,
        tolerance,
    )

    left, right = min_channels, max_channels
    best_channels = right
    best_rejection = 1.0
    best_response = None
    iterations = 0
    history = []

    while left <= right:
        mid = (left + right) // 2
        iterations += 1

        # Test with mid channels
        test_request = base_request.model_copy(
            update={"num_channels": mid}, deep=True
        )
        response = run_replications(test_request)
        rejection_prob = response.aggregated_metrics.rejection_probability

        history.append(
            {
                "iteration": iterations,
                "channels": mid,
                "rejection_prob": rejection_prob,
                "utilization": response.aggregated_metrics.avg_channel_utilization,
            }
        )

        logger.debug(
            "Iteration %d: channels=%d, rejection=%.4f",
            iterations,
            mid,
            rejection_prob,
        )

        if rejection_prob <= target_rejection_prob + tolerance:
            # Meets target, try fewer channels
            best_channels = mid
            best_rejection = rejection_prob
            best_response = response
            right = mid - 1
        else:
            # Need more channels
            left = mid + 1

    if best_response is None:
        # Fallback: use max_channels
        logger.warning(
            "Could not meet target with %d channels, using maximum",
            max_channels,
        )
        test_request = base_request.model_copy(
            update={"num_channels": max_channels}, deep=True
        )
        best_response = run_replications(test_request)
        best_channels = max_channels
        best_rejection = best_response.aggregated_metrics.rejection_probability

    logger.info(
        "Binary search complete: optimal_channels=%d, rejection=%.4f, "
        "iterations=%d",
        best_channels,
        best_rejection,
        iterations,
    )

    return OptimizationResult(
        optimal_channels=best_channels,
        achieved_rejection_prob=best_rejection,
        achieved_utilization=best_response.aggregated_metrics.avg_channel_utilization,
        throughput=best_response.aggregated_metrics.throughput,
        iterations=iterations,
        convergence_history=history,
    )


def minimize_cost(
    base_request: SimulationRequest,
    cost_function: CostFunction,
    max_channels: int = 50,
    min_channels: int = 1,
) -> OptimizationResult:
    """
    Find optimal number of channels by minimizing total cost.

    Cost = channel_cost * num_channels + rejection_penalty * rejected_requests

    Args:
        base_request: Base simulation parameters
        cost_function: Cost function with channel cost and rejection penalty
        max_channels: Maximum channels to evaluate
        min_channels: Minimum channels to start with

    Returns:
        OptimizationResult with cost-optimal configuration

    Example:
        >>> cost_fn = CostFunction(channel_cost=10.0, rejection_penalty=100.0)
        >>> result = minimize_cost(
        ...     base_request=request,
        ...     cost_function=cost_fn,
        ...     max_channels=20,
        ... )
    """
    logger.info(
        "Starting cost minimization: channel_cost=%.2f, rejection_penalty=%.2f",
        cost_function.channel_cost,
        cost_function.rejection_penalty,
    )

    best_channels = min_channels
    best_cost = float("inf")
    best_response = None
    history = []

    for channels in range(min_channels, max_channels + 1):
        test_request = base_request.model_copy(
            update={"num_channels": channels}, deep=True
        )
        response = run_replications(test_request)

        rejection_prob = response.aggregated_metrics.rejection_probability
        total_requests = response.aggregated_metrics.total_requests

        cost = cost_function.calculate(
            num_channels=channels,
            rejection_prob=rejection_prob,
            total_requests=total_requests,
        )

        history.append(
            {
                "channels": channels,
                "rejection_prob": rejection_prob,
                "cost": cost,
                "utilization": response.aggregated_metrics.avg_channel_utilization,
            }
        )

        logger.debug(
            "Channels=%d: rejection=%.4f, cost=%.2f",
            channels,
            rejection_prob,
            cost,
        )

        if cost < best_cost:
            best_cost = cost
            best_channels = channels
            best_response = response

    logger.info(
        "Cost minimization complete: optimal_channels=%d, min_cost=%.2f",
        best_channels,
        best_cost,
    )

    return OptimizationResult(
        optimal_channels=best_channels,
        achieved_rejection_prob=best_response.aggregated_metrics.rejection_probability,
        achieved_utilization=best_response.aggregated_metrics.avg_channel_utilization,
        throughput=best_response.aggregated_metrics.throughput,
        total_cost=best_cost,
        iterations=max_channels - min_channels + 1,
        convergence_history=history,
    )


def gradient_descent_channels(
    base_request: SimulationRequest,
    objective: Literal["rejection", "cost", "utilization"] = "rejection",
    cost_function: CostFunction | None = None,
    initial_channels: int | None = None,
    max_iterations: int = 20,
    step_size: int = 2,
    tolerance: float = 0.001,
) -> OptimizationResult:
    """
    Optimize channels using gradient descent approach.

    Args:
        base_request: Base simulation parameters
        objective: Optimization objective ("rejection", "cost", "utilization")
        cost_function: Cost function (required if objective="cost")
        initial_channels: Starting point (default: half of max expected)
        max_iterations: Maximum optimization iterations
        step_size: Step size for gradient estimation
        tolerance: Convergence tolerance

    Returns:
        OptimizationResult with optimized configuration
    """
    if objective == "cost" and cost_function is None:
        raise ValueError("cost_function required when objective='cost'")

    if initial_channels is None:
        # Estimate using Erlang-B as starting point
        initial_channels = max(
            1,
            int(
                base_request.arrival_process.rate
                / base_request.service_process.rate
                * 1.5
            ),
        )

    logger.info(
        "Starting gradient descent: objective=%s, initial_channels=%d",
        objective,
        initial_channels,
    )

    current_channels = initial_channels
    history = []

    for iteration in range(max_iterations):
        # Evaluate at current point
        current_response = run_replications(
            base_request.model_copy(
                update={"num_channels": current_channels}, deep=True
            )
        )

        # Evaluate at current + step_size
        next_channels = current_channels + step_size
        next_response = run_replications(
            base_request.model_copy(
                update={"num_channels": next_channels}, deep=True
            )
        )

        # Calculate objective values
        if objective == "rejection":
            current_obj = (
                current_response.aggregated_metrics.rejection_probability
            )
            next_obj = next_response.aggregated_metrics.rejection_probability
        elif objective == "utilization":
            # Maximize utilization = minimize (1 - utilization)
            current_obj = (
                1.0
                - current_response.aggregated_metrics.avg_channel_utilization
            )
            next_obj = (
                1.0 - next_response.aggregated_metrics.avg_channel_utilization
            )
        else:  # cost
            current_obj = cost_function.calculate(
                current_channels,
                current_response.aggregated_metrics.rejection_probability,
                current_response.aggregated_metrics.total_requests,
            )
            next_obj = cost_function.calculate(
                next_channels,
                next_response.aggregated_metrics.rejection_probability,
                next_response.aggregated_metrics.total_requests,
            )

        gradient = (next_obj - current_obj) / step_size

        history.append(
            {
                "iteration": iteration,
                "channels": current_channels,
                "objective_value": current_obj,
                "gradient": gradient,
            }
        )

        logger.debug(
            "Iteration %d: channels=%d, objective=%.4f, gradient=%.4f",
            iteration,
            current_channels,
            current_obj,
            gradient,
        )

        # Check convergence
        if abs(gradient) < tolerance:
            logger.info("Converged at iteration %d", iteration)
            break

        # Update (move against gradient to minimize)
        if gradient < 0:
            # Objective decreasing, continue in same direction
            current_channels = next_channels
        else:
            # Objective increasing, move in opposite direction
            current_channels = max(1, current_channels - step_size)

    # Final evaluation
    final_response = run_replications(
        base_request.model_copy(
            update={"num_channels": current_channels}, deep=True
        )
    )

    final_cost = None
    if objective == "cost":
        final_cost = cost_function.calculate(
            current_channels,
            final_response.aggregated_metrics.rejection_probability,
            final_response.aggregated_metrics.total_requests,
        )

    logger.info(
        "Gradient descent complete: optimal_channels=%d, iterations=%d",
        current_channels,
        len(history),
    )

    return OptimizationResult(
        optimal_channels=current_channels,
        achieved_rejection_prob=final_response.aggregated_metrics.rejection_probability,
        achieved_utilization=final_response.aggregated_metrics.avg_channel_utilization,
        throughput=final_response.aggregated_metrics.throughput,
        total_cost=final_cost,
        iterations=len(history),
        convergence_history=history,
    )


def erlang_b_estimate(
    arrival_rate: float,
    service_rate: float,
    target_rejection_prob: float,
    max_channels: int = 100,
) -> int:
    """
    Estimate required channels using Erlang B formula (M/M/c/c).

    This provides a theoretical starting point for optimization.

    Args:
        arrival_rate: Arrival rate (λ)
        service_rate: Service rate (μ)
        target_rejection_prob: Target blocking probability
        max_channels: Maximum channels to consider

    Returns:
        Estimated number of channels
    """
    traffic_intensity = arrival_rate / service_rate  # ρ = λ/μ

    logger.debug(
        "Erlang B estimation: λ=%.2f, μ=%.2f, ρ=%.2f, target=%.4f",
        arrival_rate,
        service_rate,
        traffic_intensity,
        target_rejection_prob,
    )

    for c in range(1, max_channels + 1):
        # Calculate Erlang B formula iteratively
        erlang_b = 1.0
        for k in range(1, c + 1):
            erlang_b = (traffic_intensity * erlang_b) / (
                k + traffic_intensity * erlang_b
            )

        if erlang_b <= target_rejection_prob:
            logger.debug(
                "Erlang B estimate: %d channels for B=%.4f",
                c,
                erlang_b,
            )
            return c

    logger.warning(
        "Erlang B could not meet target with %d channels",
        max_channels,
    )
    return max_channels


def multi_objective_optimization(
    base_request: SimulationRequest,
    rejection_weight: float = 0.5,
    utilization_weight: float = 0.3,
    cost_weight: float = 0.2,
    channel_cost: float = 1.0,
    rejection_penalty: float = 100.0,
    max_channels: int = 50,
    min_channels: int = 1,
) -> OptimizationResult:
    """
    Multi-objective optimization balancing rejection, utilization, and cost.

    Objective = w1*rejection + w2*(1-utilization) + w3*normalized_cost

    Args:
        base_request: Base simulation parameters
        rejection_weight: Weight for rejection probability (0-1)
        utilization_weight: Weight for utilization (0-1)
        cost_weight: Weight for cost (0-1)
        channel_cost: Cost per channel
        rejection_penalty: Penalty per rejected request
        max_channels: Maximum channels to evaluate
        min_channels: Minimum channels to start with

    Returns:
        OptimizationResult with Pareto-optimal configuration
    """
    weights_sum = rejection_weight + utilization_weight + cost_weight
    if abs(weights_sum - 1.0) > 0.01:
        logger.warning(
            "Weights sum to %.2f, normalizing to 1.0",
            weights_sum,
        )
        rejection_weight /= weights_sum
        utilization_weight /= weights_sum
        cost_weight /= weights_sum

    logger.info(
        "Multi-objective optimization: w_rejection=%.2f, w_util=%.2f, w_cost=%.2f",
        rejection_weight,
        utilization_weight,
        cost_weight,
    )

    best_channels = min_channels
    best_objective = float("inf")
    best_response = None
    history = []

    # First pass: collect data for normalization
    results = []
    for channels in range(min_channels, max_channels + 1):
        test_request = base_request.model_copy(
            update={"num_channels": channels}, deep=True
        )
        response = run_replications(test_request)
        results.append((channels, response))

    # Normalize factors
    rejections = [
        r.aggregated_metrics.rejection_probability for _, r in results
    ]
    utilizations = [
        r.aggregated_metrics.avg_channel_utilization for _, r in results
    ]
    costs = [
        channel_cost * c
        + rejection_penalty
        * r.aggregated_metrics.rejection_probability
        * r.aggregated_metrics.total_requests
        for c, r in results
    ]

    max_rejection = max(rejections) if rejections else 1.0
    max_cost = max(costs) if costs else 1.0

    # Second pass: evaluate normalized objectives
    for channels, response in results:
        rejection_norm = (
            response.aggregated_metrics.rejection_probability / max_rejection
            if max_rejection > 0
            else 0
        )
        utilization_norm = (
            1.0 - response.aggregated_metrics.avg_channel_utilization
        )
        cost_raw = (
            channel_cost * channels
            + rejection_penalty
            * response.aggregated_metrics.rejection_probability
            * response.aggregated_metrics.total_requests
        )
        cost_norm = cost_raw / max_cost if max_cost > 0 else 0

        objective = (
            rejection_weight * rejection_norm
            + utilization_weight * utilization_norm
            + cost_weight * cost_norm
        )

        history.append(
            {
                "channels": channels,
                "objective": objective,
                "rejection_prob": response.aggregated_metrics.rejection_probability,
                "utilization": response.aggregated_metrics.avg_channel_utilization,
                "cost": cost_raw,
            }
        )

        logger.debug(
            "Channels=%d: objective=%.4f (rej=%.4f, util=%.4f, cost=%.2f)",
            channels,
            objective,
            rejection_norm,
            utilization_norm,
            cost_norm,
        )

        if objective < best_objective:
            best_objective = objective
            best_channels = channels
            best_response = response

    best_cost = (
        channel_cost * best_channels
        + rejection_penalty
        * best_response.aggregated_metrics.rejection_probability
        * best_response.aggregated_metrics.total_requests
    )

    logger.info(
        "Multi-objective optimization complete: optimal_channels=%d, "
        "objective=%.4f",
        best_channels,
        best_objective,
    )

    return OptimizationResult(
        optimal_channels=best_channels,
        achieved_rejection_prob=best_response.aggregated_metrics.rejection_probability,
        achieved_utilization=best_response.aggregated_metrics.avg_channel_utilization,
        throughput=best_response.aggregated_metrics.throughput,
        total_cost=best_cost,
        iterations=len(results),
        convergence_history=history,
    )
