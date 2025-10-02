import heapq
import math
import logging

from scipy import stats as scipy_stats
import numpy as np

from src.simulations.core.distributions import get_distribution, Distribution
from src.simulations.core.enums import EventType
from src.simulations.core.events import Event
from src.simulations.core.statistics import SimulationStatistics
from src.simulations.core.schemas import (
    SimulationRequest,
    SimulationMetrics,
    GanttChartItem,
    SimulationResult,
    SimulationResponse,
    ConfidenceInterval,
    AggregatedMetrics,
    SweepResultItem,
    SweepResponse,
    SweepRequest,
)
from src.simulations.core.utils import set_nested_attr

logger = logging.getLogger(__name__)


class ChannelManager:
    """Manage channel availability and allocation for the G/G/c/c system"""

    def __init__(self, num_channels: int):
        if num_channels <= 0:
            raise ValueError(
                f"Number of channels must be positive, got {num_channels}"
            )
        self.num_channels = num_channels
        self.channels_free_time = [0.0] * num_channels

    def get_available_channel(self, current_time: float) -> int | None:
        """
        Find the first available channel at current_time.

        Returns:
            Channel ID if available, None if all busy.
        """
        earliest_free_time = min(self.channels_free_time)

        if earliest_free_time <= current_time:
            return self.channels_free_time.index(earliest_free_time)

        return None

    def allocate_channel(self, channel_id: int, free_at_time: float) -> None:
        """Mark a channel as busy until free_at_time"""
        if not 0 <= channel_id < self.num_channels:
            raise ValueError(f"Invalid channel_id: {channel_id}")
        self.channels_free_time[channel_id] = free_at_time

    def get_utilization_stats(self, simulation_time: float) -> dict[str, float]:
        """Calculate channel utilization statistics"""
        total_channel_time = self.num_channels * simulation_time
        return {
            "total_channel_time": total_channel_time,
            "num_channels": self.num_channels,
        }


class NonStationaryExponential(Distribution):
    """Non-stationary exponential distribution with piecewise-constant rates"""

    def __init__(
        self,
        schedule: list[dict],
        simulator: "Simulator",
        rng: np.random.Generator,
    ):
        super().__init__(rng)
        self.schedule = schedule
        self.simulator = simulator

    def generate(self) -> float:
        """Generate inter-arrival time based on current simulation time"""
        self._generation_count += 1
        current_rate = self._get_current_rate()
        return self.rng.exponential(1.0 / current_rate)

    def _get_current_rate(self) -> float:
        """Get rate for current simulation time"""
        for interval in self.schedule:
            if self.simulator.clock < interval["end_time"]:
                return interval["rate"]

        return self.schedule[-1]["rate"]

    def get_mean(self) -> float:
        """Average mean across all intervals"""
        return sum(1.0 / interval["rate"] for interval in self.schedule) / len(
            self.schedule
        )

    def get_params(self) -> dict:
        return {"schedule": self.schedule, "type": "non_stationary"}

    def __repr__(self) -> str:
        return f"NonStationaryExponential(intervals={len(self.schedule)})"


class Simulator:
    """
    Discrete-Event Simulator for G/G/c/c queuing system (no queue, rejections only).

    This simulator implements a loss system where:
    - Arrivals follow a general distribution (or non-stationary exponential)
    - Service times follow a general distribution
    - c channels are available
    - No waiting queue exists (c = queue capacity)
    - Requests are rejected if all channels are busy
    """

    def __init__(self, params: SimulationRequest) -> None:
        self.params = params
        self._validate_parameters()

        if params.random_seed is not None:
            self.rng = np.random.default_rng(params.random_seed)
        else:
            self.rng = np.random.default_rng()

        self.clock = 0.0
        self.event_queue: list[Event] = []
        self.channels = ChannelManager(params.num_channels)

        self.stats = SimulationStatistics()
        self.stats.configure_limits(
            max_service_samples=params.max_service_time_samples,
            max_gantt_items=params.max_gantt_items,
            collect_service_times=params.collect_service_times,
            collect_gantt_data=params.collect_gantt_data,
        )

        self.arrival_distribution = self._create_arrival_distribution()
        self.service_distribution = get_distribution(
            params.service_process, rng=self.rng
        )

        logger.debug(
            "Initialized simulator: channels=%d, sim_time=%.2f, "
            "collect_gantt=%s (max=%s), collect_service_times=%s (max=%s)",
            params.num_channels,
            params.simulation_time,
            params.collect_gantt_data,
            params.max_gantt_items,
            params.collect_service_times,
            params.max_service_time_samples,
        )

    def _validate_parameters(self) -> None:
        """Validate simulation parameters"""
        if self.params.num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if self.params.simulation_time <= 0:
            raise ValueError("simulation_time must be positive")
        if self.params.num_replications < 1:
            raise ValueError("num_replications must be at least 1")

    def _create_arrival_distribution(self) -> Distribution:
        """Create arrival distribution (stationary or non-stationary)"""
        if self.params.arrival_schedule:
            # Non-stationary: piecewise-constant rates
            processed_schedule = []
            current_time = 0.0

            for item in self.params.arrival_schedule:
                current_time += item.duration
                processed_schedule.append(
                    {"end_time": current_time, "rate": item.rate}
                )

            logger.debug(
                "Using non-stationary arrivals with %d intervals",
                len(processed_schedule),
            )

            return NonStationaryExponential(processed_schedule, self, self.rng)

        # Stationary distribution
        return get_distribution(self.params.arrival_process, rng=self.rng)

    def run(self) -> SimulationResult:
        """
        Execute the discrete-event simulation.

        Returns:
            SimulationResult containing metrics and visualization data
        """
        logger.debug(
            "Starting simulation run (T=%.2f)", self.params.simulation_time
        )

        # Schedule first arrival
        self._schedule_event(
            delay=self.arrival_distribution.generate(),
            event_type=EventType.ARRIVAL,
        )

        events_processed = 0
        while self.event_queue and self.clock < self.params.simulation_time:
            event = heapq.heappop(self.event_queue)
            self.clock = event.time

            if self.clock > self.params.simulation_time:
                break

            if event.event_type == EventType.ARRIVAL:
                self._handle_arrival()
            elif event.event_type == EventType.DEPARTURE:
                self._handle_departure(event.data.get("channel_id"))

            events_processed += 1

        logger.debug(
            "Simulation complete: %d events processed, final_time=%.2f",
            events_processed,
            self.clock,
        )

        return self._calculate_results()

    def _schedule_event(
        self, delay: float, event_type: EventType, data: dict | None = None
    ) -> None:
        """Schedule a future event"""
        event = Event(
            time=self.clock + delay, event_type=event_type, data=data or {}
        )
        heapq.heappush(self.event_queue, event)

    def _handle_arrival(self) -> None:
        """
        Handle customer arrival event.

        G/G/c/c logic:
        1. Check if any channel is free
        2. If yes: allocate channel, schedule departure
        3. If no: reject (loss system - no queue)
        4. Schedule next arrival
        """
        self.stats.total_requests += 1

        # Schedule next arrival
        self._schedule_event(
            delay=self.arrival_distribution.generate(),
            event_type=EventType.ARRIVAL,
        )

        # Try to allocate a channel
        channel_id = self.channels.get_available_channel(self.clock)

        if channel_id is not None:
            # Channel available - process request
            self._process_request(channel_id)
        else:
            # All channels busy - reject
            self._reject_request()

    def _process_request(self, channel_id: int) -> None:
        """Process an accepted request"""
        self.stats.processed_requests += 1
        service_time = self.service_distribution.generate()

        # Record service time (respects limits)
        self.stats.record_service_time(service_time)

        departure_time = self.clock + service_time
        self.channels.allocate_channel(channel_id, departure_time)
        self.stats.total_busy_time += service_time

        self._schedule_event(
            delay=service_time,
            event_type=EventType.DEPARTURE,
            data={"channel_id": channel_id},
        )

        # Record Gantt item (respects limits)
        gantt_item = GanttChartItem(
            channel=channel_id,
            start=self.clock,
            end=departure_time,
            duration=service_time,
        )
        self.stats.record_gantt_item(gantt_item)

        logger.debug(
            "Request accepted: channel=%d, service_time=%.4f, departure=%.4f",
            channel_id,
            service_time,
            departure_time,
        )

        logger.debug(
            "Request accepted: channel=%d, service_time=%.4f, departure=%.4f",
            channel_id,
            service_time,
            departure_time,
        )

    def _reject_request(self) -> None:
        """Handle a rejected request (all channels busy)"""
        self.stats.rejected_requests += 1
        logger.debug(
            "Request rejected: all %d channels busy at t=%.4f",
            self.params.num_channels,
            self.clock,
        )

    def _handle_departure(self, channel_id: int | None) -> None:
        """
        Handle departure event (request completes service).

        Note: In G/G/c/c system, departures just free up channels.
        No queue to process.
        """
        if channel_id is not None:
            logger.debug(
                "Request departed: channel=%d, t=%.4f", channel_id, self.clock
            )
        # Channel is automatically freed by time-based logic

    def _calculate_results(self) -> SimulationResult:
        """Calculate final simulation results"""
        total_time = self.params.simulation_time
        channel_stats = self.channels.get_utilization_stats(total_time)
        total_channel_time = channel_stats["total_channel_time"]

        metrics = SimulationMetrics(
            total_requests=self.stats.total_requests,
            processed_requests=self.stats.processed_requests,
            rejected_requests=self.stats.rejected_requests,
            rejection_probability=(
                self.stats.rejected_requests / self.stats.total_requests
                if self.stats.total_requests > 0
                else 0.0
            ),
            avg_channel_utilization=(
                self.stats.total_busy_time / total_channel_time
                if total_channel_time > 0
                else 0.0
            ),
            throughput=self.stats.processed_requests / total_time,
        )

        # Log collection info
        collection_info = self.stats.get_collection_info()
        logger.debug("Data collection summary: %s", collection_info)

        if self.stats.is_service_times_truncated:
            logger.info(
                "Service time data was truncated at %d samples "
                "(total processed: %d)",
                collection_info["service_times_limit"],
                collection_info["total_service_time_count"],
            )

        if self.stats.is_gantt_truncated:
            logger.info(
                "Gantt chart data was truncated at %d items",
                collection_info["gantt_items_limit"],
            )

        logger.debug(
            "Final metrics: requests=%d, processed=%d, rejected=%d, "
            "rejection_prob=%.4f, utilization=%.4f",
            metrics.total_requests,
            metrics.processed_requests,
            metrics.rejected_requests,
            metrics.rejection_probability,
            metrics.avg_channel_utilization,
        )

        return SimulationResult(
            metrics=metrics,
            gantt_chart=self.stats.gantt_items,
            raw_service_times=self.stats.service_times
            if self.params.collect_service_times
            else None,
        )


def run_replications(params: SimulationRequest) -> SimulationResponse:
    """
    Run multiple replications of the simulation and aggregate results.

    Args:
        params: Simulation configuration

    Returns:
        SimulationResponse with aggregated metrics and individual replications
    """
    logger.info(
        "Running %d replications with seed=%s",
        params.num_replications,
        params.random_seed,
    )

    replication_results = []

    for i in range(params.num_replications):
        # Each replication gets a different seed (if seed is set)
        if params.random_seed is not None:
            sim_params = params.model_copy(
                update={"random_seed": params.random_seed + i}, deep=True
            )
        else:
            sim_params = params

        simulator = Simulator(sim_params)
        result = simulator.run()
        replication_results.append(result)

        logger.debug(
            "Replication %d/%d complete: rejection_prob=%.4f",
            i + 1,
            params.num_replications,
            result.metrics.rejection_probability,
        )

    num_reps = len(replication_results)

    if num_reps < 2:
        # Single replication - no confidence intervals
        metrics = replication_results[0].metrics
        agg_metrics = AggregatedMetrics(
            num_replications=num_reps,
            total_requests=metrics.total_requests,
            processed_requests=metrics.processed_requests,
            rejected_requests=metrics.rejected_requests,
            rejection_probability=metrics.rejection_probability,
            avg_channel_utilization=metrics.avg_channel_utilization,
            throughput=metrics.throughput,
        )

        logger.info("Single replication complete")

        return SimulationResponse(
            aggregated_metrics=agg_metrics,
            replications=replication_results,
        )

    # Multiple replications - calculate statistics
    rejection_probs = [
        r.metrics.rejection_probability for r in replication_results
    ]
    utilizations = [
        r.metrics.avg_channel_utilization for r in replication_results
    ]
    throughputs = [r.metrics.throughput for r in replication_results]

    # Point estimates
    mean_rejection_prob = np.mean(rejection_probs)
    mean_utilization = np.mean(utilizations)
    mean_throughput = np.mean(throughputs)

    # Standard deviations
    std_rejection_prob = float(np.std(rejection_probs, ddof=1))
    std_utilization = float(np.std(utilizations, ddof=1))
    std_throughput = float(np.std(throughputs, ddof=1))

    # Confidence intervals (95% by default)
    confidence_level = 0.95
    degrees_freedom = num_reps - 1
    t_critical = scipy_stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    sqrt_n = math.sqrt(float(num_reps))

    h_rejection = t_critical * (std_rejection_prob / sqrt_n)
    h_utilization = t_critical * (std_utilization / sqrt_n)
    h_throughput = t_critical * (std_throughput / sqrt_n)

    aggregated_metrics = AggregatedMetrics(
        num_replications=num_reps,
        total_requests=float(
            np.mean([r.metrics.total_requests for r in replication_results])
        ),
        processed_requests=float(
            np.mean([r.metrics.processed_requests for r in replication_results])
        ),
        rejected_requests=float(
            np.mean([r.metrics.rejected_requests for r in replication_results])
        ),
        rejection_probability=float(mean_rejection_prob),
        avg_channel_utilization=float(mean_utilization),
        throughput=float(mean_throughput),
        rejection_probability_ci=ConfidenceInterval(
            lower_bound=mean_rejection_prob - h_rejection,
            upper_bound=mean_rejection_prob + h_rejection,
        ),
        avg_channel_utilization_ci=ConfidenceInterval(
            lower_bound=mean_utilization - h_utilization,
            upper_bound=mean_utilization + h_utilization,
        ),
        throughput_ci=ConfidenceInterval(
            lower_bound=mean_throughput - h_throughput,
            upper_bound=mean_throughput + h_throughput,
        ),
    )

    logger.info(
        "Replications complete: mean_rejection=%.4f ± %.4f",
        mean_rejection_prob,
        h_rejection,
    )

    return SimulationResponse(
        aggregated_metrics=aggregated_metrics,
        replications=replication_results,
    )


def run_sweep(params: SweepRequest) -> SweepResponse:
    """
    Run parameter sweep (sensitivity analysis).

    Args:
        params: Sweep configuration with base request and parameter to vary

    Returns:
        SweepResponse containing results for each parameter value
    """
    param_name = params.sweep_parameter.name
    param_values = params.sweep_parameter.values

    logger.info(
        "Running sweep: parameter=%s, values=%s", param_name, param_values
    )

    sweep_results = []

    for value in param_values:
        # Create copy with modified parameter
        request_for_value = params.base_request.model_copy(deep=True)
        set_nested_attr(request_for_value, param_name, value)

        logger.debug("Sweep: %s = %s", param_name, value)

        # Run replications
        simulation_response = run_replications(request_for_value)

        sweep_results.append(
            SweepResultItem(parameter_value=value, result=simulation_response)
        )

    logger.info(
        "Sweep complete: %d parameter values tested", len(sweep_results)
    )

    return SweepResponse(results=sweep_results)
