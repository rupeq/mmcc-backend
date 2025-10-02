import heapq
import math

from scipy import stats as scipy_stats
import numpy as np

from src.simulations.core.distributions import get_distribution, Distribution
from src.simulations.core.enums import EventType
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


class Simulator:
    """Discrete-event simulator for G/G/c/c queuing system.

    Attributes:
        params: Simulation configuration parameters.
        clock: Current simulation time.
        event_queue: Priority queue of future events.
        channels_free_time: Array tracking when each channel becomes free.
        arrival_distribution: Distribution object for inter-arrival times.
        service_distribution: Distribution object for service times.
        stats: Dictionary collecting simulation statistics.
        gantt_chart_data: List of Gantt chart items for visualization.
        service_times_log: List of all service times for analysis.
    """

    def __init__(self, params: SimulationRequest) -> None:
        """Initialize simulator with given parameters.

        Args:
            params: Configuration parameters for the simulation.
        """
        self.params = params

        # Setup RNG for reproducibility
        if params.random_seed is not None:
            self.rng = np.random.default_rng(params.random_seed)
        else:
            self.rng = np.random.default_rng()

        self.clock = 0.0
        self.event_queue: list = []
        self.channels_free_time = [0.0] * params.num_channels

        # Process non-stationary schedule if provided
        self.processed_schedule = None
        if self.params.arrival_schedule:
            self.processed_schedule = []
            current_time = 0.0
            for item in self.params.arrival_schedule:
                current_time += item.duration
                self.processed_schedule.append(
                    {"end_time": current_time, "rate": item.rate}
                )

        # Create distribution objects
        self.arrival_distribution = self._get_arrival_distribution()
        self.service_distribution = get_distribution(
            params.service_process, rng=self.rng
        )

        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "processed_requests": 0,
            "rejected_requests": 0,
            "total_busy_time": 0.0,
        }
        self.gantt_chart_data: list[GanttChartItem] = []
        self.service_times_log: list[float] = []

    def _get_arrival_distribution(self) -> Distribution:
        """Create arrival time distribution.

        Returns:
            Distribution object for generating inter-arrival times.

        Note:
            For non-stationary arrivals, returns a wrapper that dynamically
            adjusts the rate based on current simulation time.
        """
        if self.processed_schedule:
            # Non-stationary: return exponential with time-varying rate
            class NonStationaryExponential(Distribution):
                def __init__(
                    self,
                    schedule: list,
                    simulator: "Simulator",
                    rng: np.random.Generator,
                ):
                    super().__init__(rng)
                    self.schedule = schedule
                    self.simulator = simulator

                def generate(self) -> float:
                    """Generate inter-arrival time based on current clock."""
                    current_rate = None
                    for interval in self.schedule:
                        if self.simulator.clock < interval["end_time"]:
                            current_rate = interval["rate"]
                            break

                    if current_rate is None:
                        current_rate = self.schedule[-1]["rate"]

                    return self.rng.exponential(1.0 / current_rate)

                def get_mean(self) -> float:
                    # Average rate across schedule
                    return sum(
                        1.0 / interval["rate"] for interval in self.schedule
                    ) / len(self.schedule)

                def get_params(self) -> dict:
                    return {"schedule": self.schedule}

                def __repr__(self) -> str:
                    return f"NonStationaryExponential(intervals={len(self.schedule)})"

            return NonStationaryExponential(
                self.processed_schedule, self, self.rng
            )

        return get_distribution(self.params.arrival_process, rng=self.rng)

    def run(self) -> SimulationResult:
        """Execute the simulation.

        Returns:
            SimulationResult containing metrics and visualization data.
        """
        self._schedule_event(
            self.arrival_distribution.generate(), EventType.ARRIVAL
        )

        while self.event_queue and self.clock < self.params.simulation_time:
            time, event_type, data = heapq.heappop(self.event_queue)
            self.clock = time

            if self.clock > self.params.simulation_time:
                break

            if event_type == EventType.ARRIVAL:
                self._handle_arrival()
            elif event_type == EventType.DEPARTURE:
                pass  # Departures don't need additional handling

        return self._calculate_results()

    def _schedule_event(
        self, delay: float, event_type: str, data: dict | None = None
    ) -> None:
        """Schedule a future event.

        Args:
            delay: Time delay from current clock.
            event_type: Type of event to schedule.
            data: Optional event data dictionary.
        """
        heapq.heappush(
            self.event_queue, (self.clock + delay, event_type, data or {})
        )

    def _handle_arrival(self) -> None:
        """Process an arrival event."""
        self.stats["total_requests"] += 1

        # Schedule next arrival
        self._schedule_event(
            self.arrival_distribution.generate(), EventType.ARRIVAL
        )

        # Find earliest available channel
        earliest_free_time = min(self.channels_free_time)

        if earliest_free_time <= self.clock:
            # Channel available - accept request
            channel_id = self.channels_free_time.index(earliest_free_time)
            self.stats["processed_requests"] += 1

            service_time = self.service_distribution.generate()
            self.service_times_log.append(service_time)

            departure_time = self.clock + service_time
            self.stats["total_busy_time"] += service_time

            self.channels_free_time[channel_id] = departure_time

            self._schedule_event(
                service_time, EventType.DEPARTURE, {"channel_id": channel_id}
            )

            self.gantt_chart_data.append(
                GanttChartItem(
                    channel=channel_id,
                    start=self.clock,
                    end=departure_time,
                    duration=service_time,
                )
            )
        else:
            # All channels busy - reject
            self.stats["rejected_requests"] += 1

    def _calculate_results(self) -> SimulationResult:
        """Calculate final simulation metrics.

        Returns:
            SimulationResult with computed metrics.
        """
        total_time = self.params.simulation_time
        total_channel_time = self.params.num_channels * total_time

        metrics = SimulationMetrics(
            total_requests=self.stats["total_requests"],
            processed_requests=self.stats["processed_requests"],
            rejected_requests=self.stats["rejected_requests"],
            rejection_probability=(
                self.stats["rejected_requests"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0
                else 0
            ),
            avg_channel_utilization=(
                self.stats["total_busy_time"] / total_channel_time
                if total_channel_time > 0
                else 0
            ),
            throughput=self.stats["processed_requests"] / total_time,
        )

        return SimulationResult(
            metrics=metrics,
            gantt_chart=self.gantt_chart_data,
            raw_service_times=self.service_times_log,
        )


def run_replications(params: SimulationRequest) -> SimulationResponse:
    """Run multiple simulation replications and aggregates results.

    Args:
        params: The parameters for the simulation series.

    Returns:
        A response object containing aggregated metrics and results from
        each individual replication.
    """
    replication_results = []
    for i in range(params.num_replications):
        if params.random_seed is not None:
            sim_params = params.model_copy(
                update={"random_seed": params.random_seed + i}, deep=True
            )
        else:
            sim_params = params
        simulator = Simulator(sim_params)
        replication_results.append(simulator.run())

    num_reps = len(replication_results)

    if num_reps < 2:
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
        return SimulationResponse(
            aggregated_metrics=agg_metrics,
            replications=replication_results,
        )

    rejection_probs = [
        r.metrics.rejection_probability for r in replication_results
    ]
    utilizations = [
        r.metrics.avg_channel_utilization for r in replication_results
    ]
    throughputs = [r.metrics.throughput for r in replication_results]

    mean_rejection_prob = np.mean(rejection_probs)
    mean_utilization = np.mean(utilizations)
    mean_throughput = np.mean(throughputs)

    std_rejection_prob = float(np.std(rejection_probs, ddof=1))
    std_utilization = float(np.std(utilizations, ddof=1))
    std_throughput = float(np.std(throughputs, ddof=1))

    # level = 95%
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

    return SimulationResponse(
        aggregated_metrics=aggregated_metrics,
        replications=replication_results,
    )


def run_sweep(params: SweepRequest) -> SweepResponse:
    """Run a series of simulations (a parameter sweep) by varying a single parameter.

    Iterates through a predefined list of values for a specified parameter,
    running a full simulation (including replications) for each value.
    The aggregated results for each parameter value are then collected.
    Supports nested parameters using dot notation (e.g., 'arrival_process.rate').

    Args:
        params (SweepRequest): A `SweepRequest` object containing the base
            simulation parameters and the definition of the parameter to sweep.

    Returns:
        SweepResponse: A `SweepResponse` object containing a list of
            `SweepResultItem`s, where each item holds the parameter value
            and the corresponding `SimulationResponse`.
    """
    sweep_results = []
    param_name = params.sweep_parameter.name

    for value in params.sweep_parameter.values:
        request_for_value = params.base_request.model_copy(deep=True)

        set_nested_attr(request_for_value, param_name, value)

        simulation_response = run_replications(request_for_value)

        sweep_results.append(
            SweepResultItem(parameter_value=value, result=simulation_response)
        )

    return SweepResponse(results=sweep_results)
