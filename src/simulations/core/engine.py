import heapq
import math
from typing import Callable

from scipy import stats as scipy_stats
import numpy as np

from src.simulations.core.distributions import get_generator
from src.simulations.core.enums import EventType
from src.simulations.core.schemas import (
    SimulationRequest,
    SimulationMetrics,
    GanttChartItem,
    SimulationResult,
    SimulationResponse,
    ConfidenceInterval,
    AggregatedMetrics,
)


class Simulator:
    """Run a single G/G/c/c queuing system simulation."""

    def __init__(self, params: SimulationRequest):
        """Initialize the simulator with given parameters.

        Args:
            params: A Pydantic model containing all simulation settings.
        """
        self.params = params

        if params.random_seed is not None:
            np.random.seed(params.random_seed)

        self.clock = 0.0
        self.event_queue: list = []
        self.channels_free_time = [0.0] * params.num_channels

        self.processed_schedule = None
        if self.params.arrival_schedule:
            self.processed_schedule = []
            current_time = 0.0
            for item in self.params.arrival_schedule:
                current_time += item.duration
                self.processed_schedule.append(
                    {"end_time": current_time, "rate": item.rate}
                )

        self.arrival_generator = self._get_arrival_generator()
        self.service_generator = get_generator(params.service_process)

        self.stats = {
            "total_requests": 0,
            "processed_requests": 0,
            "rejected_requests": 0,
            "total_busy_time": 0.0,
        }
        self.gantt_chart_data: list[GanttChartItem] = []

    def _get_arrival_generator(self) -> Callable[[], float]:
        """Create a generator for inter-arrival times.

        Returns:
            A function that, when called, returns the time until the next arrival.
            This function handles both stationary and non-stationary (scheduled)
            arrival processes.
        """
        if self.processed_schedule:

            def non_stationary_generator() -> float:
                """Generate next arrival time based on the current clock."""
                current_rate = None

                for interval in self.processed_schedule:
                    if self.clock < interval["end_time"]:
                        current_rate = interval["rate"]
                        break

                if current_rate is None:
                    current_rate = self.processed_schedule[-1]["rate"]

                return np.random.exponential(1.0 / current_rate)

            return non_stationary_generator

        return get_generator(self.params.arrival_process)

    def run(self) -> SimulationResult:
        """Execute the main simulation loop.

        The loop processes events from the event queue until the simulation
        time is exceeded or the queue is empty.

        Returns:
            An object containing the calculated metrics and Gantt chart data.
        """
        self._schedule_event(self.arrival_generator(), EventType.ARRIVAL)

        while self.event_queue and self.clock < self.params.simulation_time:
            time, event_type, data = heapq.heappop(self.event_queue)

            self.clock = time
            if self.clock > self.params.simulation_time:
                break

            if event_type == EventType.ARRIVAL:
                self._handle_arrival()

            elif event_type == EventType.DEPARTURE:
                pass

        return self._calculate_results()

    def _schedule_event(self, delay: float, event_type: str, data=None):
        """Add a new event to the event queue.

        Args:
            delay: The time from now when the event should occur.
            event_type: The type of the event (e.g., ARRIVAL).
            data: Optional dictionary with event-specific data.
        """
        heapq.heappush(
            self.event_queue, (self.clock + delay, event_type, data or {})
        )

    def _handle_arrival(self):
        """Process an arrival event.

        Checks for an available channel. If found, occupies it and schedules
        a departure. Otherwise, the request is rejected. Also, schedules the
        next arrival.
        """
        self.stats["total_requests"] += 1

        self._schedule_event(self.arrival_generator(), EventType.ARRIVAL)

        earliest_free_time = min(self.channels_free_time)

        if earliest_free_time <= self.clock:
            channel_id = self.channels_free_time.index(earliest_free_time)

            self.stats["processed_requests"] += 1
            service_time = self.service_generator()
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
            self.stats["rejected_requests"] += 1

    def _calculate_results(self) -> SimulationResult:
        """Calculate and aggregates simulation metrics.

        This method is called after the simulation loop finishes.

        Returns:
            A SimulationResult object with final metrics.
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
            metrics=metrics, gantt_chart=self.gantt_chart_data
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
