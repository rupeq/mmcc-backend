from dataclasses import dataclass, field

from src.simulations.core.schemas import GanttChartItem


@dataclass
class SimulationStatistics:
    """Collect simulation statistics efficiently"""

    total_requests: int = 0
    processed_requests: int = 0
    rejected_requests: int = 0
    total_busy_time: float = 0.0
    service_times: list[float] = field(default_factory=list)
    gantt_items: list[GanttChartItem] = field(default_factory=list)

    # Online statistics (Welford's algorithm for variance)
    _count: int = 0
    _mean: float = 0.0
    _m2: float = 0.0

    def record_service_time(self, service_time: float):
        """Record service time with online mean/variance calculation"""

        self._count += 1
        delta = service_time - self._mean
        self._mean += delta / self._count
        delta2 = service_time - self._mean
        self._m2 += delta * delta2

        # Optional: only keep raw times if needed
        if len(self.service_times) < 1000:  # Limit memory
            self.service_times.append(service_time)

    @property
    def service_time_std(self) -> float:
        if self._count < 2:
            return 0.0
        return (self._m2 / (self._count - 1)) ** 0.5
